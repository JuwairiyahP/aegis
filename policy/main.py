# policy/main.py - AEGIS controller.
# Every POLL_SECONDS: ask the predictor for the risk of an SLO breach. If the risk is high, raise the
# HPA's floor (minReplicas) so new pods start BEFORE the slowdown; after BOOST_TTL seconds without a new
# boost, put the floor back so the HPA handles scale-down as usual.
# It only ever changes the HPA's spec.minReplicas (RBAC allows only get + patch on HPAs). Never the Deployment.
import json
import os
import signal
import sys
import time

import requests
from prometheus_client import Counter, Gauge, start_http_server

RISK_URL = os.getenv("RISK_URL", "http://predictor.aegis.svc/risk")
TAU = os.getenv("TAU", "0.6")                   # a number, or "auto" = use the threshold the predictor reports
STEP = int(os.getenv("STEP", "2"))              # pods to add above the current count on a boost
COOLDOWN = float(os.getenv("COOLDOWN", "60"))   # seconds between two boosts
BOOST_TTL = float(os.getenv("BOOST_TTL", "180"))  # seconds a boost lasts if not renewed
POLL = float(os.getenv("POLL_SECONDS", "10"))
ON_ERROR = os.getenv("ON_ERROR", "release")     # predictor unreachable: "release" boost now, or "hold" until TTL
BASE_FLOOR = int(os.getenv("BASE_FLOOR", "1"))  # the normal minReplicas (hpa.yaml says 1)
NS, HPA = os.getenv("NAMESPACE", "aegis"), os.getenv("HPA_NAME", "demo")
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"      # 1 = don't touch Kubernetes, just print what it would do

SA = "/var/run/secrets/kubernetes.io/serviceaccount"   # token and CA that Kubernetes mounts into the pod
API = os.getenv("K8S_API") or "https://%s:%s" % (os.getenv("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc"),
                                                os.getenv("KUBERNETES_SERVICE_PORT", "443"))
HPA_URL = f"{API}/apis/autoscaling/v2/namespaces/{NS}/horizontalpodautoscalers/{HPA}"

ACTIONS = Counter("aegis_action", "Controller decisions", ["action"])   # exported as aegis_action_total
FLOOR = Gauge("aegis_policy_floor", "minReplicas last set by the controller")
fake_hpa = {"min": BASE_FLOOR, "max": 6, "current": 1}               # used only in DRY_RUN


def log(**kw):
    print(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw}), flush=True)


def k8s_session():
    s = requests.Session()
    if os.path.exists(f"{SA}/token"):
        with open(f"{SA}/token") as f:
            s.headers["Authorization"] = "Bearer " + f.read().strip()
    s.verify = f"{SA}/ca.crt" if os.path.exists(f"{SA}/ca.crt") else True
    return s


def get_hpa(s):
    """returns (minReplicas, maxReplicas, currentReplicas)"""
    if DRY_RUN:
        return fake_hpa["min"], fake_hpa["max"], fake_hpa["current"]
    r = s.get(HPA_URL, timeout=5)
    r.raise_for_status()
    h = r.json()
    return h["spec"].get("minReplicas", 1), h["spec"]["maxReplicas"], h.get("status", {}).get("currentReplicas", 1)


def set_floor(s, n, action, reason):
    log(action=action, new_min_replicas=n, reason=reason, dry_run=DRY_RUN)
    if DRY_RUN:
        fake_hpa["min"], fake_hpa["current"] = n, max(fake_hpa["current"], n)
    else:
        r = s.patch(HPA_URL, data=json.dumps({"spec": {"minReplicas": n}}),
                    headers={"Content-Type": "application/merge-patch+json"}, timeout=5)
        r.raise_for_status()
    ACTIONS.labels(action).inc()
    FLOOR.set(n)


def main():
    start_http_server(int(os.getenv("METRICS_PORT", "8000")))         # /metrics for Prometheus
    s = k8s_session()
    state = {"boost_until": 0.0, "last_boost": -1e9}
    try:
        min_r, max_r, cur = get_hpa(s)                                # fail now if we can't read the HPA
    except requests.RequestException as e:
        log(action="cannot_read_hpa", hpa=f"{NS}/{HPA}", error=str(e)[:160],
            hint="check the ServiceAccount aegis-policy, its Role (get/patch HPAs) and the HPA name")
        sys.exit(1)
    FLOOR.set(min_r)
    log(action="start", hpa=f"{NS}/{HPA}", min_replicas=min_r, max_replicas=max_r, tau=TAU, step=STEP,
        cooldown=COOLDOWN, boost_ttl=BOOST_TTL, poll=POLL, on_error=ON_ERROR, dry_run=DRY_RUN)

    def shutdown(*_):                    # never leave a raised floor behind (it would spoil the next baseline run)
        if state["boost_until"]:
            set_floor(s, BASE_FLOOR, "release", "controller stopping")
        sys.exit(0)
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    while True:
        now = time.time()
        try:
            r = requests.get(RISK_URL, timeout=3)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            data = r.json()
            risk = float(data["risk"])
            tau = float(data["threshold"]) if TAU == "auto" else float(TAU)
        except Exception as e:                                         # predictor down -> behave like plain HPA
            ACTIONS.labels("risk_unavailable").inc()
            log(action="risk_unavailable", error=str(e)[:120])
            if state["boost_until"] and (ON_ERROR == "release" or now >= state["boost_until"]):
                set_floor(s, BASE_FLOOR, "release", "predictor unavailable")
                state["boost_until"] = 0.0
            time.sleep(POLL)
            continue

        try:
            min_r, max_r, cur = get_hpa(s)
            if risk >= tau and now - state["last_boost"] >= COOLDOWN:
                target = min(max_r, max(min_r, cur) + STEP)            # never above the HPA's maxReplicas
                if target > min_r:
                    set_floor(s, target, "boost", f"risk {risk:.2f} >= tau {tau:.2f}, replicas {cur}")
                else:
                    ACTIONS.labels("at_max").inc()
                    log(action="at_max", risk=round(risk, 3), min_replicas=min_r)
                state["boost_until"], state["last_boost"] = now + BOOST_TTL, now
            elif state["boost_until"] and now >= state["boost_until"]:
                set_floor(s, BASE_FLOOR, "release", "boost expired")
                state["boost_until"] = 0.0
        except requests.RequestException as e:                         # Kubernetes API problem: log, keep going
            ACTIONS.labels("k8s_error").inc()
            log(action="k8s_error", error=str(e)[:120])
        time.sleep(POLL)


if __name__ == "__main__":
    main()
