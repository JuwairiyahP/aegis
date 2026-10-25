# harness/capacity_native.py - estimate how many req/s ONE demo pod can take. Works on plain Windows
# (also Linux/Mac). No Docker, no Kubernetes.
#
# It runs the demo app natively, uses k6 to find the most traffic ONE process can handle, and scales that
# to a pod's CPU limit (0.25 CPU). The demo app is CPU-bound and single-threaded, so this is a fair estimate.
# It is rougher than harness/capacity_local.sh (which really caps the CPU with Docker): treat the result as
# +-30%, and confirm on the cluster later with loadtest/probe.js.
#
#   python harness\capacity_native.py                      (Windows PowerShell, from the repo folder)
#   $env:RATES="5 10 20 40 80"; python harness\capacity_native.py       (own rates, optional)
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
PORT = 8001
URL = f"http://127.0.0.1:{PORT}/work"            # 127.0.0.1, not "localhost" (avoids IPv6 surprises on Windows)
RATES = [float(x) for x in os.environ.get("RATES", "11 12 13 14 15 16 17 18").split()]
DURATION = os.environ.get("DURATION", "20s")
CPU_LIMIT = 0.25                                  # the pod's CPU limit in infra/k8s/app.yaml


def wait_ready(timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/readyz", timeout=2).status == 200:
                return True
        except Exception:
            time.sleep(0.5)
    return False


def run_k6(rate, summary_file):
    env = dict(os.environ, RATE=str(rate), DURATION=DURATION, URL=URL, SUMMARY=str(summary_file))
    subprocess.run(["k6", "run", "--quiet", str(ROOT / "loadtest" / "capacity.js")], env=env,
                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    m = json.load(open(summary_file))["metrics"]
    g = lambda name, key: m.get(name, {}).get("values", {}).get(key, 0)
    return {"target": rate, "achieved": g("http_reqs", "rate"), "median": g("http_req_duration", "med"),
            "p95": g("http_req_duration", "p(95)"), "failed": 100 * g("http_req_failed", "rate"),
            "dropped": int(g("dropped_iterations", "count"))}


def main():
    if shutil.which("k6") is None:
        sys.exit("k6 not found. Install it (Windows: winget install k6 --source winget), then open a NEW terminal.")
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        sys.exit("uvicorn not found. Activate your venv and run: pip install -r app/demo/requirements.txt")

    env = dict(os.environ, STARTUP_DELAY="0")
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1",
                               "--port", str(PORT), "--log-level", "warning"],
                              cwd=ROOT / "app" / "demo", env=env)
    rows = []
    try:
        if not wait_ready():
            sys.exit("the demo app did not start (is port %d already in use?)" % PORT)
        tmp = pathlib.Path(tempfile.gettempdir()) / "aegis_capacity.json"
        print("%8s %10s %10s %8s %8s %8s" % ("target", "achieved", "median_ms", "p95_ms", "failed%", "dropped"))
        saturated_in_a_row = 0
        for rate in RATES:
            r = run_k6(rate, tmp)
            r["saturated"] = r["achieved"] < 0.95 * rate or r["p95"] > 1000 or r["dropped"] > 0 or r["failed"] > 1
            rows.append(r)
            print("%8.0f %10.1f %10.0f %8.0f %8.1f %8d%s" % (r["target"], r["achieved"], r["median"], r["p95"],
                                                          r["failed"], r["dropped"], "   <- saturated" if r["saturated"] else ""))
            saturated_in_a_row = saturated_in_a_row + 1 if r["saturated"] else 0
            if saturated_in_a_row >= 2:
                break
            time.sleep(3)                                      # let the queue drain before the next step
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()

    ok = [r for r in rows if not r["saturated"] and r["p95"] < 250]
    best_achieved = max(r["achieved"] for r in rows)
    comfortable = max((r["achieved"] for r in ok), default=0.0)
    if len(ok) == len(rows):
        print("\nNothing saturated: this laptop can send more than the highest step. Run again with higher RATES.")
        return
    print("\n--- what this means ---")
    print(f"One app process tops out at about {best_achieved:.0f} req/s on this laptop")
    print(f"  = about {1000 / best_achieved:.0f} ms of CPU per request.")
    print(f"A pod is limited to {CPU_LIMIT} CPU, so it can do at most about {best_achieved * CPU_LIMIT:.1f} req/s,")
    print(f"and about {comfortable * CPU_LIMIT:.1f} req/s while p95 stays under 250 ms.")
    print(f"=> C1 (one pod's capacity) is roughly {comfortable * CPU_LIMIT:.1f} to {best_achieved * CPU_LIMIT:.1f} req/s.")
    if ok:
        print(f"Idle latency: p95 was {ok[0]['p95']:.0f} ms at {ok[0]['target']:.0f} req/s. Inside a 0.25-CPU pod it is a few "
              f"times higher because of CPU throttling.")
    print("Caveat: k6 and the app share this laptop, and a real pod is throttled in bursts, so keep +-30% in mind.")


if __name__ == "__main__":
    main()
