"""Sanity-check that Prometheus returns usable data for the 5 predictor queries.

    kubectl -n monitoring port-forward svc/mon-kube-prometheus-stack-prometheus 9090:9090
    python -m predictor.check --prom-url http://localhost:9090

Run it while the demo app is receiving traffic (e.g. a short k6 run).
Exit code 0 = all required series OK, 1 = something needs fixing.
"""
from __future__ import annotations

import argparse
import sys
import time

from .config import PromConfig, STEP_S
from .prom_client import PromClient, PromError


def run_checks(client: PromClient, cfg: PromConfig, minutes: float = 10.0,
               now: float | None = None) -> list[dict]:
    end = (int(time.time() if now is None else now) // STEP_S) * STEP_S
    start = end - int(minutes * 60)
    out = []
    for name, q in cfg.queries().items():
        try:
            s = client.query_range(q, start, end)
        except PromError as e:
            msg = str(e)
            status = "MULTI" if "expected 1 series" in msg else "ERROR"
            out.append({"name": name, "status": status, "detail": msg, "query": q})
            continue
        frac = float(s.notna().mean())
        if frac == 0:
            if name == "err":
                out.append({"name": name, "status": "OK",
                            "detail": "no 5xx observed (treated as 0)", "query": q})
            else:
                out.append({"name": name, "status": "EMPTY", "query": q,
                            "detail": "no data in the last %g min - is traffic flowing, "
                                      "and do namespace/route/pod labels match?" % minutes})
        else:
            out.append({"name": name, "status": "OK", "query": q,
                        "detail": f"{frac*100:.0f}% of points present, last={s.dropna().iloc[-1]:.4g}"})
    # which route labels exist? (helps pick AEGIS_ROUTE)
    try:
        rows = client.instant(
            f'sum by (route) (rate(http_requests_total{{namespace="{cfg.namespace}"}}[1m]))')
        routes = sorted(m.get("route", "<none>") for m, _ in rows)
    except PromError as e:
        routes = []
        out.append({"name": "routes", "status": "ERROR", "detail": str(e), "query": ""})
    if routes:
        ok = (not cfg.route) or cfg.route in routes
        out.append({"name": "routes", "status": "OK" if ok else "BAD_ROUTE", "query": "",
                    "detail": f"route labels seen: {routes}; AEGIS_ROUTE={cfg.route!r}"
                              + ("" if ok else " is NOT among them - set AEGIS_ROUTE")})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check Prometheus data for the predictor")
    ap.add_argument("--prom-url", default=PromConfig.from_env().base_url)
    ap.add_argument("--minutes", type=float, default=10.0)
    a = ap.parse_args(argv)
    cfg = PromConfig.from_env()
    results = run_checks(PromClient(a.prom_url), cfg, a.minutes)
    bad = 0
    for r in results:
        print(f"[{r['status']:9}] {r['name']:6} {r['detail']}")
        if r["status"] not in ("OK",):
            bad += 1
            if r["query"]:
                print(f"            query: {r['query']}")
    print("\nALL OK" if not bad else f"\n{bad} problem(s) - fix before collecting data")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
