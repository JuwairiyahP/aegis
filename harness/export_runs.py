# harness/export_runs.py - pull each run's metrics out of Prometheus into one CSV per run.
import argparse, csv, json, os, sys, urllib.parse, urllib.request

Q = {
    "rps":        'sum(rate(http_requests_total{namespace="aegis"}[30s]))',
    "p95":        'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{namespace="aegis"}[30s])) by (le))',
    "error_rate": '(sum(rate(http_requests_total{namespace="aegis",status=~"5.."}[30s])) or vector(0)) / sum(rate(http_requests_total{namespace="aegis"}[30s]))',
    "cpu":        'sum(rate(container_cpu_usage_seconds_total{namespace="aegis",pod=~"demo-.*",container!=""}[30s]))',
    "replicas":   'kube_deployment_status_replicas_available{namespace="aegis",deployment="demo"}',
    "hpa_min":    'kube_horizontalpodautoscaler_spec_min_replicas{namespace="aegis",horizontalpodautoscaler="demo"}',
}
META = ["run_id", "scenario", "peak", "ramp", "hold", "bursts"]

def query_range(prom, expr, start, end, step):
    url = prom.rstrip("/") + "/api/v1/query_range?" + urllib.parse.urlencode(
        {"query": expr, "start": start, "end": end, "step": step})
    with urllib.request.urlopen(url, timeout=30) as r:
        res = json.load(r)
    if res.get("status") != "success":
        raise RuntimeError(res)
    out = {}
    for series in res["data"]["result"]:
        for ts, val in series["values"]:
            out[int(float(ts))] = None if val in ("NaN", "+Inf", "-Inf") else float(val)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prom", default="http://localhost:9090")
    ap.add_argument("--runs", default="harness/runs.csv")
    ap.add_argument("--out", default="data/runs")
    ap.add_argument("--step", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = list(csv.DictReader(open(a.runs)))
    if not runs:
        sys.exit(f"no runs in {a.runs}")
    for run in runs:
        start, end = int(run["start_unix"]), int(run["end_unix"]) + 10
        cols = {name: query_range(a.prom, expr, start, end, a.step) for name, expr in Q.items()}
        times = sorted(set().union(*[c.keys() for c in cols.values()]))
        path = os.path.join(a.out, f"{run['run_id']}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts_unix", "t_sec"] + list(Q) + META)
            for t in times:
                w.writerow([t, t - start] + ["" if cols[n].get(t) is None else round(cols[n][t], 6) for n in Q]
                           + [run.get(m, "") for m in META])
        n_p95 = sum(1 for t in times if cols["p95"].get(t) is not None)
        breach = sum(1 for t in times if (cols["p95"].get(t) or 0) > 0.25)
        reps = [v for v in cols["replicas"].values() if v is not None]
        print(f"{run['run_id']:8s} {run['scenario']:5s} rows={len(times):4d} p95_points={n_p95:4d} "
              f"breach_points={breach:3d} max_replicas={int(max(reps, default=0))}  -> {path}")
    print("done. zip with: zip -r training_data.zip", a.out, a.runs)

if __name__ == "__main__":
    main()
