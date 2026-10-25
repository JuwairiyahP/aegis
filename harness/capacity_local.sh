#!/usr/bin/env bash
# harness/capacity_local.sh - how many req/s can ONE demo pod handle on this laptop?
# Runs the demo app in Docker limited to 0.25 CPU (same limit as the pod) and sends a fixed
# rate for 30 s per step. No Kubernetes, no Prometheus, no money. Needs Docker and k6.
#
#   harness/capacity_local.sh                     # default rates
#   RATES="2 4 6 8 10 12 15 20" harness/capacity_local.sh
set -euo pipefail

RATES=${RATES:-"1 2 3 4 5 6 8 10 12 15 20"}
PORT=8001
OUT=harness/capacity_local.csv

docker build -q -t aegis-demo:local app/demo >/dev/null
docker rm -f aegis-cap >/dev/null 2>&1 || true
docker run -d --name aegis-cap --cpus=0.25 -e STARTUP_DELAY=0 -p "$PORT:8000" aegis-demo:local >/dev/null
trap 'docker rm -f aegis-cap >/dev/null 2>&1 || true' EXIT
sleep 3

echo "rate_rps,achieved_rps,median_ms,p95_ms,failed_pct,dropped" > "$OUT"
printf "%8s %10s %10s %8s %8s %8s\n" target achieved median_ms p95_ms failed% dropped
for r in $RATES; do
  RATE=$r DURATION=30s URL="http://localhost:$PORT/work" SUMMARY=/tmp/aegis_cap.json \
    k6 run --quiet loadtest/capacity.js >/dev/null </dev/null || true
  row=$(python3 - "$r" << 'PY'
import json, sys
m = json.load(open("/tmp/aegis_cap.json"))["metrics"]
g = lambda n, k: m.get(n, {}).get("values", {}).get(k, 0)
print(f"{sys.argv[1]},{g('http_reqs','rate'):.2f},{g('http_req_duration','med'):.0f},"
      f"{g('http_req_duration','p(95)'):.0f},{100*g('http_req_failed','rate'):.1f},{int(g('dropped_iterations','count'))}")
PY
)
  echo "$row" >> "$OUT"
  echo "$row" | awk -F, '{printf "%8s %10s %10s %8s %8s %8s\n", $1, $2, $3, $4, $5, $6}'
  sleep 5                                 # let the queue drain before the next step
done
echo
echo "saved $OUT"
echo "Your one-pod capacity (C1) = the highest rate whose p95 is still under 250 ms and achieved ~= target."
echo "Also note the p95 at 1-2 req/s: that is the idle latency. If it is already near 250 ms, the SLO and WORK_ITERS don't fit."
