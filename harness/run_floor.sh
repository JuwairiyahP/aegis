#!/usr/bin/env bash
set -uo pipefail
LOG=harness/runs.csv
URL=${URL:-http://localhost:8080/work}
mkdir -p harness/k6
trap 'kubectl -n aegis patch hpa demo -p "{\"spec\":{\"minReplicas\":1}}" >/dev/null; echo "floor reset to 1"' EXIT
run() {
  local id=$1 floor=$2 peak=$3 ramp=$4 hold=120
  if grep -q "^$id," "$LOG"; then echo "skip $id"; return; fi
  echo "=== $id: floor $floor pods, peak $peak req/s, ramp ${ramp}s ==="
  kubectl -n aegis patch hpa demo -p "{\"spec\":{\"minReplicas\":$floor}}" >/dev/null
  echo "  waiting for $floor ready pods ..."
  while true; do
    r=$(kubectl -n aegis get deploy demo -o jsonpath='{.status.readyReplicas}' </dev/null)
    [ "${r:-0}" -ge "$floor" ] && break
    sleep 5
  done
  sleep 30
  t0=$(date -u +%s)
  URL=$URL PEAK=$peak RAMP=$ramp HOLD=$hold SUMMARY="harness/k6/$id.json" \
    k6 run --quiet loadtest/ramp.js </dev/null || echo "  k6 exited with an error"
  t1=$(date -u +%s)
  dropped=$(python3 -c "import json,sys; m=json.load(open(sys.argv[1]))['metrics']; print(int(m.get('dropped_iterations',{}).get('values',{}).get('count',0)))" "harness/k6/$id.json" 2>/dev/null || echo "?")
  echo "$id,train,$peak,$ramp,$hold,1,$(date -u -d "@$t0" +%FT%TZ),$(date -u -d "@$t1" +%FT%TZ),$t0,$t1,$dropped" >> "$LOG"
  echo "  done after $(( (t1 - t0) / 60 )) min"
}
run fl-001 2 12 60
run fl-002 3 14 90
run fl-003 3 16 60
run fl-004 4 16 90
echo "all floor runs done"
