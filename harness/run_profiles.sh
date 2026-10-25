#!/usr/bin/env bash
# harness/run_profiles.sh - run every load profile in a CSV, one after another, and log when each ran.
#
#   harness/run_profiles.sh loadtest/eval.csv
#   EXPORT=1 PREDICTOR_DIR=~/aegis-predictor harness/run_profiles.sh loadtest/eval.csv   # also export with B's tool
#
# CSV columns: run_id,scenario,peak,ramp,hold,bursts   (scenario = train or eval)
# Writes harness/runs.csv (UTC times) and harness/k6/<run_id>.json (k6 summary).
# Needs: kubectl pointing at the cluster, k6, python3, and the Ingress on :8080.
# Safe to stop with Ctrl-C between runs: finished runs are skipped next time.
set -euo pipefail

CSV=${1:?usage: harness/run_profiles.sh PROFILES.csv}
URL=${URL:-http://localhost:8080/work}
NS=aegis
DEPLOY=demo
LOG=harness/runs.csv
mkdir -p harness/k6
[ -f "$LOG" ] || echo "run_id,scenario,peak,ramp,hold,bursts,start_utc,end_utc,start_unix,end_unix,dropped_iterations" > "$LOG"

curl -fsS -o /dev/null "$URL" || { echo "cannot reach $URL - is the cluster up and the Ingress on :8080?"; exit 1; }

wait_for_one_replica() {
  echo "  waiting until demo is back to 1 ready replica ..."
  for _ in $(seq 1 90); do            # up to 15 minutes
    want=$(kubectl -n "$NS" get deploy "$DEPLOY" -o jsonpath='{.spec.replicas}' </dev/null)
    ready=$(kubectl -n "$NS" get deploy "$DEPLOY" -o jsonpath='{.status.readyReplicas}' </dev/null)
    if [ "$want" = "1" ] && [ "${ready:-0}" = "1" ]; then
      sleep "${SETTLE:-30}"            # let the metrics calm down before the next run
      return 0
    fi
    sleep 10
  done
  echo "  demo did not go back to 1 replica in 15 min (is the HPA floor still raised?) - stopping"
  exit 1
}

while IFS=, read -r run_id scenario peak ramp hold bursts <&3; do
  [ -z "$run_id" ] && continue
  bursts=${bursts:-1}
  if grep -q "^$run_id," "$LOG"; then echo "skip $run_id (already in $LOG)"; continue; fi
  echo "=== $run_id ($scenario): peak $peak req/s, ramp ${ramp}s, hold ${hold}s, bursts $bursts ==="
  wait_for_one_replica

  t0=$(date -u +%s)
  URL=$URL PEAK=$peak RAMP=$ramp HOLD=$hold BURSTS=$bursts SUMMARY="harness/k6/$run_id.json" \
    k6 run --quiet loadtest/ramp.js </dev/null || echo "  k6 exited with an error - check the output above"
  t1=$(date -u +%s)

  dropped=$(python3 -c "import json,sys; m=json.load(open(sys.argv[1]))['metrics']; print(int(m.get('dropped_iterations',{}).get('values',{}).get('count',0)))" "harness/k6/$run_id.json" 2>/dev/null || echo "?")
  echo "$run_id,$scenario,$peak,$ramp,$hold,$bursts,$(date -u -d "@$t0" +%FT%TZ),$(date -u -d "@$t1" +%FT%TZ),$t0,$t1,$dropped" >> "$LOG"
  echo "  finished after $(( (t1 - t0) / 60 )) min, dropped iterations: $dropped"
  [ "$dropped" != "0" ] && echo "  WARNING: k6 could not send every request, so this run got less load than planned (raise MAX_VUS)"

  cmd="python -m predictor.collect --run-id $run_id --scenario $scenario --condition baseline --startup-delay 30 --start $t0 --end $((t1 + 10))"
  if [ "${EXPORT:-0}" = "1" ]; then
    sleep 15                           # let the last Prometheus scrapes land
    (cd "${PREDICTOR_DIR:?set PREDICTOR_DIR to the predictor folder}" && $cmd </dev/null)
  else
    echo "  export later (in B's predictor folder): $cmd"
  fi
done 3< <(tail -n +2 "$CSV")
echo "all done - times are in $LOG"
