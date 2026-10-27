#!/usr/bin/env bash
# Protocol v2 runner (docs/protocol_v2.md). Run ON THE LOADGEN inside tmux.
#   NODE_PRIV=<node private ip> harness/run_v2.sh loadtest/v2_train.csv
# CSV: run_id,scenario,condition,base,calm,peak,ramp,hold,bursts,gap,floor
# condition: hpa (HPA only) | floor (HPA floor held at <floor>) | static (HPA floor 4) |
#            aegis (HPA + controller; controller switched on CALM_ON s into the run, so the
#                   predictor's 120 s window holds only this run's calm traffic)
set -uo pipefail
CSV=${1:?usage: harness/run_v2.sh PROFILES.csv}
: "${NODE_PRIV:?set NODE_PRIV}"
URL="http://${NODE_PRIV}/work"; LOG=${LOG:-harness/runs_v2.csv}; CALM_ON=${CALM_ON:-130}
cd "$(dirname "$0")/.."; mkdir -p harness/k6
[ -f "$LOG" ] || echo "run_id,scenario,condition,base,calm,peak,ramp,hold,bursts,gap,floor,start_utc,end_utc,start_unix,end_unix,dropped_iterations" > "$LOG"
k() { kubectl -n aegis "$@" </dev/null; }
policy_off() { k scale deploy policy --replicas=0 >/dev/null; k wait --for=delete pod -l app=policy --timeout=90s >/dev/null 2>&1; }
set_floor() { k patch hpa demo -p "{\"spec\":{\"minReplicas\":$1}}" >/dev/null; }
wait_pods() {   # wait until HPA wants exactly $1 pods and $1 are ready
  for _ in $(seq 1 120); do
    want=$(k get deploy demo -o jsonpath='{.spec.replicas}'); ready=$(k get deploy demo -o jsonpath='{.status.readyReplicas}')
    [ "$want" = "$1" ] && [ "${ready:-0}" = "$1" ] && return 0; sleep 10
  done; echo "  pods did not settle at $1 within 20 min - stopping"; exit 1
}
curl -fsS -o /dev/null "$URL" || { echo "cannot reach $URL"; exit 1; }
while IFS=, read -r id scen cond base calm peak ramp hold bursts gap floor <&3; do
  [ -z "$id" ] && continue
  grep -q "^$id," "$LOG" && { echo "skip $id"; continue; }
  echo "=== $id [$cond] base $base calm ${calm}s peak $peak ramp ${ramp}s hold ${hold}s bursts $bursts gap ${gap}s floor $floor ==="
  policy_off
  case "$cond" in static) f=4 ;; floor) f=$floor ;; *) f=1 ;; esac
  set_floor "$f"; wait_pods "$f"; sleep "${SETTLE:-30}"
  t0=$(date -u +%s)
  URL=$URL BASE=$base CALM=$calm PEAK=$peak RAMP=$ramp HOLD=$hold BURSTS=$bursts GAP=$gap MAX_VUS=600 \
    SUMMARY="harness/k6/$id.json" k6 run --quiet loadtest/ramp.js </dev/null &
  kpid=$!
  if [ "$cond" = aegis ]; then
    sleep "$CALM_ON"; k scale deploy policy --replicas=1 >/dev/null
    k rollout status deploy/policy --timeout=90s >/dev/null && echo "  controller ON at ${CALM_ON}s"
  fi
  wait $kpid || echo "  k6 exited with an error"
  t1=$(date -u +%s)
  [ "$cond" = aegis ] && { k logs deploy/policy > "harness/k6/$id.policy.log" 2>/dev/null; policy_off; }
  set_floor 1
  dropped=$(python3 -c "import json,sys; m=json.load(open(sys.argv[1]))['metrics']; print(int(m.get('dropped_iterations',{}).get('values',{}).get('count',0)))" "harness/k6/$id.json" 2>/dev/null || echo "?")
  echo "$id,$scen,$cond,$base,$calm,$peak,$ramp,$hold,$bursts,$gap,$floor,$(date -u -d @$t0 +%FT%TZ),$(date -u -d @$t1 +%FT%TZ),$t0,$t1,$dropped" >> "$LOG"
  echo "  done in $(( (t1-t0)/60 )) min, dropped $dropped"
done 3< <(tail -n +2 "$CSV")
sleep 15
python3 harness/export_runs.py --runs "$LOG" --out data/ec2v2
echo "all done -> data/ec2v2"
