#!/usr/bin/env bash
# EC2 training-data collection (frozen protocol, see docs/protocol_ec2.md). Run ON THE LOADGEN.
#   NODE_PRIV=<node private ip> harness/ec2_train.sh
set -uo pipefail
: "${NODE_PRIV:?set NODE_PRIV}"
export URL="http://${NODE_PRIV}/work" BASE=4 MAX_VUS=600 LOG=harness/runs_ec2.csv
cd "$(dirname "$0")/.."
kubectl -n aegis scale deploy policy --replicas=0 >/dev/null
kubectl -n aegis patch hpa demo -p '{"spec":{"minReplicas":1,"maxReplicas":4}}' >/dev/null

# 1) ramp runs, HPA only
harness/run_profiles.sh loadtest/ec2_train.csv

# 2) pre-scaled floor runs (HPA floor held at 2-3 pods): teach "high load + many pods = fine"
floor_run() {   # id floor peak ramp
  local id=$1 floor=$2 peak=$3 ramp=$4 hold=120
  grep -q "^$id," "$LOG" 2>/dev/null && { echo "skip $id"; return; }
  echo "=== $id: floor $floor, peak $peak, ramp ${ramp}s ==="
  kubectl -n aegis patch hpa demo -p "{\"spec\":{\"minReplicas\":$floor}}" >/dev/null
  until [ "$(kubectl -n aegis get deploy demo -o jsonpath='{.status.readyReplicas}')" -ge "$floor" ] 2>/dev/null; do sleep 5; done
  sleep 30
  t0=$(date -u +%s)
  PEAK=$peak RAMP=$ramp HOLD=$hold SUMMARY="harness/k6/$id.json" k6 run --quiet loadtest/ramp.js </dev/null || true
  t1=$(date -u +%s)
  echo "$id,train,$peak,$ramp,$hold,1,$(date -u -d @$t0 +%FT%TZ),$(date -u -d @$t1 +%FT%TZ),$t0,$t1,0" >> "$LOG"
  kubectl -n aegis patch hpa demo -p '{"spec":{"minReplicas":1}}' >/dev/null
}
mkdir -p harness/k6
floor_run ef-001 2 42 60
floor_run ef-002 3 54 30
floor_run ef-003 2 30 30
kubectl -n aegis patch hpa demo -p '{"spec":{"minReplicas":1}}' >/dev/null

sleep 15
python3 harness/export_runs.py --runs "$LOG" --out data/ec2
echo "EC2 training data done -> data/ec2"
