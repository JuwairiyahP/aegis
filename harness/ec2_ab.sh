#!/usr/bin/env bash
# Interleaved HPA-only vs HPA+AEGIS on the HELD-OUT profile (frozen, see docs/protocol_ec2.md).
# Run ON THE LOADGEN after the model is trained and deployed.   NODE_PRIV=... PAIRS=5 harness/ec2_ab.sh
set -uo pipefail
: "${NODE_PRIV:?set NODE_PRIV}"
export URL="http://${NODE_PRIV}/work" BASE=4 MAX_VUS=600 LOG=harness/runs_ec2.csv
PAIRS=${PAIRS:-5}
cd "$(dirname "$0")/.."
set_condition() {   # 0 = HPA only, 1 = HPA + AEGIS
  kubectl -n aegis scale deploy policy --replicas="$1" >/dev/null
  if [ "$1" = 0 ]; then
    kubectl -n aegis wait --for=delete pod -l app=policy --timeout=90s >/dev/null 2>&1
    kubectl -n aegis patch hpa demo -p '{"spec":{"minReplicas":1}}' >/dev/null
  else
    kubectl -n aegis rollout status deploy/policy --timeout=90s >/dev/null
  fi
}
one_run() {         # id condition scenario
  grep -q "^$1," "$LOG" 2>/dev/null && { echo "skip $1"; return; }
  set_condition "$2"
  printf 'run_id,scenario,peak,ramp,hold,bursts\n%s,%s,42,45,120,1\n' "$1" "$3" > harness/ab_tmp.csv
  harness/run_profiles.sh harness/ab_tmp.csv
}
for i in $(seq 1 "$PAIRS"); do
  n=$(printf %03d "$i")
  one_run "xb-$n" 0 eval     # baseline = also the model's held-out test set
  one_run "xa-$n" 1 aegis
done
set_condition 0
rm -f harness/ab_tmp.csv
sleep 15
python3 harness/export_runs.py --runs "$LOG" --out data/ec2
kubectl -n aegis logs deploy/policy --tail=0 >/dev/null 2>&1 || true
echo "A/B done -> data/ec2 (xb-* = HPA only, xa-* = HPA+AEGIS)"
