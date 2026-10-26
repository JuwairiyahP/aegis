# AEGIS: Prediction-aware autoscaling for SLO protection

AEGIS predicts short-horizon SLO breaches (p95 latency > 250 ms within 120 s) from Prometheus
telemetry with XGBoost, and pre-scales a Kubernetes workload by raising the HPA's `minReplicas`
before the breach. Plain HPA keeps running underneath as the reactive safety net.

CCD Mini Project (CSE 3130), Manipal Institute of Technology, 2026.

## Architecture

```
k6 load generator (EC2 t3.small)
        |  HTTP /work
        v
K3s node (EC2 m7i-flex.large)
  demo app (FastAPI, 1-4 pods, 250m CPU each, 30 s startup)  <-- HPA (60 % CPU)
        |  /metrics (5 s)                                         ^
        v                                                         | patch minReplicas
  Prometheus --> predictor /risk (XGBoost) --> controller (policy) -+
        |
        v
  Grafana dashboard
```

## Repository layout

| Path | Owner | Contents |
|---|---|---|
| `app/demo/` | C | CPU-bound demo service (`/work`, `/readyz`, `/metrics`) |
| `loadtest/` | C | k6 scripts (`ramp.js`, `probe.js`) and load-profile CSVs |
| `harness/` | C / A | run orchestration, export from Prometheus, A/B script |
| `policy/` | C | AEGIS controller (raises HPA floor on high risk) |
| `predictor/` | B | features, labels, training, evaluation, FastAPI `/risk` service, models |
| `infra/k8s/` | A | manifests: app, HPA, RBAC, predictor + controller, Prometheus values, Grafana dashboard |
| `infra/terraform/` | A | AWS: EC2 node + load generator, security group, key pair |
| `infra/ec2/` | A | setup scripts for both instances |
| `docs/protocol_ec2.md` | all | frozen experiment protocol |
| `data/pilot_laptop/` | - | pilot runs on a laptop k3d cluster (not used for results) |
| `data/probe_ec2/` | - | clean capacity probe on EC2 |
| `data/ec2/` | - | final EC2 runs: `et-*`/`ef-*` training, `xb-*` HPA-only, `xa-*` HPA+AEGIS |

## Experiment history

1. **Laptop pilot** (k3d on WSL): 34 runs. Breaches turned out to be driven by host contention
   (one pod breached at ~8 req/s on a busy laptop but handled 15 req/s when idle), so models
   trained on it did not generalise. Kept in `data/pilot_laptop/` and `predictor/experiments/`.
2. **EC2 protocol** (`docs/protocol_ec2.md`), frozen before collecting data:
   - clean probe: one pod handles ~24 req/s, breaches at ~26-28 req/s
   - 17 training runs (14 ramp + 3 pre-scaled), model `ec2-v1`, threshold 0.43 chosen on validation runs
   - held-out profile (peak 42, ramp 45 s): 5 interleaved pairs HPA-only vs HPA+AEGIS

## Key results so far

| | XGBoost `ec2-v1` | Best reactive rule (p95/SLO) |
|---|---|---|
| PR-AUC, held-out runs of training family (chance 0.44) | 0.78 | 0.56 |
| PR-AUC, leave-one-run-out CV (chance 0.41) | 0.69 | 0.42 |
| Breaches warned >= 30 s ahead (test) | 6 / 6 | 6 / 6 |

HPA vs HPA+AEGIS: see `results/` (added after the A/B runs).

## Reproduce

```bash
# AWS (Free plan, ap-south-1)
cd infra/terraform && terraform init && terraform apply -var my_ip=$(curl -s https://checkip.amazonaws.com)
# node: build images into K3s
ssh -i ~/.ssh/aegis ubuntu@<node_ip> 'bash -s' < infra/ec2/setup_node.sh
# loadgen: monitoring + app + predictor + controller
scp -i ~/.ssh/aegis ubuntu@<node_ip>:/etc/rancher/k3s/k3s.yaml /tmp/k3s.yaml
scp -i ~/.ssh/aegis /tmp/k3s.yaml ubuntu@<loadgen_ip>:~/k3s.yaml
ssh -i ~/.ssh/aegis ubuntu@<loadgen_ip> "NODE_PRIV=<node_private_ip> bash -s" < infra/ec2/setup_loadgen.sh
# on the loadgen
NODE_PRIV=<node_private_ip> harness/ec2_train.sh         # training data
python -m predictor.train --data <parquet dir> --out predictor/models/<ver>
NODE_PRIV=<node_private_ip> PAIRS=5 harness/ec2_ab.sh    # A/B
# teardown
terraform destroy
```

## Limitations

- Single node, single synthetic workload, k6 traffic; 2-vCPU burstable-flex instance.
- Small sample (17 training runs, 5 A/B pairs); results are preliminary.
- Model trained on HPA-only runs plus 3 pre-scaled runs; behaviour after AEGIS actions is extrapolated.
