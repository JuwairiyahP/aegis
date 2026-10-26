# AEGIS EC2 experiment protocol (frozen 2026-10-26, before any EC2 data was collected)

Pilot runs on a shared laptop showed breaches driven by host contention, not app load
(one pod breached at ~8 req/s on a busy laptop but handled 15 req/s on a quiet one).
All results below come from this protocol only. Laptop runs are reported as a pilot study.

## Environment
- Node: EC2 m7i-flex.large (2 vCPU, 8 GiB), K3s, ap-south-1. Load generator: separate t3.small.
- App: CPU-bound /work, 250m CPU limit/pod, 30 s startup delay. HPA 1-4 pods, 60 % CPU, 60 s scale-down window.
- Measured capacity (clean probe, 1 pod): p95 0.10 s at 24 req/s; breach at ~26-28 req/s (CPU limit).
- k6 arrival rate from the load generator to the node's private IP. Base rate 4 req/s.

## Training data (HPA only)
- 12 ramp runs: peak {30, 38, 46, 54} x ramp {30, 60, 120} s, hold 120 s.
- 2 slow-ramp runs: (34, 180 s), (50, 180 s) - near-capacity runs that may not breach.
- 3 pre-scaled runs (HPA floor held): (floor 2, peak 42, ramp 60), (3, 54, 30), (2, 30, 30).

## Model
- B's pipeline (predictor/), v3 settings (depth 3, min_child_weight 5, monotone).
- Train/validation split by run on training data only; threshold chosen on validation (F1) and frozen.

## Held-out test + A/B (never used for training or tuning)
- Profile: peak 42 req/s, ramp 45 s, hold 120 s (neither value appears in training).
- 5 interleaved pairs: xb-* = HPA only (also the model's held-out test set), xa-* = HPA + AEGIS.
- Controller: TAU = model threshold (auto), STEP 2, COOLDOWN 60 s, BOOST_TTL 180 s, poll 10 s.

## Metrics
- SLO: 30 s p95 > 250 ms. SLO violation % = share of 5 s samples in breach.
- Peak p95; recovery time (first breach until 60 s below SLO); replica-minutes (cost).
- Model: PR-AUC, precision/recall at frozen tau, breaches warned >= 30 s ahead, false alarms/h.
