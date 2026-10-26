# Model report - ec2-v1

- Data source: prometheus-csv; training condition(s): baseline
- Evaluation: **held-out runs from the SAME profile family (not a held-out profile)**
- Runs - train/val/test: 10/3/4 (split by whole run)
- Usable rows (train/val/test): 506/141/200; positives: 192/68/88
- Threshold tau = 0.425, chosen on validation runs (F1), then frozen

## Held-out results

| Scorer | PR-AUC | Precision | Recall | Median lead (s) | Actionable* | False alarms/h |
|---|---|---|---|---|---|---|
| XGBoost | 0.782 | 0.643 | 0.818 | 120 | 100.0% | 10.8 |
| reactive: p95/SLO | 0.563 | 0.440 | 1.000 | 120 | 100.0% | 14.4 |
| reactive: CPU per replica | 0.463 | 0.440 | 1.000 | 120 | 100.0% | 14.4 |
| reactive: RPS per replica | 0.453 | 0.509 | 0.989 | 120 | 100.0% | 10.8 |

No-skill PR-AUC (base rate) = 0.440.
\*Actionable = share of held-out breaches warned at least 30 s ahead (the pod startup delay); earlier warnings cannot help.

Held-out breach onsets evaluated: **6** (detected 6).

**Not a held-out profile:** test runs share the training load family, so these numbers are optimistic for the final held-out scenario.

## Leave-one-run-out cross-validation (all training-family runs)

17 runs; each run is scored by models trained on the other runs, and each threshold (F1) is chosen only from those other runs. Same load family as training, so this is NOT a held-out profile.

| Scorer | Pooled PR-AUC | Median per-run PR-AUC | Precision | Recall | Breaches warned in time* | Median lead (s) | False alarms/h |
|---|---|---|---|---|---|---|---|
| XGBoost | 0.694 | 0.997 | 0.655 | 0.796 | 80% of 25 | 120 | 10.2 |
| XGBoost (per-replica features only) | 0.721 | 0.994 | 0.677 | 0.851 | 84% of 25 | 120 | 17.9 |
| reactive: p95/SLO | 0.420 | 0.552 | 0.411 | 1.000 | 92% of 25 | 120 | 14.5 |
| reactive: CPU per replica | 0.399 | 0.607 | 0.411 | 1.000 | 92% of 25 | 120 | 14.5 |
| reactive: RPS per replica | 0.418 | 0.634 | 0.412 | 0.911 | 88% of 25 | 120 | 13.6 |

No-skill PR-AUC (base rate of usable rows) = 0.411.
\*Warned in time = warning at least the pod startup delay before the breach began.

## Top features (gain)

- rps_mean_60s: 45.7
- rps_mean_120s: 38.4
- reps_mean_120s: 35.8
- p95_max_120s: 24.3
- rps_max_120s: 21.3
- reps_mean_30s: 14.3
- p95_mean_30s: 13.3
- rps_mean_30s: 10.9
- rps_per_replica_slope_30s: 10.8
- rps_per_replica_now: 10.6

## Limitations

- Trained on HPA-only runs; once AEGIS raises the replica floor the replica features shift, so risk may fall after an action (use hysteresis in the controller).
- Scores are not calibrated probabilities (class-weighted); use tau, not 0.5.
- Lead time is capped at the 120 s look-back window.
