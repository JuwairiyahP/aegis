# Model report - v1

- Data source: prometheus-csv; training condition(s): baseline
- Evaluation: **held-out runs from the SAME profile family (not a held-out profile)**
- Runs - train/val/test: 8/4/4 (split by whole run)
- Usable rows (train/val/test): 441/210/256; positives: 224/133/96
- Threshold tau = 0.093, chosen on validation runs (F1), then frozen

## Held-out results

| Scorer | PR-AUC | Precision | Recall | Median lead (s) | Actionable* | False alarms/h |
|---|---|---|---|---|---|---|
| XGBoost | 0.803 | 0.462 | 1.000 | 120 | 100.0% | 42.2 |
| reactive: p95/SLO | 0.376 | 0.382 | 1.000 | 120 | 100.0% | 11.2 |
| reactive: CPU per replica | 0.340 | 0.375 | 1.000 | 120 | 100.0% | 11.2 |
| reactive: RPS per replica | 0.348 | 0.375 | 1.000 | 120 | 100.0% | 11.2 |

No-skill PR-AUC (base rate) = 0.375.
\*Actionable = share of held-out breaches warned at least 30 s ahead (the pod startup delay); earlier warnings cannot help.

Held-out breach onsets evaluated: **4** (detected 4).

**Too few held-out breaches (< 5, a rule of thumb).** Treat these numbers as anecdotal and say so in the report.

**Not a held-out profile:** test runs share the training load family, so these numbers are optimistic for the final held-out scenario.

## Leave-one-run-out cross-validation (all training-family runs)

16 runs; each run is scored by models trained on the other runs, and each threshold (F1) is chosen only from those other runs. Same load family as training, so this is NOT a held-out profile.

| Scorer | Pooled PR-AUC | Median per-run PR-AUC | Precision | Recall | Breaches warned in time* | Median lead (s) | False alarms/h |
|---|---|---|---|---|---|---|---|
| XGBoost | 0.862 | 0.965 | 0.786 | 0.870 | 92% of 24 | 120 | 4.8 |
| XGBoost (per-replica features only) | 0.852 | 0.939 | 0.753 | 0.812 | 92% of 24 | 120 | 10.3 |
| reactive: p95/SLO | 0.469 | 0.535 | 0.496 | 0.985 | 100% of 24 | 120 | 11.9 |
| reactive: CPU per replica | 0.453 | 0.399 | 0.499 | 1.000 | 100% of 24 | 120 | 11.9 |
| reactive: RPS per replica | 0.468 | 0.418 | 0.492 | 0.967 | 100% of 24 | 120 | 11.9 |

No-skill PR-AUC (base rate of usable rows) = 0.499.
\*Warned in time = warning at least the pod startup delay before the breach began.

## Top features (gain)

- cpu_per_replica_mean_120s: 65.7
- rps_per_replica_mean_120s: 21.0
- rps_per_replica_max_60s: 16.3
- cpu_mean_120s: 11.7
- cpu_per_replica_max_60s: 10.8
- rps_mean_120s: 8.9
- rps_per_replica_max_30s: 8.9
- rps_slope_30s: 7.4
- p95_max_120s: 7.0
- rps_mean_30s: 5.9

## Limitations

- Trained on HPA-only runs; once AEGIS raises the replica floor the replica features shift, so risk may fall after an action (use hysteresis in the controller).
- Scores are not calibrated probabilities (class-weighted); use tau, not 0.5.
- Lead time is capped at the 120 s look-back window.
