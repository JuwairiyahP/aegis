# Model report - v3

- Data source: prometheus-csv; training condition(s): baseline
- Evaluation: **held-out profile**
- Runs - train/val/test: 19/5/10 (split by whole run)
- Usable rows (train/val/test): 1181/310/684; positives: 487/95/88
- Threshold tau = 0.522, chosen on validation runs (F1), then frozen

## Held-out results

| Scorer | PR-AUC | Precision | Recall | Median lead (s) | Actionable* | False alarms/h |
|---|---|---|---|---|---|---|
| XGBoost | 0.213 | 0.349 | 0.170 | 70 | 50.0% | 2.1 |
| reactive: p95/SLO | 0.325 | 0.234 | 0.557 | 95 | 100.0% | 32.6 |
| reactive: CPU per replica | 0.318 | 0.129 | 1.000 | 120 | 100.0% | 10.5 |
| reactive: RPS per replica | 0.282 | 0.126 | 0.977 | 120 | 100.0% | 10.5 |

No-skill PR-AUC (base rate) = 0.129.
\*Actionable = share of held-out breaches warned at least 30 s ahead (the pod startup delay); earlier warnings cannot help.

Held-out breach onsets evaluated: **6** (detected 3).

## Top features (gain)

- rps_mean_60s: 115.0
- rps_mean_120s: 107.9
- rps_per_replica_max_60s: 69.6
- cpu_max_120s: 62.5
- rps_per_replica_max_120s: 42.6
- p95_max_120s: 41.9
- rps_max_120s: 29.3
- rps_slope_30s: 28.1
- cpu_mean_120s: 27.3
- rps_per_replica_mean_120s: 26.6

## Limitations

- Trained on HPA-only runs; once AEGIS raises the replica floor the replica features shift, so risk may fall after an action (use hysteresis in the controller).
- Scores are not calibrated probabilities (class-weighted); use tau, not 0.5.
- Lead time is capped at the 120 s look-back window.
