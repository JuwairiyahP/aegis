# Pilot experiments (laptop data)

Scripts used while analysing the laptop pilot runs (`data/pilot_laptop/`). Kept for transparency;
their results are NOT used in the final evaluation, because the pilot data turned out to be
dominated by host contention (see `docs/protocol_ec2.md`).

| Script | What it did |
|---|---|
| `pilot_xgb_v1.py` | Stand-alone XGBoost on 16 pilot runs; validation = 4 runs |
| `pilot_xgb_v2.py` | Same, on 24 pilot runs (added edge-of-capacity runs) |
| `pilot_variant_cv.py` | Leave-one-run-out CV comparing 4 model variants with B's pipeline |
| `pilot_leave_one_peak_out.py` | Leave-one-peak-out CV (simulates an unseen load level) |

The chosen settings (depth 3, min_child_weight 5, monotone constraints) became the defaults
in `predictor/train.py` and were used for the final model `models/ec2-v1`.
