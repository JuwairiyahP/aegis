"""Gate before deployment: does the model warn because a surge is building, or because of the schedule?
   python -m predictor.diagnose <parquet dir> <model dir> <run-id prefixes, comma separated>
"""
# Diagnostic: does the model warn because a surge is building, or because of the schedule?
import sys, json, numpy as np, pandas as pd

from predictor.export import load_runs
from predictor.dataset import build_dataset, surge_starts
from predictor.model_io import load_model, predict_risk
from predictor.features import FEATURE_NAMES
data, model_dir, runs_glob = sys.argv[1], sys.argv[2], sys.argv[3]
m, meta = load_model(model_dir); tau = meta['threshold']
R = {r.run_id: r for r in load_runs(data)}
ds = build_dataset(list(R.values()))
pre_alarm = []; rows = []
for rid in sorted(ds.run_id.unique()):
    if not rid.startswith(tuple(runs_glob.split(','))): continue
    d = ds[ds.run_id == rid].copy(); w = d['warm'].values
    d['risk'] = np.nan; d.loc[w, 'risk'] = predict_risk(m, d.loc[w, FEATURE_NAMES], meta['best_iteration'])
    t = (d.index - d.index[0]).total_seconds().to_numpy()
    st = surge_starts(R[rid].meta.get('profile') or {})
    calm = d['pre_surge'].values & w
    pa = float((d.loc[calm, 'risk'] >= tau).mean()) if calm.any() else float('nan')
    onsets = t[d['onset'].values & (t >= 120)]
    alarm = (d['risk'].values >= tau) & ~d['in_breach'].values
    first_alarm_after_surge = None
    if st:
        a = t[alarm & (t >= st[0])]
        first_alarm_after_surge = (a[0] - st[0]) if len(a) else None
    lead = (onsets[0] - (t[alarm & (t >= (st[0] if st else 0)) & (t < onsets[0])][0])) if (len(onsets) and (alarm & (t >= (st[0] if st else 0)) & (t < (onsets[0] if len(onsets) else 0))).any()) else None
    rows.append((rid, round(pa, 2), st[0] if st else None, onsets[0] if len(onsets) else None, first_alarm_after_surge, lead))
    pre_alarm.append(pa)
print(f"tau {tau:.3f}")
print("run      pre-surge-alarm-share  surge_start  first_breach  alarm_after_surge_s  lead_before_breach_s")
for r in rows: print("  ".join(str(x) for x in r))
print("mean pre-surge alarm share:", round(float(np.nanmean(pre_alarm)), 3))
