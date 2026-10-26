import sys, json, time, numpy as np, pandas as pd
sys.path.insert(0, '.')
from predictor import cv as CV
from predictor.export import load_runs
from predictor.dataset import build_dataset
from predictor.features import FEATURE_NAMES
from xgboost import XGBClassifier

runs = load_runs('data/runs')
ds = build_dataset(runs)
pool = ds[ds['scenario'] != 'eval']
print('cv runs', pool['run_id'].nunique())

ABS = ('rps_', 'cpu_', 'reps_')   # absolute-level window features
def load_invariant(f):
    if 'per_replica' in f or f.startswith(('p95', 'err')): return True
    if f == 'reps_now': return True
    return not f.startswith(ABS) and f not in ('rps_now', 'cpu_now')
LI = [f for f in FEATURE_NAMES if load_invariant(f)]
MONO_UP = lambda f: (('per_replica' in f or f.startswith('p95') or f.startswith('err')) and 'slope' not in f)
def mono(feats): return tuple(1 if MONO_UP(f) else 0 for f in feats)

USE_MONO = {'on': False}
def _fit(df, params, seed, feats):
    u = df[df['usable']]; pos, neg = int(u['y'].sum()), int((u['y'] == 0).sum())
    if pos == 0 or neg == 0: return None
    kw = dict(monotone_constraints=mono(feats)) if USE_MONO['on'] else {}
    m = XGBClassifier(max_depth=params['max_depth'], n_estimators=params['n_estimators'],
                      learning_rate=params['learning_rate'], subsample=0.8, colsample_bytree=0.8,
                      min_child_weight=params.get('mcw', 1), scale_pos_weight=neg/pos,
                      random_state=seed, n_jobs=4, **kw)
    m.fit(u[feats], u['y']); return m
CV._fit = _fit

variants = {
  'V0 B default (all feats, depth4)':      (dict(max_depth=4, n_estimators=150, learning_rate=0.05), FEATURE_NAMES, False),
  'V1 load-invariant feats, depth3':       (dict(max_depth=3, n_estimators=150, learning_rate=0.05, mcw=5), LI, False),
  'V2 load-invariant + monotone, depth3':  (dict(max_depth=3, n_estimators=150, learning_rate=0.05, mcw=5), LI, True),
  'V3 all feats + monotone, depth3':       (dict(max_depth=3, n_estimators=150, learning_rate=0.05, mcw=5), FEATURE_NAMES, True),
}
res = {}
for name, (params, feats, m) in variants.items():
    USE_MONO['on'] = m; t = time.time()
    r = CV.loro_cv(pool, params, models={name: feats}, beta=1.0, seed=0, horizon_s=120, step_s=5, startup_delay_s=30)
    e = r['scorers'][name]; res[name] = e
    print(f"{name:40s} PR-AUC {e['pooled_pr_auc']:.3f} med/run {e['per_run_pr_auc_median']:.3f} P {e['precision']:.2f} R {e['recall']:.2f} "
          f"warned-in-time {e['actionable_rate']:.2f} of {e['n_onsets_evaluable']} FA/h {e['false_per_hour']:.1f}  ({time.time()-t:.0f}s)", flush=True)
b = r['scorers']
for k in CV.BASELINE_COLS:
    e = b[k]; print(f"{k:40s} PR-AUC {e['pooled_pr_auc']:.3f} P {e['precision']:.2f} R {e['recall']:.2f} warned {e['actionable_rate']:.2f} FA/h {e['false_per_hour']:.1f}")
print('n LI feats', len(LI), 'of', len(FEATURE_NAMES))
