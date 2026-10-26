import sys, numpy as np, pandas as pd
sys.path.insert(0,'.')
exec(open('predictor/experiments/pilot_variant_cv.py').read().split("variants = {")[0])
from sklearn.metrics import average_precision_score
import glob, csv, os
peak = {os.path.basename(f)[:-4]: int(next(csv.DictReader(open(f)))['peak']) for f in glob.glob('data/pilot_laptop/*.csv')}
pool = pool.assign(peak=pool['run_id'].map(peak))
variants = {
  'V0 B default':        (dict(max_depth=4, n_estimators=150, learning_rate=0.05), FEATURE_NAMES, False),
  'V1 load-invariant':   (dict(max_depth=3, n_estimators=150, learning_rate=0.05, mcw=5), LI, False),
  'V2 LI + monotone':    (dict(max_depth=3, n_estimators=150, learning_rate=0.05, mcw=5), LI, True),
  'V3 all + monotone':   (dict(max_depth=3, n_estimators=150, learning_rate=0.05, mcw=5), FEATURE_NAMES, True),
}
print('leave-one-PEAK-out (simulates an unseen load level); groups:', sorted(pool['peak'].unique()))
for name,(params,feats,m) in variants.items():
    USE_MONO['on']=m; ys=[];ss=[];per=[]
    for g in sorted(pool['peak'].unique()):
        tr=pool[pool['peak']!=g]; te=pool[(pool['peak']==g)&pool['usable']]
        mod=_fit(tr,params,0,feats); s=mod.predict_proba(te[feats])[:,1]; y=te['y'].values
        ys.append(y); ss.append(s)
        if 0<y.sum()<len(y): per.append(average_precision_score(y,s))
    y=np.concatenate(ys); s=np.concatenate(ss)
    print(f"{name:20s} pooled PR-AUC {average_precision_score(y,s):.3f}  median per-peak {np.median(per):.3f}  (base {y.mean():.3f})")
for c in ['p95_over_slo_now','cpu_per_replica_now','rps_per_replica_now']:
    te=pool[pool['usable']]; print(f"rule {c:22s} PR-AUC {average_precision_score(te['y'],te[c]):.3f}")
