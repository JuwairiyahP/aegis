# Plan-B predictor: XGBoost "SLO breach starts within 120 s" on AEGIS runs.
import glob, os, sys, json
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score

DATA = sys.argv[1] if len(sys.argv) > 1 else "data/runs"
SLO, H, STEP, WARMUP = 0.25, 120, 5, 120

def load(path):
    d = pd.read_csv(path)
    d["p95"] = d["p95"].ffill().fillna(0)
    for c in ["rps", "cpu", "replicas", "error_rate", "hpa_min"]:
        d[c] = d[c].ffill().bfill().fillna(0)
    return d

def slope(s, n):
    x = np.arange(n) * STEP
    return s.rolling(n, min_periods=2).apply(lambda y: np.polyfit(x[-len(y):], y, 1)[0], raw=True)

def features(d):
    f = pd.DataFrame(index=d.index)
    rep = d["replicas"].clip(lower=1)
    f["rps_per_rep"] = d["rps"] / rep
    f["cpu_per_rep"] = d["cpu"] / rep
    f["p95_ratio"] = d["p95"] / SLO
    f["replicas"] = d["replicas"]
    for w in (30, 60, 120):
        n = w // STEP
        for c, s in [("rps_per_rep", f["rps_per_rep"]), ("p95", d["p95"]), ("cpu_per_rep", f["cpu_per_rep"])]:
            f[f"{c}_mean{w}"] = s.rolling(n, min_periods=1).mean()
            f[f"{c}_max{w}"] = s.rolling(n, min_periods=1).max()
            f[f"{c}_slope{w}"] = slope(s, n)
    return f.fillna(0)

def labels(d):
    br = d["p95"] > SLO
    onset = br & ~br.shift(1, fill_value=False) & (d["t_sec"] >= WARMUP)
    t = d["t_sec"].values; on_t = t[onset.values]
    y = np.array([((on_t > ti) & (on_t <= ti + H)).any() for ti in t]).astype(int)
    lead = np.array([(on_t[on_t > ti] - ti).min() if ((on_t > ti) & (on_t <= ti + H)).any() else np.nan for ti in t])
    return y, br.values, onset.values, lead

runs = {}
for p in sorted(glob.glob(f"{DATA}/*.csv")):
    d = load(p); rid = os.path.basename(p)[:-4]
    y, br, on, lead = labels(d)
    X = features(d)
    keep = ~br                          # drop rows already in breach
    runs[rid] = dict(d=d, X=X, y=y, br=br, on=on, keep=keep)

train_ids = [r for r in runs if r.startswith(("tr-", "fl-"))]
eval_ids = [r for r in runs if r.startswith("ev-")]
# validation = 4 runs held out from training (mix of peaks + 1 floor run)
val_ids = ["tr-003", "tr-108", "tr-111", "fl-003", "tr-202", "tr-204", "fl-005"]
fit_ids = [r for r in train_ids if r not in val_ids]

def stack(ids):
    X = pd.concat([runs[r]["X"][runs[r]["keep"]] for r in ids])
    y = np.concatenate([runs[r]["y"][runs[r]["keep"]] for r in ids])
    return X, y

Xf, yf = stack(fit_ids); Xv, yv = stack(val_ids); Xe, ye = stack(eval_ids)
spw = (yf == 0).sum() / max(1, (yf == 1).sum())
m = XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=3, scale_pos_weight=spw, eval_metric="aucpr", early_stopping_rounds=40,
                  random_state=0, n_jobs=2)
m.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)

pv = m.predict_proba(Xv)[:, 1]
# threshold on validation: best F1
best = max(np.arange(0.1, 0.91, 0.05), key=lambda t: (2*precision_score(yv, pv>=t, zero_division=0)*recall_score(yv, pv>=t)) /
           max(1e-9, precision_score(yv, pv>=t, zero_division=0)+recall_score(yv, pv>=t)))
tau = round(float(best), 2)

def event_metrics(ids, scorer, tau):
    """per breach onset: was risk >= tau at any point in the 120 s before it? lead = earliest such warning.
       false alarms: alarms (risk>=tau, not in breach) with no onset in next 120 s, per run."""
    leads, caught, total, fa_rows, alarm_rows = [], 0, 0, 0, 0
    for r in ids:
        R = runs[r]; t = R["d"]["t_sec"].values; s = scorer(R["X"]); on_t = t[R["on"]]
        for ot in on_t:
            total += 1
            w = (t >= ot - H) & (t < ot) & ~R["br"] & (s >= tau)
            if w.any(): caught += 1; leads.append(ot - t[w].min())
        alarm = (s >= tau) & ~R["br"]
        alarm_rows += alarm.sum(); fa_rows += (alarm & (R["y"] == 0)).sum()
    return dict(onsets=total, caught=caught, lead_median=float(np.median(leads)) if leads else None,
                lead_min=float(np.min(leads)) if leads else None, leads=leads,
                false_alarm_share=round(fa_rows/alarm_rows, 2) if alarm_rows else None)

xgb_score = lambda X: m.predict_proba(X)[:, 1]
# simple rule baseline: rps_per_rep > k (k tuned on validation the same way)
def rule(k): return lambda X: (X["rps_per_rep_mean30"].values > k).astype(float)
ks = np.arange(3, 9.01, 0.5)
kbest = max(ks, key=lambda k: (lambda p: 2*precision_score(yv,p,zero_division=0)*recall_score(yv,p)/max(1e-9,precision_score(yv,p,zero_division=0)+recall_score(yv,p)))(rule(k)(Xv)>=0.5))

out = {}
for name, ids, X, y in [("validation", val_ids, Xv, yv), ("eval (held-out profile)", eval_ids, Xe, ye)]:
    p = m.predict_proba(X)[:, 1]
    rp = rule(kbest)(X)
    out[name] = dict(
        rows=int(len(y)), positives=int(y.sum()),
        xgb=dict(pr_auc=round(average_precision_score(y, p), 3) if y.sum() else None,
                 precision=round(precision_score(y, p >= tau, zero_division=0), 3),
                 recall=round(recall_score(y, p >= tau, zero_division=0), 3),
                 **{k: v for k, v in event_metrics(ids, xgb_score, tau).items() if k != "leads"}),
        rule=dict(precision=round(precision_score(y, rp >= 0.5, zero_division=0), 3),
                  recall=round(recall_score(y, rp >= 0.5, zero_division=0), 3),
                  **{k: v for k, v in event_metrics(ids, rule(kbest), 0.5).items() if k != "leads"}),
        base_rate=round(float(y.mean()), 3))
print(json.dumps(dict(fit_runs=fit_ids, val_runs=val_ids, eval_runs=eval_ids, best_iteration=int(m.best_iteration),
                      tau=tau, rule=f"rps_per_rep_mean30 > {kbest}", results=out), indent=1))
imp = pd.Series(m.feature_importances_, index=Xf.columns).sort_values(ascending=False).head(8)
print("\ntop features:\n", imp.round(3).to_string())
m.save_model("models/pilot_aegis_xgb_planb_v2.json")
json.dump(dict(tau=tau, features=list(Xf.columns)), open("models/pilot_model_meta_v2.json", "w"))
