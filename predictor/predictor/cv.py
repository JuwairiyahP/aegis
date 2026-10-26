"""Nested leave-one-run-out cross-validation (the honest estimate when runs are few).

For every run h:  train on the other runs, score h.  The alarm threshold for that fold
is chosen ONLY from the other runs (an inner leave-one-run-out), never from h.
Scores for all runs are then pooled. Reactive baselines get the same treatment
(no model, but their threshold is also chosen without looking at h).

This still measures generalisation WITHIN the training load family. It is not a
held-out profile.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from .evaluate import MIN_HELDOUT_ONSETS, choose_threshold, false_alarms, lead_time_stats, pr_at
from .features import FEATURE_NAMES

BASELINE_COLS = {"reactive: p95/SLO": "p95_over_slo_now",
                 "reactive: CPU per replica": "cpu_per_replica_now",
                 "reactive: RPS per replica": "rps_per_replica_now"}
PER_REPLICA_FEATURES = [f for f in FEATURE_NAMES if "per_replica" in f]
MODELS = {"XGBoost": list(FEATURE_NAMES),
          "XGBoost (per-replica features only)": PER_REPLICA_FEATURES}


def _fit(df: pd.DataFrame, params: dict, seed: int, feats: list[str]):
    u = df[df["usable"]]
    pos, neg = int(u["y"].sum()), int((u["y"] == 0).sum())
    if pos == 0 or neg == 0:
        return None
    from .train import monotone_for
    m = XGBClassifier(max_depth=params["max_depth"], n_estimators=params["n_estimators"],
                      learning_rate=params["learning_rate"], subsample=0.8, colsample_bytree=0.8,
                      min_child_weight=params.get("min_child_weight", 1),
                      monotone_constraints=monotone_for(feats) if params.get("monotone") else None,
                      scale_pos_weight=neg / pos, eval_metric="aucpr", random_state=seed, n_jobs=2)
    m.fit(u[feats], u["y"])
    return m


def _score(model, df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    out = np.full(len(df), np.nan)
    w = df["warm"].to_numpy()
    if w.any():
        out[w] = model.predict_proba(df.loc[w, feats])[:, 1]
    return out


def _tau(ys, ss, beta):
    y, s = np.concatenate(ys), np.concatenate(ss)
    ok = ~np.isnan(s)
    return choose_threshold(y[ok], s[ok], beta)[0]


def loro_cv(ds: pd.DataFrame, params: dict, *, models: dict | None = None, beta: float = 1.0,
            seed: int = 0, horizon_s: int, step_s: int, startup_delay_s: int) -> dict:
    models = models or MODELS
    run_ids = sorted(ds["run_id"].unique())
    by = {r: ds[ds["run_id"] == r].copy() for r in run_ids}
    scorers = list(models) + list(BASELINE_COLS)
    rec = {s: [] for s in scorers}     # per-fold records
    skipped = []
    for h in run_ids:
        rest = [r for r in run_ids if r != h]
        # ---- inner out-of-fold scores on the other runs, to choose each scorer's threshold
        inner = {s: ([], []) for s in scorers}
        for r2 in rest:
            tr = pd.concat([by[x] for x in rest if x != r2])
            te = by[r2]
            u = te["usable"].to_numpy()
            y_u = te["y"].to_numpy()[u]
            for name, feats in models.items():
                m = _fit(tr, params, seed, feats)
                if m is not None:
                    inner[name][0].append(y_u)
                    inner[name][1].append(_score(m, te, feats)[u])
            for name, col in BASELINE_COLS.items():
                inner[name][0].append(y_u)
                inner[name][1].append(te[col].to_numpy(dtype=float)[u])
        full = pd.concat([by[x] for x in rest])
        te = by[h]
        score = {}
        for name, feats in models.items():
            m = _fit(full, params, seed, feats)
            score[name] = _score(m, te, feats) if m is not None else None
        for name, col in BASELINE_COLS.items():
            score[name] = te[col].to_numpy(dtype=float)
        for s in scorers:
            try:
                if score[s] is None or not inner[s][0]:
                    raise ValueError("no usable model/positives")
                tau = _tau(*inner[s], beta)
            except ValueError:
                skipped.append((h, s))
                continue
            f = te.copy()
            f["score"] = score[s]
            rec[s].append({"run": h, "tau": tau, "frame": f})
    out = {"n_runs": len(run_ids), "beta": beta, "skipped_folds": skipped, "scorers": {}}
    for s in scorers:
        out["scorers"][s] = _aggregate(rec[s], horizon_s, step_s, startup_delay_s)
    return out


def _aggregate(folds: list[dict], horizon_s: int, step_s: int, startup_delay_s: int) -> dict:
    if not folds:
        return {"n_folds": 0}
    ys, ss, per_run_auc = [], [], []
    tp = fp = fn = 0
    leads, n_eval, episodes, false_eps, hours = [], 0, 0, 0, 0.0
    for f in folds:
        fr = f["frame"]
        u = fr[fr["usable"] & fr["score"].notna()]
        y, s = u["y"].to_numpy(), u["score"].to_numpy(dtype=float)
        ys.append(y)
        ss.append(s)
        if 0 < y.sum() < len(y):
            per_run_auc.append(float(average_precision_score(y, s)))
        c = pr_at(y, s, f["tau"])
        tp, fp, fn = tp + c["tp"], fp + c["fp"], fn + c["fn"]
        lt = lead_time_stats({f["run"]: fr}, "score", f["tau"], horizon_s, step_s, startup_delay_s)
        leads += lt["leads_s"]
        n_eval += lt["n_evaluable"]
        fa = false_alarms({f["run"]: fr}, "score", f["tau"], horizon_s, step_s)
        episodes += fa["alarm_episodes"]
        false_eps += fa["false_episodes"]
        hours += len(u) * step_s / 3600
    y, s = np.concatenate(ys), np.concatenate(ss)
    det = np.array([x for x in leads if x is not None], dtype=float)
    prec = tp / (tp + fp) if tp + fp else None
    rec_ = tp / (tp + fn) if tp + fn else None
    return {
        "n_folds": len(folds), "n_rows": int(len(y)), "n_pos": int(y.sum()),
        "base_rate": float(y.mean()),
        "pooled_pr_auc": float(average_precision_score(y, s)) if y.sum() > 0 else None,
        "per_run_pr_auc_median": float(np.median(per_run_auc)) if per_run_auc else None,
        "precision": prec, "recall": rec_,
        "n_onsets_evaluable": n_eval, "n_detected": int(len(det)),
        "detection_rate": float(len(det) / n_eval) if n_eval else None,
        "median_lead_s": float(np.median(det)) if len(det) else None,
        "actionable_rate": float((det >= startup_delay_s).sum() / n_eval) if n_eval else None,
        "false_alarm_episodes": false_eps, "alarm_episodes": episodes,
        "false_per_hour": float(false_eps / hours) if hours else None,
        "sufficient_onsets": n_eval >= MIN_HELDOUT_ONSETS,
    }
