"""Honest evaluation: PR-AUC, precision/recall at a frozen threshold, lead time,
false-alarm rate, and simple reactive baselines to show whether ML adds value."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

# Rule of thumb (NOT a statistical test): fewer held-out breach onsets than this
# means the numbers below are anecdotal and the report must say so.
MIN_HELDOUT_ONSETS = 5


def _f(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def choose_threshold(y, score, beta: float = 2.0) -> tuple[float, float]:
    """Threshold maximising F-beta (beta=2 favours recall). Use on VALIDATION runs only."""
    y = np.asarray(y)
    s = np.asarray(score, dtype=float)
    if y.sum() == 0:
        raise ValueError("validation set has no positives - cannot choose a threshold")
    p, r, thr = precision_recall_curve(y, s)
    p, r = p[:-1], r[:-1]
    b2 = beta ** 2
    denom = b2 * p + r
    f = np.where(denom > 0, (1 + b2) * p * r / np.where(denom > 0, denom, 1), 0.0)
    best = f.max()
    if best <= 0:
        raise ValueError("no threshold achieves F-beta > 0 on validation")
    k = np.where(f >= best - 1e-12)[0][-1]  # highest threshold among ties
    return float(thr[k]), float(best)


def pr_at(y, score, tau: float) -> dict:
    y = np.asarray(y)
    pred = np.asarray(score) >= tau
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}


def lead_time_stats(frames: dict[str, pd.DataFrame], score_col: str, tau: float,
                    horizon_s: int, step_s: int, startup_delay_s: int) -> dict:
    """Lead time = seconds between the score first crossing tau (within the H seconds
    before an onset) and the onset. Capped at H by construction."""
    n = horizon_s // step_s
    leads, n_onsets, n_eval = [], 0, 0
    for f in frames.values():
        onset = f["onset"].to_numpy()
        valid = (f["warm"] & ~f["in_breach"]).to_numpy()
        score = f[score_col].to_numpy(dtype=float)
        for i in np.flatnonzero(onset):
            n_onsets += 1
            idx = np.arange(max(0, i - n), i)
            idx = idx[valid[idx]]
            if len(idx) == 0:
                continue  # onset too early in the run to have a valid look-back
            n_eval += 1
            hit = idx[score[idx] >= tau]
            leads.append((i - hit[0]) * step_s if len(hit) else np.nan)
    arr = np.array(leads, dtype=float)
    det = arr[~np.isnan(arr)]
    return {
        "leads_s": [None if np.isnan(x) else float(x) for x in arr],
        "n_onsets": n_onsets,
        "n_evaluable": n_eval,
        "n_detected": int(len(det)),
        "detection_rate": _f(len(det) / n_eval) if n_eval else None,
        "median_s": _f(np.median(det)) if len(det) else None,
        "mean_s": _f(det.mean()) if len(det) else None,
        "p10_s": _f(np.percentile(det, 10)) if len(det) else None,
        "startup_delay_s": startup_delay_s,
        "actionable_rate": _f((det >= startup_delay_s).sum() / n_eval) if n_eval else None,
        "sufficient_heldout_breaches": n_eval >= MIN_HELDOUT_ONSETS,
    }


def false_alarms(frames: dict[str, pd.DataFrame], score_col: str, tau: float,
                 horizon_s: int, step_s: int) -> dict:
    n = horizon_s // step_s
    rows, episodes, false = 0, 0, 0
    for f in frames.values():
        usable = f["usable"].to_numpy()
        alarm = usable & (f[score_col].to_numpy(dtype=float) >= tau)
        onset = f["onset"].to_numpy()
        rows += int(usable.sum())
        i, L = 0, len(alarm)
        while i < L:
            if alarm[i]:
                j = i
                while j + 1 < L and alarm[j + 1]:
                    j += 1
                episodes += 1
                if not onset[i + 1: j + n + 1].any():
                    false += 1
                i = j + 1
            else:
                i += 1
    hours = rows * step_s / 3600
    return {"alarm_episodes": episodes, "false_episodes": false,
            "false_per_hour": _f(false / hours) if hours else None}


def evaluate_scorer(val: pd.DataFrame, test: pd.DataFrame, score_col: str, *,
                    horizon_s: int, step_s: int, startup_delay_s: int,
                    beta: float = 2.0) -> dict:
    """Pick tau on `val`, then report on `test` (never tune on test)."""
    vu = val[val["usable"]]
    tau, vf = choose_threshold(vu["y"], vu[score_col], beta)
    tu = test[test["usable"]]
    y, s = tu["y"].to_numpy(), tu[score_col].to_numpy(dtype=float)
    out = {
        "tau": tau, "val_fbeta": vf, "beta": beta,
        "n_test_rows": int(len(tu)), "n_test_pos": int(y.sum()),
        "base_rate": _f(y.mean()) if len(y) else None,
        "pr_auc": _f(average_precision_score(y, s)) if y.sum() > 0 else None,
    }
    out.update(pr_at(y, s, tau))
    frames = {rid: g for rid, g in test.groupby("run_id")}
    out["lead_time"] = lead_time_stats(frames, score_col, tau, horizon_s, step_s, startup_delay_s)
    out["false_alarms"] = false_alarms(frames, score_col, tau, horizon_s, step_s)
    if y.sum() == 0:
        out["note"] = "test set has no positives: PR-AUC undefined"
    return out
