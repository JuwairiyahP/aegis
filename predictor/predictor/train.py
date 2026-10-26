"""Train the XGBoost breach-risk model with honest, run-level evaluation.

    python -m predictor.train --data data/runs --out models/v1 --version v1
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBClassifier

from .config import HORIZON_S, SLO_S, STARTUP_DELAY_S, STEP_S, WINDOWS_S
from .cv import BASELINE_COLS, loro_cv
from .dataset import RunInfo, build_dataset, split_runs
from .evaluate import MIN_HELDOUT_ONSETS, evaluate_scorer
from .export import RunData, load_runs
from .features import FEATURE_NAMES
from .model_io import predict_risk, save_model

# Monotone constraints: risk may only rise with load per replica, latency and errors
# (levels, not slopes). Encodes domain knowledge so the model can't learn odd dips
# from a few noisy runs.
def monotone_for(feats):
    up = lambda f: ("per_replica" in f or f.startswith(("p95", "err"))) and "slope" not in f
    return tuple(1 if up(f) else 0 for f in feats)

BASELINES = BASELINE_COLS


class TrainingError(RuntimeError):
    pass


def train_model(runs: list[RunData], out_dir, version: str = "v1", *, seed: int = 0,
                val_frac: float = 0.2, test_frac: float = 0.2, max_depth: int = 4,
                n_estimators: int = 400, learning_rate: float = 0.05, beta: float = 1.0,
                min_child_weight: float = 5.0, monotone: bool = True,
                startup_delay_s: int = STARTUP_DELAY_S, heldout=("eval",),
                make_plots: bool = True, cv: bool | None = None) -> dict:
    sources = sorted({str(r.meta.get("source", "unknown")) for r in runs})
    synthetic = "synthetic" in sources
    if synthetic and "synthetic" not in version:
        version += "-synthetic"
    conditions = sorted({str(r.meta.get("condition", "unknown")) for r in runs})

    ds = build_dataset(runs)
    infos = [RunInfo(rid, str(g["scenario"].iloc[0]), bool(g["onset"].any()))
             for rid, g in ds.groupby("run_id")]
    split = split_runs(infos, val_frac, test_frac, seed, heldout)
    part = {k: ds[ds["run_id"].isin(getattr(split, k))].copy() for k in ("train", "val", "test")}

    tr, va = part["train"][part["train"]["usable"]], part["val"][part["val"]["usable"]]
    if tr["y"].sum() == 0 or va["y"].sum() == 0:
        raise TrainingError(
            f"need positives in train and validation (train={int(tr['y'].sum())}, "
            f"val={int(va['y'].sum())}). Collect more runs that contain SLO breaches.")
    spw = float((tr["y"] == 0).sum() / (tr["y"] == 1).sum())
    model = XGBClassifier(
        max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw, min_child_weight=min_child_weight,
        monotone_constraints=monotone_for(FEATURE_NAMES) if monotone else None,
        eval_metric="aucpr", early_stopping_rounds=30, random_state=seed, n_jobs=2)
    model.fit(tr[FEATURE_NAMES], tr["y"], eval_set=[(va[FEATURE_NAMES], va["y"])], verbose=False)
    best_it = int(model.best_iteration)

    for k, d in part.items():  # score every warm row (needed for lead-time look-backs)
        d["risk"] = np.nan
        w = d["warm"].to_numpy()
        d.loc[w, "risk"] = predict_risk(model, d.loc[w, FEATURE_NAMES], best_it)

    kw = dict(horizon_s=HORIZON_S, step_s=STEP_S, startup_delay_s=startup_delay_s, beta=beta)
    model_eval = evaluate_scorer(part["val"], part["test"], "risk", **kw)
    baselines = {name: evaluate_scorer(part["val"], part["test"], col, **kw)
                 for name, col in BASELINES.items()}

    cv_res = None
    cv_pool = ds[~ds["scenario"].isin(heldout)]
    n_cv_runs = cv_pool["run_id"].nunique()
    if cv is None:
        cv = 4 <= n_cv_runs <= 20          # cheap enough; skip for large datasets
    if cv and n_cv_runs >= 4:
        # fixed tree count (not early-stopped) so no fold's score leaks into another's tuning
        cv_res = loro_cv(cv_pool, {"max_depth": max_depth, "n_estimators": 150,
                                   "learning_rate": learning_rate, "min_child_weight": min_child_weight,
                                   "monotone": monotone}, beta=beta, seed=seed,
                         horizon_s=HORIZON_S, step_s=STEP_S, startup_delay_s=startup_delay_s)

    metrics = {
        "version": version, "data_source": sources, "synthetic": synthetic,
        "training_conditions": conditions,
        "evaluation_type": split.evaluation_type,
        "split": {"train": split.train, "val": split.val, "test": split.test},
        "n_rows": {k: int(d["usable"].sum()) for k, d in part.items()},
        "n_pos": {k: int(d.loc[d["usable"], "y"].sum()) for k, d in part.items()},
        "best_iteration": best_it, "scale_pos_weight": spw,
        "model": model_eval, "baselines": baselines, "cv": cv_res,
        "min_heldout_onsets_rule_of_thumb": MIN_HELDOUT_ONSETS,
    }
    meta = {
        "version": version, "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_names": FEATURE_NAMES, "threshold": model_eval["tau"],
        "best_iteration": best_it, "slo_s": SLO_S, "horizon_s": HORIZON_S, "step_s": STEP_S,
        "windows_s": list(WINDOWS_S), "startup_delay_s": startup_delay_s,
        "data_source": sources, "synthetic": synthetic, "evaluation_type": split.evaluation_type,
        "xgboost": xgboost.__version__, "python": platform.python_version(),
        "params": {"max_depth": max_depth, "learning_rate": learning_rate,
                   "min_child_weight": min_child_weight, "monotone": monotone,
                   "n_estimators_max": n_estimators, "scale_pos_weight": spw, "seed": seed},
    }
    out = Path(out_dir)
    save_model(model, meta, out)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    imp = pd.Series(model.get_booster().get_score(importance_type="gain"))
    imp = imp.reindex(FEATURE_NAMES).fillna(0.0).sort_values(ascending=False)
    imp.to_csv(out / "feature_importance.csv", header=["gain"])
    if make_plots:
        from . import plots
        plots.feature_importance(imp, out / "feature_importance.png")
        te = part["test"].assign(baseline_p95=part["test"]["p95_over_slo_now"],
                                 baseline_cpu=part["test"]["cpu_per_replica_now"])
        plots.pr_curves(te, {"XGBoost": "risk", "reactive: p95/SLO": "baseline_p95",
                             "reactive: CPU/replica": "baseline_cpu"}, out / "pr_curve.png")
        pick = max(split.test, key=lambda r: int(part["test"].query("run_id == @r")["onset"].sum()))
        plots.timeline(part["test"].query("run_id == @pick"), meta["threshold"], SLO_S,
                       out / "example_test_run.png", f"Held-out run {pick}")
    (out / "report.md").write_text(render_report(metrics, imp))
    return metrics


def _fmt(x, pct=False, nd=3):
    if x is None:
        return "n/a"
    return f"{x*100:.1f}%" if pct else f"{x:.{nd}f}"


def _degenerate(e: dict) -> bool:
    p, r, b = e.get("precision"), e.get("recall"), e.get("base_rate")
    return None not in (p, r, b) and r >= 0.95 and p <= b + 0.05


def render_cv(cv: dict) -> list[str]:
    L = ["## Leave-one-run-out cross-validation (all training-family runs)", "",
         f"{cv['n_runs']} runs; each run is scored by models trained on the other runs, and each "
         f"threshold (F{cv['beta']:g}) is chosen only from those other runs. Same load family as "
         "training, so this is NOT a held-out profile.", "",
         "| Scorer | Pooled PR-AUC | Median per-run PR-AUC | Precision | Recall | "
         "Breaches warned in time* | Median lead (s) | False alarms/h |",
         "|---|---|---|---|---|---|---|---|"]
    base = None
    for name, e in cv["scorers"].items():
        if not e.get("n_folds"):
            L.append(f"| {name} | n/a | | | | | | |")
            continue
        base = e["base_rate"]
        warned = (f"{e['actionable_rate']*100:.0f}% of {e['n_onsets_evaluable']}"
                  if e["actionable_rate"] is not None else "n/a")
        L.append(f"| {name} | {_fmt(e['pooled_pr_auc'])} | {_fmt(e['per_run_pr_auc_median'])} | "
                 f"{_fmt(e['precision'])} | {_fmt(e['recall'])} | {warned} | "
                 f"{_fmt(e['median_lead_s'], nd=0)} | {_fmt(e['false_per_hour'], nd=1)} |")
    L += ["", f"No-skill PR-AUC (base rate of usable rows) = {_fmt(base)}.",
          "\\*Warned in time = warning at least the pod startup delay before the breach began.", ""]
    if cv.get("skipped_folds"):
        L += [f"Folds skipped (no positives to learn from): {len(cv['skipped_folds'])}.", ""]
    return L


def render_report(m: dict, imp: pd.Series) -> str:
    me, lt, fa = m["model"], m["model"]["lead_time"], m["model"]["false_alarms"]
    L = [f"# Model report - {m['version']}", ""]
    if m["synthetic"]:
        L += ["> **SYNTHETIC DATA.** These numbers test the pipeline only. They say nothing",
              "> about the real AEGIS system and must NOT go in the report as results.", ""]
    L += [f"- Data source: {', '.join(m['data_source'])}; training condition(s): "
          f"{', '.join(m['training_conditions'])}",
          f"- Evaluation: **{m['evaluation_type']}**",
          f"- Runs - train/val/test: {len(m['split']['train'])}/{len(m['split']['val'])}/"
          f"{len(m['split']['test'])} (split by whole run)",
          f"- Usable rows (train/val/test): {m['n_rows']['train']}/{m['n_rows']['val']}/"
          f"{m['n_rows']['test']}; positives: {m['n_pos']['train']}/{m['n_pos']['val']}/"
          f"{m['n_pos']['test']}",
          f"- Threshold tau = {me['tau']:.3f}, chosen on validation runs (F{m['model']['beta']:g}), "
          "then frozen", "",
          "## Held-out results", "",
          "| Scorer | PR-AUC | Precision | Recall | Median lead (s) | Actionable* | False alarms/h |",
          "|---|---|---|---|---|---|---|"]
    for name, e in [("XGBoost", me)] + list(m["baselines"].items()):
        l, f = e["lead_time"], e["false_alarms"]
        L.append(f"| {name} | {_fmt(e['pr_auc'])} | {_fmt(e['precision'])} | {_fmt(e['recall'])} | "
                 f"{_fmt(l['median_s'], nd=0)} | {_fmt(l['actionable_rate'], pct=True)} | "
                 f"{_fmt(f['false_per_hour'], nd=1)} |")
    L += ["", f"No-skill PR-AUC (base rate) = {_fmt(me['base_rate'])}.",
          f"\\*Actionable = share of held-out breaches warned at least {lt['startup_delay_s']} s "
          "ahead (the pod startup delay); earlier warnings cannot help.", "",
          f"Held-out breach onsets evaluated: **{lt['n_evaluable']}** "
          f"(detected {lt['n_detected']}).", ""]
    if not lt["sufficient_heldout_breaches"]:
        L += [f"**Too few held-out breaches (< {MIN_HELDOUT_ONSETS}, a rule of thumb).** "
              "Treat these numbers as anecdotal and say so in the report.", ""]
    if "SAME profile" in m["evaluation_type"]:
        L += ["**Not a held-out profile:** test runs share the training load family, so these "
              "numbers are optimistic for the final held-out scenario.", ""]
    if _degenerate(me):
        L += ["**Uninformative operating point:** at the frozen threshold the model flags almost "
              "every row (precision ~ base rate, recall ~ 1). Judge ranking quality (PR-AUC), not "
              "the threshold numbers, until there is more data.", ""]
    if m.get("cv"):
        L += render_cv(m["cv"])
    L += ["## Top features (gain)", ""] + [f"- {k}: {v:.1f}" for k, v in imp.head(10).items()]
    L += ["", "## Limitations", "",
          "- Trained on HPA-only runs; once AEGIS raises the replica floor the replica "
          "features shift, so risk may fall after an action (use hysteresis in the controller).",
          "- Scores are not calibrated probabilities (class-weighted); use tau, not 0.5.",
          "- Lead time is capped at the 120 s look-back window."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the AEGIS breach-risk model")
    ap.add_argument("--data", required=True, help="dir with run_*.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--min-child-weight", type=float, default=5.0)
    ap.add_argument("--no-monotone", action="store_true", help="disable monotone constraints")
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--startup-delay", type=int, default=STARTUP_DELAY_S)
    ap.add_argument("--heldout-scenarios", default="eval")
    ap.add_argument("--beta", type=float, default=1.0,
                    help="F-beta used to choose the threshold (1 = balanced, 2 favours recall)")
    ap.add_argument("--no-cv", action="store_true", help="skip leave-one-run-out cross-validation")
    a = ap.parse_args(argv)
    m = train_model(load_runs(a.data), a.out, a.version, seed=a.seed, val_frac=a.val_frac,
                    test_frac=a.test_frac, max_depth=a.max_depth, n_estimators=a.n_estimators,
                    startup_delay_s=a.startup_delay, beta=a.beta,
                    min_child_weight=a.min_child_weight, monotone=not a.no_monotone,
                    heldout=tuple(a.heldout_scenarios.split(",")),
                    cv=False if a.no_cv else None)
    if m["synthetic"]:
        print("!! TRAINED ON SYNTHETIC DATA - pipeline test only, not a result !!", file=sys.stderr)
    print((Path(a.out) / "report.md").read_text())


if __name__ == "__main__":
    main()
