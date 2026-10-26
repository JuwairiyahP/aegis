"""Report plots (matplotlib, headless). Imported lazily - never by the service."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

BLUE, ORANGE, GRAY, INK = "#2a6fdb", "#d9660b", "#8a8f98", "#222222"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)


def feature_importance(imp: pd.Series, path, top: int = 20):
    imp = imp.sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(imp) + 1.2))
    ax.barh(imp.index, imp.values, color=BLUE, height=0.7)
    ax.set_xlabel("XGBoost gain")
    ax.set_title("Top feature importances (gain)", loc="left", color=INK)
    _style(ax)
    ax.grid(axis="x", color="#e5e7eb")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pr_curves(test: pd.DataFrame, cols: dict[str, str], path):
    u = test[test["usable"]]
    y = u["y"].to_numpy()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    colors = [BLUE, ORANGE, GRAY]
    for (label, col), c in zip(cols.items(), colors):
        p, r, _ = precision_recall_curve(y, u[col].to_numpy(dtype=float))
        ax.plot(r, p, color=c, linewidth=2, label=label)
    ax.axhline(y.mean(), color=INK, linestyle=":", linewidth=1, label="no skill (base rate)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-recall on held-out runs", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def timeline(run: pd.DataFrame, tau: float, slo_s: float, path, title: str):
    x = (run.index - run.index[0]).total_seconds()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    a1.plot(x, run["p95_s"] * 1000, color=BLUE, linewidth=2)
    a1.axhline(slo_s * 1000, color=ORANGE, linestyle="--", linewidth=1.2)
    a1.text(x[-1], slo_s * 1000, " SLO", color=ORANGE, va="bottom", ha="right", fontsize=8)
    a1.set_ylabel("p95 (ms)")
    a1.set_yscale("symlog", linthresh=500)
    a1.set_title(title, loc="left", color=INK)
    a2.plot(x, run["risk"], color=BLUE, linewidth=2)
    a2.axhline(tau, color=ORANGE, linestyle="--", linewidth=1.2)
    a2.text(x[-1], tau, " threshold", color=ORANGE, va="bottom", ha="right", fontsize=8)
    for xi in x[run["onset"].to_numpy()]:
        for a in (a1, a2):
            a.axvline(xi, color=INK, linewidth=1, alpha=0.5)
    a2.set_ylabel("predicted risk")
    a2.set_xlabel("seconds since run start (vertical lines = breach onsets)")
    a2.set_ylim(0, 1.02)
    for a in (a1, a2):
        _style(a)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
