"""Breach onsets and future-breach labels (contract 3.3).

y = 1  iff a NEW breach begins within the next HORIZON_S seconds.
Rows already in breach are excluded (predicting that a breach continues is
trivial and would inflate every metric). Rows whose look-ahead window runs past
the end of the run with no onset seen are censored (label unknown) and excluded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HORIZON_S, SLO_S, STEP_S


def breach_flags(p95: pd.Series, slo_s: float = SLO_S) -> pd.Series:
    return (p95 > slo_s).fillna(False).astype(bool)


def breach_onsets(flag: pd.Series) -> pd.Series:
    prev = flag.shift(1, fill_value=False)
    onset = flag & ~prev
    if len(onset):
        onset.iloc[0] = False  # a breach already under way at t0 has unobserved onset
    return onset.astype(bool)


def future_onset(onset: pd.Series, horizon_s: int = HORIZON_S,
                 step_s: int = STEP_S) -> tuple[np.ndarray, np.ndarray]:
    """For each row t: (any onset in (t, t+H], full look-ahead window available)."""
    n = horizon_s // step_s
    o = onset.to_numpy().astype(int)
    L = len(o)
    csum = np.concatenate([[0], np.cumsum(o)])
    idx = np.arange(L)
    lo = np.minimum(idx + 1, L)
    hi = np.minimum(idx + n + 1, L)
    future = (csum[hi] - csum[lo]) > 0
    full = (idx + n) <= (L - 1)
    return future, full


def label_run(p95: pd.Series, warm: pd.Series, slo_s: float = SLO_S,
              horizon_s: int = HORIZON_S, step_s: int = STEP_S) -> pd.DataFrame:
    flag = breach_flags(p95, slo_s)
    onset = breach_onsets(flag)
    future, full = future_onset(onset, horizon_s, step_s)
    y = pd.Series(future, index=p95.index)
    censored = pd.Series(~future & ~full, index=p95.index)
    warm = warm.astype(bool)
    usable = warm & ~flag & ~censored
    return pd.DataFrame({
        "in_breach": flag, "onset": onset, "y": y.astype(int),
        "censored": censored, "warm": warm, "usable": usable,
    })
