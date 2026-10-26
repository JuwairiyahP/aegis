"""Feature builder: the ONE function used by training AND the /risk service.

All features are strictly trailing (use only data at or before time t), so
nothing from the future can leak into a row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RAW_COLS, SLO_S, STEP_S, WINDOWS_S

STATS = ("mean", "max", "slope")
WINDOW_SERIES = ("p95", "rps", "err", "cpu", "reps", "rps_per_replica", "cpu_per_replica")


def clean_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Put raw series on a regular grid and fill gaps.

    p95/rps/err/cpu: no data -> 0 (no traffic means no latency, no errors).
    reps: forward-fill only (never back-fill: that would use future data).
    """
    if raw is None or len(raw) == 0:
        raise ValueError("empty raw frame")
    df = raw.reindex(columns=list(RAW_COLS)).astype(float).sort_index()
    idx = pd.date_range(df.index[0], df.index[-1], freq=f"{STEP_S}s")
    df = df.reindex(idx)
    # A short gap in p95 usually means NO request completed (the service is saturated),
    # not that it is idle: carry the last latency forward for up to 30 s (causal ffill),
    # then treat longer gaps as "no traffic" -> 0.
    df["p95"] = df["p95"].ffill(limit=6)
    for c in ("p95", "rps", "err", "cpu"):
        df[c] = df[c].fillna(0.0)
    df["reps"] = df["reps"].ffill()
    df.index.name = "ts"
    return df


def _slope_weights(n: int, step: int) -> np.ndarray:
    t = np.arange(n, dtype=float) * step
    t -= t.mean()
    return t / (t ** 2).sum()


def rolling_slope(s: pd.Series, n: int, step: int = STEP_S) -> pd.Series:
    """Least-squares slope (units per second) over the trailing n samples."""
    w = _slope_weights(n, step)
    return s.rolling(n, min_periods=n).apply(lambda a: float(a @ w), raw=True)


def feature_names() -> list[str]:
    names = [f"{c}_now" for c in RAW_COLS]
    names += ["rps_per_replica_now", "cpu_per_replica_now", "p95_over_slo_now"]
    for s in WINDOW_SERIES:
        for w in WINDOWS_S:
            for stat in STATS:
                names.append(f"{s}_{stat}_{w}s")
    return names


FEATURE_NAMES = feature_names()


def build_features(raw: pd.DataFrame, slo_s: float = SLO_S,
                   step: int = STEP_S) -> pd.DataFrame:
    """Return one feature row per timestamp (NaN until the longest window is full)."""
    df = clean_raw(raw)
    reps = df["reps"].clip(lower=1.0)
    base = {
        "p95": df["p95"], "rps": df["rps"], "err": df["err"],
        "cpu": df["cpu"], "reps": df["reps"],
        "rps_per_replica": df["rps"] / reps,
        "cpu_per_replica": df["cpu"] / reps,
    }
    out = {f"{c}_now": df[c] for c in RAW_COLS}
    out["rps_per_replica_now"] = base["rps_per_replica"]
    out["cpu_per_replica_now"] = base["cpu_per_replica"]
    out["p95_over_slo_now"] = df["p95"] / slo_s
    for s in WINDOW_SERIES:
        ser = base[s]
        for w in WINDOWS_S:
            n = w // step + 1
            roll = ser.rolling(n, min_periods=n)
            out[f"{s}_mean_{w}s"] = roll.mean()
            out[f"{s}_max_{w}s"] = roll.max()
            out[f"{s}_slope_{w}s"] = rolling_slope(ser, n, step)
    feats = pd.DataFrame(out, index=df.index)[FEATURE_NAMES]
    return feats
