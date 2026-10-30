# harness/metrics.py - the metrics used in the report, computed from one run's CSV (5 s samples).
# Columns used: t_sec, rps, p95 (seconds, 30 s window), replicas (ready pods), hpa_min.
import numpy as np
import pandas as pd

SLO = 0.25          # seconds: the SLO is "30-second p95 latency <= 250 ms"
STEP = 5            # seconds between samples


def load_run(path):
    d = pd.read_csv(path)
    d["p95"] = pd.to_numeric(d["p95"], errors="coerce")
    return d


def breach_flags(d, slo=SLO, fill_gaps=False):
    """True where p95 is above the SLO. A missing p95 is normally 'no traffic' = not a breach.
    fill_gaps=True is the worst-case reading used as a sensitivity check: a missing p95 that follows a breach sample counts
    as breach until the next valid sample (on EC2 these gaps sit inside severe breaches: the overloaded pod answered nothing)."""
    p = d["p95"]
    if not fill_gaps:
        return p > slo
    out, last = [], False
    for x in p:
        last = last if pd.isna(x) else bool(x > slo)
        out.append(last)
    return pd.Series(out, index=p.index)


def run_metrics(d, slo=SLO, fill_gaps=False):
    p, v = d["p95"], breach_flags(d, slo, fill_gaps)
    onset = v & ~v.shift(1, fill_value=False)
    valid = int(p.notna().sum()) + (int((v & p.isna()).sum()) if fill_gaps else 0)
    out = {
        "run_id": d["run_id"].iloc[0],
        "samples": len(d), "valid_samples": valid,
        "slo_violation_pct": 100 * v.sum() / valid if valid else np.nan,   # share of timestamps with p95 > SLO
        "breached": bool(v.any()),
        "n_breach_onsets": int(onset.sum()),
        "peak_p95_s": p.max(),
        "max_replicas": d["replicas"].max(),
        "replica_minutes": (d["replicas"] * STEP).sum() / 60,
        "peak_rps": d["rps"].max(),
        "first_breach_s": np.nan, "recovery_s": np.nan, "reaction_s": np.nan,
    }
    if v.any():
        first = d["t_sec"][onset].iloc[0]
        out["first_breach_s"] = first
        # recovery = from the first breach until p95 is back under the SLO for good
        out["recovery_s"] = recovery_time(d, slo, fill_gaps=fill_gaps)
        # reaction = from the first breach until an extra pod (more than at breach start) is ready
        base = d["replicas"][d["t_sec"] == first].iloc[0]
        later = d[(d["t_sec"] >= first) & (d["replicas"] > base)]
        out["reaction_s"] = later["t_sec"].iloc[0] - first if len(later) else np.nan
    return out


def recovery_time(d, slo=SLO, stable_s=60, fill_gaps=False):
    """Frozen protocol (docs/protocol_ec2.md): from the first breach until p95 has stayed below the SLO for 60 s.
    Measured to the START of that 60 s calm period (the protocol wording is ambiguous; add stable_s if the end is meant).
    A calm period cut short by the end of the run counts if the rest of the run is calm. NaN = never recovered."""
    v = breach_flags(d, slo, fill_gaps).fillna(False).values
    t = d["t_sec"].values
    onset = [i for i in range(len(v)) if v[i] and (i == 0 or not v[i - 1])]
    if not onset:
        return np.nan
    n = int(stable_s // STEP)
    for i in range(onset[0] + 1, len(v)):
        if not v[i:i + n].any():
            return float(t[i] - t[onset[0]])
    return np.nan


def reaction_times(d, slo=SLO, plain_only=False):
    """one value per breach onset: seconds until an extra pod is ready (NaN if none came).
    plain_only=True skips breaches that started while the HPA floor was raised (not plain HPA)."""
    v = breach_flags(d, slo)
    res = []
    for t in d["t_sec"][v & ~v.shift(1, fill_value=False)]:
        if plain_only and d["hpa_min"][d["t_sec"] == t].iloc[0] > 1:
            continue
        base = d["replicas"][d["t_sec"] == t].iloc[0]
        later = d[(d["t_sec"] >= t) & (d["replicas"] > base)]
        res.append(later["t_sec"].iloc[0] - t if len(later) else np.nan)
    return res


def summarize(series):
    s = pd.Series(series).dropna()
    if not len(s):
        return {"n": 0}
    return {"n": len(s), "mean": s.mean(), "sd": s.std(ddof=1) if len(s) > 1 else np.nan, "median": s.median(),
            "q1": s.quantile(.25), "q3": s.quantile(.75), "min": s.min(), "max": s.max()}
