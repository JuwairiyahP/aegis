"""Turn runs into a labelled feature table, and split BY RUN (never by row)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import HORIZON_S, SLO_S, STEP_S
from .export import RunData
from .features import FEATURE_NAMES, build_features, clean_raw
from .labels import label_run


def prepare_run(run: RunData, slo_s: float = SLO_S, horizon_s: int = HORIZON_S,
                step_s: int = STEP_S) -> pd.DataFrame:
    raw = clean_raw(run.raw)
    feats = build_features(raw, slo_s, step_s)
    warm = feats.notna().all(axis=1)
    labels = label_run(raw["p95"], warm, slo_s, horizon_s, step_s)
    df = pd.concat([feats, labels], axis=1)
    df["p95_s"] = raw["p95"]
    df["run_id"] = run.run_id
    df["scenario"] = run.scenario
    return df


def build_dataset(runs: list[RunData], **kw) -> pd.DataFrame:
    return pd.concat([prepare_run(r, **kw) for r in runs])


@dataclass
class RunInfo:
    run_id: str
    scenario: str
    has_onset: bool


@dataclass
class Split:
    train: list[str]
    val: list[str]
    test: list[str]
    evaluation_type: str


def split_runs(infos: list[RunInfo], val_frac: float = 0.2, test_frac: float = 0.2,
               seed: int = 0, heldout_scenarios=("eval",)) -> Split:
    """Split whole runs. Scenarios in `heldout_scenarios` are test-only.

    If none exist, test runs are drawn from the training family and the result is
    labelled as NOT a held-out profile. Stratified so runs with breaches are spread
    across train/val/test.
    """
    rng = np.random.RandomState(seed)
    held = [i for i in infos if i.scenario in heldout_scenarios]
    pool = [i for i in infos if i.scenario not in heldout_scenarios]

    def pick(items, frac):
        picked, rest = [], []
        for flag in (True, False):
            g = sorted((i for i in items if i.has_onset == flag), key=lambda i: i.run_id)
            rng.shuffle(g)
            k = int(round(frac * len(g)))
            if flag and frac > 0 and len(g) >= 3:
                k = max(k, 1)
            picked += g[:k]
            rest += g[k:]
        return picked, rest

    if held:
        test = held
        evaluation_type = "held-out profile"
        val, train = pick(pool, val_frac)
    else:
        evaluation_type = "held-out runs from the SAME profile family (not a held-out profile)"
        test, rest = pick(pool, test_frac)
        val, train = pick(rest, val_frac / max(1 - test_frac, 1e-9))
    if not train or not any(i.has_onset for i in train):
        raise ValueError("training split has no run with a breach - collect more runs")
    ids = lambda xs: sorted(i.run_id for i in xs)
    s = Split(ids(train), ids(val), ids(test), evaluation_type)
    assert not (set(s.train) & set(s.val) | set(s.train) & set(s.test) | set(s.val) & set(s.test))
    return s


def xy(df: pd.DataFrame):
    u = df[df["usable"]]
    return u[FEATURE_NAMES], u["y"].to_numpy()
