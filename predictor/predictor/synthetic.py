"""SYNTHETIC run generator - for testing the pipeline before real data exists.

A crude queueing + HPA simulation. It is NOT evidence about the real system.
Every run it writes is tagged source="synthetic", and models trained on it get
"-synthetic" appended to their version so they can't be mistaken for real results.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import SLO_S, STARTUP_DELAY_S, STEP_S
from .export import save_run
from .prom_client import to_index

CAP = 40.0          # req/s one pod can serve
POD_CPU = 0.25      # cores (request = limit)
S0 = 0.02           # base service time (s)
T0 = 1_700_000_000  # arbitrary epoch start


@dataclass
class Profile:
    duration_s: int
    base_rps: float
    bursts: list = field(default_factory=list)  # dicts: start, ramp, peak, plateau, down
    startup_delay_s: int = STARTUP_DELAY_S


def sample_train_profile(seed: int) -> Profile:
    rng = np.random.default_rng(seed)
    base = float(rng.uniform(5, 15))
    t = int(rng.integers(150, 220))
    bursts = []
    for _ in range(int(rng.choice([1, 2], p=[0.6, 0.4]))):
        mild = rng.random() < 0.25
        peak = float(rng.uniform(15, 30) if mild else rng.uniform(35, 230))
        b = dict(start=t, ramp=int(rng.integers(45, 181)), peak=peak,
                 plateau=int(rng.integers(90, 200)), down=60)
        bursts.append(b)
        t += b["ramp"] + b["plateau"] + b["down"] + int(rng.integers(120, 200))
    return Profile(duration_s=t + 60, base_rps=base, bursts=bursts)


def eval_profile() -> Profile:
    """Fixed held-out evaluation scenario (same shape every time)."""
    return Profile(duration_s=780, base_rps=10.0, bursts=[
        dict(start=240, ramp=90, peak=150.0, plateau=180, down=60)])


def _offered(p: Profile, t: int) -> float:
    r = p.base_rps
    for b in p.bursts:
        x = t - b["start"]
        if x < 0:
            continue
        extra = b["peak"] - p.base_rps
        if x < b["ramp"]:
            r += extra * x / b["ramp"]
        elif x < b["ramp"] + b["plateau"]:
            r += extra
        elif x < b["ramp"] + b["plateau"] + b["down"]:
            r += extra * (1 - (x - b["ramp"] - b["plateau"]) / b["down"])
    return r


def simulate(p: Profile, noise_seed: int, hpa_min: int = 1, hpa_max: int = 6,
             target: float = 0.6) -> pd.DataFrame:
    rng = np.random.default_rng(noise_seed)
    ready_times = [-1e9] * hpa_min      # one entry per pod (spec replicas)
    recs, queue = [], 0.0
    rows = []
    for t in range(p.duration_s):
        ready = sum(1 for rt in ready_times if rt <= t)
        rps = max(0.0, _offered(p, t) * (1 + 0.05 * rng.standard_normal()))
        cap = max(ready, 1e-9) * CAP
        rho = rps / cap if ready else 10.0
        queue = max(0.0, queue + rps - cap)
        rc = min(rho, 0.97)
        lat = (S0 * (1 + 3 * rc / (1 - rc)) + queue / cap) * math.exp(0.08 * rng.standard_normal())
        lat = min(lat, 3.0)  # client timeout cap
        served = min(rps, cap if ready else 0.0)
        cpu = served / CAP * POD_CPU * (1 + 0.03 * rng.standard_normal())
        err = max(0.0, (rps - cap) / rps) * 0.3 if rps > 0 else 0.0
        rows.append((rps, lat, cpu, err, ready))
        if t % 15 == 0 and ready:                         # HPA sync
            util = cpu / (ready * POD_CPU)
            if abs(util / target - 1) > 0.1:
                desired = int(np.clip(math.ceil(ready * util / target), hpa_min, hpa_max))
            else:
                desired = len(ready_times)
            recs.append((t, desired))
            recent = [d for (tt, d) in recs if tt > t - 300]   # scale-down stabilisation
            eff = max(recent) if desired < len(ready_times) else desired
            if eff > len(ready_times):
                ready_times += [t + p.startup_delay_s] * (eff - len(ready_times))
            elif eff < len(ready_times):
                ready_times.sort(reverse=True)               # drop not-ready pods first
                ready_times = ready_times[len(ready_times) - eff:]
    a = np.array(rows)
    s = lambda col: pd.Series(a[:, col]).rolling(30, min_periods=1).mean()
    df = pd.DataFrame({"rps": s(0), "p95": s(1), "cpu": s(2), "err": s(3), "reps": a[:, 4]})
    df = df.iloc[::STEP_S].reset_index(drop=True)
    df.index = to_index(T0 + STEP_S * np.arange(len(df)))
    df.index.name = "ts"
    return df[["p95", "rps", "err", "cpu", "reps"]]


def make_dataset(out_dir, n_train: int = 30, n_eval: int = 6, seed: int = 0) -> list[str]:
    ids = []
    for i in range(n_train):
        prof = sample_train_profile(seed * 1000 + i)
        raw = simulate(prof, noise_seed=seed * 1000 + 500 + i)
        rid = f"syn-train-{i:03d}"
        save_run(raw, rid, dict(scenario="train", source="synthetic", condition="baseline",
                                seed=seed * 1000 + i, startup_delay_s=prof.startup_delay_s,
                                slo_s=SLO_S, step_s=STEP_S), out_dir)
        ids.append(rid)
    for i in range(n_eval):
        prof = eval_profile()
        raw = simulate(prof, noise_seed=900_000 + seed * 100 + i)
        rid = f"syn-eval-{i:03d}"
        save_run(raw, rid, dict(scenario="eval", source="synthetic", condition="baseline",
                                seed=900_000 + seed * 100 + i,
                                startup_delay_s=prof.startup_delay_s,
                                slo_s=SLO_S, step_s=STEP_S), out_dir)
        ids.append(rid)
    return ids


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate SYNTHETIC runs (pipeline testing only)")
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n-train", type=int, default=30)
    ap.add_argument("--n-eval", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    ids = make_dataset(a.out, a.n_train, a.n_eval, a.seed)
    print(f"wrote {len(ids)} SYNTHETIC runs to {a.out}")


if __name__ == "__main__":
    main()
