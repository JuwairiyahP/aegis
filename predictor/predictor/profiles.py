"""Training-family load profiles for k6 `ramping-arrival-rate` (real cluster runs).

Calibrated from A's handoff: one pod handles ~3-4 req/s at 250m CPU, 6 pods ~20 req/s,
so a ~2 req/s base ramping to ~8-18 req/s makes HPA alone breach the SLO briefly.
Mild profiles (peak 4-6 req/s) are included on purpose: runs WITHOUT a breach are
needed as honest negatives.

The fixed HELD-OUT evaluation profile belongs to Person C. Do not generate it here
and never train on it.

    python -m predictor.profiles --seed 3            # stages JSON for k6
    python -m predictor.profiles --seed 3 --summary  # human-readable
    python -m predictor.profiles --seed 3 --duration # total seconds
"""
from __future__ import annotations

import argparse
import json

import numpy as np

BASE_RPS = 2.0
IDLE_S = 180      # >120 s so features warm up AND ~60 s of calm rows remain
GAP_S = 150       # between two bursts, lets HPA scale back down (window 60 s)
TAIL_S = 30


def sample_profile(seed: int, base_rps: float = BASE_RPS) -> dict:
    rng = np.random.default_rng(seed)
    n_bursts = int(rng.choice([1, 2], p=[0.6, 0.4]))
    stages = [{"duration": f"{IDLE_S}s", "target": base_rps}]
    bursts = []
    for b in range(n_bursts):
        mild = rng.random() < 0.2
        peak = float(rng.uniform(4, 6) if mild else rng.uniform(8, 18))
        peak = round(peak, 1)
        ramp = int(rng.integers(60, 181))
        plateau = int(rng.integers(90, 151))
        down = 60
        stages += [{"duration": f"{ramp}s", "target": peak},
                   {"duration": f"{plateau}s", "target": peak},
                   {"duration": f"{down}s", "target": base_rps}]
        bursts.append({"peak_rps": peak, "ramp_s": ramp, "plateau_s": plateau, "down_s": down})
        if b < n_bursts - 1:
            stages.append({"duration": f"{GAP_S}s", "target": base_rps})
    stages.append({"duration": f"{TAIL_S}s", "target": base_rps})
    return {"seed": seed, "base_rps": base_rps, "bursts": bursts, "stages": stages}


def duration_s(profile: dict) -> int:
    return sum(int(s["duration"].rstrip("s")) for s in profile["stages"])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Training load profile for k6")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--duration", action="store_true")
    ap.add_argument("--full", action="store_true", help="print the whole profile JSON")
    a = ap.parse_args(argv)
    p = sample_profile(a.seed)
    if a.duration:
        print(duration_s(p))
    elif a.summary:
        print(f"seed {a.seed}: {duration_s(p)} s total")
        for i, b in enumerate(p["bursts"], 1):
            print(f"  burst {i}: peak {b['peak_rps']} req/s, ramp {b['ramp_s']} s, "
                  f"plateau {b['plateau_s']} s")
    elif a.full:
        print(json.dumps(p))
    else:
        print(json.dumps(p["stages"]))


if __name__ == "__main__":
    main()
