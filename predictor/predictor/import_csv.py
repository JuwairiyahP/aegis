"""Convert run CSVs exported by A's tooling into the Parquet runs the trainer reads.

    python -m predictor.import_csv --src data\\runs_csv --out data\\runs

Expected CSV columns (as in the repo's data/runs/*.csv):
    ts_unix, rps, p95 (seconds), error_rate, cpu (cores), replicas   [+ hpa_min, run_id,
    scenario, peak, ramp, hold, bursts]
A run is held out (scenario=eval) if its `scenario` column says "eval" or its file name
contains "eval"/"ev-". Everything else is scenario=train.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW_COLS, SLO_S, STARTUP_DELAY_S, STEP_S
from .export import save_run
from .prom_client import to_index

RENAME = {"error_rate": "err", "replicas": "reps"}
REQUIRED = ["ts_unix", "rps", "p95", "cpu", "replicas"]


def convert_file(path: Path, out_dir, condition: str = "baseline",
                 startup_delay_s: int = STARTUP_DELAY_S) -> dict:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing} (has {list(df.columns)})")
    if "error_rate" not in df.columns:
        df["error_rate"] = np.nan
    ts = df["ts_unix"].astype(float).to_numpy()
    if not np.all(np.diff(ts) > 0):
        raise ValueError(f"{path.name}: ts_unix is not strictly increasing")
    raw = pd.DataFrame({"p95": df["p95"], "rps": df["rps"], "err": df["error_rate"],
                        "cpu": df["cpu"], "reps": df["replicas"]}).astype(float)
    raw.index = to_index(ts)
    raw.index.name = "ts"
    grid = pd.date_range(raw.index[0], raw.index[-1], freq=f"{STEP_S}s")
    gaps = len(grid) - len(raw)
    raw = raw.reindex(grid)[list(RAW_COLS)]
    raw.index.name = "ts"
    first = df.iloc[0]
    rid = str(first["run_id"]) if "run_id" in df.columns else path.stem
    scen = str(first["scenario"]).lower() if "scenario" in df.columns else ""
    is_eval = scen == "eval" or "eval" in path.stem.lower() or path.stem.lower().startswith("ev-")
    prof = {k: (None if pd.isna(first[k]) else float(first[k]))
            for k in ("base", "calm", "peak", "ramp", "hold", "bursts", "gap", "floor")
            if k in df.columns and not isinstance(first[k], str)}
    if prof.get("ramp") is not None:          # defaults of loadtest/ramp.js
        prof.setdefault("calm", 180.0); prof.setdefault("bursts", 1.0)
        prof.setdefault("gap", 150.0); prof.setdefault("down", 60.0)
        for k in ("calm", "bursts", "gap"):
            if prof.get(k) is None:
                prof[k] = {"calm": 180.0, "bursts": 1.0, "gap": 150.0}[k]
    if "condition" in df.columns and isinstance(first["condition"], str):
        condition = first["condition"]
    meta = dict(scenario="eval" if is_eval else "train", source="prometheus-csv",
                condition=condition, seed=0, startup_delay_s=startup_delay_s, slo_s=SLO_S,
                step_s=STEP_S, started_at=str(raw.index[0]), ended_at=str(raw.index[-1]),
                profile=prof, imported_from=path.name)
    save_run(raw, rid, meta, out_dir)
    return {"run_id": rid, "scenario": meta["scenario"], "rows": len(raw), "gaps_filled": gaps,
            "max_p95_ms": float(raw["p95"].max() * 1000), "max_reps": float(raw["reps"].max()),
            "breach_rows": int((raw["p95"] > SLO_S).sum())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Convert run CSVs to Parquet runs")
    ap.add_argument("--src", required=True, help="folder with the run CSV files")
    ap.add_argument("--out", default="data/runs")
    ap.add_argument("--condition", default="baseline")
    ap.add_argument("--startup-delay", type=int, default=STARTUP_DELAY_S)
    a = ap.parse_args(argv)
    files = sorted(Path(a.src).glob("*.csv"))
    if not files:
        print(f"no .csv files in {a.src}")
        return 1
    bad = 0
    for f in files:
        try:
            s = convert_file(f, a.out, a.condition, a.startup_delay)
            print(f"{s['run_id']:<8} {s['scenario']:<5} rows={s['rows']:<4} gaps={s['gaps_filled']} "
                  f"max_p95={s['max_p95_ms']:.0f}ms max_reps={s['max_reps']:.0f} "
                  f"breach_rows={s['breach_rows']}")
        except Exception as e:  # keep going, report at the end
            bad += 1
            print(f"FAILED {f.name}: {e}")
    print(f"\nconverted {len(files) - bad}/{len(files)} run(s) to {a.out}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
