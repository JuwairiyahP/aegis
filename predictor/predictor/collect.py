"""Export one finished run from live Prometheus to Parquet.

    # inside WSL, with the port-forward running:
    #   kubectl -n monitoring port-forward svc/mon-kube-prometheus-stack-prometheus 9090:9090
    python -m predictor.collect --run-id r001 --scenario train --last-minutes 12
    python -m predictor.collect --run-id r001 --start 2026-10-25T09:10:00Z --end 2026-10-25T09:22:00Z
    python -m predictor.collect --run-id r001 --start 1793000000 --end 1793000700   # unix seconds

Times without a zone are read as UTC. India time is UTC+5:30.
To export many runs from a CSV, use `python -m predictor.export_runs` instead.
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from .config import SLO_S, STARTUP_DELAY_S, STEP_S, PromConfig
from .export import save_run
from .labels import breach_flags, breach_onsets
from .prom_client import PromClient


def parse_time(s) -> float:
    """Unix seconds (or milliseconds), or an ISO timestamp (naive = UTC)."""
    try:
        v = float(s)
        return v / 1000.0 if v > 1e11 else v   # 1.79e12 = milliseconds
    except (TypeError, ValueError):
        t = pd.Timestamp(str(s).strip())
        t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        return t.timestamp()


def export_one(client: PromClient, cfg: PromConfig, run_id: str, start: float, end: float,
               out, *, scenario: str = "train", condition: str = "baseline", seed: int = 0,
               startup_delay_s: int = STARTUP_DELAY_S, profile: dict | None = None,
               prom_url: str = "") -> dict:
    """Fetch the raw series for [start, end], save one Parquet file, return a summary."""
    start = (int(start) // STEP_S) * STEP_S
    end = (int(end) // STEP_S) * STEP_S
    if end <= start:
        raise ValueError(f"end must be after start (start={start}, end={end})")
    raw = client.fetch_raw(start, end, cfg)
    flag = breach_flags(raw["p95"].fillna(0.0))
    meta = dict(scenario=scenario, source="prometheus", condition=condition, seed=seed,
                startup_delay_s=startup_delay_s, slo_s=SLO_S, step_s=STEP_S,
                started_at=str(raw.index[0]), ended_at=str(raw.index[-1]),
                prom_url=prom_url, namespace=cfg.namespace, route=cfg.route)
    if profile:
        meta["profile"] = profile
    path = save_run(raw, run_id, meta, out)
    return {
        "path": str(path), "rows": len(raw),
        "missing": raw.isna().mean().round(3).to_dict(),
        "max_p95_ms": float(raw["p95"].max() * 1000) if raw["p95"].notna().any() else float("nan"),
        "breach_rows": int(flag.sum()), "onsets": int(breach_onsets(flag).sum()),
        "max_replicas": float(raw["reps"].max()) if raw["reps"].notna().any() else float("nan"),
        "app_empty": bool(raw["p95"].isna().all() or raw["rps"].isna().all()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Export a run from Prometheus")
    ap.add_argument("--prom-url", default=PromConfig.from_env().base_url)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--scenario", default="train", choices=["train", "eval"])
    ap.add_argument("--condition", default="baseline", help="baseline (HPA only) or aegis")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--startup-delay", type=int, default=STARTUP_DELAY_S)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--last-minutes", type=float)
    ap.add_argument("--profile", help="JSON describing the load profile (stored as metadata)")
    ap.add_argument("--out", default="data/runs")
    a = ap.parse_args(argv)

    if a.last_minutes:
        end = (int(time.time()) // STEP_S) * STEP_S
        start = end - int(a.last_minutes * 60)
    elif a.start and a.end:
        start, end = parse_time(a.start), parse_time(a.end)
    else:
        ap.error("give --start and --end, or --last-minutes")

    s = export_one(PromClient(a.prom_url), PromConfig.from_env(), a.run_id, start, end, a.out,
                   scenario=a.scenario, condition=a.condition, seed=a.seed,
                   startup_delay_s=a.startup_delay,
                   profile=json.loads(a.profile) if a.profile else None, prom_url=a.prom_url)
    print(f"saved {s['path']}  rows={s['rows']}  missing-fraction={s['missing']}")
    print(f"max p95={s['max_p95_ms']:.0f} ms  breach rows={s['breach_rows']}  "
          f"onsets={s['onsets']}  max replicas={s['max_replicas']}")
    if s["app_empty"]:
        print("WARNING: application series are empty - run `python -m predictor.check`")


if __name__ == "__main__":
    main()
