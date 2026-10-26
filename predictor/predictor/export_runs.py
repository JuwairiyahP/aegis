"""Export EVERY run listed in a CSV (e.g. harness/runs.csv) with one command.

    # Prometheus port-forward must be running:
    #   kubectl -n monitoring port-forward svc/mon-kube-prometheus-stack-prometheus 9090:9090
    python -m predictor.export_runs --csv ~/aegis/harness/runs.csv --out data/runs

Look first without writing anything:
    python -m predictor.export_runs --csv ~/aegis/harness/runs.csv --dry-run

The CSV columns are auto-detected: run id, profile name, start and end (unix seconds,
milliseconds, or ISO/UTC text; unix columns are preferred when both exist). A run whose
id or profile contains "eval" is exported as scenario=eval (held out, never trained on);
everything else is scenario=train. Override with --id-col/--profile-col/--start-col/--end-col.
Runs already exported are skipped unless --force. Prometheus keeps 7 days of data.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

from .collect import export_one, parse_time
from .config import STARTUP_DELAY_S, PromConfig
from .export import _safe
from .prom_client import PromClient, PromError

ID_CANDIDATES = ["run_id", "run", "id", "run_name", "name", "label", "run_label"]
PROFILE_CANDIDATES = ["profile", "profile_name", "profile_file", "scenario", "load_profile",
                      "script", "name", "label"]


def _rank_time_col(header: str) -> int:
    h = header.lower()
    if "unix" in h or "epoch" in h:
        return 0
    if re.search(r"(^|_)ts$", h) or h.endswith("_s"):
        return 1
    return 2


def detect_columns(headers: list[str], id_col=None, profile_col=None,
                   start_col=None, end_col=None) -> dict:
    low = {h.lower().strip(): h for h in headers}

    def first(cands):
        return next((low[c] for c in cands if c in low), None)

    def time_col(key):
        cands = [h for h in headers if key in h.lower() and "delay" not in h.lower()]
        return min(cands, key=_rank_time_col) if cands else None

    cols = {
        "id": id_col or first(ID_CANDIDATES),
        "profile": profile_col or first(PROFILE_CANDIDATES),
        "start": start_col or time_col("start"),
        "end": end_col or time_col("end"),
    }
    missing = [k for k in ("start", "end") if not cols[k]]
    if missing:
        raise ValueError(f"cannot find {missing} column(s) in CSV headers {headers}; "
                         "pass --start-col / --end-col")
    return cols


def load_rows(csv_path, cols_override: dict, pre_roll: float, post_roll: float,
              eval_match: str = "eval") -> tuple[list[dict], list[str]]:
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        headers = [h for h in (rd.fieldnames or []) if h]
        cols = detect_columns(headers, **cols_override)
        raw_rows = list(rd)
    rows, warnings, seen = [], [], {}
    for i, r in enumerate(raw_rows, start=1):
        try:
            start, end = parse_time(r[cols["start"]]), parse_time(r[cols["end"]])
        except Exception as e:
            warnings.append(f"row {i}: cannot parse start/end ({e}); skipped")
            continue
        if end <= start:
            warnings.append(f"row {i}: end is not after start; skipped")
            continue
        rid = (r.get(cols["id"]) or "").strip() if cols["id"] else ""
        rid = rid or f"run-{i:02d}"
        seen[rid] = seen.get(rid, 0) + 1
        if seen[rid] > 1:
            rid = f"{rid}-{seen[rid]}"
        profile = (r.get(cols["profile"]) or "").strip() if cols["profile"] else ""
        is_eval = eval_match.lower() in f"{rid} {profile}".lower()
        rows.append({"run_id": rid, "profile": profile, "scenario": "eval" if is_eval else "train",
                     "start": start - pre_roll, "end": end + post_roll, "seed": i,
                     "orig_start": start, "orig_end": end})
    return rows, warnings + [f"columns used: {cols}"]


def run_export(rows: list[dict], out, client: PromClient, cfg: PromConfig, *,
               condition: str, startup_delay_s: int, force: bool, prom_url: str = "") -> list[dict]:
    results = []
    for r in rows:
        path = Path(out) / f"run_{_safe(r['run_id'])}.parquet"
        if path.exists() and not force:
            results.append({**r, "status": "skipped (already exported)"})
            continue
        try:
            s = export_one(client, cfg, r["run_id"], r["start"], r["end"], out,
                           scenario=r["scenario"], condition=condition, seed=r["seed"],
                           startup_delay_s=startup_delay_s, prom_url=prom_url,
                           profile={"name": r["profile"], "csv_start": r["orig_start"],
                                    "csv_end": r["orig_end"]})
            note = "OK"
            if s["app_empty"]:
                note = "EMPTY application series"
            elif s["onsets"] == 0:
                note = "OK (no breach in this run)"
            results.append({**r, "status": note, **s})
        except (PromError, ValueError) as e:
            results.append({**r, "status": f"FAILED: {e}"})
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export all runs from a runs CSV")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="data/runs")
    ap.add_argument("--prom-url", default="http://localhost:9090")
    ap.add_argument("--condition", default="baseline", help="baseline (HPA only) or aegis")
    ap.add_argument("--startup-delay", type=int, default=STARTUP_DELAY_S)
    ap.add_argument("--pre-roll", type=float, default=0.0, help="seconds to include before start")
    ap.add_argument("--post-roll", type=float, default=30.0, help="seconds to include after end")
    ap.add_argument("--eval-match", default="eval", help="substring marking held-out runs")
    ap.add_argument("--id-col"); ap.add_argument("--profile-col")
    ap.add_argument("--start-col"); ap.add_argument("--end-col")
    ap.add_argument("--force", action="store_true", help="re-export runs that already exist")
    ap.add_argument("--dry-run", action="store_true", help="show what would be exported")
    a = ap.parse_args(argv)

    rows, notes = load_rows(
        a.csv, dict(id_col=a.id_col, profile_col=a.profile_col,
                    start_col=a.start_col, end_col=a.end_col),
        a.pre_roll, a.post_roll, a.eval_match)
    for n in notes:
        print(n)
    if not rows:
        print("no usable rows in the CSV")
        return 1
    print(f"\n{len(rows)} runs: {sum(r['scenario'] == 'train' for r in rows)} train, "
          f"{sum(r['scenario'] == 'eval' for r in rows)} eval")
    for r in rows:
        print(f"  {r['run_id']:<20} {r['scenario']:<5} {int(r['orig_end'] - r['orig_start']):>5} s  "
              f"{r['profile']}")
    if a.dry_run:
        print("\n(dry run: nothing exported)")
        return 0

    res = run_export(rows, a.out, PromClient(a.prom_url), PromConfig.from_env(),
                     condition=a.condition, startup_delay_s=a.startup_delay, force=a.force,
                     prom_url=a.prom_url)
    print("\nRESULT")
    bad = 0
    for r in res:
        extra = ""
        if "onsets" in r:
            extra = (f"rows={r['rows']} onsets={r['onsets']} max_p95={r['max_p95_ms']:.0f}ms "
                     f"max_reps={r['max_replicas']:.0f}")
        print(f"  {r['run_id']:<20} {r['status']:<28} {extra}")
        bad += r["status"].startswith(("FAILED", "EMPTY"))
    done = [r for r in res if "onsets" in r]
    train_breach = sum(r["onsets"] > 0 for r in done if r["scenario"] == "train")
    print(f"\nexported {len(done)} run(s) to {a.out}; train runs with a breach: {train_breach}")
    if bad:
        print(f"{bad} run(s) need attention (see above). Check the port-forward and "
              "`python -m predictor.check`.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
