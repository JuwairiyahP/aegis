"""Per-run Parquet export. One file per run, keyed by run_id.

We store RAW series (not features) so features can be rebuilt whenever the
feature code changes, and so the split can always be done by whole run.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

META_KEY = b"aegis_meta"


@dataclass
class RunData:
    run_id: str
    raw: pd.DataFrame
    meta: dict

    @property
    def scenario(self) -> str:
        return str(self.meta.get("scenario", "train"))


def _safe(run_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)


def save_run(raw: pd.DataFrame, run_id: str, meta: dict, out_dir) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {**meta, "run_id": run_id}
    df = raw.copy()
    df.index.name = "ts"
    table = pa.Table.from_pandas(df, preserve_index=True)
    md = dict(table.schema.metadata or {})
    md[META_KEY] = json.dumps(meta, default=str).encode()
    table = table.replace_schema_metadata(md)
    path = out_dir / f"run_{_safe(run_id)}.parquet"
    pq.write_table(table, path)
    return path


def load_run(path) -> RunData:
    table = pq.read_table(path)
    meta = json.loads((table.schema.metadata or {}).get(META_KEY, b"{}"))
    df = table.to_pandas()
    if "ts" in df.columns:
        df = df.set_index("ts")
    df.index = pd.DatetimeIndex(df.index)
    run_id = meta.get("run_id") or Path(path).stem.removeprefix("run_")
    return RunData(run_id=run_id, raw=df, meta=meta)


def load_runs(data_dir) -> list[RunData]:
    files = sorted(Path(data_dir).glob("run_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no run_*.parquet files in {data_dir}")
    return [load_run(f) for f in files]
