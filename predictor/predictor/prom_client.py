"""Minimal Prometheus HTTP API client returning regular 5 s pandas frames."""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests

from .config import RAW_COLS, STEP_S, PromConfig


class PromError(RuntimeError):
    """Prometheus unreachable, returned an error, or returned unexpected data."""


def to_index(epoch_seconds) -> pd.DatetimeIndex:
    ms = np.round(np.asarray(epoch_seconds, dtype=float) * 1000).astype("int64")
    return pd.DatetimeIndex(pd.to_datetime(ms, unit="ms", utc=True))


def grid(start: float, end: float, step: int = STEP_S) -> pd.DatetimeIndex:
    n = int(np.floor((end - start) / step + 1e-9)) + 1
    return to_index(start + step * np.arange(n))


class PromClient:
    def __init__(self, base_url: str, timeout: float = 10.0, session=None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def query_range(self, query: str, start: float, end: float,
                    step: int = STEP_S) -> pd.Series:
        """Run a range query; return a float Series on the regular grid (NaN = no data)."""
        try:
            resp = self.session.get(
                f"{self.base_url}/api/v1/query_range",
                params={"query": query, "start": start, "end": end, "step": step},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            body = resp.json()
        except (requests.RequestException, ValueError) as e:
            raise PromError(f"Prometheus request failed: {e}") from e
        if body.get("status") != "success":
            raise PromError(f"Prometheus error: {body.get('error', body)}")
        result = body["data"]["result"]
        idx = grid(start, end, step)
        if len(result) == 0:
            return pd.Series(np.nan, index=idx, dtype=float)
        if len(result) > 1:
            raise PromError(
                f"expected 1 series, got {len(result)} for query: {query} "
                "(check label selectors / duplicate exporters)"
            )
        vals = result[0]["values"]
        s = pd.Series(
            [float(v) for _, v in vals],  # "NaN" strings parse to nan
            index=to_index([t for t, _ in vals]),
            dtype=float,
        )
        s = s[~s.index.duplicated(keep="last")]
        return s.reindex(idx)

    def instant(self, query: str, ts: float | None = None) -> list[tuple[dict, float]]:
        """Instant query -> [(labels, value), ...]. Used by the `check` command."""
        params = {"query": query}
        if ts is not None:
            params["time"] = ts
        try:
            resp = self.session.get(f"{self.base_url}/api/v1/query",
                                    params=params, timeout=self.timeout)
            resp.raise_for_status()
            body = resp.json()
        except (requests.RequestException, ValueError) as e:
            raise PromError(f"Prometheus request failed: {e}") from e
        if body.get("status") != "success":
            raise PromError(f"Prometheus error: {body.get('error', body)}")
        return [(r.get("metric", {}), float(r["value"][1]))
                for r in body["data"]["result"]]

    def fetch_raw(self, start: float, end: float, cfg: PromConfig,
                  step: int = STEP_S) -> pd.DataFrame:
        """The five raw series (p95 in seconds, rps, err fraction, cpu cores, replicas)."""
        cols = {name: self.query_range(q, start, end, step)
                for name, q in cfg.queries().items()}
        df = pd.DataFrame(cols)[list(RAW_COLS)]
        df.index.name = "ts"
        return df
