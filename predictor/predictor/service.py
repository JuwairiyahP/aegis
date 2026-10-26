"""FastAPI /risk service (contract 3.4) plus Prometheus /metrics.

    GET /risk     -> {"risk": 0..1, "ts": "...", "model_version": "...", ...}
    GET /healthz  -> 200 when the model is loaded
    GET /metrics  -> aegis_risk gauge etc. (for Grafana / the report)

The service queries Prometheus itself and builds features with the SAME
`build_features` used in training. If Prometheus is unreachable or data is
missing, /risk returns HTTP 503 (never a made-up risk), so the policy engine can
fall back to plain HPA and log the failure.

A background task re-evaluates every RISK_POLL_SECONDS (default 5; 0 = off) so the
aegis_risk gauge is recorded in BOTH conditions, including baseline runs where no
controller is calling /risk.
"""
from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse

from .config import STEP_S, WINDOWS_S, PromConfig
from .features import FEATURE_NAMES, build_features
from .model_io import load_model, predict_risk
from .prom_client import PromClient, PromError

LOOKBACK_S = max(WINDOWS_S) + 30  # a little more than the longest window
STALE_AFTER_S = 60                # drop aegis_risk from /metrics if older than this


class RiskPredictor:
    def __init__(self, model_dir, client: PromClient, cfg: PromConfig):
        self.model, self.meta = load_model(model_dir)
        if list(self.meta["feature_names"]) != FEATURE_NAMES:
            raise RuntimeError("model feature list differs from current feature code - retrain")
        self.client, self.cfg = client, cfg
        self.last: dict | None = None
        self.last_ok_ts: float | None = None
        self.errors = 0
        self.up = 0

    @property
    def version(self) -> str:
        return self.meta["version"]

    def risk(self, now: float | None = None) -> dict:
        try:
            res = self._risk(now)
        except PromError:
            self.errors += 1
            self.up = 0
            raise
        self.last, self.last_ok_ts, self.up = res, time.time(), 1
        return res

    def _risk(self, now: float | None) -> dict:
        now = time.time() if now is None else now
        end = (int(now) // STEP_S) * STEP_S
        raw = self.client.fetch_raw(end - LOOKBACK_S, end, self.cfg)
        if raw["reps"].isna().all():
            raise PromError("replica metric missing (kube-state-metrics / labels?)")
        if raw[["p95", "rps", "cpu"]].isna().all().all():
            raise PromError("no application metrics returned (check namespace/route labels)")
        feats = build_features(raw, self.meta["slo_s"], self.meta["step_s"])
        last = feats.iloc[[-1]]
        if last.isna().any(axis=None):
            bad = list(last.columns[last.isna().any()])[:5]
            raise PromError(f"incomplete features at latest timestamp: {bad}")
        risk = float(predict_risk(self.model, last[FEATURE_NAMES], self.meta["best_iteration"])[0])
        row = raw.iloc[-1]
        return {
            "risk": risk,
            "ts": last.index[-1].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model_version": self.version,
            "in_breach": bool(row["p95"] > self.meta["slo_s"]),
            "threshold": self.meta["threshold"],
            "inputs": {"p95_ms": round(float(row["p95"]) * 1000, 1),
                       "rps": round(float(row["rps"]), 2),
                       "replicas": None if pd.isna(row["reps"]) else float(row["reps"])},
        }

    def metrics_text(self, now: float | None = None) -> str:
        now = time.time() if now is None else now
        L = [
            "# HELP aegis_predictor_up 1 if the latest risk evaluation succeeded",
            "# TYPE aegis_predictor_up gauge",
            f"aegis_predictor_up {self.up}",
            "# HELP aegis_predictor_errors_total Failed risk evaluations (telemetry problems)",
            "# TYPE aegis_predictor_errors_total counter",
            f"aegis_predictor_errors_total {self.errors}",
            "# HELP aegis_model_info Loaded model version",
            "# TYPE aegis_model_info gauge",
            f'aegis_model_info{{version="{self.version}"}} 1',
            "# HELP aegis_risk_threshold Frozen alarm threshold tau",
            "# TYPE aegis_risk_threshold gauge",
            f"aegis_risk_threshold {self.meta['threshold']}",
        ]
        if self.last is not None and self.last_ok_ts and now - self.last_ok_ts <= STALE_AFTER_S:
            L += ["# HELP aegis_risk Predicted risk of an SLO breach within 120 s",
                  "# TYPE aegis_risk gauge",
                  f"aegis_risk {self.last['risk']}"]
        if self.last_ok_ts:
            L += ["# HELP aegis_predictor_last_success_timestamp_seconds Unix time of last success",
                  "# TYPE aegis_predictor_last_success_timestamp_seconds gauge",
                  f"aegis_predictor_last_success_timestamp_seconds {self.last_ok_ts}"]
        return "\n".join(L) + "\n"


async def _poll(app: FastAPI, every_s: float):
    while True:
        p = app.state.predictor
        if p is not None:
            try:
                await asyncio.to_thread(p.risk)
            except PromError:
                pass  # counted in p.errors; /risk will surface the detail
            except Exception:  # never let the poller die
                p.errors += 1
                p.up = 0
        await asyncio.sleep(every_s)


def create_app(predictor: RiskPredictor | None = None, poll_s: float | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.predictor is None:
            cfg = PromConfig.from_env()
            app.state.predictor = RiskPredictor(
                os.getenv("MODEL_DIR", "models/current"), PromClient(cfg.base_url), cfg)
        every = float(os.getenv("RISK_POLL_SECONDS", "5")) if poll_s is None else poll_s
        task = asyncio.create_task(_poll(app, every)) if every > 0 else None
        yield
        if task:
            task.cancel()

    app = FastAPI(title="AEGIS risk predictor", lifespan=lifespan)
    app.state.predictor = predictor

    @app.get("/healthz")
    def healthz(request: Request):
        p = request.app.state.predictor
        if p is None:
            raise HTTPException(503, "model not loaded")
        return {"status": "ok", "model_version": p.version,
                "synthetic": bool(p.meta.get("synthetic", False))}

    @app.get("/risk")
    def risk(request: Request):
        p = request.app.state.predictor
        if p is None:
            raise HTTPException(503, "model not loaded")
        try:
            return p.risk()
        except PromError as e:
            raise HTTPException(503, f"telemetry unavailable: {e}")

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics(request: Request):
        p = request.app.state.predictor
        if p is None:
            raise HTTPException(503, "model not loaded")
        return PlainTextResponse(p.metrics_text(),
                                 media_type="text/plain; version=0.0.4; charset=utf-8")

    return app


app = create_app()
