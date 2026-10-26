"""Frozen contract constants for the AEGIS predictor (Person B).

Everything here mirrors the Day-1 "freeze" list in the roadmap. If the team
changes a contract value, change it HERE only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

STEP_S = 5                 # Prometheus scrape / query step (seconds)
SLO_S = 0.250              # breach = 30 s p95 latency > 250 ms
HORIZON_S = 120            # label: a NEW breach begins within the next 120 s
WINDOWS_S = (30, 60, 120)  # trailing windows for mean / max / slope features
STARTUP_DELAY_S = 30       # pod readiness delay used in the main experiment
RAW_COLS = ("p95", "rps", "err", "cpu", "reps")

# Never use these as model inputs: they are labels / bookkeeping.
LABEL_COLS = ("in_breach", "onset", "y", "censored", "warm", "usable")


@dataclass(frozen=True)
class PromConfig:
    # In-cluster address from A's handoff (kube-prometheus-stack, release "mon").
    # From a laptop use http://localhost:9090 with the kubectl port-forward running.
    base_url: str = "http://mon-kube-prometheus-stack-prometheus.monitoring.svc:9090"
    namespace: str = "aegis"
    deployment: str = "demo"
    pod_regex: str = "demo-.*"
    route: str = "/work"          # only /work requests count (not /metrics, /readyz)
    rate_window: str = "30s"

    @classmethod
    def from_env(cls) -> "PromConfig":
        d = cls()
        return cls(
            base_url=os.getenv("PROM_URL", d.base_url),
            namespace=os.getenv("AEGIS_NAMESPACE", d.namespace),
            deployment=os.getenv("AEGIS_DEPLOYMENT", d.deployment),
            pod_regex=os.getenv("AEGIS_POD_REGEX", d.pod_regex),
            route=os.getenv("AEGIS_ROUTE", d.route),
            rate_window=os.getenv("AEGIS_RATE_WINDOW", d.rate_window),
        )

    def queries(self) -> dict[str, str]:
        w = self.rate_window
        # Empty AEGIS_ROUTE disables the route filter (A's handoff queries have none).
        route_sel = f',route="{self.route}"' if self.route else ""
        sel = f'namespace="{self.namespace}"{route_sel}'
        return {
            "p95": (
                "histogram_quantile(0.95, sum(rate("
                f"http_request_duration_seconds_bucket{{{sel}}}[{w}])) by (le))"
            ),
            "rps": f"sum(rate(http_requests_total{{{sel}}}[{w}]))",
            "err": (
                f'sum(rate(http_requests_total{{{sel},status=~"5.."}}[{w}]))'
                f" / sum(rate(http_requests_total{{{sel}}}[{w}]))"
            ),
            "cpu": (
                "sum(rate(container_cpu_usage_seconds_total{"
                f'namespace="{self.namespace}",pod=~"{self.pod_regex}",'
                f'container!="",container!="POD"}}[{w}]))'
            ),
            "reps": (
                "kube_deployment_status_replicas_available{"
                f'namespace="{self.namespace}",deployment="{self.deployment}"}}'
            ),
        }
