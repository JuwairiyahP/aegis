"""Save / load / predict. Shared by training and the service so outputs match exactly."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


def predict_risk(model: XGBClassifier, X: pd.DataFrame, best_iteration: int | None) -> np.ndarray:
    rng = (0, int(best_iteration) + 1) if best_iteration is not None else None
    return model.predict_proba(X, iteration_range=rng)[:, 1]


def save_model(model: XGBClassifier, meta: dict, out_dir) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(out / "model.json")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


def load_model(model_dir) -> tuple[XGBClassifier, dict]:
    d = Path(model_dir)
    meta = json.loads((d / "meta.json").read_text())
    model = XGBClassifier()
    model.load_model(d / "model.json")
    return model, meta
