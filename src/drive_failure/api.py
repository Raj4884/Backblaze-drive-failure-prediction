"""FastAPI service: POST /predict with SMART readings, get a calibrated failure probability."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from drive_failure.features import add_static_features
from drive_failure.pipeline import load_bundle
from drive_failure.schema import FEATURES

DEFAULT_MODEL_PATH = "models/model.joblib"


class Drive(BaseModel):
    serial_number: str | None = None
    model: str = Field("unknown", description="Drive model string, e.g. 'ST4000DM000'")
    capacity_bytes: float = 4e12
    readings: dict[str, float] = Field(
        default_factory=dict,
        description="Feature values by name, e.g. smart_5_raw, d30_smart_5_raw. Missing = NaN.",
    )


class PredictRequest(BaseModel):
    drives: list[Drive] = Field(min_length=1, max_length=1000)


def create_app(model_path: str | None = None) -> FastAPI:
    app = FastAPI(title="drive-failure", version="0.1.0")
    state: dict = {}
    path = Path(model_path or os.environ.get("DRIVE_FAILURE_MODEL", DEFAULT_MODEL_PATH))

    def bundle() -> dict:
        if "b" not in state:
            try:
                state["b"] = load_bundle(path)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        return state["b"]

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": "b" in state or path.exists()}

    @app.post("/predict")
    def predict(req: PredictRequest):
        b = bundle()
        rows = []
        unknown: set[str] = set()
        for d in req.drives:
            unknown |= {k for k in d.readings if k not in FEATURES}
            rows.append({"model": d.model, "capacity_bytes": d.capacity_bytes, **d.readings})
        df = add_static_features(pd.DataFrame(rows))
        for col in FEATURES:
            if col not in df.columns:
                df[col] = float("nan")
        raw = b["model"].predict_proba(df[FEATURES])[:, 1]
        prob = b["calibrator"].predict(raw)
        return {
            "horizon_days": b["horizon"],
            "model": b["model_name"],
            "alert_threshold": b["alert_threshold"],
            "ignored_fields": sorted(unknown),
            "predictions": [
                {
                    "serial_number": d.serial_number,
                    "failure_probability": float(p),
                    "alert": bool(p >= b["alert_threshold"]),
                }
                for d, p in zip(req.drives, prob, strict=True)
            ],
        }

    return app


app = create_app()
