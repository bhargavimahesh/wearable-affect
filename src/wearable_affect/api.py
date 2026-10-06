"""HTTP API around the StressDetector: calibrate once per user, then predict per 60 s window."""

import math
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from wearable_affect.data import PROJECT_ROOT, WRIST_FS
from wearable_affect.detector import CALIBRATION_MIN, WINDOW_S, StressDetector, cut_windows

DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "stress_detector.joblib"
MAX_CALIBRATION_S = 15 * 60


class WristSignals(BaseModel):
    """Raw wrist signals at Empatica E4 rates: BVP 64 Hz, EDA 4 Hz, TEMP 4 Hz, ACC 32 Hz (x, y, z)."""

    bvp: list[float]
    eda: list[float]
    temp: list[float]
    acc: list[tuple[float, float, float]]

    @model_validator(mode="after")
    def check_signals(self) -> "WristSignals":
        arrays = self.as_arrays()
        durations = {name: len(arrays[name]) / fs for name, fs in WRIST_FS.items()}
        if len(set(durations.values())) != 1:
            raise ValueError(f"All signals must cover the same duration; got seconds {durations}")
        for name, values in arrays.items():
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains NaN or infinite values")
        return self

    @classmethod
    def from_arrays(cls, signals: dict[str, np.ndarray]) -> "WristSignals":
        """Build a request body from arrays in the detector's format (the inverse of as_arrays)."""
        return cls(
            bvp=signals["BVP"][:, 0].tolist(),
            eda=signals["EDA"][:, 0].tolist(),
            temp=signals["TEMP"][:, 0].tolist(),
            acc=signals["ACC"].tolist(),
        )

    def duration_s(self) -> float:
        return len(self.eda) / WRIST_FS["EDA"]

    def as_arrays(self) -> dict[str, np.ndarray]:
        """In the shapes the detector expects: (n, 1) for single channels, (n, 3) for ACC."""
        return {
            "BVP": np.asarray(self.bvp, dtype=float)[:, None],
            "EDA": np.asarray(self.eda, dtype=float)[:, None],
            "TEMP": np.asarray(self.temp, dtype=float)[:, None],
            "ACC": np.asarray(self.acc, dtype=float).reshape(-1, 3),
        }


class CalibrationResponse(BaseModel):
    baseline: dict[str, float | None]
    n_windows: int


class PredictRequest(BaseModel):
    signals: WristSignals
    baseline: dict[str, float | None] = Field(description="Exactly as returned by /calibrate")


class PredictResponse(BaseModel):
    stress_probability: float
    is_stress: bool
    threshold: float
    missing_features: list[str]


def _to_json(values: dict[str, float]) -> dict[str, float | None]:
    """JSON has no NaN, so missing values travel as null."""
    return {k: None if math.isnan(v) else v for k, v in values.items()}


def _from_json(values: dict[str, float | None]) -> dict[str, float]:
    return {k: float("nan") if v is None else v for k, v in values.items()}


def create_app(detector: StressDetector | None = None) -> FastAPI:
    """Build the API. Without a detector, load one from $MODEL_PATH (default: artifacts/)."""
    if detector is None:
        detector = StressDetector.load(Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)))

    app = FastAPI(title="Wearable stress detection", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": detector.metadata}

    @app.post("/calibrate", response_model=CalibrationResponse)
    def calibrate(signals: WristSignals) -> CalibrationResponse:
        duration = signals.duration_s()
        if not WINDOW_S <= duration <= MAX_CALIBRATION_S:
            raise HTTPException(
                400,
                f"Calibration must last {WINDOW_S}-{MAX_CALIBRATION_S} s, got {duration:.0f} s "
                f"(recommended: {CALIBRATION_MIN} min of calm sitting)",
            )
        arrays = signals.as_arrays()
        baseline = detector.baseline_from_calibration(arrays)
        return CalibrationResponse(baseline=_to_json(baseline), n_windows=len(cut_windows(arrays)))

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        if request.signals.duration_s() != WINDOW_S:
            raise HTTPException(400, f"A prediction needs exactly {WINDOW_S} s of signals")
        missing_from_baseline = set(detector.feature_names) - set(request.baseline)
        if missing_from_baseline:
            raise HTTPException(400, f"Baseline lacks {sorted(missing_from_baseline)}; call /calibrate first")

        prob, missing = detector.predict_with_quality(request.signals.as_arrays(), _from_json(request.baseline))
        return PredictResponse(
            stress_probability=prob,
            is_stress=prob >= detector.threshold,
            threshold=detector.threshold,
            missing_features=missing,
        )

    return app