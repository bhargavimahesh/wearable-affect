"""The deployable stress detector: features -> personal baseline -> model -> probability."""

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from wearable_affect.data import WRIST_FS
from wearable_affect.evaluation import feature_columns
from wearable_affect.features import features_from_signals
from wearable_affect.models import make_logreg
from wearable_affect.personalisation import split_calibration, subtract_baseline

WINDOW_S = 60
STEP_S = 30
CALIBRATION_MIN = 5


def cut_windows(signals: dict[str, np.ndarray], window_s: int = WINDOW_S, step_s: int = STEP_S) -> list[dict]:
    """Cut an unlabelled recording into windows, exactly like make_windows does for WESAD."""
    duration_s = min(len(signals[name]) // fs for name, fs in WRIST_FS.items())
    return [
        {name: signals[name][start * fs : (start + window_s) * fs] for name, fs in WRIST_FS.items()}
        for start in range(0, duration_s - window_s + 1, step_s)
    ]


@dataclass
class StressDetector:
    """A trained model plus everything needed to apply it to raw wrist signals."""

    model: object
    feature_names: list[str]
    threshold: float = 0.5
    metadata: dict = field(default_factory=dict)

    def baseline_from_calibration(self, signals: dict[str, np.ndarray]) -> dict[str, float]:
        """A person's calm baseline (mean of each feature) from a calm calibration recording."""
        windows = cut_windows(signals)
        if not windows:
            raise ValueError(f"Calibration recording is shorter than one {WINDOW_S} s window")
        features = pd.DataFrame([features_from_signals(w) for w in windows])[self.feature_names]
        return features.mean().to_dict()

    def predict_proba(self, window_signals: dict[str, np.ndarray], baseline: dict[str, float]) -> float:
        """Probability of stress for one 60 s window, relative to the person's baseline."""
        features = pd.DataFrame([features_from_signals(window_signals)])[self.feature_names]
        relative = features - pd.Series(baseline)[self.feature_names]
        return float(self.model.predict_proba(relative)[0, 1])

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "StressDetector":
        """Load a saved detector. Only load files you created yourself: loading can run code."""
        detector = joblib.load(path)
        trained_with = detector.metadata.get("sklearn_version")
        if trained_with != sklearn.__version__:
            warnings.warn(f"Detector trained with scikit-learn {trained_with}, running {sklearn.__version__}")
        return detector


def train_detector(features: pd.DataFrame, calibration_min: int = CALIBRATION_MIN) -> StressDetector:
    """Train the deployed configuration on all subjects: baseline-normalised logistic regression."""
    calibration, rest = split_calibration(features, calibration_min)
    rest = subtract_baseline(rest, calibration)
    cols = feature_columns(rest)

    model = make_logreg()
    model.fit(rest[cols], rest["target"])

    return StressDetector(
        model=model,
        feature_names=cols,
        metadata={
            "model": "logreg",
            "calibration_min": calibration_min,
            "window_s": WINDOW_S,
            "n_subjects": int(features["subject_id"].nunique()),
            "n_training_windows": len(rest),
            "sklearn_version": sklearn.__version__,
        },
    )