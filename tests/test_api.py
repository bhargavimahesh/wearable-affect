import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from wearable_affect.api import WristSignals, create_app
from wearable_affect.detector import StressDetector, cut_windows
from wearable_affect.features import features_from_signals
from wearable_affect.models import make_logreg
from wearable_affect.synthetic import synthetic_signals as fake_signals

def as_json(signals: dict[str, np.ndarray]) -> dict:
    """Signals in the API's request format."""
    return WristSignals.from_arrays(signals).model_dump()


@pytest.fixture(scope="module")
def client() -> TestClient:
    """An API backed by a small detector trained on fake data, so tests don't need the real model."""
    signals = fake_signals(360)
    names = list(features_from_signals(cut_windows(signals)[0]).keys())
    rng = np.random.default_rng(0)
    model = make_logreg().fit(pd.DataFrame(rng.normal(size=(20, len(names))), columns=names), [0, 1] * 10)
    detector = StressDetector(model=model, feature_names=names, metadata={"model": "test"})
    return TestClient(create_app(detector))


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_calibrate_then_predict(client):
    calibration = client.post("/calibrate", json=as_json(fake_signals(300)))
    assert calibration.status_code == 200
    assert calibration.json()["n_windows"] == 9

    window = fake_signals(60, seed=1)
    response = client.post("/predict", json={"signals": as_json(window), "baseline": calibration.json()["baseline"]})
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["stress_probability"] <= 1
    assert body["is_stress"] == (body["stress_probability"] >= body["threshold"])


def test_predict_rejects_wrong_duration(client):
    calibration = client.post("/calibrate", json=as_json(fake_signals(300))).json()
    response = client.post("/predict", json={"signals": as_json(fake_signals(90)), "baseline": calibration["baseline"]})
    assert response.status_code == 400


def test_signals_of_different_duration_are_rejected(client):
    body = as_json(fake_signals(60))
    body["eda"] = body["eda"][:-4]  # EDA one second shorter than the rest
    assert client.post("/calibrate", json=body).status_code == 422


def test_calibration_too_short_is_rejected(client):
    assert client.post("/calibrate", json=as_json(fake_signals(30))).status_code == 400


def test_flat_pulse_is_reported_as_missing_heart_features(client):
    calibration = client.post("/calibrate", json=as_json(fake_signals(300))).json()
    window = fake_signals(60)
    window["BVP"] = np.zeros_like(window["BVP"])  # e.g. sensor lost contact
    response = client.post("/predict", json={"signals": as_json(window), "baseline": calibration["baseline"]})
    assert response.status_code == 200
    assert "hr_mean" in response.json()["missing_features"]