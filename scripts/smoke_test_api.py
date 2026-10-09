"""Smoke test a running API: health, calibrate, predict, and compare with the detector used directly.

Usage: python scripts/smoke_test_api.py [API_URL] [MODEL_PATH]
"""

import sys
import time

import httpx
import numpy as np

from wearable_affect.api import WristSignals
from wearable_affect.synthetic import synthetic_signals as fake_signals
from wearable_affect.detector import StressDetector




api = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
model_path = sys.argv[2] if len(sys.argv) > 2 else "artifacts/stress_detector.joblib"

calibration, window = fake_signals(300), fake_signals(60, seed=1)

with httpx.Client(base_url=api, timeout=30) as client:
    health = client.get("/health")
    health.raise_for_status()
    print("health:", health.json())

    r = client.post("/calibrate", json=WristSignals.from_arrays(calibration).model_dump())
    r.raise_for_status()
    baseline = r.json()["baseline"]

    t0 = time.perf_counter()
    r = client.post("/predict", json={"signals": WristSignals.from_arrays(window).model_dump(), "baseline": baseline})
    latency_ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    served = r.json()["stress_probability"]

detector = StressDetector.load(model_path)
direct = detector.predict_proba(window, detector.baseline_from_calibration(calibration))

print(f"served {served:.6f}   direct {direct:.6f}   latency {latency_ms:.0f} ms")
assert abs(served - direct) < 1e-4, "API and direct detector disagree"
print("SMOKE TEST PASSED")