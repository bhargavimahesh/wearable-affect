"""Synthetic wrist signals for tests, smoke tests and the CI stand-in model. Not real physiology."""

import numpy as np

from wearable_affect.data import WRIST_FS


def synthetic_signals(seconds: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Plausible-looking calm wrist signals: a 66 bpm pulse, slowly varying EDA, noise."""
    rng = np.random.default_rng(seed)
    t = {name: np.arange(seconds * fs) / fs for name, fs in WRIST_FS.items()}
    return {
        "BVP": (np.sin(2 * np.pi * 1.1 * t["BVP"]) + rng.normal(0, 0.05, len(t["BVP"])))[:, None],
        "EDA": (1.0 + 0.1 * np.sin(2 * np.pi * t["EDA"] / 90) + rng.normal(0, 0.003, len(t["EDA"])))[:, None],
        "TEMP": (33 + rng.normal(0, 0.01, len(t["TEMP"])))[:, None],
        "ACC": rng.normal(0, 1, (len(t["ACC"]), 3)),
    }