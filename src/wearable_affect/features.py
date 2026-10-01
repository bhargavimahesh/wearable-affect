"""Handcrafted physiological features for one window, using NeuroKit2."""

import numpy as np
import neurokit2 as nk
import pandas as pd

from wearable_affect.data import WRIST_FS
from wearable_affect.windows import Window

HEART_FEATURES = ["hr_mean", "hr_std", "hrv_sdnn", "hrv_rmssd"]


def _slope(x: np.ndarray, fs: int) -> float:
    """Slope of a straight-line fit through the signal, in units per second."""
    t = np.arange(len(x)) / fs
    return float(np.polyfit(t, x, deg=1)[0])


def heart_features(bvp: np.ndarray, fs: int) -> dict[str, float]:
    """Heart rate and HRV from a BVP (PPG) segment, using NeuroKit2's Elgendi peak detector."""
    try:
        cleaned = nk.ppg_clean(bvp, sampling_rate=fs)
        peaks = nk.ppg_findpeaks(cleaned, sampling_rate=fs)["PPG_Peaks"]
    except Exception:
        # NeuroKit2 can fail on flat or extremely noisy segments; treat as missing.
        return {name: np.nan for name in HEART_FEATURES}

    ibi = np.diff(peaks) / fs  # inter-beat intervals in seconds
    if len(ibi) < 10:  # too few beats to trust
        return {name: np.nan for name in HEART_FEATURES}

    hr = 60.0 / ibi
    return {
        "hr_mean": float(np.mean(hr)),
        "hr_std": float(np.std(hr)),
        "hrv_sdnn": float(np.std(ibi) * 1000),  # ms
        "hrv_rmssd": float(np.sqrt(np.mean(np.diff(ibi) ** 2)) * 1000),  # ms
    }


def eda_features(eda: np.ndarray, fs: int) -> dict[str, float]:
    """Tonic level (SCL) and phasic responses (SCRs) from an EDA segment."""
    decomposed = nk.eda_phasic(eda, sampling_rate=fs, method="smoothmedian")
    tonic = decomposed["EDA_Tonic"].to_numpy()
    phasic = decomposed["EDA_Phasic"].to_numpy()

    try:
        _, info = nk.eda_peaks(phasic, sampling_rate=fs)
        amplitudes = np.asarray(info["SCR_Amplitude"], dtype=float)
        amplitudes = amplitudes[~np.isnan(amplitudes)]
    except Exception:
        # No detectable responses (e.g. a flat segment) counts as zero SCRs.
        amplitudes = np.array([])

    return {
        "scl_mean": float(np.mean(tonic)),
        "scl_slope": _slope(tonic, fs),
        "phasic_std": float(np.std(phasic)),
        "scr_count": float(len(amplitudes)),
        "scr_amplitude_mean": float(np.mean(amplitudes)) if len(amplitudes) else 0.0,
    }


def extract_features(window: Window) -> dict[str, float]:
    """All handcrafted features for one window, as a flat name -> value dict."""
    eda = window.signals["EDA"][:, 0]
    temp = window.signals["TEMP"][:, 0]
    bvp = window.signals["BVP"][:, 0]
    acc_magnitude = np.linalg.norm(window.signals["ACC"], axis=1)

    features = {}
    features.update(eda_features(eda, WRIST_FS["EDA"]))
    features["temp_mean"] = float(np.mean(temp))
    features["temp_std"] = float(np.std(temp))
    features["temp_slope"] = _slope(temp, WRIST_FS["TEMP"])
    features["acc_mean"] = float(np.mean(acc_magnitude))
    features["acc_std"] = float(np.std(acc_magnitude))
    features.update(heart_features(bvp, WRIST_FS["BVP"]))
    return features


def build_feature_table(windows: list[Window]) -> pd.DataFrame:
    """One row per window: identifiers, target, and all features."""
    rows = []
    for w in windows:
        row = {"subject_id": w.subject_id, "start_s": w.start_s, "target": w.target}
        row.update(extract_features(w))
        rows.append(row)
    return pd.DataFrame(rows)