import neurokit2 as nk
import numpy as np

from wearable_affect.features import extract_features, heart_features
from wearable_affect.windows import Window


def fake_window(heart_hz: float = 1.2, eda: np.ndarray | None = None) -> Window:
    """A 60 s window with a clean sine 'pulse'; EDA defaults to a linear rise."""
    t_bvp = np.arange(60 * 64) / 64
    if eda is None:
        eda = 1.0 + 0.01 * np.arange(60 * 4) / 4  # rises 0.01 µS per second
    signals = {
        "BVP": np.sin(2 * np.pi * heart_hz * t_bvp)[:, None],
        "EDA": np.asarray(eda)[:, None],
        "TEMP": np.full((60 * 4, 1), 33.0),
        "ACC": np.zeros((60 * 32, 3)),
    }
    return Window(subject_id="S_fake", start_s=0, target=0, signals=signals)


def test_heart_rate_of_clean_pulse():
    features = extract_features(fake_window(heart_hz=1.2))
    assert abs(features["hr_mean"] - 72) < 1  # 1.2 beats/s = 72 bpm


def test_regular_pulse_has_low_hrv():
    features = extract_features(fake_window(heart_hz=1.2))
    # Peaks can only land on whole samples: at 64 Hz one sample is 15.6 ms,
    # so even a perfectly regular pulse shows a little HRV.
    assert features["hrv_rmssd"] < 20


def test_flat_bvp_gives_missing_heart_features():
    features = heart_features(np.zeros(60 * 64), fs=64)
    assert all(np.isnan(v) for v in features.values())


def test_rising_eda_has_positive_tonic_slope():
    features = extract_features(fake_window())
    assert abs(features["scl_slope"] - 0.01) < 0.001


def test_flat_eda_has_no_responses():
    features = extract_features(fake_window(eda=np.full(60 * 4, 2.0)))
    assert features["scr_count"] == 0
    assert features["scl_mean"] == 2.0


def test_simulated_responses_are_detected():
    eda = nk.eda_simulate(duration=60, sampling_rate=4, scr_number=4, random_state=0)
    features = extract_features(fake_window(eda=eda))
    assert features["scr_count"] >= 2

def test_calm_noisy_eda_has_few_responses():
    rng = np.random.default_rng(0)
    eda = 1.3 + rng.normal(0, 0.003, 60 * 4)  # calm level with small sensor noise
    features = extract_features(fake_window(eda=eda))
    assert features["scr_count"] <= 5  # real resting SCR rates are a few per minute