import numpy as np
import pandas as pd
import pytest
import sklearn

from wearable_affect.data import LABEL_FS, WRIST_FS, SubjectRecording
from wearable_affect.detector import StressDetector, cut_windows
from wearable_affect.features import build_feature_table, features_from_signals
from wearable_affect.models import make_logreg
from wearable_affect.personalisation import split_calibration, subtract_baseline
from wearable_affect.windows import make_windows


def fake_signals(seconds: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Plausible-looking calm wrist signals: a 66 bpm pulse, slowly varying EDA, noise."""
    rng = np.random.default_rng(seed)
    t = {name: np.arange(seconds * fs) / fs for name, fs in WRIST_FS.items()}
    return {
        "BVP": (np.sin(2 * np.pi * 1.1 * t["BVP"]) + rng.normal(0, 0.05, len(t["BVP"])))[:, None],
        "EDA": (1.0 + 0.1 * np.sin(2 * np.pi * t["EDA"] / 90) + rng.normal(0, 0.003, len(t["EDA"])))[:, None],
        "TEMP": (33 + rng.normal(0, 0.01, len(t["TEMP"])))[:, None],
        "ACC": rng.normal(0, 1, (len(t["ACC"]), 3)),
    }


def test_five_minutes_give_nine_windows():
    assert len(cut_windows(fake_signals(300))) == 9  # starts at 0, 30, ..., 240 s


def test_calibration_shorter_than_one_window_is_rejected():
    detector = StressDetector(model=None, feature_names=[])
    with pytest.raises(ValueError):
        detector.baseline_from_calibration(fake_signals(30))


def test_serving_matches_training_pipeline():
    """The detector applied to raw signals must give exactly what the training pipeline computes."""
    signals = fake_signals(420)
    recording = SubjectRecording("S_fake", signals, np.ones(420 * LABEL_FS, dtype=np.int8))  # all baseline

    # Training path: labelled windows -> feature table -> calibration split -> baseline subtraction
    table = build_feature_table(make_windows(recording))
    calibration, rest = split_calibration(table, minutes=5)
    rest = subtract_baseline(rest, calibration)
    cols = [c for c in rest.columns if c not in ("subject_id", "start_s", "target")]
    model = make_logreg().fit(rest[cols], [0, 1, 0, 1])  # any fitted model will do
    expected = model.predict_proba(rest[cols])[:, 1]

    # Serving path: raw signals only
    detector = StressDetector(model=model, feature_names=cols)
    baseline = detector.baseline_from_calibration({k: v[: 300 * WRIST_FS[k]] for k, v in signals.items()})
    served = [detector.predict_proba(w, baseline) for w in cut_windows(signals)[9:]]

    assert np.allclose(served, expected)


def test_save_and_load_give_identical_predictions(tmp_path):
    signals = fake_signals(360)
    names = list(features_from_signals(cut_windows(signals)[0]).keys())
    rng = np.random.default_rng(1)
    model = make_logreg().fit(pd.DataFrame(rng.normal(size=(20, len(names))), columns=names), [0, 1] * 10)
    detector = StressDetector(model=model, feature_names=names, metadata={"sklearn_version": sklearn.__version__})
    baseline = detector.baseline_from_calibration(signals)

    path = tmp_path / "detector.joblib"
    detector.save(path)
    loaded = StressDetector.load(path)

    window = cut_windows(signals)[-1]
    assert loaded.predict_proba(window, baseline) == detector.predict_proba(window, baseline)


def test_loading_warns_about_version_mismatch(tmp_path):
    path = tmp_path / "detector.joblib"
    StressDetector(model=None, feature_names=[], metadata={"sklearn_version": "0.0"}).save(path)
    with pytest.warns(UserWarning):
        StressDetector.load(path)