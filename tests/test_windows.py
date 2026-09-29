import numpy as np

from wearable_affect.data import LABEL_FS, WRIST_FS, SubjectRecording
from wearable_affect.windows import make_windows


def fake_recording(conditions: list[tuple[int, int]]) -> SubjectRecording:
    """Build a recording from (condition, seconds) pairs, with all-zero signals."""
    labels = np.concatenate(
        [np.full(seconds * LABEL_FS, condition, dtype=np.int8) for condition, seconds in conditions]
    )
    total_s = sum(seconds for _, seconds in conditions)
    signals = {
        name: np.zeros((total_s * fs, 3 if name == "ACC" else 1), dtype=np.float32)
        for name, fs in WRIST_FS.items()
    }
    return SubjectRecording(subject_id="S_fake", signals=signals, labels=labels)


def test_window_shapes():
    rec = fake_recording([(1, 120)])
    windows = make_windows(rec, window_s=60, step_s=30)

    assert len(windows) == 3  # starts at 0, 30, 60
    for w in windows:
        assert w.signals["BVP"].shape == (60 * 64, 1)
        assert w.signals["ACC"].shape == (60 * 32, 3)
        assert w.signals["EDA"].shape == (60 * 4, 1)


def test_windows_spanning_two_conditions_are_dropped():
    rec = fake_recording([(1, 90), (2, 90)])  # baseline, then stress
    windows = make_windows(rec, window_s=60, step_s=30)

    # Starts 0, 30 are pure baseline; 60 crosses the boundary at 90s; 90, 120 are pure stress.
    assert [w.start_s for w in windows] == [0, 30, 90, 120]
    assert [w.target for w in windows] == [0, 0, 1, 1]


def test_excluded_conditions_are_dropped():
    rec = fake_recording([(0, 60), (4, 60), (1, 60)])  # transition, meditation, baseline
    windows = make_windows(rec, window_s=60, step_s=30)

    assert [w.start_s for w in windows] == [120]
    assert windows[0].target == 0