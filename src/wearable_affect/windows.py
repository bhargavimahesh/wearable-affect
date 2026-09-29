"""Cut WESAD recordings into fixed-length, single-condition windows."""

from dataclasses import dataclass

import numpy as np

from wearable_affect.data import LABEL_FS, WRIST_FS, SubjectRecording

# WESAD condition -> binary target. Conditions not listed here are dropped.
CONDITION_TO_TARGET = {
    1: 0,  # baseline  -> non-stress
    3: 0,  # amusement -> non-stress
    2: 1,  # stress    -> stress
}


@dataclass
class Window:
    """One fixed-length piece of a recording with a single binary label."""

    subject_id: str
    start_s: int
    target: int  # 1 = stress, 0 = non-stress
    signals: dict[str, np.ndarray]


def make_windows(rec: SubjectRecording, window_s: int = 60, step_s: int = 30) -> list[Window]:
    """Cut one recording into windows that lie entirely within a single kept condition."""
    duration_s = len(rec.labels) // LABEL_FS
    windows = []

    for start_s in range(0, duration_s - window_s + 1, step_s):
        end_s = start_s + window_s

        window_labels = rec.labels[start_s * LABEL_FS : end_s * LABEL_FS]
        condition = int(window_labels[0])
        if condition not in CONDITION_TO_TARGET:
            continue  # transition, meditation, or ignored label
        if not np.all(window_labels == condition):
            continue  # window spans two conditions

        signals = {
            name: rec.signals[name][start_s * fs : end_s * fs]
            for name, fs in WRIST_FS.items()
        }
        if any(len(signals[name]) != window_s * fs for name, fs in WRIST_FS.items()):
            continue  # a signal ended before the labels did

        windows.append(
            Window(
                subject_id=rec.subject_id,
                start_s=start_s,
                target=CONDITION_TO_TARGET[condition],
                signals=signals,
            )
        )

    return windows