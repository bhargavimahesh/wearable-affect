"""Load WESAD wrist recordings."""

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np

WRIST_FS = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4}  # Hz, from WESAD docs
LABEL_FS = 700  # labels are sampled at the chest device's rate
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "WESAD"/ "WESAD"

@dataclass
class SubjectRecording:
    """One subject's wrist signals and study-condition labels."""

    subject_id: str
    signals: dict[str, np.ndarray]
    labels: np.ndarray


def list_subjects(data_dir: Path) -> list[str]:
    """Return the subject IDs (e.g. 'S2') found in the WESAD folder."""
    data_dir = Path(data_dir)
    return sorted(p.name for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("S"))


def load_subject(data_dir: Path, subject_id: str) -> SubjectRecording:
    """Load one subject's wrist signals and labels from its WESAD pickle file."""
    path = Path(data_dir) / subject_id / f"{subject_id}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No WESAD file for {subject_id} at {path}")

    with open(path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")  # WESAD was saved with Python 2

    signals = {
        name: np.asarray(raw["signal"]["wrist"][name], dtype=np.float32)
        for name in WRIST_FS
    }
    labels = np.asarray(raw["label"], dtype=np.int8)
    return SubjectRecording(subject_id=subject_id, signals=signals, labels=labels)