"""PaPaGei-S adapter: turn 60 s BVP windows into one embedding per window."""

import numpy as np
import pandas as pd
import torch
from scipy import signal

from wearable_affect.data import PROJECT_ROOT, WRIST_FS
from wearable_affect.third_party.papagei_resnet import ResNet1DMoE
from wearable_affect.windows import Window

FS_TARGET = 125  # Hz, PaPaGei's input rate
SEGMENT_S = 10  # seconds per PaPaGei input segment
EMBEDDING_DIM = 512
WEIGHTS_PATH = PROJECT_ROOT / "weights" / "papagei_s.pt"
ID_COLUMNS = ["subject_id", "start_s", "target"]


def preprocess_bvp(bvp: np.ndarray, fs: int) -> np.ndarray:
    """Replicate PaPaGei's preprocessing. Returns (n_segments, SEGMENT_S * FS_TARGET)."""
    b, a = signal.cheby2(4, 20, [0.5, 12], btype="bandpass", fs=fs)
    filtered = signal.filtfilt(b, a, bvp)

    seg_len = SEGMENT_S * fs
    n_segments = len(filtered) // seg_len
    segments = filtered[: n_segments * seg_len].reshape(n_segments, seg_len)

    mean = segments.mean(axis=1, keepdims=True)
    std = segments.std(axis=1, keepdims=True)
    segments = (segments - mean) / np.where(std > 0, std, 1.0)  # flat segment -> zeros, not NaN

    return signal.resample_poly(segments, up=FS_TARGET, down=fs, axis=1)


def load_papagei(device: str | None = None) -> tuple[ResNet1DMoE, str]:
    """Build PaPaGei-S, load the pretrained weights, and move it to the GPU if available."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet1DMoE(
        in_channels=1, base_filters=32, kernel_size=3, stride=2,
        groups=1, n_block=18, n_classes=EMBEDDING_DIM, n_experts=3,
    )
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
    return model.to(device).eval(), device


def embed_windows(windows: list[Window], model: ResNet1DMoE, device: str, batch_windows: int = 64) -> np.ndarray:
    """One embedding per window: the mean of its 10 s segment embeddings. Shape (n_windows, 512)."""
    embeddings = []
    for start in range(0, len(windows), batch_windows):
        batch = windows[start : start + batch_windows]
        segments = np.stack([preprocess_bvp(w.signals["BVP"][:, 0], WRIST_FS["BVP"]) for w in batch])
        n_win, n_seg, length = segments.shape  # e.g. (64, 6, 1250)

        x = torch.from_numpy(segments.reshape(n_win * n_seg, 1, length).astype(np.float32)).to(device)
        with torch.inference_mode():
            segment_embeddings = model(x)[0].cpu().numpy()  # (n_win * n_seg, 512)

        embeddings.append(segment_embeddings.reshape(n_win, n_seg, -1).mean(axis=1))
    return np.vstack(embeddings)


def build_embedding_table(windows: list[Window], model: ResNet1DMoE, device: str) -> pd.DataFrame:
    """One row per window: identifiers, target, and the PaPaGei embedding as columns."""
    embeddings = embed_windows(windows, model, device)
    ids = pd.DataFrame(
        {
            "subject_id": [w.subject_id for w in windows],
            "start_s": [w.start_s for w in windows],
            "target": [w.target for w in windows],
        }
    )
    columns = [f"ppg_emb_{i:03d}" for i in range(embeddings.shape[1])]
    return pd.concat([ids, pd.DataFrame(embeddings, columns=columns)], axis=1)