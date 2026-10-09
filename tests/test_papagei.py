import numpy as np
import pytest
torch = pytest.importorskip("torch")  # PaPaGei tests need the optional `fm` group
from wearable_affect.papagei import WEIGHTS_PATH, embed_windows, load_papagei, preprocess_bvp
from wearable_affect.windows import Window


def test_preprocessing_shape_and_normalisation():
    t = np.arange(60 * 64) / 64
    segments = preprocess_bvp(np.sin(2 * np.pi * 1.2 * t), fs=64)
    assert segments.shape == (6, 1250)  # 6 x 10 s at 125 Hz
    assert np.allclose(segments.mean(axis=1), 0, atol=0.05)
    assert np.allclose(segments.std(axis=1), 1, atol=0.05)


def test_resampling_preserves_pulse_frequency():
    t = np.arange(60 * 64) / 64
    segments = preprocess_bvp(np.sin(2 * np.pi * 1.2 * t), fs=64)
    freqs = np.fft.rfftfreq(1250, d=1 / 125)
    for seg in segments:
        dominant = freqs[np.argmax(np.abs(np.fft.rfft(seg)))]
        assert abs(dominant - 1.2) < 0.11  # 72 bpm stays 72 bpm


def test_flat_signal_gives_no_nan():
    segments = preprocess_bvp(np.zeros(60 * 64), fs=64)
    assert np.isfinite(segments).all()


@pytest.mark.skipif(not WEIGHTS_PATH.exists(), reason="PaPaGei weights not downloaded")
def test_one_embedding_per_window():
    t = np.arange(60 * 64) / 64
    bvp = np.sin(2 * np.pi * 1.2 * t)[:, None]
    windows = [Window("S_fake", i * 30, 0, {"BVP": bvp}) for i in range(3)]
    model, device = load_papagei()
    embeddings = embed_windows(windows, model, device)
    assert embeddings.shape == (3, 512)
    assert np.isfinite(embeddings).all()