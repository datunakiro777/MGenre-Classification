import wave

import numpy as np
import torch

from music_genre.audio import SAMPLE_RATE
from music_genre.dataset import GENRES
from music_genre.inference import classify, load_model
from music_genre.model import GenreCNN


def test_real_decode_spectrogram_model_pipeline(tmp_path):
    path = tmp_path / "tone.wav"
    time = np.arange(4 * SAMPLE_RATE) / SAMPLE_RATE
    samples = (np.sin(2 * np.pi * 440 * time) * 15000).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(samples.tobytes())
    model_path = tmp_path / "test_model.pt"
    torch.save({
        "model_state": GenreCNN().state_dict(),
        "labels": list(GENRES), "mean": -30.0, "std": 20.0,
    }, model_path)
    result = classify(path, model_path=model_path)
    assert len(result["scores"]) == 8
    assert abs(sum(item["probability"] for item in result["scores"]) - 1.0) < 1e-5
    assert result["genre"] in GENRES
    assert len(result["spectrogram"]) > 100
    load_model.cache_clear()
