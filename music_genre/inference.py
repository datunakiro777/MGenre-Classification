"""Load a trained checkpoint and classify uploaded audio."""

from __future__ import annotations

import base64
import io
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import librosa.display
import numpy as np
import torch

from .audio import AudioError, load_audio, log_mel, prediction_windows, probe_duration
from .dataset import GENRES
from .model import GenreCNN

DEFAULT_MODEL_PATH = Path("models/genre_cnn.pt")


@lru_cache(maxsize=2)
def load_model(path: str):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if tuple(checkpoint["labels"]) != GENRES:
        raise ValueError("Model genre order does not match the application.")
    model = GenreCNN(len(GENRES)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, device, float(checkpoint["mean"]), float(checkpoint["std"])


def render_spectrogram(spectrogram: np.ndarray) -> str:
    neon = LinearSegmentedColormap.from_list(
        "neon", ["#0e0e0e", "#34205f", "#7947d4", "#35f5ce", "#cbf86a"]
    )
    fig, ax = plt.subplots(figsize=(10, 2.7), facecolor="#171717")
    ax.set_facecolor("#0e0e0e")
    image = librosa.display.specshow(
        spectrogram, x_axis="time", y_axis="mel", sr=22050,
        hop_length=512, ax=ax, cmap=neon,
    )
    ax.set(xlabel="Time (seconds)", ylabel="Frequency (Hz)")
    ax.tick_params(colors="#c8c8c8")
    ax.xaxis.label.set_color("#c8c8c8")
    ax.yaxis.label.set_color("#c8c8c8")
    for spine in ax.spines.values():
        spine.set_color("#666666")
    colorbar = fig.colorbar(image, ax=ax, pad=0.01)
    colorbar.set_label("dB", color="#c8c8c8")
    colorbar.ax.tick_params(colors="#c8c8c8")
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@torch.no_grad()
def classify(path: str | Path, *, model_path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    duration = probe_duration(path)
    if duration < 3:
        raise AudioError("Audio must be at least three seconds long.")
    if duration > 3600:
        raise AudioError("Audio must be no longer than one hour.")
    start = max(0.0, (duration - 30.0) / 2)
    audio = load_audio(path, start=start, max_duration=30)
    spectrogram = log_mel(audio)
    model, device, mean, std = load_model(str(model_path))
    windows = prediction_windows(spectrogram)
    tensor = torch.from_numpy(np.ascontiguousarray(((windows - mean) / std)[:, None])).to(device)
    scores = torch.softmax(model(tensor).mean(dim=0), dim=0).cpu().numpy()
    ranked = sorted(zip(GENRES, scores.tolist()), key=lambda item: item[1], reverse=True)
    return {
        "genre": ranked[0][0],
        "confidence": ranked[0][1],
        "scores": [{"genre": genre, "probability": score, "percent": round(score * 100, 1)} for genre, score in ranked],
        "spectrogram": render_spectrogram(spectrogram),
        "analyzed_from": round(start, 1),
        "analyzed_to": round(start + len(audio) / 22050, 1),
    }
