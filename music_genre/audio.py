"""One audio and spectrogram pipeline for training and prediction."""

from __future__ import annotations

import subprocess
import re
from pathlib import Path

import imageio_ffmpeg
import librosa
import numpy as np

SAMPLE_RATE = 22_050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
WINDOW_SECONDS = 3
MAX_CLIP_SECONDS = 30
WINDOW_FRAMES = round(WINDOW_SECONDS * SAMPLE_RATE / HOP_LENGTH)


class AudioError(ValueError):
    """The uploaded or dataset audio cannot be decoded."""


def probe_duration(path: str | Path) -> float:
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", str(path)]
    try:
        result = subprocess.run(cmd, capture_output=True, check=False, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AudioError("Could not inspect this audio file.") from exc
    details = result.stderr.decode("utf-8", errors="replace")
    match = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", details)
    if not match:
        raise AudioError("Could not determine the audio duration.")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def load_audio(path: str | Path, *, start: float = 0, max_duration: float | None = None) -> np.ndarray:
    """Decode to 22.05 kHz mono float32 with the bundled FFmpeg binary."""
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin"]
    if start > 0:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(path)]
    if max_duration is not None:
        cmd += ["-t", str(max_duration)]
    cmd += ["-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"]
    try:
        result = subprocess.run(cmd, capture_output=True, check=False, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AudioError("Could not read this audio file.") from exc
    if result.returncode != 0:
        raise AudioError("Could not read this audio file. Check that it contains valid audio.")
    audio = np.frombuffer(result.stdout, dtype="<f4").copy()
    if audio.size < WINDOW_SECONDS * SAMPLE_RATE:
        raise AudioError("Audio must be at least three seconds long.")
    if not np.all(np.isfinite(audio)):
        raise AudioError("Audio contains invalid samples.")
    return audio


def central_clip(audio: np.ndarray, *, seconds: int = MAX_CLIP_SECONDS) -> np.ndarray:
    limit = seconds * SAMPLE_RATE
    if audio.size <= limit:
        return audio
    start = (audio.size - limit) // 2
    return audio[start : start + limit]


def log_mel(audio: np.ndarray) -> np.ndarray:
    """Return a 128-band power mel spectrogram in dB, fixed reference."""
    power = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, power=2.0,
    )
    return librosa.power_to_db(power, ref=1.0, top_db=80.0).astype(np.float32)


def crop_window(spectrogram: np.ndarray, start: int) -> np.ndarray:
    crop = spectrogram[:, start : start + WINDOW_FRAMES]
    if crop.shape[1] < WINDOW_FRAMES:
        crop = np.pad(crop, ((0, 0), (0, WINDOW_FRAMES - crop.shape[1])))
    return crop.astype(np.float32, copy=False)


def prediction_windows(spectrogram: np.ndarray) -> np.ndarray:
    """Ten reproducible, evenly spaced three-second views of a clip."""
    last_start = max(0, spectrogram.shape[1] - WINDOW_FRAMES)
    starts = np.linspace(0, last_start, 10, dtype=int)
    return np.stack([crop_window(spectrogram, int(start)) for start in starts])
