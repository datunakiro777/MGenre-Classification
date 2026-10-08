"""FMA Small metadata validation and spectrogram caching."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .audio import AudioError, N_MELS, WINDOW_FRAMES, crop_window, load_audio, log_mel

GENRES = (
    "Electronic", "Experimental", "Folk", "Hip-Hop",
    "Instrumental", "International", "Pop", "Rock",
)
SPLITS = ("training", "validation", "test")


def audio_path(audio_root: Path, track_id: int) -> Path:
    name = f"{track_id:06d}"
    return audio_root / name[:3] / f"{name}.mp3"


def read_tracks(metadata_path: Path) -> pd.DataFrame:
    tracks = pd.read_csv(metadata_path, header=[0, 1], index_col=0)
    small = tracks.loc[tracks[("set", "subset")] == "small"]
    frame = pd.DataFrame({
        "track_id": small.index.astype(int),
        "genre": small[("track", "genre_top")].values,
        "split": small[("set", "split")].values,
        "artist_id": small[("artist", "id")].values,
    })
    if len(frame) != 8000 or set(frame.genre) != set(GENRES):
        raise ValueError("FMA Small metadata does not contain the expected 8,000 tracks and eight genres.")
    if not set(frame.split).issubset(SPLITS):
        raise ValueError("Unexpected dataset split name.")
    if frame.artist_id.isna().any():
        raise ValueError("Some FMA Small tracks have no artist ID.")
    split_count = frame.groupby("artist_id").split.nunique()
    if (split_count > 1).any():
        raise ValueError("Artist leakage detected between dataset splits.")
    return frame


def prepare_cache(metadata_path: Path, audio_root: Path, cache_root: Path) -> Path:
    frame = read_tracks(metadata_path)
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / "manifest.csv"
    problems_path = cache_root / "skipped.json"
    rows: list[dict] = []
    problems: list[dict] = []
    for i, row in enumerate(frame.itertuples(index=False), 1):
        source = audio_path(audio_root, row.track_id)
        target = cache_root / f"{row.track_id:06d}.npy"
        try:
            if not target.exists():
                if not source.exists():
                    raise AudioError("Audio file missing from archive.")
                spectrogram = log_mel(load_audio(source, max_duration=30))
                if spectrogram.shape[0] != N_MELS or spectrogram.shape[1] < WINDOW_FRAMES:
                    raise AudioError("Unexpected spectrogram shape.")
                np.save(target, spectrogram.astype(np.float16))
            rows.append({
                "track_id": row.track_id, "genre": row.genre, "split": row.split,
                "artist_id": row.artist_id, "cache_file": target.name,
            })
        except (AudioError, ValueError, OSError) as exc:
            problems.append({"track_id": row.track_id, "reason": str(exc)})
        if i % 100 == 0:
            print(f"Prepared {i}/8000 tracks; skipped {len(problems)}", flush=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["track_id", "genre", "split", "artist_id", "cache_file"])
        writer.writeheader()
        writer.writerows(rows)
    problems_path.write_text(json.dumps(problems, indent=2) + "\n")
    return manifest_path


def read_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty or set(frame.genre) != set(GENRES):
        raise ValueError("No usable cached FMA Small data or missing genres.")
    if (frame.groupby("artist_id").split.nunique() > 1).any():
        raise ValueError("Artist leakage detected in cached manifest.")
    return frame


class CropDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, cache_root: Path, mean: float, std: float) -> None:
        self.rows = list(frame.itertuples(index=False))
        self.cache_root = cache_root
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        spectrogram = np.load(self.cache_root / row.cache_file, mmap_mode="r")
        last_start = max(0, spectrogram.shape[1] - WINDOW_FRAMES)
        start = int(np.random.randint(0, last_start + 1))
        crop = crop_window(spectrogram, start)
        return np.ascontiguousarray(((crop - self.mean) / self.std)[None]), GENRES.index(row.genre)


def training_stats(frame: pd.DataFrame, cache_root: Path) -> tuple[float, float]:
    total = 0.0
    squared = 0.0
    count = 0
    for name in frame.loc[frame.split == "training", "cache_file"]:
        values = np.load(cache_root / name, mmap_mode="r")
        total += float(np.sum(values, dtype=np.float64))
        squared += float(np.sum(np.square(values.astype(np.float64))))
        count += values.size
    mean = total / count
    std = max((squared / count - mean * mean) ** 0.5, 1e-6)
    return mean, std
