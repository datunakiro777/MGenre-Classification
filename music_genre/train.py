"""Prepare FMA Small, train the CNN, and write a held-out evaluation report."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .audio import prediction_windows
from .dataset import GENRES, CropDataset, prepare_cache, read_manifest, training_stats
from .model import GenreCNN


class TrackDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, cache_root: Path, mean: float, std: float) -> None:
        self.rows = list(frame.itertuples(index=False))
        self.cache_root = cache_root
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        spec = np.load(self.cache_root / row.cache_file, mmap_mode="r")
        windows = prediction_windows(spec)
        windows = np.ascontiguousarray(((windows - self.mean) / self.std)[:, None])
        return windows, GENRES.index(row.genre)


@torch.no_grad()
def evaluate_split(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    losses: list[float] = []
    criterion = nn.CrossEntropyLoss(reduction="none")
    for windows, targets in loader:
        batch, views = windows.shape[:2]
        logits = model(windows.reshape(batch * views, 1, *windows.shape[-2:]).to(device))
        track_logits = logits.reshape(batch, views, -1).mean(dim=1)
        targets = targets.to(device)
        losses.extend(criterion(track_logits, targets).cpu().tolist())
        labels.extend(targets.cpu().tolist())
        predictions.extend(track_logits.argmax(dim=1).cpu().tolist())
    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, labels=list(range(len(GENRES))), average="macro", zero_division=0),
        "labels": labels,
        "predictions": predictions,
    }


def save_plots(history: list[dict], labels: list[int], predictions: list[int], report_dir: Path) -> None:
    epochs = [item["epoch"] for item in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [item["train_loss"] for item in history], label="Training")
    axes[0].plot(epochs, [item["validation_loss"] for item in history], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")
    axes[0].legend()
    axes[1].plot(epochs, [item["train_accuracy"] for item in history], label="Training crop")
    axes[1].plot(epochs, [item["validation_accuracy"] for item in history], label="Validation track")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy", ylim=(0, 1))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(report_dir / "training_curves.png", dpi=150)
    plt.close(fig)

    matrix = confusion_matrix(labels, predictions, labels=list(range(len(GENRES))))
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(8), yticks=range(8), xticklabels=GENRES, yticklabels=GENRES,
           xlabel="Predicted", ylabel="Actual", title="Held-out test confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(report_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    cache_root = Path(args.cache_root)
    manifest_path = cache_root / "manifest.csv"
    if args.prepare or not manifest_path.exists():
        if not args.metadata or not args.audio_root:
            raise SystemExit("Preparation needs --metadata and --audio-root.")
        prepare_cache(Path(args.metadata), Path(args.audio_root), cache_root)

    frame = read_manifest(manifest_path)
    counts = frame.groupby("split").size().to_dict()
    print(f"Usable tracks by split: {counts}", flush=True)
    mean, std = training_stats(frame, cache_root)
    print(f"Training spectrogram mean={mean:.4f}, std={std:.4f}", flush=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on {device}", flush=True)

    train_frame = frame.loc[frame.split == "training"]
    validation_frame = frame.loc[frame.split == "validation"]
    test_frame = frame.loc[frame.split == "test"]
    train_loader = DataLoader(CropDataset(train_frame, cache_root, mean, std), batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(TrackDataset(validation_frame, cache_root, mean, std), batch_size=8)
    test_loader = DataLoader(TrackDataset(test_frame, cache_root, mean, std), batch_size=8)

    model = GenreCNN(len(GENRES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        for windows, targets in train_loader:
            windows, targets = windows.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(windows)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_count += len(targets)
        validation = evaluate_split(model, validation_loader, device)
        epoch_record = {
            "epoch": epoch, "train_loss": total_loss / total_count,
            "train_accuracy": total_correct / total_count,
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "validation_macro_f1": validation["macro_f1"],
        }
        history.append(epoch_record)
        (report_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(epoch_record), flush=True)
        if validation["loss"] < best_loss - 0.001:
            best_loss = validation["loss"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                "model_state": model.state_dict(), "labels": list(GENRES),
                "mean": mean, "std": std, "epoch": epoch,
            }, model_path)
        else:
            stale_epochs += 1
        if epoch >= 8 and stale_epochs >= args.patience:
            print("Early stopping after validation loss stopped improving.", flush=True)
            break

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test = evaluate_split(model, test_loader, device)
    class_report = classification_report(
        test["labels"], test["predictions"], labels=list(range(8)),
        target_names=GENRES, output_dict=True, zero_division=0,
    )
    summary = {
        "dataset": "FMA Small", "seed": 42, "device": str(device),
        "usable_tracks_by_split": counts, "skipped_tracks": json.loads((cache_root / "skipped.json").read_text()) if (cache_root / "skipped.json").exists() else [],
        "best_epoch": best_epoch, "test_accuracy": test["accuracy"],
        "test_macro_f1": test["macro_f1"], "test_loss": test["loss"],
        "per_class": {genre: class_report[genre] for genre in GENRES},
        "confusion_matrix": confusion_matrix(test["labels"], test["predictions"], labels=list(range(8))).tolist(),
    }
    (report_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_plots(history, test["labels"], test["predictions"], report_dir)
    print(f"Test accuracy={test['accuracy']:.4f}, macro-F1={test['macro_f1']:.4f}; best epoch={best_epoch}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", help="Path to extracted fma_metadata/tracks.csv")
    parser.add_argument("--audio-root", help="Path to extracted fma_small directory")
    parser.add_argument("--cache-root", default="data/cache")
    parser.add_argument("--model-path", default="models/genre_cnn.pt")
    parser.add_argument("--report-dir", default="reports/generated")
    parser.add_argument("--prepare", action="store_true", help="Recheck and build cached spectrograms")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
