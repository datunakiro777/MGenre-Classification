"""Build a readable report from the generated evaluation metrics."""

from __future__ import annotations

import json
from pathlib import Path

from .dataset import GENRES


def build_report(report_dir: Path = Path("reports/generated")) -> Path:
    metrics = json.loads((report_dir / "metrics.json").read_text())
    history = json.loads((report_dir / "history.json").read_text())
    best_validation = next(item for item in history if item["epoch"] == metrics["best_epoch"])
    counts = metrics["usable_tracks_by_split"]
    lines = [
        "# FMA Small CNN evaluation", "",
        "## Dataset and method", "",
        "The model was trained on the [FMA Small dataset](https://github.com/mdeff/fma), introduced in the [FMA paper](https://arxiv.org/abs/1612.01840). The official split keeps artists separate across training, validation, and test sets. FMA audio uses artist-selected licenses; the audio is not included in this repository.",
        "",
        f"Usable tracks: {counts['training']:,} training, {counts['validation']:,} validation, and {counts['test']:,} test. {len(metrics['skipped_tracks'])} source clips were skipped and are listed below. A fixed seed of {metrics['seed']} was used. The best checkpoint was selected at epoch {metrics['best_epoch']} using validation loss, and the held-out test set was evaluated afterward.",
        "",
        "Audio was decoded to mono 22,050 Hz and represented as a 128-band log-mel spectrogram (FFT 2,048, hop 512). The four-block CNN trained on random three-second crops. Track-level predictions average logits from ten evenly spaced crops before softmax.",
        "", "## Held-out test results", "",
        f"- Accuracy: **{metrics['test_accuracy']:.1%}**",
        f"- Macro F1: **{metrics['test_macro_f1']:.3f}**",
        f"- Cross-entropy loss: **{metrics['test_loss']:.3f}**",
        f"- Validation accuracy at the selected checkpoint: **{best_validation['validation_accuracy']:.1%}**",
        "", "| Genre | Precision | Recall | F1 | Tracks |", "| --- | ---: | ---: | ---: | ---: |",
    ]
    for genre in GENRES:
        values = metrics["per_class"][genre]
        lines.append(f"| {genre} | {values['precision']:.3f} | {values['recall']:.3f} | {values['f1-score']:.3f} | {int(values['support'])} |")
    lines += [
        "", "## Interpretation", "",
        "The held-out test accuracy is lower than validation accuracy at the selected checkpoint. Performance also varies considerably by genre; inspect the per-genre scores and confusion matrix before using a prediction. This is a research baseline for FMA Small's eight labels, rather than a calibrated measure of every style present in a song.",
        "", "## Figures", "",
        "![Training and validation curves](training_curves.png)", "",
        "![Held-out test confusion matrix](confusion_matrix.png)", "",
        "## Skipped source tracks", "",
    ]
    skipped = metrics["skipped_tracks"]
    if skipped:
        lines += ["| Track ID | Reason |", "| ---: | --- |"]
        lines += [f"| {item['track_id']} | {item['reason']} |" for item in skipped]
    else:
        lines.append("None.")
    lines += ["", "Scores describe this dataset's eight single-label genres and may not represent every genre in an uploaded song.", ""]
    path = report_dir / "report.md"
    path.write_text("\n".join(lines))
    return path


if __name__ == "__main__":
    print(build_report())
