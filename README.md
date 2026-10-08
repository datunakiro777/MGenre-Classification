# Music genre classification

A local FastAPI website that classifies uploaded music or video, or a directly linked media file, into the eight genres of FMA Small. A four-block PyTorch CNN learns from log-mel spectrograms and returns the predicted genre, all class scores, and the spectrogram used for prediction.

## Dataset and research basis

The [FMA Small dataset](https://github.com/mdeff/fma) contains 8,000 30-second MP3 clips, with 1,000 per genre. Its [paper](https://arxiv.org/abs/1612.01840) provides an 80/10/10 training, validation, and test split that keeps each artist in one split. The eight genres are Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, and Rock. FMA audio remains under each artist's license; this repository stores only code and locally generated artifacts, not the training audio. The FMA paper and metadata are CC BY 4.0, and the dataset code is MIT licensed. Please cite the FMA paper when using the dataset.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). On the development Mac, `uv sync --extra test` installs PyTorch, the audio processor, the website, and test dependencies. The bundled FFmpeg binary from `imageio-ffmpeg` handles MP3, WAV, and FLAC decoding; no separate FFmpeg installation is needed.

```bash
uv sync --extra test
mkdir -p data/raw data/extracted
curl -L --fail -C - https://os.unil.cloud.switch.ch/fma/fma_metadata.zip -o data/raw/fma_metadata.zip
curl -L --fail -C - https://os.unil.cloud.switch.ch/fma/fma_small.zip -o data/raw/fma_small.zip
shasum -a 1 data/raw/fma_metadata.zip data/raw/fma_small.zip
```

Expected SHA-1 values, [published by FMA](https://github.com/mdeff/fma#usage):

| Archive | SHA-1 |
| --- | --- |
| `fma_metadata.zip` | `f0df49ffe5f2a6008d7dc83c6915b31835dfe733` |
| `fma_small.zip` | `ade154f733639d52e35e32f5593efe5be76c6d70` |

Extract with `bsdtar` on macOS, because the built-in `unzip` may not support these archives:

```bash
bsdtar -xf data/raw/fma_metadata.zip -C data/extracted
bsdtar -xf data/raw/fma_small.zip -C data/extracted
```

## Train and evaluate

```bash
uv run python -m music_genre.train \
  --metadata data/extracted/fma_metadata/tracks.csv \
  --audio-root data/extracted/fma_small \
  --prepare
```

The first run caches 128-band log-mel spectrograms in `data/cache` (22,050 Hz mono, FFT 2,048, hop 512), then trains on random three-second crops. Validation and test predictions average ten evenly spaced crop logits per track. Training uses Apple MPS when available and otherwise CPU, with seed 42, AdamW, and early stopping on validation loss. The best checkpoint is `models/genre_cnn.pt`. The held-out evaluation and skipped-track log are in `reports/generated`:

- `metrics.json`: track-level test accuracy, macro-F1, per-genre precision/recall/F1, confusion matrix, usable split sizes, and skipped tracks.
- `training_curves.png`: training and validation loss/accuracy.
- `confusion_matrix.png`: actual versus predicted genres on the test set.
- `history.json`: per-epoch training history.
- `report.md`: readable evaluation summary with figures and skipped-track details; generate it with `uv run python -m music_genre.report` after training.

To repeat training without recomputing cached spectrograms, omit `--prepare` and the dataset path arguments.

## Run the website

```bash
uv run uvicorn music_genre.app:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Upload an MP3, WAV, FLAC, MP4, WebM, or MOV file up to 50 MiB and at least three seconds long, or paste a public HTTPS link directly to one of those file types. The website extracts audio from video files, analyzes up to the central 30 seconds, and uses the same preprocessing and class order as training. Linked media must be publicly accessible without sign-in; links to local or private network addresses are rejected.

The [History page](http://127.0.0.1:8000/history) saves every successful new analysis, including the original media file, genre scores, and spectrogram. It shows ten analyses per page, newest first, and lets you play or download a saved file when the browser supports its format. History is stored locally in `data/history` (ignored by Git); it is shared by anyone using this local installation and grows as you analyze more media. Uploads made before the History feature was added were temporary and cannot be recovered.

Spotify and YouTube page links cannot provide audio to this classifier. [Spotify's developer rules](https://developer.spotify.com/documentation/web-api/reference/get-track) prohibit downloading its audio or ingesting it into an ML model, and [YouTube's developer rules](https://developers.google.com/youtube/terms/developer-policies-guide) prohibit separating audio tracks through its API. If it is your own YouTube upload, [YouTube Studio lets you download the MP4](https://support.google.com/youtube/answer/56100?hl=en-GB-&ref_topic=9257441); you can then upload that file here. The scores are softmax model outputs, not a guarantee that a song has only one genre.

## Verify

```bash
uv run pytest -q
```
