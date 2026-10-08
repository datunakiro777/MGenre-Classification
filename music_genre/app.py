"""Local media upload and direct-link classification website."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .audio import AudioError
from .inference import classify
from .links import ALLOWED_EXTENSIONS, LinkError, download_media

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
ALLOWED_SUFFIXES = ALLOWED_EXTENSIONS
MODEL_PATH = Path(os.environ.get("MGENRE_MODEL_PATH", "models/genre_cnn.pt"))
TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app = FastAPI(title="Music Genre Classifier")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


def page(request: Request, **context) -> HTMLResponse:
    return TEMPLATES.TemplateResponse(request=request, name="index.html", context=context)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return page(request)


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, audio: UploadFile = File(...)):
    filename = Path(audio.filename or "").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        return page(request, error="Choose an MP3, WAV, FLAC, MP4, WebM, or MOV file.")
    if not MODEL_PATH.exists():
        return page(request, error="The trained model is missing. Run the training command in the README first.")
    try:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"upload{suffix}"
            size = 0
            with path.open("wb") as destination:
                while chunk := await audio.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        return page(request, error="The file is too large. The limit is 50 MB.")
                    destination.write(chunk)
            if size == 0:
                return page(request, error="The selected file is empty.")
            result = classify(path, model_path=MODEL_PATH)
    except AudioError as exc:
        return page(request, error=str(exc))
    return page(request, result=result, filename=filename)


@app.post("/predict-link", response_class=HTMLResponse)
async def predict_link(request: Request, media_url: str = Form(...)):
    if not MODEL_PATH.exists():
        return page(request, error="The trained model is missing. Run the training command in the README first.")
    try:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "linked_media"
            source_host = await download_media(media_url.strip(), path)
            result = classify(path, model_path=MODEL_PATH)
    except (AudioError, LinkError) as exc:
        return page(request, error=str(exc), media_url=media_url)
    return page(request, result=result, filename=f"Media from {source_host}")
