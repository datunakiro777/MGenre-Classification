"""Local media upload and direct-link classification website."""

from __future__ import annotations

import os
import logging
import mimetypes
import sqlite3
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .audio import AudioError
from . import history
from .inference import classify
from .links import ALLOWED_EXTENSIONS, LinkError, download_media

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
ALLOWED_SUFFIXES = ALLOWED_EXTENSIONS
MODEL_PATH = Path(os.environ.get("MGENRE_MODEL_PATH", "models/genre_cnn.pt"))
TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app = FastAPI(title="Music Genre Classifier")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
logger = logging.getLogger(__name__)
EXAMPLE_SCORES = [
    {"genre": "Electronic", "percent": 46.8},
    {"genre": "Experimental", "percent": 18.4},
    {"genre": "Hip-Hop", "percent": 11.3},
    {"genre": "Instrumental", "percent": 9.7},
    {"genre": "Rock", "percent": 6.2},
    {"genre": "Pop", "percent": 4.1},
    {"genre": "Folk", "percent": 2.3},
    {"genre": "International", "percent": 1.2},
]


def page(request: Request, **context) -> HTMLResponse:
    context.setdefault("example_scores", EXAMPLE_SCORES)
    return TEMPLATES.TemplateResponse(request=request, name="index.html", context=context)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return page(request)


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request, page_number: int = Query(1, ge=1, alias="page")):
    try:
        entries, total = history.list_analyses(page=page_number)
        max_page = max(1, (total + history.PAGE_SIZE - 1) // history.PAGE_SIZE)
        if page_number > max_page:
            page_number = max_page
            entries, total = history.list_analyses(page=page_number)
        error = None
    except (OSError, sqlite3.Error) as exc:
        logger.warning("Could not read analysis history: %s", exc)
        entries, total, max_page = [], 0, 1
        error = "History is unavailable right now. Check that the local data folder is writable."
    return TEMPLATES.TemplateResponse(
        request=request,
        name="history.html",
        context={"entries": entries, "total": total, "page": page_number, "max_page": max_page, "error": error},
    )


@app.get("/history/{analysis_id}/media")
async def history_media(analysis_id: int):
    try:
        path = history.media_path(analysis_id)
    except (OSError, sqlite3.Error) as exc:
        logger.warning("Could not read saved media: %s", exc)
        path = None
    if path is None:
        raise HTTPException(status_code=404, detail="Saved media was not found.")
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, headers={"Content-Disposition": "inline"})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, audio: UploadFile = File(...)):
    filename = Path(audio.filename or "").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        return page(request, error="Choose an MP3, WAV, FLAC, MP4, WebM, or MOV file.")
    if not MODEL_PATH.exists():
        return page(request, error="The trained model is missing. Run the training command in the README first.")
    history_id = None
    history_error = None
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
            try:
                history_id = history.save_analysis(
                    path, filename=filename, source_type="upload", suffix=suffix, result=result
                )
            except (OSError, sqlite3.Error) as exc:
                logger.warning("Could not save upload to history: %s", exc)
                history_error = "Prediction succeeded, but it could not be saved to history."
    except AudioError as exc:
        return page(request, error=str(exc))
    return page(request, result=result, filename=filename, history_id=history_id, history_error=history_error)


@app.post("/predict-link", response_class=HTMLResponse)
async def predict_link(request: Request, media_url: str = Form(...)):
    if not MODEL_PATH.exists():
        return page(request, error="The trained model is missing. Run the training command in the README first.")
    history_id = None
    history_error = None
    try:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "linked_media"
            source_host = await download_media(media_url.strip(), path)
            result = classify(path, model_path=MODEL_PATH)
            filename = f"Media from {source_host}"
            try:
                history_id = history.save_analysis(
                    path,
                    filename=filename,
                    source_type="link",
                    suffix=Path(urlsplit(media_url.strip()).path).suffix.lower(),
                    result=result,
                )
            except (OSError, sqlite3.Error) as exc:
                logger.warning("Could not save linked media to history: %s", exc)
                history_error = "Prediction succeeded, but it could not be saved to history."
    except (AudioError, LinkError) as exc:
        return page(request, error=str(exc), media_url=media_url)
    return page(request, result=result, filename=filename, history_id=history_id, history_error=history_error)


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    audio: UploadFile | None = File(None),
    media_url: str = Form(""),
):
    """Handle the single input form while keeping the original routes available."""
    has_file = bool(audio and audio.filename)
    has_link = bool(media_url.strip())
    if has_file and has_link:
        return page(request, error="Choose either a file or a direct media link.", media_url=media_url)
    if has_file:
        return await predict(request, audio)
    if has_link:
        return await predict_link(request, media_url)
    return page(request, error="Choose a file or paste a direct media link to analyze.")
