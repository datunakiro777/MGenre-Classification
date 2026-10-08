from fastapi.testclient import TestClient
import pytest

from music_genre import app as site


client = TestClient(site.app)


@pytest.fixture(autouse=True)
def isolated_history(monkeypatch, tmp_path):
    monkeypatch.setattr(site.history, "HISTORY_ROOT", tmp_path / "history")


def test_home_page_has_upload_form():
    response = client.get("/")
    assert response.status_code == 200
    assert 'action="/analyze"' in response.text
    assert 'name="audio"' in response.text
    assert 'name="media_url"' in response.text
    assert "Example result" in response.text
    assert 'href="/history"' in response.text


def test_history_starts_empty():
    response = client.get("/history")
    assert response.status_code == 200
    assert "No saved tracks yet" in response.text
    assert "Earlier uploads were temporary" in response.text


def test_analyze_requires_one_source():
    response = client.post("/analyze", data={"media_url": ""})
    assert "Choose a file or paste a direct media link" in response.text


def test_analyze_rejects_two_sources():
    response = client.post(
        "/analyze",
        data={"media_url": "https://example.com/song.mp3"},
        files={"audio": ("song.mp3", b"sample", "audio/mpeg")},
    )
    assert "Choose either a file or a direct media link" in response.text


def test_rejects_unsupported_file():
    response = client.post("/predict", files={"audio": ("notes.txt", b"hello", "text/plain")})
    assert "Choose an MP3, WAV, FLAC, MP4, WebM, or MOV" in response.text


def test_rejects_oversized_file(monkeypatch, tmp_path):
    model = tmp_path / "model.pt"
    model.touch()
    monkeypatch.setattr(site, "MODEL_PATH", model)
    monkeypatch.setattr(site, "MAX_UPLOAD_BYTES", 10)
    response = client.post("/predict", files={"audio": ("music.mp3", b"12345678901", "audio/mpeg")})
    assert "too large" in response.text


def test_rejects_unreadable_audio(monkeypatch, tmp_path):
    model = tmp_path / "model.pt"
    model.touch()
    monkeypatch.setattr(site, "MODEL_PATH", model)
    response = client.post("/predict", files={"audio": ("broken.mp3", b"not audio", "audio/mpeg")})
    assert "Could not determine the audio duration" in response.text


def test_shows_prediction_and_scores(monkeypatch, tmp_path):
    model = tmp_path / "model.pt"
    model.touch()
    monkeypatch.setattr(site, "MODEL_PATH", model)
    monkeypatch.setattr(site, "classify", lambda *args, **kwargs: {
        "genre": "Rock", "confidence": 0.6,
        "scores": [{"genre": "Rock", "percent": 60.0}],
        "spectrogram": "aGVsbG8=", "analyzed_from": 0.0, "analyzed_to": 30.0,
    })
    response = client.post("/analyze", files={"audio": ("song.mp3", b"sample", "audio/mpeg")})
    assert response.status_code == 200
    assert "Rock" in response.text
    assert "60.0%" in response.text
    assert "data:image/png;base64,aGVsbG8=" in response.text
    assert "Saved to history" in response.text
    saved = client.get("/history")
    assert "song.mp3" in saved.text
    assert "Rock" in saved.text
    assert "60.0%" in saved.text
    assert "data:image/png;base64,aGVsbG8=" in saved.text
    media = client.get("/history/1/media")
    assert media.status_code == 200
    assert media.content == b"sample"
    assert client.get("/history/999/media").status_code == 404


def test_link_form_rejects_spotify(monkeypatch, tmp_path):
    model = tmp_path / "model.pt"
    model.touch()
    monkeypatch.setattr(site, "MODEL_PATH", model)
    response = client.post(
        "/analyze",
        data={"media_url": "https://open.spotify.com/track/abc"},
        files={"audio": ("", b"", "application/octet-stream")},
    )
    assert response.status_code == 200
    assert "Spotify links cannot provide audio" in response.text


def test_link_form_classifies_direct_media(monkeypatch, tmp_path):
    model = tmp_path / "model.pt"
    model.touch()
    monkeypatch.setattr(site, "MODEL_PATH", model)

    async def fake_download(url, destination):
        destination.write_bytes(b"media")
        return "example.com"

    monkeypatch.setattr(site, "download_media", fake_download)
    monkeypatch.setattr(site, "classify", lambda *args, **kwargs: {
        "genre": "Rock", "confidence": 0.6,
        "scores": [{"genre": "Rock", "percent": 60.0}],
        "spectrogram": "aGVsbG8=", "analyzed_from": 0.0, "analyzed_to": 30.0,
    })
    response = client.post("/analyze", data={"media_url": "https://example.com/song.mp3"})
    assert response.status_code == 200
    assert "Media from example.com" in response.text
    assert "60.0%" in response.text
    saved = client.get("/history")
    assert "Media from example.com" in saved.text
    assert "Direct link" in saved.text
    assert client.get("/history/1/media").content == b"media"
