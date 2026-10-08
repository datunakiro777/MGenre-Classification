import sqlite3

from music_genre import history


def test_saved_media_and_results_survive_new_queries(monkeypatch, tmp_path):
    monkeypatch.setattr(history, "HISTORY_ROOT", tmp_path / "history")
    source = tmp_path / "track.mp3"
    source.write_bytes(b"original audio")
    result = {
        "genre": "Rock",
        "confidence": 0.6,
        "scores": [{"genre": "Rock", "percent": 60.0}],
        "spectrogram": "aGVsbG8=",
        "analyzed_from": 0.0,
        "analyzed_to": 30.0,
    }
    first_id = history.save_analysis(
        source, filename="track.mp3", source_type="upload", suffix=".mp3", result=result
    )
    source.unlink()

    entries, total = history.list_analyses()
    assert total == 1
    assert entries[0]["id"] == first_id
    assert entries[0]["result"] == result
    assert entries[0]["media_exists"] is True
    assert history.media_path(first_id).read_bytes() == b"original audio"


def test_history_paginates_newest_first(monkeypatch, tmp_path):
    monkeypatch.setattr(history, "HISTORY_ROOT", tmp_path / "history")
    source = tmp_path / "track.mp3"
    source.write_bytes(b"audio")
    for number in range(history.PAGE_SIZE + 1):
        history.save_analysis(
            source,
            filename=f"track-{number}.mp3",
            source_type="upload",
            suffix=".mp3",
            result={"genre": "Rock", "confidence": 0.5, "scores": [], "spectrogram": ""},
        )

    first_page, total = history.list_analyses(page=1)
    second_page, _ = history.list_analyses(page=2)
    assert total == history.PAGE_SIZE + 1
    assert len(first_page) == history.PAGE_SIZE
    assert first_page[0]["filename"] == f"track-{history.PAGE_SIZE}.mp3"
    assert [entry["filename"] for entry in second_page] == ["track-0.mp3"]


def test_media_path_cannot_escape_history_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(history, "HISTORY_ROOT", tmp_path / "history")
    source = tmp_path / "track.mp3"
    source.write_bytes(b"audio")
    analysis_id = history.save_analysis(
        source, filename="track.mp3", source_type="upload", suffix=".mp3", result={}
    )
    with sqlite3.connect(history.HISTORY_ROOT / "history.sqlite3") as connection:
        connection.execute(
            "UPDATE analyses SET media_file = ? WHERE id = ?", ("../track.mp3", analysis_id)
        )
    assert history.media_path(analysis_id) is None
