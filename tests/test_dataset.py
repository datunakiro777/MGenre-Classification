from pathlib import Path

import pytest

from music_genre.dataset import GENRES, read_tracks


def test_official_metadata_has_artist_separated_splits():
    path = Path("data/extracted/fma_metadata/tracks.csv")
    if not path.exists():
        pytest.skip("FMA metadata is not downloaded")
    frame = read_tracks(path)
    assert len(frame) == 8000
    assert set(frame.genre) == set(GENRES)
    assert (frame.groupby("artist_id").split.nunique() == 1).all()
