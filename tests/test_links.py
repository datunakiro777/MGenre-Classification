import asyncio

import httpx
import pytest

from music_genre import links


@pytest.mark.parametrize("url,message", [
    ("https://open.spotify.com/track/123", "Spotify links"),
    ("https://www.youtube.com/watch?v=abc", "YouTube video links"),
    ("https://youtu.be/abc", "YouTube video links"),
    ("http://example.com/song.mp3", "public HTTPS"),
    ("https://127.0.0.1/song.mp3", "local or private"),
])
def test_rejects_unsupported_and_private_urls(url, message):
    with pytest.raises(links.LinkError, match=message):
        asyncio.run(links.validate_public_url(url))


def test_downloads_bounded_direct_media(monkeypatch, tmp_path):
    async def allow_example(url):
        return url

    monkeypatch.setattr(links, "validate_public_url", allow_example)
    transport = httpx.MockTransport(lambda request: httpx.Response(
        200, headers={"content-type": "audio/mpeg"}, content=b"mock mp3 bytes",
    ))

    async def run():
        async with httpx.AsyncClient(transport=transport) as client:
            return await links.download_media("https://example.com/song.mp3", tmp_path / "song", client=client)

    assert asyncio.run(run()) == "example.com"
    assert (tmp_path / "song").read_bytes() == b"mock mp3 bytes"


def test_rejects_oversized_linked_file(monkeypatch, tmp_path):
    async def allow_example(url):
        return url

    monkeypatch.setattr(links, "validate_public_url", allow_example)
    monkeypatch.setattr(links, "MAX_MEDIA_BYTES", 10)
    transport = httpx.MockTransport(lambda request: httpx.Response(
        200, headers={"content-type": "audio/mpeg"}, content=b"12345678901",
    ))

    async def run():
        async with httpx.AsyncClient(transport=transport) as client:
            await links.download_media("https://example.com/song.mp3", tmp_path / "song", client=client)

    with pytest.raises(links.LinkError, match="too large"):
        asyncio.run(run())


def test_redirect_to_private_address_is_rejected(monkeypatch, tmp_path):
    original = links.validate_public_url

    async def validate(url):
        if url.startswith("https://example.com/"):
            return url
        return await original(url)

    monkeypatch.setattr(links, "validate_public_url", validate)
    transport = httpx.MockTransport(lambda request: httpx.Response(
        302, headers={"location": "https://127.0.0.1/private.mp3"},
    ))

    async def run():
        async with httpx.AsyncClient(transport=transport) as client:
            await links.download_media("https://example.com/song.mp3", tmp_path / "song", client=client)

    with pytest.raises(links.LinkError, match="local or private"):
        asyncio.run(run())
