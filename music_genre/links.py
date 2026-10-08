"""Fetch directly hosted media for local classification, with URL safeguards."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx

MAX_MEDIA_BYTES = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".mp4", ".webm", ".mov"}
ALLOWED_MIME_TYPES = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/wave",
    "audio/flac", "audio/x-flac", "video/mp4", "video/webm", "video/quicktime",
    "application/octet-stream",
}


class LinkError(ValueError):
    """A media link cannot be used for classification."""


def platform_message(host: str) -> str | None:
    host = host.lower().rstrip(".")
    if host == "spotify.com" or host.endswith(".spotify.com") or host == "spotify.link":
        return "Spotify links cannot provide audio to this classifier. Upload an audio file you own or are allowed to analyze."
    if host in {"youtube.com", "youtu.be", "youtube-nocookie.com"} or host.endswith((".youtube.com", ".youtube-nocookie.com")):
        return "YouTube video links cannot provide audio to this classifier. Upload an authorized audio or video file instead."
    return None


async def validate_public_url(url: str) -> str:
    if len(url) > 2048:
        raise LinkError("The link is too long.")
    try:
        parts = urlsplit(url)
        valid_port = parts.port in (None, 443)
    except ValueError as exc:
        raise LinkError("Enter a valid HTTPS media link.") from exc
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password or not valid_port:
        raise LinkError("Enter a public HTTPS link directly to an audio or video file.")
    if message := platform_message(parts.hostname):
        raise LinkError(message)
    try:
        addresses = await asyncio.wait_for(
            asyncio.get_running_loop().getaddrinfo(parts.hostname, 443, type=socket.SOCK_STREAM),
            timeout=5,
        )
    except (OSError, asyncio.TimeoutError) as exc:
        raise LinkError("The link's host could not be reached.") from exc
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise LinkError("Links to local or private network addresses are not allowed.")
    return url


async def _download(url: str, destination: Path, client: httpx.AsyncClient) -> str:
    current = url
    for _ in range(4):
        await validate_public_url(current)
        async with client.stream("GET", current) as response:
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("location")
                if not location:
                    raise LinkError("The media link redirected without a destination.")
                current = urljoin(current, location)
                continue
            if response.status_code != 200:
                raise LinkError(f"The media link returned HTTP {response.status_code}.")
            mime = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            suffix = Path(urlsplit(current).path).suffix.lower()
            if mime not in ALLOWED_MIME_TYPES and suffix not in ALLOWED_EXTENSIONS:
                raise LinkError("The link must point directly to an MP3, WAV, FLAC, MP4, WebM, or MOV file.")
            if mime and mime not in ALLOWED_MIME_TYPES:
                raise LinkError("The link returned a page instead of an audio or video file.")
            declared = response.headers.get("content-length")
            if declared:
                try:
                    declared_size = int(declared)
                except ValueError as exc:
                    raise LinkError("The media server returned an invalid file size.") from exc
                if declared_size > MAX_MEDIA_BYTES:
                    raise LinkError("The linked file is too large. The limit is 50 MB.")
            total = 0
            with destination.open("wb") as output:
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > MAX_MEDIA_BYTES:
                        raise LinkError("The linked file is too large. The limit is 50 MB.")
                    output.write(chunk)
            if total == 0:
                raise LinkError("The linked file is empty.")
            return urlsplit(current).hostname or "linked media"
    raise LinkError("The media link redirected too many times.")


async def download_media(url: str, destination: Path, *, client: httpx.AsyncClient | None = None) -> str:
    """Download one public HTTPS media file and return its host for display."""
    try:
        async with asyncio.timeout(90):
            if client is not None:
                return await _download(url, destination, client)
            timeout = httpx.Timeout(connect=5, read=15, write=5, pool=5)
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=False, trust_env=False) as actual:
                return await _download(url, destination, actual)
    except (httpx.HTTPError, OSError, TimeoutError) as exc:
        raise LinkError("Could not download this media file. Check the link and try again.") from exc
