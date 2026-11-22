"""External links and thumbnails for the chat UI's recommendation cards."""

from __future__ import annotations

import csv
import re
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Any
from urllib.parse import urlsplit

import httpx
from PIL import Image, ImageOps
from pydantic import SecretStr

_MOVIE_POSTER = re.compile(r"/[a-zA-Z0-9_-]+\.(?:jpg|jpeg|png|webp)")
_POOL = ThreadPoolExecutor(max_workers=3, thread_name_prefix="card-media")
_CACHE_SIZE = 128
_MAX_IMAGE_BYTES = 2 * 1024 * 1024
_CACHE: OrderedDict[tuple[str, int, SecretStr | None], tuple[float, Future[bytes | None]]] = (
    OrderedDict()
)
_LOCK = Lock()


@dataclass(frozen=True)
class CardMedia:
    label: str | None
    url: str | None
    image: Future[bytes | None] | None = None


def _positive_id(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    if not re.fullmatch(r"[0-9]{1,10}", str(value).strip()):
        return None
    return int(value) or None


@lru_cache(maxsize=4)
def _movie_ids(path: Path, _modified_ns: int) -> dict[int, int]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            result = {}
            for row in csv.DictReader(handle):
                movie_id = _positive_id(row.get("movieId"))
                tmdb_id = _positive_id(row.get("tmdbId"))
                if movie_id and tmdb_id:
                    result[movie_id] = tmdb_id
            return result
    except (OSError, UnicodeError, csv.Error):
        return {}


def _image_url(
    client: httpx.Client, domain: str, item_id: int, token: SecretStr | None
) -> str | None:
    if domain == "movie":
        if not token:
            return None
        response = client.get(
            f"https://api.themoviedb.org/3/movie/{item_id}",
            headers={"Authorization": f"Bearer {token.get_secret_value()}"},
        )
        response.raise_for_status()
        path = response.json().get("poster_path")
        if isinstance(path, str) and _MOVIE_POSTER.fullmatch(path):
            return f"https://image.tmdb.org/t/p/w500{path}"
        return None

    response = client.get(
        "https://store.steampowered.com/api/appdetails", params={"appids": item_id}
    )
    response.raise_for_status()
    entry = response.json().get(str(item_id)) or {}
    if not entry.get("success"):
        return None
    data = entry.get("data") or {}
    url = data.get("header_image")
    if _positive_id(data.get("steam_appid")) != item_id or not isinstance(url, str):
        return None
    parsed = urlsplit(url)
    if (
        parsed.scheme == "https"
        and parsed.port in (None, 443)
        and parsed.username is None
        and (parsed.hostname or "").endswith(".steamstatic.com")
    ):
        return url
    return None


def _load_thumbnail(domain: str, item_id: int, token: SecretStr | None) -> bytes | None:
    try:
        with httpx.Client(timeout=3.0) as client:
            url = _image_url(client, domain, item_id, token)
            if url is None:
                return None
            content = BytesIO()
            with client.stream("GET", url) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    if content.tell() + len(chunk) > _MAX_IMAGE_BYTES:
                        return None
                    content.write(chunk)
            content.seek(0)
            with Image.open(content) as image:
                if image.width * image.height > 8_000_000:
                    return None
                thumbnail = ImageOps.pad(image.convert("RGB"), (320, 180), color="#27272a")
                result = BytesIO()
                thumbnail.save(result, format="JPEG", quality=85)
                return result.getvalue()
    except (
        httpx.HTTPError, ValueError, AttributeError, OSError, SyntaxError,
        Image.DecompressionBombError,
    ):
        return None


def _request_thumbnail(
    domain: str, item_id: int, token: SecretStr | None = None
) -> Future[bytes | None] | None:
    key = (domain, item_id, token)
    now = monotonic()
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            started, image = cached
            ttl = 3600 if image.done() and image.result() is not None else 60
            if not image.done() or now - started < ttl:
                _CACHE.move_to_end(key)
                return image
            del _CACHE[key]
        if len(_CACHE) >= _CACHE_SIZE:
            oldest_done = next((entry for entry, (_, job) in _CACHE.items() if job.done()), None)
            if oldest_done is None:
                return None
            del _CACHE[oldest_done]
        image = _POOL.submit(_load_thumbnail, domain, item_id, token)
        _CACHE[key] = (now, image)
        return image


class RecommendationMedia:
    def __init__(self, links_path: Path, tmdb_token: SecretStr | None) -> None:
        self.links_path = links_path
        self.tmdb_token = tmdb_token

    def for_item(self, item: dict[str, Any]) -> CardMedia | None:
        domain = item.get("domain")
        item_id = _positive_id(item.get("id"))
        if domain not in ("movie", "game"):
            return None  # older history has no unambiguous per-item domain
        if item_id is None:
            return CardMedia(None, None)
        if domain == "game":
            return CardMedia(
                "Mở trên Steam", f"https://store.steampowered.com/app/{item_id}/",
                _request_thumbnail("game", item_id),
            )
        try:
            modified = self.links_path.stat().st_mtime_ns
        except OSError:
            return CardMedia(None, None)
        tmdb_id = _movie_ids(self.links_path, modified).get(item_id)
        if tmdb_id is None:
            return CardMedia(None, None)
        image = _request_thumbnail("movie", tmdb_id, self.tmdb_token) if self.tmdb_token else None
        return CardMedia("Xem trên TMDB", f"https://www.themoviedb.org/movie/{tmdb_id}", image)
