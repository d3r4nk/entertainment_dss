from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path

from pydantic import SecretStr

from entdec.ui import recommendation_media


def test_movie_poster_resolves_movie_lens_id_to_tmdb_id(
    tmp_path: Path, monkeypatch
) -> None:
    links = tmp_path / "links.csv"
    links.write_text("movieId,imdbId,tmdbId\n42,0114709,862\n", encoding="utf-8")
    token = SecretStr("test-token")
    future: Future[bytes | None] = Future()
    calls = []

    def request_image(purpose, domain, item_id, received_token):
        calls.append((purpose, domain, item_id, received_token))
        return future

    monkeypatch.setattr(recommendation_media, "_request_image", request_image)
    service = recommendation_media.RecommendationMedia(links, token)

    assert service.poster_for({"domain": "movie", "id": 42}) is future
    assert calls == [("poster", "movie", 862, token)]


def test_poster_rejects_invalid_item_ids(tmp_path: Path) -> None:
    service = recommendation_media.RecommendationMedia(tmp_path / "links.csv", None)

    assert service.poster_for({"domain": "movie", "id": "not-an-id"}) is None
    assert service.poster_for({"domain": "music", "id": 42}) is None
