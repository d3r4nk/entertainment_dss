"""Tests for :mod:`entdec.nlp` (chunker, textdocs, embedder)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from entdec.config import Config
from entdec.data import movielens, steam
from entdec.nlp import chunker, embedder, textdocs


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
class FakeOllama:
    """Deterministic embedder that records how many documents it embedded."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self.calls = 0
        self.embedded: list[str] = []

    def embed_batch(
        self, texts, batch_size: int = 24, dimensions: int | None = None, **_: object
    ) -> np.ndarray:
        self.calls += 1
        vectors = []
        for text in texts:
            self.embedded.append(str(text))
            seed = abs(hash(str(text))) % 1000
            rng = np.random.default_rng(seed)
            vector = rng.normal(size=self.dim).astype(np.float32)
            vectors.append(vector / np.linalg.norm(vector))
        return np.vstack(vectors)


@pytest.fixture
def nlp_config(make_config, raw_datasets) -> Config:
    cfg = make_config(
        "tiny",
        overrides={
            "profiles": {
                "tiny": {
                    "max_items": 6,
                    "max_users": 5,
                    "min_movie_ratings": 2,
                    "min_game_reviews": 2,
                    "min_interactions_per_user": 2,
                    "min_positive_per_user": 3,
                }
            },
            "nlp": {
                "embed_dim": 16,
                "embed_batch_size": 4,
                "chunk_max_tokens": 40,
                "chunk_target_tokens": 30,
                "chunk_overlap_tokens": 5,
            },
        },
    )
    movielens.run(cfg, force=True)
    steam.run(cfg, force=True)
    return cfg


# --------------------------------------------------------------------------- #
# chunker
# --------------------------------------------------------------------------- #
def test_estimate_tokens() -> None:
    assert chunker.estimate_tokens("") == 0
    assert chunker.estimate_tokens("one two three") == 4  # 3 words * 1.3


def test_short_document_is_one_chunk() -> None:
    chunks = chunker.chunk_text("A short document. Two sentences.")
    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].text == "A short document. Two sentences."


def test_split_sentences_handles_punctuation_and_newlines() -> None:
    text = "First sentence. Second one! Third?\nFourth."
    assert chunker.split_sentences(text) == [
        "First sentence.",
        "Second one!",
        "Third?",
        "Fourth.",
    ]


def test_long_document_is_split_into_overlapping_chunks() -> None:
    sentences = [f"Sentence number {index} with a few extra words in it." for index in range(40)]
    text = " ".join(sentences)
    chunks = chunker.chunk_text(text, max_tokens=100, target_tokens=80, overlap_tokens=20)
    assert len(chunks) > 1
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_chunks_reuse_the_overlap_tail() -> None:
    text = ". ".join(f"Part {i} of the story" for i in range(30)) + "."
    chunks = chunker.chunk_text(text, max_tokens=60, target_tokens=50, overlap_tokens=20)
    assert len(chunks) > 1
    first_words = set(chunks[0].text.split())
    second_words = set(chunks[1].text.split())
    assert first_words & second_words, "consecutive chunks should share the overlap"


def test_single_over_long_sentence_is_split_by_words() -> None:
    text = " ".join(f"word{index}" for index in range(400))
    chunks = chunker.chunk_text(text, max_tokens=100, target_tokens=80, overlap_tokens=10)
    assert len(chunks) > 1
    assert all(chunk.text for chunk in chunks)


def test_empty_document_yields_one_empty_chunk() -> None:
    chunks = chunker.chunk_text("")
    assert len(chunks) == 1
    assert chunks[0].tokens == 0


def test_all_words_survive_chunking() -> None:
    text = ". ".join(f"Token{i}" for i in range(200)) + "."
    chunks = chunker.chunk_text(text, max_tokens=50, target_tokens=40, overlap_tokens=10)
    joined = " ".join(chunk.text for chunk in chunks)
    for index in range(200):
        assert f"Token{index}" in joined


# --------------------------------------------------------------------------- #
# documents
# --------------------------------------------------------------------------- #
def test_movie_document_matches_the_spec_format() -> None:
    text = textdocs.build_movie_document(
        "Inception", 2010, ["Action", "Sci-Fi"], ["dreams", "heist"], ["great soundtrack"]
    )
    assert text == (
        "Inception (2010). Genres: Action, Sci-Fi. Themes: dreams, heist. "
        "User tags: great soundtrack."
    )


def test_game_document_matches_the_spec_format() -> None:
    text = textdocs.build_game_document(
        "Hades", 2020, ["Action", "Roguelike"], "Escape the underworld."
    )
    assert text == "Hades (2020). Tags: Action, Roguelike. Escape the underworld."


def test_documents_skip_missing_pieces() -> None:
    assert textdocs.build_movie_document("Alien", 0, [], [], []) == "Alien."
    assert textdocs.build_game_document("Grounded", None, [], "") == "Grounded."


def test_query_text_prepends_instruction() -> None:
    instruction = "Instruct: retrieve\nQuery: "
    assert textdocs.query_text("phim kinh di", instruction) == f"{instruction}phim kinh di"
    assert textdocs.query_text("x", "") == "x"


def test_documents_to_frames_counts_chunks(nlp_config: Config) -> None:
    documents = [
        {"domain": "movie", "item_id": 1, "text": "short one."},
        {
            "domain": "game",
            "item_id": 2,
            "text": " ".join(f"Sentence {i} here." for i in range(30)),
        },
    ]
    texts, chunks, stats = textdocs.documents_to_frames(documents, nlp_config)
    assert len(texts) == 2
    assert len(chunks) == stats.n_chunks
    assert stats.n_chunks > 2
    assert set(texts.columns) == set(textdocs.ITEM_TEXT_COLUMNS)
    assert set(chunks.columns) == set(textdocs.ITEM_CHUNK_COLUMNS)


def test_texts_step_writes_parquet(nlp_config: Config) -> None:
    assert textdocs.run(nlp_config, None) == 0
    texts = pd.read_parquet(nlp_config.paths.processed / textdocs.ITEM_TEXTS_PARQUET)
    chunks = pd.read_parquet(nlp_config.paths.processed / textdocs.ITEM_CHUNKS_PARQUET)
    assert set(texts["domain"]) == {"movie", "game"}
    assert set(chunks["domain"]) == {"movie", "game"}
    assert (chunks.groupby(["domain", "item_id"]).size() >= 1).all()
    assert nlp_config.report_path("texts.md").exists()


def test_real_documents_follow_the_spec(nlp_config: Config) -> None:
    documents = textdocs.build_movie_documents(nlp_config) + textdocs.build_game_documents(
        nlp_config
    )
    assert documents
    for document in documents:
        text = document["text"]
        assert text, "every document must have content"
        assert "." in text
        if document["domain"] == "movie":
            assert ". Genres: " in text
            assert ". Themes: " in text
        else:
            assert ". Tags: " in text or text.count(".") == 1


# --------------------------------------------------------------------------- #
# embedder
# --------------------------------------------------------------------------- #
def test_text_hash_is_stable_and_sensitive() -> None:
    assert embedder.text_hash("abc") == embedder.text_hash("abc")
    assert embedder.text_hash("abc") != embedder.text_hash("abd")


def test_chunk_key_format() -> None:
    assert embedder.chunk_key("movie", 7, 0, "deadbeef") == "movie:7:0:deadbeef"


def test_embedding_cache_round_trip(nlp_config: Config) -> None:
    directory = embedder.cache_dir(nlp_config, "test-model", 4)
    cache = embedder.EmbeddingCache(directory)
    key = embedder.chunk_key("movie", 1, 0, "aaaa")
    cache.put(key, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    cache.save_shard([key])

    reloaded = embedder.EmbeddingCache(directory).load()
    assert reloaded.n_loaded == 1
    assert reloaded.get(key) is not None


def test_clear_cache_removes_shards(nlp_config: Config) -> None:
    directory = embedder.cache_dir(nlp_config, "clear-me", 4)
    embedder.EmbeddingCache(directory).save_shard([])
    (directory / "shard_00000.parquet").write_bytes(b"not a parquet")
    assert embedder.clear_cache(nlp_config, "clear-me", 4) == 1


def test_select_chunks_filters(nlp_config: Config) -> None:
    chunks = pd.DataFrame(
        {
            "domain": ["movie", "movie", "game", "game"],
            "item_id": [1, 2, 3, 4],
            "chunk_index": [0, 0, 0, 0],
            "text": ["a", "b", "c", "d"],
            "n_tokens": [1, 1, 1, 1],
        }
    )
    assert len(embedder.select_chunks(chunks, domain="game")) == 2
    # limit_items keeps the N lowest (domain, item_id) pairs -> ('game', 3).
    limited = embedder.select_chunks(chunks, limit_items=1)
    assert len(limited) == 1
    assert limited.iloc[0]["item_id"] == 3


def test_aggregate_item_vectors_is_normalised_mean() -> None:
    frame = pd.DataFrame(
        {
            "domain": ["movie", "movie"],
            "item_id": [1, 1],
            "chunk_index": [0, 1],
            "text_hash": ["a", "b"],
            "embedding": [
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
            ],
        }
    )
    result = embedder.aggregate_item_vectors(frame)
    assert len(result) == 1
    vector = np.asarray(result.iloc[0]["embedding"])
    assert np.linalg.norm(vector) == pytest.approx(1.0)
    assert vector[0] == pytest.approx(vector[1])


def test_aggregate_item_vectors_empty() -> None:
    assert embedder.aggregate_item_vectors(pd.DataFrame()).empty


def test_embed_chunks_caches_and_resumes(nlp_config: Config) -> None:
    chunks = pd.DataFrame(
        {
            "domain": ["movie", "movie", "game"],
            "item_id": [1, 1, 2],
            "chunk_index": [0, 1, 0],
            "text": ["alpha", "beta", "gamma"],
            "n_tokens": [1, 1, 1],
        }
    )
    fake = FakeOllama()
    vectors, stats = embedder.embed_chunks(nlp_config, chunks, client=fake)
    assert len(vectors) == 3
    assert stats.n_new_chunks == 3
    assert fake.calls == 1

    vectors2, stats2 = embedder.embed_chunks(nlp_config, chunks, client=fake)
    assert stats2.n_new_chunks == 0
    assert stats2.n_cached_chunks == 3
    assert fake.calls == 1, "a resumed run must not call Ollama again"
    assert len(vectors2) == 3


def test_embed_chunks_reembeds_changed_text_only(nlp_config: Config) -> None:
    chunks = pd.DataFrame(
        {
            "domain": ["movie"],
            "item_id": [1],
            "chunk_index": [0],
            "text": ["original text"],
            "n_tokens": [2],
        }
    )
    fake = FakeOllama()
    embedder.embed_chunks(nlp_config, chunks, client=fake)
    changed = chunks.assign(text=["changed text"])
    _, stats = embedder.embed_chunks(nlp_config, changed, client=fake)
    assert stats.n_new_chunks == 1
    assert stats.n_cached_chunks == 0


def test_embed_chunks_force_clears_cache(nlp_config: Config) -> None:
    chunks = pd.DataFrame(
        {
            "domain": ["movie"],
            "item_id": [1],
            "chunk_index": [0],
            "text": ["force me"],
            "n_tokens": [2],
        }
    )
    fake = FakeOllama()
    embedder.embed_chunks(nlp_config, chunks, client=fake)
    _, stats = embedder.embed_chunks(nlp_config, chunks, client=fake, force=True)
    assert stats.n_new_chunks == 1


def test_embedder_run_end_to_end(nlp_config: Config) -> None:
    textdocs.run(nlp_config, None)
    fake = FakeOllama()
    assert embedder.run(nlp_config, None, client=fake) == 0

    items = pd.read_parquet(nlp_config.paths.processed / embedder.ITEM_EMBEDDINGS_PARQUET)
    chunks = pd.read_parquet(nlp_config.paths.processed / embedder.CHUNK_EMBEDDINGS_PARQUET)
    assert set(items["domain"]) == {"movie", "game"}
    assert (items["dim"] > 0).all()
    assert chunks["text_hash"].str.len().eq(16).all()
    assert nlp_config.report_path("embeddings.md").exists()

    norms = np.linalg.norm(np.vstack(items["embedding"].to_numpy()), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_embedder_run_without_texts_fails(make_config) -> None:
    cfg = make_config("tiny")
    assert embedder.run(cfg, None, client=FakeOllama()) == 1


def test_embedder_limit_keeps_output_complete(nlp_config: Config) -> None:
    textdocs.run(nlp_config, None)
    fake = FakeOllama()
    embedder.run(nlp_config, None, client=fake)
    full = len(pd.read_parquet(nlp_config.paths.processed / embedder.ITEM_EMBEDDINGS_PARQUET))

    class Args:
        domain = None
        limit = 1
        force = False

    embedder.run(nlp_config, Args(), client=fake)
    after = len(pd.read_parquet(nlp_config.paths.processed / embedder.ITEM_EMBEDDINGS_PARQUET))
    assert after == full
