"""Chatbot pipeline (section 8): intent -> retrieval -> rerank -> explanation.

* **Intent** (:mod:`entdec.llm.intent`) parses the question into a validated
  schema (LLM, one retry, keyword fallback).
* **Retrieval** resolves ``reference_titles`` with rapidfuzz against the graph
  (:data:`entdec.graph.queries.ALL_ITEM_TITLES`), embeds the question with the
  Qwen3 instruction prefix and queries the Neo4j vector index
  (:data:`VECTOR_QUERY`), then expands through the parameterized theme/genre/
  SIMILAR_TO templates. When Neo4j is unreachable the equivalent searches run
  over the local Parquet artefacts so the chatbot still works. The LLM never
  writes Cypher (v1 rule, spec section 8 step 2).
* **Rerank** blends the locally computed hybrid parts (content embedding,
  KG theme overlap, popularity) with the per-profile/domain chat weights in
  ``config/tuned_weights.yaml``. These weights are validation-tuned using the
  user's liked items as an item-to-item query; MF and ItemKNN are excluded.
* **Explanation** builds the evidence strings in code and qwen3:4b only phrases
  them in the user's language (Vietnamese by default). Only retrieved items are
  visible to the LLM. If generation fails, a deterministic local phrasing is
  returned instead.

Embedding and generation run strictly in sequence inside :meth:`ChatPipeline.run`
(one model resident in VRAM at a time). The query embedding is computed once
per turn and reused by both domains.
"""

from __future__ import annotations

import contextlib
import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from entdec.chatbot.progress import Progress
from entdec.chatbot.session import ChatSession
from entdec.config import Config
from entdec.llm.generator import generate_answer
from entdec.llm.intent import (
    Intent,
    canonical_genres,
    normalize_theme_names,
    parse_intent,
    rules_intent,
)
from entdec.logging_setup import get_logger
from entdec.nlp.ollama_client import OllamaClient, l2_normalize
from entdec.recsys.common import DomainData, load_domain, top_k_positions
from entdec.recsys.evaluate import CHAT_MODEL_NAMES, chat_weights_from_cfg
from entdec.recsys.hybrid import min_max_normalise

logger = get_logger(__name__)


def normalise_chat_weights(
    weights: dict[str, float], available: set[str] | None = None
) -> dict[str, float]:
    """Filter and normalise chat weights, with a non-zero safe fallback."""
    allowed = set(CHAT_MODEL_NAMES) if available is None else available
    filtered = {
        name: float(weight)
        for name, weight in weights.items()
        if name in allowed and np.isfinite(weight) and float(weight) >= 0.0
    }
    total = sum(filtered.values())
    if total > 0.0:
        return {name: weight / total for name, weight in filtered.items()}

    fallback = {
        name: weight
        for name, weight in {
            "popularity": 0.05,
            "content": 0.475,
            "kg": 0.475,
        }.items()
        if name in allowed
    }
    fallback_total = sum(fallback.values())
    return {name: weight / fallback_total for name, weight in fallback.items()}


EVIDENCE_LABELS = {
    "en": {"themes": "themes", "genres": "genres", "similar": "similar to", "match": "match"},
    "vi": {"themes": "chủ đề", "genres": "thể loại", "similar": "tương tự", "match": "điểm"},
}

NO_RESULT_MESSAGE = {
    "vi": "Chưa tìm được gợi ý phù hợp trong danh mục hiện có. Bạn thử diễn đạt lại giúp mình nhé?",
    "en": "No matching item was found in the current catalogue - please try rephrasing.",
}

_FOLLOW_UP_RE = re.compile(
    r"\b(?:more|another|additional|again|lighter|violence|violent|it|that|one|"
    r"thêm|nữa|nó|đó|cái|nhẹ nhàng|bạo lực)\b",
    re.I,
)
_REFERENCE_QUESTION_RE = re.compile(
    r"(?:là gì|là ai|mô tả|nói về|what is|what's|describe|tell me about|which one)",
    re.I,
)
_REFERENCE_CUE_RE = re.compile(
    r"\b(?:like|similar to|love|loved|liked|likes|fans of|enjoy|enjoyed|"
    r"inspired by|already played|already seen|played|seen|don't like|do not like|"
    r"thích|giống|hay chơi|cho người thích|đã xem|đã chơi)\b",
    re.I,
)


def _is_short_follow_up(question: str) -> bool:
    return len(question.split()) <= 12 and bool(_FOLLOW_UP_RE.search(question))


def _has_constraint_change(question: str) -> bool:
    return bool(re.search(r"(?:less|more|không|ít|nhiều|tránh|avoid|without)", question, re.I))


def _is_lighter_request(question: str) -> bool:
    return bool(re.search(
        r"(?:ít bạo lực hơn|nhẹ nhàng hơn|something lighter|lighter|less violent|less violence)",
        question,
        re.I,
    ))


def _is_less_violent_request(question: str) -> bool:
    return bool(re.search(r"(?:ít bạo lực hơn|less violent|less violence)", question, re.I))


def _matches_constraint_exclusion(
    candidate: Candidate, ranker: ChatRanker, terms: list[str]
) -> bool:
    if not terms:
        return False
    labels = list(ranker.theme_vectors.get(candidate.item_id, {}))
    labels.extend(
        str(row.genre)
        for row in ranker.data.genres.itertuples(index=False)
        if int(row.item_id) == candidate.item_id
    )
    return any(
        term in label.casefold()
        or (term == "violence" and ("violent" in label.casefold() or "gore" in label.casefold()))
        for term in terms
        for label in labels
    )


def _asks_about_reference(question: str) -> bool:
    return bool(_REFERENCE_QUESTION_RE.search(question))


def _asks_repeat(question: str) -> bool:
    return bool(re.search(r"(?:lặp lại|nói lại|repeat|again|show again)", question, re.I))


# --------------------------------------------------------------------------- #
# result types
# --------------------------------------------------------------------------- #
@dataclass
class Candidate:
    """One retrieved item plus the evidence the answer will quote."""

    item_id: int
    title: str
    position: int  # row in the DomainData catalog
    domain: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    shared_themes: list[str] = field(default_factory=list)
    shared_genres: list[str] = field(default_factory=list)
    similar_to: dict[str, float] = field(default_factory=dict)  # reference title -> cosine

    def evidence_line(self, language: str = "en") -> str:
        """One grounded evidence line, rendered in the user's language.

        The theme/genre names themselves stay in the catalogue language; only
        the labels around them are translated.
        """
        labels = EVIDENCE_LABELS.get(language[:2].lower(), EVIDENCE_LABELS["en"])
        parts: list[str] = []
        if self.shared_themes:
            parts.append(f"{labels['themes']}: " + ", ".join(self.shared_themes[:3]))
        if self.shared_genres:
            parts.append(f"{labels['genres']}: " + ", ".join(self.shared_genres[:3]))
        if self.similar_to:
            best = max(self.similar_to.items(), key=lambda kv: kv[1])
            parts.append(f"{labels['similar']} {best[0]} ({best[1]:.2f})")
        parts.append(f"{labels['match']} {self.scores.get('total', 0.0):.2f}")
        return f"{self.title}: " + "; ".join(parts)


@dataclass
class ChatResult:
    """Everything one chat turn produces (shared by CLI, API and UI)."""

    question: str
    language: str
    intent: Intent
    domains_queried: list[str]
    items: list[Candidate]
    answer: str
    used_graph: bool
    intent_llm: bool
    answer_llm: bool
    # catalogue titles the user's references resolved to (both domains)
    resolved_references: list[str] = field(default_factory=list)
    durations: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        """JSON-ready view (for the API)."""
        return {
            "question": self.question,
            "language": self.language,
            "intent": self.intent.model_dump(),
            "domains_queried": self.domains_queried,
            "used_graph": self.used_graph,
            "intent_llm": self.intent_llm,
            "answer_llm": self.answer_llm,
            "answer": self.answer,
            "resolved_references": self.resolved_references,
            "durations": self.durations,
            "items": [
                {
                    "id": item.item_id,
                    "domain": item.domain,
                    "title": item.title,
                    "score": round(item.scores.get("total", 0.0), 4),
                    "shared_themes": item.shared_themes,
                    "shared_genres": item.shared_genres,
                    "similar_to": item.similar_to,
                    "evidence": item.evidence_line(self.language),
                }
                for item in self.items
            ],
        }


# --------------------------------------------------------------------------- #
# title resolution (shared by the graph and local backends)
# --------------------------------------------------------------------------- #
def fuzzy_resolve(wanted: list[str], titles: dict[str, Any]) -> dict[str, Any]:
    """Map free-text titles onto catalogue titles with rapidfuzz (best match).

    ``titles`` maps ``title -> value``; the value may be a raw item id or a
    ``(domain, raw id)`` pair. The result maps resolved title -> value.
    """
    from rapidfuzz import process

    resolved: dict[str, Any] = {}
    if not titles or not wanted:
        return {}
    # Steam/MovieLens titles carry ®/™ and trailing articles; matching on a
    # normalised form lets "The Witcher 3" reach "The Witcher® 3: Wild Hunt".
    normalised: dict[str, str] = {}
    for title in titles:
        normalised.setdefault(normalise_title(title), title)
    choices = list(normalised)
    for want in wanted:
        matches = process.extract(normalise_title(want), choices, limit=1, score_cutoff=70.0)
        if not matches:
            logger.info("unresolved catalogue reference")
            continue
        match_title = normalised[str(matches[0][0])]
        resolved[match_title] = titles[match_title]
        logger.info("resolved catalogue reference")
    return resolved


def normalise_title(title: str) -> str:
    """Lowercase a title and drop trademark symbols for fuzzy matching."""
    cleaned = title.lower()
    for symbol in ("®", "™", "©"):
        cleaned = cleaned.replace(symbol, "")
    return " ".join(cleaned.split())


def _normalise_reference_text(text: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", text.casefold()).split())


def node_id_of(domain: str, raw: int) -> str:
    from entdec.graph.ids import item_node_id

    return item_node_id(domain, int(raw))


# --------------------------------------------------------------------------- #
# graph retrieval backend
# --------------------------------------------------------------------------- #
class GraphRetriever:
    """Reference resolution + candidate discovery through Neo4j templates."""

    def __init__(self, cfg: Config) -> None:
        from entdec.config import PROJECT_ROOT  # noqa: F401
        from entdec.graph.neo4j_client import Neo4jClient

        self.client = Neo4jClient.from_config(cfg)
        self._titles: dict[str, tuple[str, int]] | None = None  # title -> (domain, raw id)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.client.close()

    def load_titles(self) -> dict[str, tuple[str, int]]:
        """``title -> {'movie'|'game', raw id}`` for rapidfuzz resolution."""
        from entdec.graph.ids import parse_item_id
        from entdec.graph.queries import ALL_ITEM_TITLES

        if self._titles is None:
            rows = self.client.run(ALL_ITEM_TITLES.cypher)
            titles: dict[str, tuple[str, int]] = {}
            for row in rows:
                domain, raw = parse_item_id(str(row["id"]))
                titles[str(row["title"])] = (domain, raw)
            self._titles = titles
        return self._titles

    def vector_search(
        self, embedding: np.ndarray, domain: str, k: int
    ) -> dict[int, float]:
        from entdec.graph.queries import VECTOR_QUERY, vector_query_params

        raw_rows = self.client.run(
            VECTOR_QUERY.cypher,
            **vector_query_params(embedding.astype(float).tolist(), k=k, domain=domain),
        )
        from entdec.graph.ids import parse_item_id

        return {
            int(parse_item_id(str(row["id"]))[1]): float(row["score"]) for row in raw_rows
        }

    def by_theme(self, themes: list[str], domain: str, k: int) -> dict[int, float]:
        from entdec.graph.ids import parse_item_id
        from entdec.graph.queries import ITEM_IDS_BY_THEME, theme_params

        raw_rows = self.client.run(
            ITEM_IDS_BY_THEME.cypher, **theme_params(themes, domain, k)
        )
        return {int(parse_item_id(str(row["id"]))[1]): float(row["theme_score"]) for row in raw_rows}

    def by_genre(self, genres: list[str], domain: str, k: int) -> dict[int, float]:
        from entdec.graph.ids import parse_item_id
        from entdec.graph.queries import ITEM_IDS_BY_GENRE, genre_params

        raw_rows = self.client.run(
            ITEM_IDS_BY_GENRE.cypher, **genre_params(genres, domain, k)
        )
        return {int(parse_item_id(str(row["id"]))[1]): float(row["genre_hits"]) for row in raw_rows}

    def similar_to(self, reference_ids: list[int], domain: str, k: int) -> dict[int, float]:
        """SIMILAR_TO neighbours of the reference items (graph expansion)."""
        from entdec.graph.ids import parse_item_id
        from entdec.graph.queries import ITEM_IDS_SIMILAR_TO, similar_params

        rows = self.client.run(
            ITEM_IDS_SIMILAR_TO.cypher,
            **similar_params([node_id_of(domain, raw) for raw in reference_ids], domain, k),
        )
        return {int(parse_item_id(str(row["id"]))[1]): float(row["score"]) for row in rows}


# --------------------------------------------------------------------------- #
# reranker
# --------------------------------------------------------------------------- #
class ChatRanker:
    """Hybrid reranking with validation-tuned anonymous-chat weights."""

    def __init__(self, cfg: Config, domain: str) -> None:
        self.cfg = cfg
        self.domain = domain
        self.data: DomainData = load_domain(cfg, domain, with_embeddings=True)
        from entdec.recsys.itemknn import ItemKNNModel
        from entdec.recsys.kg_scoring import item_theme_vectors
        from entdec.recsys.popularity import popularity_scores

        self.knn = ItemKNNModel(self.data)
        self.theme_vectors = item_theme_vectors(self.data)
        self.popularity = popularity_scores(self.data)

        weights = chat_weights_from_cfg(cfg, domain)
        available = set(CHAT_MODEL_NAMES)
        if self.data.embeddings is None:
            available.discard("content")
        self.weights = normalise_chat_weights(weights, available)
        logger.info("chat rerank weights (%s): %s", domain, self.weights)

    def is_known_item(self, raw: int) -> bool:
        return self.data.catalog_index.get(int(raw)) is not None

    def titles_by_id(self) -> dict[int, str]:
        key = self.data.item_key
        return {
            int(row[0]): str(row[1])
            for row in zip(self.data.items[key], self.data.items["title"], strict=True)
            if int(row[0]) in self.data.catalog_index
        }

    def item_theme_names(self) -> list[str]:
        return sorted({theme for vector in self.theme_vectors.values() for theme in vector})

    def parts_for(
        self,
        liked_ids: list[int],
        query_vector: np.ndarray | None,
        extra_themes: list[str] | None = None,
        extra_theme_weights: dict[str, float] | None = None,
    ) -> dict[str, np.ndarray]:
        """Content / itemknn / kg / popularity vectors over the catalogue."""
        data = self.data
        known_positions = known_positions_for(data, liked_ids)

        content = np.zeros(len(data.catalog), dtype=np.float64)
        if data.embeddings is not None and (known_positions or query_vector is not None):
            stack = []
            if known_positions:
                stack.append(data.embeddings[known_positions].mean(axis=0))
            if query_vector is not None:
                stack.append(np.asarray(query_vector, dtype=np.float32))
            profile = np.mean(stack, axis=0)
            norm = np.linalg.norm(profile)
            if norm > 0:
                content = np.asarray(data.embeddings @ (profile / norm), dtype=np.float64)

        knn = np.zeros(len(data.catalog), dtype=np.float64)
        if known_positions:
            mask = np.zeros(len(data.catalog), dtype=np.float32)
            mask[known_positions] = 1.0
            knn = np.asarray(self.knn.similarity @ mask).ravel().astype(np.float64)

        kg = np.zeros(len(data.catalog), dtype=np.float64)
        theme_profile = self.theme_profile_for(liked_ids, extra_themes, extra_theme_weights)
        if theme_profile:
            for pos, raw in enumerate(data.catalog):
                vector = self.theme_vectors.get(int(raw))
                if vector:
                    kg[pos] = sum(
                        weight * theme_profile[theme]
                        for theme, weight in vector.items()
                        if theme in theme_profile
                    )

        return {
            "content": content,
            "itemknn": knn,
            "kg": kg,
            "popularity": self.popularity.astype(np.float64),
        }

    def theme_profile_for(
        self,
        liked_ids: list[int],
        extra_themes: list[str] | None = None,
        extra_theme_weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Normalised theme profile over reference items (+ intent/cross themes).

        ``extra_theme_weights`` carries the *weighted* theme vector of a
        cross-domain reference item (e.g. the movie "Inception" when the
        pipeline is ranking games), so its themes transfer across domains.
        """
        profile: dict[str, float] = {}
        for raw in liked_ids:
            vector = self.theme_vectors.get(int(raw), {})
            for theme, weight in vector.items():
                profile[theme] = profile.get(theme, 0.0) + float(weight)
        for theme, weight in (extra_theme_weights or {}).items():
            profile[theme] = profile.get(theme, 0.0) + float(weight)
        for theme in extra_themes or []:
            profile.setdefault(theme, 0.5)
        total = sum(profile.values())
        return {t: w / total for t, w in profile.items()} if total > 0 else {}

    def rerank(
        self,
        pool: dict[int, Candidate],
        liked_ids: list[int],
        query_vector: np.ndarray | None,
        *,
        extra_themes: list[str] | None = None,
        extra_theme_weights: dict[str, float] | None = None,
        excluded_ids: set[int] | None = None,
        k: int,
    ) -> list[Candidate]:
        """Blend parts over every pooled candidate and return the top-k."""
        data = self.data
        if not pool:
            return []

        parts = self.parts_for(liked_ids, query_vector, extra_themes, extra_theme_weights)
        normalised = {name: min_max_normalise(vector) for name, vector in parts.items()}

        blend = np.zeros(len(data.catalog), dtype=np.float64)
        for name, vector in normalised.items():
            blend += self.weights.get(name, 0.0) * vector

        candidate_mask = np.zeros(len(data.catalog), dtype=bool)
        for candidate in pool.values():
            candidate_mask[candidate.position] = True
        blend = np.where(candidate_mask, blend, -np.inf)

        excluded = set(excluded_ids or set())
        for raw in set(liked_ids) | excluded:
            pos = data.item_pos(raw)
            if pos is not None:
                blend[pos] = -np.inf

        finite = int((blend > -np.inf).sum())
        if finite == 0:
            return []
        positions = top_k_positions(blend, min(int(k), finite))
        theme_lookup = self.theme_profile_for(liked_ids, extra_themes, extra_theme_weights)
        for pos in positions:
            candidate = pool[data.catalog[pos]]
            for name, vector in normalised.items():
                if name in self.weights:
                    candidate.scores[name] = float(vector[pos])
            candidate.scores["kg_profile"] = 0.0
            candidate.scores["total"] = float(blend[pos])
            shared = [
                theme for theme, weight in theme_lookup.items() if theme in self.theme_vectors.get(candidate.item_id, {})
            ]
            candidate.shared_themes = sorted(
                shared, key=lambda t: -(self.theme_vectors[candidate.item_id].get(t, 0.0))
            )
        return [pool[data.catalog[pos]] for pos in positions]

# --------------------------------------------------------------------------- #
# position / id helpers
# --------------------------------------------------------------------------- #
def known_positions_for(data: DomainData, raw_ids) -> list[int]:
    positions: list[int] = []
    for raw in raw_ids:
        pos = data.item_pos(int(raw))
        if pos is not None:
            positions.append(pos)
    return positions


def initialise_candidates(pool: dict[int, Candidate], ranker: ChatRanker, raw: int, scores: dict[str, float]) -> None:
    """Merge one raw item into the pool, keeping its best scores."""
    if not ranker.is_known_item(raw):
        return
    candidate = pool.get(int(raw))
    if candidate is None:
        titles = ranker.titles_by_id()
        candidate = Candidate(
            item_id=int(raw),
            title=titles.get(int(raw), str(raw)),
            position=int(ranker.data.item_pos(int(raw))),
            domain=ranker.domain,
        )
        pool[int(raw)] = candidate
    for key, score in scores.items():
        candidate.scores[key] = max(candidate.scores.get(key, 0.0), float(score))


def add_genre_evidence(pool: dict[int, Candidate], ranker: ChatRanker, genres: list[str]) -> None:
    if not genres:
        return
    genre_by_item: dict[int, list[str]] = {}
    wanted = {g.lower() for g in genres}
    for row in ranker.data.genres.itertuples(index=False):
        if str(row.genre).lower() in wanted:
            genre_by_item.setdefault(int(row.item_id), []).append(str(row.genre))
    for candidate in pool.values():
        candidate.shared_genres = sorted(set(genre_by_item.get(candidate.item_id, [])))


def add_same_similar_evidence(
    pool: dict[int, Candidate],
    ranker: ChatRanker,
    reference_ids: list[int],
    reference_titles: dict[str, int],
) -> None:
    """Cosine between candidates and SAME-domain reference items (evidence)."""
    if ranker.data.embeddings is None or not reference_ids:
        return
    matrix = l2_normalize(ranker.data.embeddings.astype(np.float32))
    id_to_title = {raw: title for title, raw in reference_titles.items()}
    for candidate in pool.values():
        if candidate.item_id in reference_ids:
            continue
        for raw in reference_ids:
            ref_pos = ranker.data.item_pos(raw)
            if ref_pos is None or candidate.position < 0:
                continue
            score = float(np.dot(matrix[candidate.position], matrix[ref_pos]))
            if score > 0:
                candidate.similar_to[id_to_title.get(raw, str(raw))] = round(score, 3)


def add_cross_similar_evidence(
    pool: dict[int, Candidate],
    ranker: ChatRanker,
    cross_refs: list[tuple[str, str, int]],
    other_rankers: dict[str, ChatRanker],
) -> None:
    """Cosine between this domain's candidates and OTHER-domain reference items.

    Item embeddings of both domains live in the same space, so "similar to
    Inception (0.63)" is computed from the vectors, not guessed by the LLM.
    """
    if ranker.data.embeddings is None or not cross_refs:
        return
    for candidate in pool.values():
        if candidate.position < 0 or candidate.position >= len(ranker.data.embeddings):
            continue
        candidate_vector = ranker.data.embeddings[candidate.position]
        for title, other_domain, raw in cross_refs:
            other_ranker = other_rankers.get(other_domain)
            if other_ranker is None or other_ranker.data.embeddings is None:
                continue
            other_pos = other_ranker.data.item_pos(raw)
            if other_pos is None:
                continue
            score = float(np.dot(candidate_vector, other_ranker.data.embeddings[other_pos]))
            if score > 0:
                candidate.similar_to[title] = max(
                    candidate.similar_to.get(title, 0.0), round(score, 3)
                )


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
class ChatPipeline:
    """One chat turn, reusing long-lived state (client, graph, rankers)."""

    def __init__(
        self,
        cfg: Config,
        *,
        client: OllamaClient | None = None,
        use_graph: bool | None = None,
    ) -> None:
        self.cfg = cfg
        self.client = client or OllamaClient.from_config(cfg)
        self._graph: GraphRetriever | None = None
        graph_enabled = not cfg.env.entdec_no_graph if use_graph is None else use_graph
        self._graph_ok: bool | None = None if graph_enabled else False
        self._rankers: dict[str, ChatRanker] = {}

    # --- state management ------------------------------------------------------
    def _ranker(self, domain: str) -> ChatRanker:
        if domain not in self._rankers:
            self._rankers[domain] = ChatRanker(self.cfg, domain)
        return self._rankers[domain]

    @property
    def graph(self) -> GraphRetriever:
        if self._graph is None:
            self._graph = GraphRetriever(self.cfg)
        return self._graph

    def question_embedding(self, question: str) -> np.ndarray | None:
        """Embed the query with the Qwen3 instruction prefix (section 6)."""
        instruction = str(self.cfg.get("nlp.query_instruction", ""))
        dim = int(self.cfg.get("nlp.embed_dim", 1024))
        try:
            result = self.client.embed([instruction + question], dimensions=dim, keep_alive="5m")
        except Exception as exc:  # noqa: BLE001 - retrieval continues without it
            logger.warning("query embedding failed (%s)", exc)
            return None
        if result.vectors.shape[0] != 1 or result.vectors.shape[1] == 0:
            return None
        return result.vectors[0]

    # --- retrieval --------------------------------------------------------------
    def resolve_references(self, wanted: list[str]) -> dict[str, tuple[str, int]]:
        """Resolve free-text titles against the JOINT pool of both domains.

        Each reference is fuzzy-matched over movie AND game titles at once and
        kept only in the domain it actually belongs to, so a movie reference
        such as "Inception" is never forced into the game catalogue (which
        would happily return "Inscryption"). The graph's title list is used
        when Neo4j is reachable, the local Parquet otherwise.
        """
        if not wanted:
            return {}
        return fuzzy_resolve(wanted, self.title_catalog())

    def title_catalog(self) -> dict[str, tuple[str, int]]:
        """Return verified title-to-domain/id mappings from the active backends."""
        pool: dict[str, tuple[str, int]] = {}
        if self._graph_ok is not False:
            try:
                pool = dict(self.graph.load_titles())
            except Exception as exc:  # noqa: BLE001 - graph optional
                self._graph_ok = False
                logger.info("graph titles unavailable (%s) - resolving locally", exc)
        if not pool:
            for domain in ("movie", "game"):
                try:
                    for raw, title in self._ranker(domain).titles_by_id().items():
                        pool.setdefault(title, (domain, raw))
                except Exception as exc:  # noqa: BLE001 - one domain may be absent
                    logger.info("local titles unavailable for %s (%s)", domain, exc)
        return pool

    def resolve_message_references(self, message: str) -> dict[str, tuple[str, int]]:
        """Resolve literal catalogue-title mentions before intent parsing."""
        normalized_message = _normalise_reference_text(message)
        padded_message = f" {normalized_message} "
        found: dict[str, tuple[str, int]] = {}
        for title, reference in self.title_catalog().items():
            clean_title = normalise_title(title)
            aliases = {clean_title}
            aliases.add(re.sub(r"\s*\([^)]*\)", "", clean_title).strip())
            tokens = aliases.copy()
            for alias in aliases:
                words = alias.split()
                tokens.update(" ".join(words[:size]) for size in range(2, len(words)))
            for alias in tokens:
                normalized_alias = _normalise_reference_text(alias)
                if len(normalized_alias) >= 4 and f" {normalized_alias} " in padded_message:
                    found[title] = reference
                    break
        return found

    def retrieve(
        self,
        domains: list[str],
        intent: Intent,
        question: str,
        query_vector: np.ndarray | None,
        references: dict[str, tuple[str, int]],
    ) -> tuple[dict[str, dict[int, Candidate]], bool]:
        """Per-domain candidate pools; graph first, local Parquet as fallback."""
        pools: dict[str, dict[int, Candidate]] = {}
        used_graph = False
        for domain in domains:
            same_refs = {title: raw for title, (ref_domain, raw) in references.items() if ref_domain == domain}
            if self._graph_ok is not False:
                try:
                    pool = self._retrieve_graph(domain, intent, query_vector, same_refs)
                    self._graph_ok = True
                    used_graph = True
                    pools[domain] = pool
                    continue
                except Exception as exc:  # noqa: BLE001 - fall back below
                    self._graph_ok = False
                    logger.warning("graph retrieval failed (%s) - using local retrieval", exc)
            pools[domain] = self._retrieve_local(domain, intent, query_vector, same_refs)
        return pools, used_graph

    def _retrieve_graph(
        self,
        domain: str,
        intent: Intent,
        query_vector: np.ndarray | None,
        same_refs: dict[str, int],
    ) -> dict[int, Candidate]:
        ranker = self._ranker(domain)
        retriever = self.graph
        pool: dict[int, Candidate] = {}

        if query_vector is not None:
            for raw, score in retriever.vector_search(query_vector, domain, self.retrieval_k).items():
                initialise_candidates(pool, ranker, raw, {"embed": score})
        themes = normalize_theme_names(intent.themes, known=ranker.item_theme_names())
        if themes:
            for raw, score in retriever.by_theme(themes, domain, self.retrieval_k).items():
                initialise_candidates(pool, ranker, raw, {"themes": score})
        genres = canonical_genres(intent)
        if genres:
            for raw, hits in retriever.by_genre(genres, domain, self.retrieval_k).items():
                initialise_candidates(pool, ranker, raw, {"genres": hits})
        reference_ids = list(same_refs.values())
        if reference_ids:
            for raw, score in retriever.similar_to(reference_ids, domain, self.retrieval_k).items():
                initialise_candidates(pool, ranker, raw, {"similar": score})

        add_genre_evidence(pool, ranker, genres)
        add_same_similar_evidence(pool, ranker, reference_ids, same_refs)
        return pool

    def _retrieve_local(
        self,
        domain: str,
        intent: Intent,
        query_vector: np.ndarray | None,
        same_refs: dict[str, int],
    ) -> dict[int, Candidate]:
        """Parquet-only retrieval: content/theme/genre similarity over the catalogue."""
        ranker = self._ranker(domain)
        data = ranker.data
        reference_ids = list(same_refs.values())
        pool: dict[int, Candidate] = {}

        if query_vector is None and not reference_ids:
            popular = np.argsort(-ranker.popularity)[: self.retrieval_k]
            for pos in popular:
                initialise_candidates(pool, ranker, data.catalog[int(pos)], {"popularity": 1.0})
        else:
            intent_themes = normalize_theme_names(intent.themes, known=ranker.item_theme_names())
            parts = ranker.parts_for(reference_ids, query_vector, intent_themes)
            for pos in np.argsort(-parts["content"])[: self.retrieval_k]:
                initialise_candidates(pool, ranker, data.catalog[int(pos)], {"embed": float(parts["content"][pos])})
            genres = canonical_genres(intent)
            if genres:
                frame = data.genres[data.genres["genre"].isin(genres)]
                counts = frame["item_id"].astype(int).value_counts()
                for raw, count in counts.head(self.retrieval_k * 2).items():
                    initialise_candidates(pool, ranker, int(raw), {"genres": float(count)})
            themes = intent_themes
            if themes or reference_ids:
                profile = ranker.theme_profile_for(reference_ids, themes)
                if profile:
                    for _pos, raw in enumerate(data.catalog):
                        vector = ranker.theme_vectors.get(int(raw))
                        if not vector:
                            continue
                        overlap = sum(weight * profile[t] for t, weight in vector.items() if t in profile)
                        if overlap > 0:
                            initialise_candidates(pool, ranker, int(raw), {"themes": overlap})
            while len(pool) < self.retrieval_k:
                popular = np.argsort(-ranker.popularity)
                filled = False
                for pos in popular:
                    raw = int(data.catalog[pos])
                    if raw not in pool and raw not in reference_ids:
                        initialise_candidates(pool, ranker, raw, {"popularity": 0.0})
                        filled = True
                        if len(pool) >= self.retrieval_k:
                            break
                if not filled:
                    break

        add_genre_evidence(pool, ranker, canonical_genres(intent))
        add_same_similar_evidence(pool, ranker, reference_ids, same_refs)
        return pool

    @property
    def retrieval_k(self) -> int:
        return int(self.cfg.get("llm.retrieval_k", 30))

    @property
    def rerank_k(self) -> int:
        return int(self.cfg.get("llm.rerank_k", 10))

    # --- explanation --------------------------------------------------------------
    @staticmethod
    def _no_answer(language: str) -> str:
        return NO_RESULT_MESSAGE.get(language[:2], NO_RESULT_MESSAGE["en"])

    def _generate(
        self, question: str, evidence_lines: list[str], language: str, titles: list[str]
    ) -> tuple[str, bool]:
        answer, fell_back = generate_answer(
            self.client,
            question,
            evidence_lines,
            language,
            titles,
            temperature=float(self.cfg.get("llm.temperature", 0.2)),
            num_ctx=int(self.cfg.get("llm.num_ctx", 4096)),
            max_tokens=int(self.cfg.get("llm.max_tokens_answer", 700)),
        )
        return answer, not fell_back

    # --- the pipeline -------------------------------------------------------------
    @staticmethod
    def _split_references(
        references: dict[str, tuple[str, int]], domain: str
    ) -> tuple[dict[str, int], list[tuple[str, str, int]]]:
        """Split the joint reference set into same-domain and cross-domain parts."""
        same: dict[str, int] = {}
        cross: list[tuple[str, str, int]] = []
        for title, (ref_domain, raw) in references.items():
            if ref_domain == domain:
                same[title] = raw
            else:
                cross.append((title, ref_domain, raw))
        return same, cross

    def _cross_theme_weights(self, cross_refs: list[tuple[str, str, int]]) -> dict[str, float]:
        """Theme vector transferred from cross-domain reference items.

        The taxonomy is shared across domains, so the liked movie's themes
        (e.g. "story rich") can seed the game ranker's KG profile directly.
        """
        weights: dict[str, float] = {}
        for _title, other_domain, raw in cross_refs:
            try:
                other_ranker = self._ranker(other_domain)
            except Exception as exc:  # noqa: BLE001 - other domain may be absent
                logger.info("cross-domain themes unavailable for %s (%s)", other_domain, exc)
                continue
            for theme, weight in other_ranker.theme_vectors.get(int(raw), {}).items():
                weights[theme] = weights.get(theme, 0.0) + float(weight)
        return weights

    def run(
        self,
        question: str,
        *,
        language: str | None = None,
        top_k: int | None = None,
        session: ChatSession | None = None,
    ) -> ChatResult:
        """Full pipeline for one question; embedding and generation are sequential."""
        progress = Progress()
        progress.next("embed")
        started = time.perf_counter()
        durations: dict[str, float] = {}

        # 1. embed the question once, used for every domain (5 minutes VRAM keep alive)
        t0 = time.perf_counter()
        query_vector = self.question_embedding(question)
        durations["embed_s"] = time.perf_counter() - t0

        progress.next("session", completed_status="ok" if query_vector is not None else "skip")
        # 2. Resolve references from prior displayed evidence before parsing.
        session_references: dict[str, tuple[str, int]] = {}
        feedback_constraints: list[str] = []
        follow_up = False
        explicit_domain = None
        if session is not None:
            rules = rules_intent(question)
            follow_up = _is_short_follow_up(question)
            domain_changed = (
                session.recent_domain
                and rules.domain in {"movie", "game"}
                and rules.domain != session.recent_domain
            )
            explicit_filters_changed = (
                not follow_up
                and bool(rules.genres or rules.themes)
                and bool(set(rules.genres + rules.themes) - set(session.recent_genres + session.recent_themes))
            )
            if domain_changed or explicit_filters_changed:
                session.clear_recent_context()
            session_references = session.resolve_position(question)
            session_references.update(session.titles_in_history(question))
            if _REFERENCE_CUE_RE.search(question):
                session_references.update(self.resolve_message_references(question))
            _liked, _excluded, feedback_constraints = session.feedback_for(
                question, session_references
            )
            session.recent_constraints = list(dict.fromkeys(
                [*session.recent_constraints, *feedback_constraints]
            ))[-8:]
            explicit_domain = rules.domain if rules.domain in {"movie", "game"} else None

        progress.next("intent", completed_status="ok" if session is not None else "skip")
        # 3. intent (the parser gets the new message and a code-built summary only)
        t0 = time.perf_counter()
        intent = parse_intent(
            self.client,
            question,
            context=session.summary_for_intent() if session is not None else None,
            intent_temperature=float(self.cfg.get("llm.intent_temperature", 0.0)),
            num_ctx=int(self.cfg.get("llm.num_ctx", 4096)),
        )
        durations["intent_s"] = time.perf_counter() - t0
        language = (language or intent.language or "vi").lower()[:2]
        if session is not None:
            if session_references and _asks_about_reference(question):
                intent.reference_titles = list(session_references)
            else:
                intent.reference_titles = list(dict.fromkeys(
                    [*intent.reference_titles, *session_references]
                ))
            if follow_up and (
                not explicit_domain or explicit_domain == session.recent_domain
            ):
                if not explicit_domain and session.recent_domain and intent.domain == "both":
                    intent.domain = session.recent_domain
                if not intent.genres:
                    intent.genres = list(session.recent_genres)
                if not intent.themes:
                    intent.themes = list(session.recent_themes)
            if follow_up and session.recent_constraints and not _has_constraint_change(question):
                for constraint in session.recent_constraints:
                    if not constraint.startswith("avoid:"):
                        intent.themes = list(dict.fromkeys([*intent.themes, constraint]))
            if _is_lighter_request(question) and not _is_less_violent_request(question):
                intent.themes = list(dict.fromkeys([*intent.themes, "cosy"]))
            for constraint in feedback_constraints:
                if not constraint.startswith("avoid:"):
                    intent.themes = list(dict.fromkeys([*intent.themes, constraint]))
            if explicit_domain:
                intent.domain = explicit_domain
            if explicit_domain and session_references and _asks_about_reference(question):
                selected_domain = next(iter(session_references.values()))[0]
                intent.domain = selected_domain
        domains = [intent.domain] if intent.domain in {"movie", "game"} else ["movie", "game"]

        progress.next("resolve")
        # 4. retrieval (graph, or parquet when Neo4j is down)
        t0 = time.perf_counter()
        references = self.resolve_references(intent.reference_titles)
        references.update(session_references)
        if session is not None and follow_up:
            references = {**session.liked_references, **references}
            intent.reference_titles = list(dict.fromkeys(
                [*intent.reference_titles, *session.liked_references]
            ))
        progress.next("retrieval", references=sorted(references), intent=intent.model_dump())
        pools, used_graph = self.retrieve(domains, intent, question, query_vector, references)
        if session is not None:
            constraint_exclusions = [
                constraint.split(":", maxsplit=1)[1]
                for constraint in session.recent_constraints
                if constraint.startswith("avoid:")
            ]
            for domain, pool in pools.items():
                ranker = self._ranker(domain)
                pools[domain] = {
                    item_id: candidate
                    for item_id, candidate in pool.items()
                    if not _matches_constraint_exclusion(
                        candidate, ranker, constraint_exclusions
                    )
                }
        durations["retrieval_s"] = time.perf_counter() - t0

        progress.next("rerank", used_graph=used_graph, candidates={d: len(p) for d, p in pools.items()})
        # 5. rerank
        t0 = time.perf_counter()
        items: list[Candidate] = []
        direct_reference = None
        if session is not None and _asks_about_reference(question) and session_references:
            direct_reference = next(iter(session_references.items()))
        if direct_reference is not None:
            title, (domain, raw_id) = direct_reference
            ranker = self._ranker(domain)
            position = ranker.data.item_pos(raw_id)
            if position is not None:
                candidate = Candidate(raw_id, title, position, domain=domain)
                candidate.shared_themes = sorted(ranker.theme_vectors.get(raw_id, {}))
                candidate.shared_genres = sorted(
                    str(row.genre)
                    for row in ranker.data.genres.itertuples(index=False)
                    if int(row.item_id) == raw_id
                )
                candidate.scores["total"] = 0.0
                items = [candidate]
                domains = [domain]
        else:
            title_exclusions = list(intent.exclude)
            if session is not None:
                attribute_exclusions = {
                    _normalise_reference_text(constraint.split(":", maxsplit=1)[1])
                    for constraint in session.recent_constraints
                    if constraint.startswith("avoid:")
                }
                if "violence" in attribute_exclusions:
                    attribute_exclusions.update({"violent", "gore", "blood"})
                title_exclusions = [
                    term
                    for term in title_exclusions
                    if _normalise_reference_text(term) not in attribute_exclusions
                ]
            stateless_excluded_ids = (
                self._excluded_ids(title_exclusions, domains)
                if session is None
                else set()
            )
            for domain in domains:
                pool = pools.get(domain, {})
                ranker = self._ranker(domain)
                same_refs, cross_refs = self._split_references(references, domain)
                if cross_refs:
                    add_cross_similar_evidence(pool, ranker, cross_refs, self._rankers)
                session_likes = session.liked_ids.get(domain, set()) if session else set()
                session_exclusions = (
                    session.excluded_ids.get(domain, set())
                    | (
                        session.recommended_ids.get(domain, set())
                        if session is not None and not _asks_repeat(question)
                        else set()
                    )
                ) if session else set()
                excluded_ids = (
                    self._excluded_ids(title_exclusions, [domain])
                    if session is not None
                    else stateless_excluded_ids
                ) | session_exclusions
                items.extend(
                    ranker.rerank(
                        pool,
                        list(dict.fromkeys([*same_refs.values(), *session_likes])),
                        query_vector,
                        extra_themes=normalize_theme_names(intent.themes, known=ranker.item_theme_names()),
                        extra_theme_weights=self._cross_theme_weights(cross_refs),
                        excluded_ids=excluded_ids,
                        k=top_k or self.rerank_k,
                    )
                )
        items.sort(key=lambda c: -c.scores.get("total", 0.0))
        items = items[: top_k or self.rerank_k]
        durations["rerank_s"] = time.perf_counter() - t0

        progress.next("evidence")
        # 6. grounded explanation
        t0 = time.perf_counter()
        evidence_lines = [candidate.evidence_line(language) for candidate in items]
        progress.next("answer", evidence=evidence_lines, items=[{"title": i.title, "domain": i.domain, "scores": i.scores} for i in items])
        if evidence_lines:
            answer, answer_llm = self._generate(
                question, evidence_lines, language, [item.title for item in items]
            )
        else:
            answer, answer_llm = self._no_answer(language), False
        durations["answer_s"] = time.perf_counter() - t0
        durations["total_s"] = time.perf_counter() - started

        result = ChatResult(
            question=question,
            language=language,
            intent=intent,
            domains_queried=domains,
            items=items,
            answer=answer,
            used_graph=used_graph,
            intent_llm=intent.parsed_by == "llm",
            answer_llm=answer_llm,
            resolved_references=sorted(references),
            durations=durations,
        )
        if session is not None:
            session.language = language
            session.add_turn(
                question,
                intent,
                references,
                items,
                answer,
            )
        progress.next(None)
        return result

    def recommend(
        self,
        question: str,
        *,
        domain: str | None = None,
        k: int | None = None,
    ) -> ChatResult:
        """Retrieval + hybrid rerank WITHOUT LLM phrasing (API ``/recommend``).

        Uses the keyword-rule intent when ``domain`` is given; otherwise the
        same intent parser as :meth:`run`. The deterministic answer is simply
        the ranked evidence list, so nothing depends on Ollama generation.
        """
        started = time.perf_counter()
        durations: dict[str, float] = {}

        query_vector = self.question_embedding(question)
        from entdec.llm.intent import rules_intent

        if domain in {"movie", "game"}:
            intent = rules_intent(question)
            intent.domain = domain
        else:
            intent = parse_intent(
                self.client,
                question,
                intent_temperature=float(self.cfg.get("llm.intent_temperature", 0.0)),
                num_ctx=int(self.cfg.get("llm.num_ctx", 4096)),
            )
        language = intent.language

        domains = [intent.domain] if intent.domain in {"movie", "game"} else ["movie", "game"]
        references = self.resolve_references(intent.reference_titles)
        pools, used_graph = self.retrieve(domains, intent, question, query_vector, references)
        durations["retrieval_s"] = time.perf_counter() - started

        items: list[Candidate] = []
        excluded_ids = self._excluded_ids(intent.exclude, domains)
        for current in domains:
            pool = pools.get(current, {})
            ranker = self._ranker(current)
            same_refs, cross_refs = self._split_references(references, current)
            if cross_refs:
                add_cross_similar_evidence(pool, ranker, cross_refs, self._rankers)
            items.extend(
                ranker.rerank(
                    pool,
                    list(same_refs.values()),
                    query_vector,
                    extra_themes=normalize_theme_names(intent.themes, known=ranker.item_theme_names()),
                    extra_theme_weights=self._cross_theme_weights(cross_refs),
                    excluded_ids=excluded_ids,
                    k=int(k or self.rerank_k),
                )
            )
        items.sort(key=lambda c: -c.scores.get("total", 0.0))
        items = items[: int(k or self.rerank_k)]

        evidence_lines = [candidate.evidence_line(language) for candidate in items]
        if evidence_lines:
            answer = "\n".join(f"{i}. {line}" for i, line in enumerate(evidence_lines, start=1))
        else:
            answer = self._no_answer(language)
        durations["total_s"] = time.perf_counter() - started
        return ChatResult(
            question=question,
            language=language,
            intent=intent,
            domains_queried=domains,
            items=items,
            answer=answer,
            used_graph=used_graph,
            intent_llm=intent.parsed_by == "llm",
            answer_llm=False,
            resolved_references=sorted(references),
            durations=durations,
        )

    def _excluded_ids(self, excluded_titles: list[str], domains: list[str]) -> set[int]:
        """Resolve excluded titles to catalogue ids in either domain."""
        from entdec.recsys.cross_domain import resolve_items

        excluded: set[int] = set()
        for domain in domains:
            ranker = self._ranker(domain)
            excluded.update(resolve_items(ranker.data.items, excluded_titles, ranker.data.item_key))
        return excluded


__all__ = [
    "Candidate",
    "ChatPipeline",
    "ChatResult",
    "ChatRanker",
    "GraphRetriever",
    "fuzzy_resolve",
    "initialise_candidates",
    "known_positions_for",
    "normalise_title",
    "parse_intent",
    "normalise_chat_weights",
]
