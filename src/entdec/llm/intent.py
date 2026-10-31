"""Section 8 step 1: parse the user's request into a validated intent.

qwen3:4b (think=false, temperature 0) returns JSON that is validated by the
pydantic schema below. On an unusable reply the prompt is retried once; if
that also fails (or Ollama is unreachable) a keyword-rule fallback produces a
coarse intent so the pipeline can still answer with vector + graph search.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from entdec.logging_setup import get_logger
from entdec.nlp.ollama_client import OllamaClient, OllamaError
from entdec.taxonomy.genres import CANONICAL_GENRES

logger = get_logger(__name__)

AllowedDomain = Literal["movie", "game", "both"]

CANONICAL_LOWER = {genre.lower(): genre for genre in CANONICAL_GENRES}

# Vietnamese + English keyword rules, enough for the 5 sample queries.
MOVIE_WORDS = ("phim", "movie", "film", "xem", "watch", "cinema")
GAME_WORDS = ("game", "gaming", "chơi", "trò chơi", "play")

GENRE_RULES: list[tuple[str, str]] = [
    ("kinh dị", "horror"),
    ("horror", "horror"),
    ("scary", "horror"),
    ("hài", "comedy"),
    ("hài hước", "comedy"),
    ("comedy", "comedy"),
    ("funny", "comedy"),
    ("nhẹ nhàng", "romance"),
    ("lãng mạn", "romance"),
    ("romance", "romance"),
    ("tâm lý", "drama"),
    ("drama", "drama"),
    ("hành động", "action"),
    ("action", "action"),
    ("khoa học viễn tưởng", "sci-fi"),
    ("sci-fi", "sci-fi"),
    ("science fiction", "sci-fi"),
    ("phiêu lưu", "adventure"),
    ("adventure", "adventure"),
    ("open world", "adventure"),
    ("thám hiểm", "adventure"),
    ("huyền huyễn", "fantasy"),
    ("fantasy", "fantasy"),
    ("magic", "fantasy"),
    ("mystery", "mystery"),
    ("thriller", "thriller"),
    ("war", "war"),
    ("rpg", "rpg"),
    ("strategy", "strategy"),
    ("simulation", "simulation"),
    ("puzzle", "puzzle"),
    ("fps", "action"),
    ("shooter", "action"),
    ("platformer", "adventure"),
    ("survival", "simulation"),
]

THEME_RULES: list[tuple[str, str]] = [
    ("cốt truyện", "story rich"),
    ("story", "story rich"),
    ("nội dung sâu", "story rich"),
    ("nhẹ nhàng", "cosy"),
    ("cosy", "cosy"),
    ("cozy", "cosy"),
    ("dễ thương", "cosy"),
    ("tâm lý", "psychological"),
    ("psychological", "psychological"),
    ("khám phá", "exploration"),
    ("explore", "exploration"),
    ("thế giới mở", "exploration"),
    ("kinh dị tâm lý", "psychological"),
]

VIET_RE = re.compile(
    "[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]"
)


class Intent(BaseModel):
    """Validated output of intent parsing (section 8, step 1)."""

    domain: AllowedDomain = "both"
    reference_titles: list[str] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    language: str = "vi"
    parsed_by: Literal["llm", "rules"] = "rules"

    @field_validator("domain", mode="before")
    @classmethod
    def _normalise_domain(cls, value: Any) -> Any:
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"movie", "film", "movies", "films"}:
                return "movie"
            if lowered in {"game", "games", "video game"}:
                return "game"
            return "both"
        return value

    @field_validator("language", mode="before")
    @classmethod
    def _normalise_language(cls, value: Any) -> Any:
        if isinstance(value, str):
            code = value.strip().lower()[:2]
            return code or "vi"
        return "vi"

    @field_validator(
        "reference_titles", "genres", "themes", "keywords", "exclude", mode="before"
    )
    @classmethod
    def _ensure_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []


# --------------------------------------------------------------------------- #
# LLM parsing
# --------------------------------------------------------------------------- #
def _extract_json(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction from an LLM answer."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```\s*$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in reply")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("intent JSON is not an object")
    return payload


def parse_intent(
    client: OllamaClient,
    question: str,
    *,
    intent_temperature: float = 0.0,
    num_ctx: int = 4096,
    system_prompt: str | None = None,
) -> Intent:
    """Parse ``question`` with qwen3:4b; one retry, then the rule fallback.

    Never raises: the worst case is a coarse :class:`Intent` from
    :func:`rules_intent` (``parsed_by="rules"``), so the pipeline still answers
    with plain vector + graph search.
    """
    from entdec.llm.prompts import INTENT_RETRY_HINT, INTENT_SYSTEM

    system = system_prompt or INTENT_SYSTEM
    previous = ""
    for attempt in (1, 2):
        prompt = question if attempt == 1 else (
            INTENT_RETRY_HINT.format(previous=previous[:400]) + "\n\nUser request: " + question
        )
        try:
            result = client.generate(
                prompt,
                system=system,
                temperature=intent_temperature,
                num_ctx=num_ctx,
                think=False,
                fmt="json",
                retries=1,
            )
        except OllamaError as exc:
            logger.warning("intent LLM unreachable (%s) - falling back to keyword rules", exc)
            break
        text = result.text.strip()
        try:
            payload = _extract_json(text)
            intent = Intent.model_validate(payload)
            intent.parsed_by = "llm"
            return intent
        except (ValueError, json.JSONDecodeError) as exc:
            logger.info("intent parse attempt %d failed: %s", attempt, exc)
            previous = text
    return rules_intent(question)


# --------------------------------------------------------------------------- #
# rule fallback
# --------------------------------------------------------------------------- #
def normalize_theme_names(names: list[str], known: list[str] | None = None) -> list[str]:
    """Lowercase + optional fuzzy-map theme names onto the catalogue's themes."""
    out: list[str] = []
    for name in names:
        lowered = name.strip().lower()
        if lowered and lowered not in out:
            out.append(lowered)
    if known:
        from rapidfuzz import process

        lowered_known = [str(k).strip().lower() for k in known]
        mapped: list[str] = []
        for name in out:
            matches = process.extract(name, lowered_known, limit=1, score_cutoff=50.0)
            mapped.append(matches[0][0] if matches else name)
        return mapped
    return out

def detect_language(question: str) -> str:
    """Cheap heuristic: Vietnamese diacritics or common VN words -> vi, else en."""
    if VIET_RE.search(question) or any(
        word in question.lower() for word in ("tôi", "gợi ý", "chơi", "muốn", "giống")
    ):
        return "vi"
    return "en"


def rules_intent(question: str) -> Intent:
    """Keyword-rule fallback intent (no LLM involved)."""
    lowered = question.lower()
    movie_hits = sum(term in lowered for term in MOVIE_WORDS)
    game_hits = sum(term in lowered for term in GAME_WORDS)
    if movie_hits > game_hits:
        domain: Literal["movie", "game", "both"] = "movie"
    elif game_hits > movie_hits:
        domain = "game"
    else:
        domain = "both"

    genres: list[str] = []
    themes: list[str] = []
    for keyword, canonical in GENRE_RULES:
        if keyword in lowered:
            canonical_capitalised = CANONICAL_LOWER.get(canonical, canonical)
            if canonical_capitalised not in genres and canonical_capitalised in CANONICAL_GENRES:
                genres.append(canonical_capitalised)
    for keyword, theme in THEME_RULES:
        if keyword in lowered and theme not in themes:
            themes.append(theme)

    return Intent(
        domain=domain,
        reference_titles=[],
        genres=genres,
        themes=themes,
        keywords=[question[:200]] if question.strip() else [],
        exclude=[],
        language=detect_language(question),
        parsed_by="rules",
    )


# --------------------------------------------------------------------------- #
# helpers shared with the pipeline
# --------------------------------------------------------------------------- #
def canonical_genres(intent: Intent) -> list[str]:
    """Genre names from the intent mapped onto the canonical genre names.

    Accepts canonical names directly ("Comedy") as well as Vietnamese/
    English aliases from :data:`GENRE_RULES` ("kinh dị" -> "Horror").
    """
    out: list[str] = []
    for raw in intent.genres:
        stripped = raw.strip()
        canonical = CANONICAL_LOWER.get(stripped.lower())
        if canonical is None:
            # tolerate "Comedy " / "COMEDY"
            canonical = CANONICAL_LOWER.get(stripped.lower().replace("_", " "))
        if canonical is None:
            for keyword, alias in GENRE_RULES:
                if keyword == stripped.lower():
                    canonical = CANONICAL_LOWER.get(alias, alias)
                    break
        if canonical is not None and canonical not in out and canonical in CANONICAL_GENRES:
            out.append(canonical)
    return out


def detected_genres_from_question(question: str) -> list[str]:
    """Canonical genres suggested by keyword rules for a raw question."""
    lowered = question.lower()
    out: list[str] = []
    for keyword, canonical in GENRE_RULES:
        if keyword in lowered:
            canonical_capitalised = CANONICAL_LOWER.get(canonical, canonical)
            if canonical_capitalised not in out and canonical_capitalised in CANONICAL_GENRES:
                out.append(canonical_capitalised)
    return out


__all__ = [
    "Intent",
    "canonical_genres",
    "detect_language",
    "detected_genres_from_question",
    "parse_intent",
    "rules_intent",
]
