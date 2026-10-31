"""Prompt templates for the qwen3:4b chatbot (PROJECT_SPEC.md section 8).

The LLM only parses the user's intent and phrases a grounded answer; the
retrieved candidates, the hybrid reranking and the evidence strings are all
produced by code from the knowledge graph.
"""

from __future__ import annotations

INTENT_SYSTEM = """You translate a user's entertainment request into JSON.

Allowed keys (all optional, use [] when not stated):
  domain: "movie" | "game" | "both"
  reference_titles: ["title"]    # works the user mentions, can be games OR films
  genres: ["genre keywords"]     # e.g. horror, comedy, open world, fps, puzzle
  themes: ["story/feeling descriptors"]  # e.g. psychological, cosy, story rich
  exclude: ["what to avoid"]
  keywords: ["extra free-text words worth embedding"]
  language: "vi" | "en" | ...    # the user's language code

Rules:
- Reply with JSON ONLY. No markdown, no commentary.
- Keep the user's own words in the lists (do not translate them).
- If unsure about the domain, use "both".

Examples:
User: "Tôi thích Inception, gợi ý vài game có cốt truyện hay"
-> {"domain": "game", "reference_titles": ["Inception"], "genres": [], "themes": ["story rich", "puzzle"], "keywords": ["cốt truyện"], "exclude": [], "language": "vi"}
User: "Recommend a light comedy for the weekend"
-> {"domain": "movie", "reference_titles": [], "genres": ["comedy"], "themes": ["light", "cosy"], "keywords": [], "exclude": [], "language": "en"}"""


INTENT_RETRY_HINT = """Your previous reply was not valid JSON.
Previous reply (truncated):
{previous}

Reply again with JSON ONLY, exactly matching the schema."""


LANGUAGE_NAMES = {
    "vi": "Vietnamese (tiếng Việt)",
    "en": "English",
}

EXPLAIN_SYSTEM = """You are a movie and game recommendation assistant. You answer in {language_name}.

You see ONLY the candidate items in the user's message, each with evidence
generated from a knowledge graph (shared themes, genres, similarity to a liked
work). Never invent, infer or mention an item that is not listed.

Reply with exactly one line per candidate, in this format:
<CANDIDATE TITLE> | <one short sentence in {language_name}, at most 20 words>

The sentence must use only that candidate's evidence and read as plain prose for
the user. Do not add plot details, awards or facts that are not in the evidence.
Never write about candidates, evidence, prompts, scores, the language or your
own process, and never translate the title. No numbering, no preamble, no
closing remark, no quotes."""

EXPLAIN_PROMPT_BODY = """User request: {question}

Candidates (title | evidence from the knowledge graph):
{evidence}

Write the {n_max} lines "<TITLE> | <sentence>" now, in {language_name}."""

EXPLAIN_RETRY_HINT = """Your previous reply was rejected. Reply again with ONLY lines in
the form "<CANDIDATE TITLE> | <one short sentence in {language_name}>",
using the candidate titles exactly as given, with no other text."""


def language_name(language: str) -> str:
    """Human-readable language name used inside the prompts."""
    return LANGUAGE_NAMES.get(language[:2].lower(), "English")


def explain_system(language: str) -> str:
    """System prompt for the grounded explanation."""
    return EXPLAIN_SYSTEM.format(language_name=language_name(language))


def explain_body(question: str, evidence: str, language: str, n_max: int = 5) -> str:
    """User message carrying the grounded facts."""
    return EXPLAIN_PROMPT_BODY.format(
        question=question.strip(),
        evidence=evidence,
        n_max=n_max,
        language_name=language_name(language),
    )


def explain_retry(language: str) -> str:
    """Correction appended when the first reply was rejected."""
    return EXPLAIN_RETRY_HINT.format(language_name=language_name(language))


__all__ = [
    "EXPLAIN_PROMPT_BODY",
    "EXPLAIN_SYSTEM",
    "INTENT_RETRY_HINT",
    "INTENT_SYSTEM",
    "LANGUAGE_NAMES",
    "explain_body",
    "explain_retry",
    "explain_system",
    "language_name",
]
