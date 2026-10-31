"""Sentence-aware chunking (PROJECT_SPEC.md section 6).

A document is measured in approximate tokens (``words * 1.3``). Documents longer
than ``chunk_max_tokens`` (400 by default) are split on sentence boundaries into
chunks of roughly ``chunk_target_tokens`` (300) with ``chunk_overlap_tokens``
(50) of overlap, so a fact that straddles a boundary is still embedded intact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from entdec.logging_setup import get_logger

logger = get_logger(__name__)

WORDS_TO_TOKENS = 1.3
CHUNK_MAX_TOKENS = 400
CHUNK_TARGET_TOKENS = 300
CHUNK_OVERLAP_TOKENS = 50

# Split after . ! ? (also inside quotes/brackets) or on hard line breaks.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])[\"'’”)\]]*\s+|\n+")
_WHITESPACE = re.compile(r"\s+")


def estimate_tokens(text: str, words_to_tokens: float = WORDS_TO_TOKENS) -> int:
    """Approximate the token count of ``text``."""
    if not text:
        return 0
    return int(round(len(text.split()) * words_to_tokens))


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, dropping empties and keeping order."""
    if not text:
        return []
    parts = _SENTENCE_SPLIT.split(text.strip())
    return [part.strip() for part in parts if part and part.strip()]


def _split_long_sentence(sentence: str, target_tokens: int, words_to_tokens: float) -> list[str]:
    """Split a single over-long sentence on word boundaries."""
    words = sentence.split()
    if not words:
        return []
    words_per_chunk = max(10, int(target_tokens / max(words_to_tokens, 0.1)))
    return [
        " ".join(words[start : start + words_per_chunk])
        for start in range(0, len(words), words_per_chunk)
    ]


@dataclass
class Chunk:
    """One chunk of a document."""

    index: int
    text: str
    tokens: int


def chunk_text(
    text: str,
    *,
    max_tokens: int = CHUNK_MAX_TOKENS,
    target_tokens: int = CHUNK_TARGET_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    words_to_tokens: float = WORDS_TO_TOKENS,
) -> list[Chunk]:
    """Split ``text`` into overlapping chunks.

    Args:
        text: The document.
        max_tokens: Documents at or below this size stay in one chunk.
        target_tokens: Soft size limit for a chunk.
        overlap_tokens: Tokens repeated at the start of the next chunk.
        words_to_tokens: Words-to-tokens conversion factor.

    Returns:
        Ordered chunks; always at least one (possibly empty) chunk.
    """
    cleaned = _WHITESPACE.sub(" ", (text or "").strip())
    total = estimate_tokens(cleaned, words_to_tokens)
    if total <= max_tokens:
        return [Chunk(index=0, text=cleaned, tokens=total)]

    pieces: list[str] = []
    for sentence in split_sentences(cleaned):
        if estimate_tokens(sentence, words_to_tokens) > max_tokens:
            pieces.extend(_split_long_sentence(sentence, target_tokens, words_to_tokens))
        else:
            pieces.append(sentence)
    if not pieces:
        return [Chunk(index=0, text=cleaned, tokens=total)]

    chunks: list[Chunk] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> None:
        if not current:
            return
        joined = " ".join(current).strip()
        chunks.append(
            Chunk(index=len(chunks), text=joined, tokens=estimate_tokens(joined, words_to_tokens))
        )

    for piece in pieces:
        piece_tokens = estimate_tokens(piece, words_to_tokens)
        if current and current_tokens + piece_tokens > target_tokens:
            flush()
            # Restart with the overlapping tail of the previous chunk.
            tail: list[str] = []
            tail_tokens = 0
            for previous in reversed(current):
                previous_tokens = estimate_tokens(previous, words_to_tokens)
                if tail_tokens + previous_tokens > overlap_tokens:
                    break
                tail.insert(0, previous)
                tail_tokens += previous_tokens
            current = tail
            current_tokens = tail_tokens
        current.append(piece)
        current_tokens += piece_tokens

    flush()
    return chunks or [Chunk(index=0, text=cleaned, tokens=total)]


def chunk_word_counts(chunks: list[Chunk]) -> list[int]:
    """Word count per chunk (handy for tests and reports)."""
    return [len(chunk.text.split()) for chunk in chunks]


__all__ = [
    "CHUNK_MAX_TOKENS",
    "CHUNK_OVERLAP_TOKENS",
    "CHUNK_TARGET_TOKENS",
    "WORDS_TO_TOKENS",
    "Chunk",
    "chunk_text",
    "chunk_word_counts",
    "estimate_tokens",
    "split_sentences",
]
