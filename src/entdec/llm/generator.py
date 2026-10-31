"""Section 8 step 4: the LLM only phrases the answer.

The candidate items and their evidence are produced by the code-backed
retrieval/rerank pipeline; the prompt omits everything that was not retrieved.
qwen3:4b likes to append its planning (usually in English) around the answer, so
the reply is parsed instead of trusted: the model must return
``<candidate title> | <one sentence in the user's language>`` lines, the code
maps every line back onto a retrieved candidate and drops anything else. A
reply that does not survive parsing is retried once and then replaced by a
deterministic local phrasing of the same evidence.
"""

from __future__ import annotations

import math
import re

from entdec.llm.prompts import explain_body, explain_retry, explain_system
from entdec.logging_setup import get_logger
from entdec.nlp.ollama_client import OllamaClient, OllamaError

logger = get_logger(__name__)

# qwen3 sometimes leaks reasoning inside a think block even with think=False
_OPEN = "<" + "think" + ">"
_CLOSE = "<" + "/" + "think" + ">"
THINK_RE = re.compile(re.escape(_OPEN) + r".*?" + re.escape(_CLOSE), re.IGNORECASE | re.DOTALL)

# bullet/numbered prefixes the model adds even when asked not to
PREFIX_RE = re.compile(r"^\s*(?:(?:\d+[.)]|[-*+•])\s*)+")
# list-shaped answer line (used by callers/tests to check the final answer)
LIST_LINE_RE = re.compile(r"^\s*(?:\d+[.)]\s*|[-*+•]\s+)\S")

# Vietnamese has letters outside the ASCII alphabet; English replies do not.
_VI_CHARS = set(
    "ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ"
)

# words that only appear when the model talks about its own task instead of
# answering (the evidence lines already carry the numbers it would quote)
META_WORDS = (
    "user",
    "candidate",
    "evidence",
    "vietnamese",
    "english",
    "prompt",
    "instruction",
    "format",
    "so in",
    "score",
    "match",
    "point ",
    "reply",
)

MAX_SENTENCE_WORDS = 35

FALLBACK_TEMPLATE = {
    "vi": (
        "Gợi ý dựa trên câu hỏi của bạn:\n{bullets}\n"
        "(Trình tạo câu trả lời không khả dụng nên đây là bản rút gọn.)"
    ),
    "en": (
        "Suggestions for your request:\n{bullets}\n"
        "(The answer generator is unavailable; this is the condensed version.)"
    ),
}


def is_vietnamese(text: str) -> bool:
    """True when the text contains at least one Vietnamese-specific letter."""
    return any(char.lower() in _VI_CHARS for char in text)


def _normalise_title(title: str) -> str:
    cleaned = PREFIX_RE.sub("", title).strip().strip("\"'").lower()
    for symbol in ("®", "™", "©"):
        cleaned = cleaned.replace(symbol, "")
    return " ".join(cleaned.split())


def _usable_sentence(sentence: str, language: str) -> bool:
    """Reject sentences that talk about the task instead of the item."""
    text = sentence.strip().strip("\"'").strip()
    if not text or len(text.split()) > MAX_SENTENCE_WORDS or "|" in text:
        return False
    lowered = text.lower()
    if any(word in lowered for word in META_WORDS):
        return False
    return is_vietnamese(text) if language[:2] == "vi" else True


def cut_reasoning_tail(text: str) -> str:
    """Drop everything from a stray think tag on (qwen3 sometimes appends one)."""
    index = text.find(_OPEN)
    if index != -1:
        return text[:index].rstrip()
    return text


def parse_answer(text: str, titles: list[str], language: str) -> list[tuple[str, str]]:
    """Map ``<title> | <sentence>`` lines back onto the retrieved candidates.

    Returns ``(title, sentence)`` pairs in candidate order; every line that does
    not belong to a candidate (reasoning, greetings, invented items) is dropped.
    """
    text = cut_reasoning_tail(THINK_RE.sub("", text))
    wanted = {_normalise_title(title): title for title in titles}
    found: dict[str, str] = {}
    dropped = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        head, sep, sentence = line.partition("|")
        if not sep:
            dropped += 1
            continue
        title = wanted.get(_normalise_title(head))
        if title is None or title in found:
            dropped += 1
            continue
        if not _usable_sentence(sentence, language):
            dropped += 1
            continue
        found[title] = " ".join(sentence.split()).strip().strip("\"'")
    if dropped:
        logger.info("dropped %d generated line(s) that did not match a candidate", dropped)
    return [(title, found[title]) for title in titles if title in found]


def _enough(parsed: list[tuple[str, str]], titles: list[str]) -> bool:
    return len(parsed) >= max(1, math.ceil(len(titles) / 2))


def generate_answer(
    client: OllamaClient,
    question: str,
    evidence_lines: list[str],
    language: str,
    titles: list[str] | None = None,
    *,
    temperature: float = 0.2,
    num_ctx: int = 4096,
    max_tokens: int = 600,
    attempts: int = 2,
) -> tuple[str, bool]:
    """Phrase a short answer from the retrieved evidence lines only.

    ``titles`` defaults to the candidate titles implied by ``evidence_lines``.
    Returns ``(answer, used_fallback)``.
    """
    if titles is None:
        titles = [line.split(":", 1)[0].strip() for line in evidence_lines]
    evidence = "\n".join(f"{title} | {line}" for title, line in zip(titles, evidence_lines, strict=True))
    if len(evidence) > 9000:
        evidence = evidence[:9000]
        logger.warning("evidence was truncated before generation")
    body = explain_body(question, evidence, language, n_max=len(titles))
    # qwen3 honours this soft switch even when the server ignores think=False
    body = f"{body}\n/no_think"
    system = explain_system(language)
    for attempt in range(1, max(1, attempts) + 1):
        try:
            result = client.generate(
                body,
                system=system,
                temperature=temperature,
                num_ctx=num_ctx,
                think=False,
                options={"num_predict": int(max_tokens)},
                # answering straight away beats qwen3's long English planning
                prefill="1. ",
            )
        except OllamaError as exc:
            logger.warning("generation failed (%s) - deterministic fallback", exc)
            break
        parsed = parse_answer(result.text, titles, language)
        if _enough(parsed, titles):
            answer = "\n".join(f"{rank}. {title}: {sentence}" for rank, (title, sentence) in enumerate(parsed, start=1))
            return answer, False
        logger.warning(
            "rejected answer on attempt %d (%d of %d candidates used, language=%s)",
            attempt,
            len(parsed),
            len(titles),
            language,
        )
        body = f"{body}\n\n{explain_retry(language)}"
    logger.warning("no acceptable generated answer - deterministic fallback")
    return fallback_answer(evidence_lines, language), True


def fallback_answer(evidence_lines: list[str], language: str) -> str:
    """Deterministic, template-based answer (same facts, no LLM)."""
    bullets = "\n".join(f"  + {line}" for line in evidence_lines)
    template = FALLBACK_TEMPLATE.get(language[:2].lower(), FALLBACK_TEMPLATE["en"])
    return template.format(bullets=bullets)


__all__ = [
    "LIST_LINE_RE",
    "META_WORDS",
    "fallback_answer",
    "generate_answer",
    "is_vietnamese",
    "parse_answer",
]
