"""Thin, resilient HTTP client for a local Ollama server.

Only the endpoints the project needs are wrapped: ``/api/embed``,
``/api/generate``, ``/api/chat``, ``/api/tags`` and ``/api/ps``.

Measured on the target machine (RTX 3050 Ti 4 GB): ``qwen3-embedding:4b``
natively returns 2560 dimensions and honours the ``dimensions`` parameter
(Matryoshka truncation). Its context window is the dominant memory cost, so
embeddings default to ``num_ctx=512``: the model then keeps more layers on the
GPU and throughput improves by ~50% (9.6 vs 6.3 docs/s) while still fitting the
~400 token chunk limit.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np

from entdec.logging_setup import get_logger

logger = get_logger(__name__)

DEFAULT_EMBED_NUM_CTX = 512
DEFAULT_KEEP_ALIVE = "10m"


class OllamaError(RuntimeError):
    """Raised when Ollama cannot be reached or returns an unusable response."""


@dataclass
class EmbedResult:
    """Result of one embedding call."""

    vectors: np.ndarray
    prompt_tokens: int = 0
    duration_s: float = 0.0
    used_dimensions_param: bool = False

    @property
    def n_docs(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1]) if self.vectors.ndim == 2 else 0


@dataclass
class GenerateResult:
    """Result of one generation call."""

    text: str
    prompt_tokens: int = 0
    eval_tokens: int = 0
    duration_s: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation; all-zero rows stay zero."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


class OllamaClient:
    """HTTP client with retries for the local Ollama server."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        embed_model: str = "qwen3-embedding:4b",
        gen_model: str = "qwen3:4b",
        *,
        timeout_s: float = 300.0,
        max_retries: int = 3,
        embed_num_ctx: int = DEFAULT_EMBED_NUM_CTX,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.embed_model = embed_model
        self.gen_model = gen_model
        self.timeout_s = timeout_s
        self.max_retries = max(1, max_retries)
        self.embed_num_ctx = embed_num_ctx
        self.keep_alive = keep_alive
        self.transport = transport  # injectable for unit tests (httpx.MockTransport)

    # --- low level ------------------------------------------------------------
    def _client(self, timeout_s: float | None = None) -> httpx.Client:
        return httpx.Client(
            base_url=self.host, timeout=timeout_s or self.timeout_s, transport=self.transport
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Send a request, retrying transport errors and 5xx responses."""
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with self._client() as client:
                    response = client.request(method, path, **kwargs)
                if response.status_code >= 500:
                    raise OllamaError(
                        f"{path} returned {response.status_code}: {response.text[:200]}"
                    )
                return response
            except (httpx.HTTPError, OllamaError) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                backoff = min(2.0**attempt, 10.0)
                logger.warning(
                    "ollama %s attempt %d/%d failed (%s) - retrying in %.0fs",
                    path,
                    attempt,
                    self.max_retries,
                    exc,
                    backoff,
                )
                time.sleep(backoff)
        raise OllamaError(f"ollama request to {path} failed: {last_error}")

    # --- health / introspection ----------------------------------------------
    def list_models(self) -> list[str]:
        """Model names available on the server (empty when unreachable)."""
        try:
            response = self._request("GET", "/api/tags")
            payload = response.json()
        except (OllamaError, ValueError):
            return []
        return sorted(
            str(model.get("name")) for model in payload.get("models", []) if model.get("name")
        )

    def loaded_models(self) -> list[dict[str, Any]]:
        """Models currently resident, with their size and VRAM split."""
        try:
            response = self._request("GET", "/api/ps")
            return list(response.json().get("models", []))
        except (OllamaError, ValueError):
            return []

    def health(self) -> bool:
        """True when the server answers and both required models are present."""
        models = self.list_models()
        if not models:
            return False
        required = {self.embed_model, self.gen_model}
        return required.issubset(set(models))

    def unload(self, model: str | None = None) -> None:
        """Ask Ollama to drop a model from memory (``keep_alive: 0``)."""
        payload = {"model": model or self.gen_model, "keep_alive": 0}
        try:
            self._request("POST", "/api/generate", json=payload)
        except OllamaError as exc:  # pragma: no cover - best effort
            logger.debug("could not unload %s: %s", payload["model"], exc)

    # --- embeddings -----------------------------------------------------------
    def embed(
        self,
        texts: Sequence[str],
        *,
        model: str | None = None,
        dimensions: int | None = None,
        keep_alive: str | None = None,
        normalise: bool = True,
    ) -> EmbedResult:
        """Embed a batch of documents.

        Args:
            texts: Documents to embed (order is preserved).
            model: Override the configured embedding model.
            dimensions: Requested output dimension; falls back to client side
                truncation when the server ignores or rejects it.
            keep_alive: Ollama keep-alive string.
            normalise: L2-normalise the returned vectors.

        Returns:
            An :class:`EmbedResult` whose ``vectors`` has shape ``(n, dim)``.
        """
        if not texts:
            return EmbedResult(vectors=np.zeros((0, dimensions or 0), dtype=np.float32))

        payload: dict[str, Any] = {
            "model": model or self.embed_model,
            "input": list(texts),
            "truncate": True,
            "keep_alive": keep_alive or self.keep_alive,
            "options": {"num_ctx": self.embed_num_ctx},
        }
        if dimensions:
            payload["dimensions"] = int(dimensions)

        started = time.perf_counter()
        response = self._request("POST", "/api/embed", json=payload)
        used_param = bool(dimensions)

        if response.status_code >= 400:
            # Some models/versions reject `dimensions`: retry without it.
            if dimensions and "dimensions" in response.text.lower():
                logger.info("server rejected `dimensions`; truncating client side instead")
                payload.pop("dimensions", None)
                response = self._request("POST", "/api/embed", json=payload)
                used_param = False
            if response.status_code >= 400:
                raise OllamaError(f"embed failed: {response.status_code} {response.text[:300]}")

        try:
            body = response.json()
            vectors = np.asarray(body["embeddings"], dtype=np.float32)
        except (KeyError, ValueError, TypeError) as exc:
            raise OllamaError(f"unexpected embed response: {exc}") from exc

        if dimensions and vectors.shape[1] != dimensions:
            vectors = vectors[:, :dimensions]

        if normalise:
            vectors = l2_normalize(vectors)

        return EmbedResult(
            vectors=vectors,
            prompt_tokens=int(body.get("prompt_eval_count") or 0),
            duration_s=time.perf_counter() - started,
            used_dimensions_param=used_param,
        )

    def embed_batch(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 24,
        dimensions: int | None = None,
        progress_every: int = 1,
        on_progress: Any = None,
    ) -> np.ndarray:
        """Embed many documents, batching requests to keep VRAM stable.

        Args:
            texts: All documents.
            batch_size: Documents per HTTP call.
            dimensions: Requested output dimension.
            progress_every: Log every N batches.
            on_progress: Optional callback ``(done, total)`` after each batch.

        Returns:
            ``(len(texts), dim)`` array, rows in input order.
        """
        total = len(texts)
        if total == 0:
            return np.zeros((0, dimensions or 0), dtype=np.float32)

        chunks: list[np.ndarray] = []
        done = 0
        started = time.perf_counter()
        for start in range(0, total, batch_size):
            batch = list(texts[start : start + batch_size])
            result = self.embed(batch, dimensions=dimensions)
            chunks.append(result.vectors)
            done += len(batch)
            if on_progress is not None:
                on_progress(done, total)
            if progress_every and (start // batch_size) % progress_every == 0:
                elapsed = time.perf_counter() - started
                rate = done / elapsed if elapsed else 0.0
                eta = (total - done) / rate if rate else 0.0
                logger.info("embedded %d/%d docs (%.1f docs/s, eta %.0fs)", done, total, rate, eta)
        return np.vstack(chunks)

    def native_dimension(self, model: str | None = None) -> int:
        """Detect the embedding dimension of a model at runtime."""
        result = self.embed(["dimension probe"], model=model, dimensions=None)
        return result.dim

    # --- generation -----------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        num_ctx: int | None = None,
        think: bool | None = None,
        fmt: str | dict[str, Any] | None = None,
        model: str | None = None,
        options: dict[str, Any] | None = None,
        retries: int | None = None,
        prefill: str | None = None,
    ) -> GenerateResult:
        """Generate a completion with the local LLM.

        Args:
            prompt: User prompt.
            system: Optional system message.
            temperature: Sampling temperature (spec default 0.2).
            num_ctx: Context window (spec default 4096).
            think: Qwen3 thinking mode; the project runs with it disabled.
            fmt: ``"json"`` (or a JSON schema) to constrain the output.
            model: Override the configured generation model.
            options: Extra Ollama options merged last.
            retries: Override the client retry budget.
            prefill: Start of the assistant reply (``"1. "``). qwen3 answers
                immediately instead of planning first, so this is sent through
                ``/api/chat`` as a trailing assistant message.

        Returns:
            A :class:`GenerateResult` with the answer text.
        """
        merged_options: dict[str, Any] = {"num_ctx": int(num_ctx or 4096)}
        if temperature is not None:
            merged_options["temperature"] = float(temperature)
        if options:
            merged_options.update(options)

        payload: dict[str, Any] = {
            "model": model or self.gen_model,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": merged_options,
        }
        if prefill is None:
            payload["prompt"] = prompt
            if system:
                payload["system"] = system
            path = "/api/generate"
        else:
            messages: list[dict[str, str]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": prefill})
            payload["messages"] = messages
            path = "/api/chat"
        if think is not None:
            payload["think"] = bool(think)
        if fmt is not None:
            payload["format"] = fmt

        original = self.max_retries
        if retries is not None:
            self.max_retries = max(1, retries)
        try:
            started = time.perf_counter()
            response = self._request("POST", path, json=payload)
            elapsed = time.perf_counter() - started
        finally:
            self.max_retries = original

        if response.status_code >= 400:
            raise OllamaError(f"generate failed: {response.status_code} {response.text[:300]}")

        body = response.json()
        if prefill is not None:
            generated = str((body.get("message") or {}).get("content") or "")
            text = f"{prefill}{generated}"
        else:
            text = str(body.get("response") or "")
        return GenerateResult(
            text=text,
            prompt_tokens=int(body.get("prompt_eval_count") or 0),
            eval_tokens=int(body.get("eval_count") or 0),
            duration_s=elapsed,
            raw=body,
        )

    # --- convenience ----------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: Any) -> OllamaClient:
        """Build a client from the application :class:`~entdec.config.Config`."""
        return cls(
            host=cfg.env.ollama_host,
            embed_model=cfg.env.ollama_embed_model,
            gen_model=cfg.env.ollama_gen_model,
            timeout_s=float(cfg.get("llm.request_timeout_s", 300)),
            max_retries=int(cfg.get("llm.max_retries", 3)),
            embed_num_ctx=int(cfg.get("nlp.embed_num_ctx", DEFAULT_EMBED_NUM_CTX)),
        )


__all__ = [
    "DEFAULT_EMBED_NUM_CTX",
    "DEFAULT_KEEP_ALIVE",
    "EmbedResult",
    "GenerateResult",
    "OllamaClient",
    "OllamaError",
    "l2_normalize",
]
