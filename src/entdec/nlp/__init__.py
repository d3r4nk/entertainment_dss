"""NLP layer: item documents, chunking, the Ollama HTTP client and embeddings."""

from __future__ import annotations

from entdec.nlp.ollama_client import OllamaClient, OllamaError

__all__ = ["OllamaClient", "OllamaError"]
