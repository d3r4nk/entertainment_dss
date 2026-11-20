# Progress

## 2026-11-20 - Conversational chatbot memory

- Added bounded, code-built session memory with reference resolution, feedback,
  carried preferences, and repeat filtering across CLI, API, and Streamlit.
- Added API session IDs with in-memory TTL/capacity, reset/history commands, and
  opt-out through `--no-memory` or `ENTDEC_MEMORY_ENABLED`.
- Added a 20-conversation live evaluation suite and multi-turn integration tests.
- Ruff passed; the non-integration suite passed (243 passed, 7 deselected).
- Integration tests were attempted; all 7 were skipped because Neo4j was
  unavailable. Tiny evaluation completed against live Ollama and local Parquet
  (Neo4j unavailable): references 20/20, repeat rate 0/400, topic switches
  18/20, 1 grounding-title match flagged, fallback rate 2/60, mean latency
  51.82 s/turn (P95 78.52 s). See [chat memory evaluation](reports/tiny/chat_memory_eval.md).

## 2026-11-19 - Cross-domain evaluation

- Added the deterministic `crosseval` command, shared cross-command scoring, proxy
  metrics, paired seed bootstraps, gold-pair ranks, and optional title masking.
- Validated 45 gold pairs against the default Parquet catalogues: 19
  franchise/adaptation pairs and 26 thematic pairs; all remain `reviewed=false`.
- The tiny-profile run completed in 30.3 s, including title-masked embeddings for
  all 41 in-profile gold items.
- The default-profile run completed in 53.1 s for both directions and title-masked
  embeddings for 89 gold documents; cached item embeddings were reused.
- Ruff and all non-integration tests passed. Default and tiny reports are in
  `reports/default/` and `reports/tiny/`; the title-mask check used local Ollama.
