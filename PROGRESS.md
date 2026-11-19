# Progress

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
