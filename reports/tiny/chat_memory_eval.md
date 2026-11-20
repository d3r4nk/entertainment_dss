# Chat memory evaluation (tiny)

Scripted conversations: 20 (60 turns)
Services: live Ollama; Neo4j graph used for 0/60 turns, local Parquet used for 60/60.

| Metric | Result |
|---|---:|
| Reference-resolution accuracy | 20/20 (100.0%) |
| Repeat-recommendation rate | 0/400 (0.0%) |
| Topic-switch correctness | 18/20 (90.0%) |
| Grounding violations | 1 |
| Fallback-template rate | 2/60 (3.3%) |
| Mean latency per turn | 51.82 s |
| P95 latency per turn | 78.52 s |

Per-turn latencies (seconds): 51.88, 45.26, 50.14, 49.90, 76.40, 47.57, 53.13, 46.18, 46.31, 47.61, 45.29, 46.69, 47.72, 45.44, 50.05, 47.61, 46.74, 48.45, 49.73, 46.55, 47.91, 46.72, 46.93, 48.28, 58.66, 84.77, 54.28, 49.25, 77.06, 51.40, 49.60, 46.91, 48.74, 78.77, 78.18, 78.52, 48.40, 52.09, 48.70, 50.65, 47.90, 48.82, 49.19, 47.42, 46.04, 45.07, 45.04, 47.24, 44.30, 44.32, 75.51, 72.80, 44.52, 46.37, 44.87, 44.21, 45.69, 44.53, 44.20, 46.44

No user messages or session contents are persisted by this command beyond this report.
