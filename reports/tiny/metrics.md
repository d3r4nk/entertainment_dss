# Ablation metrics (per domain)

Generated 2026-10-31T11:52:36.355491+00:00 on profile `tiny`.

Split: last positive = test, second-last = validation; ranked against the
full sampled catalogue excluding train items. Diversity is mean pairwise
theme-Jaccard distance inside the top-10; coverage is over the catalogue.

## movies

| model | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | HitRate@10 | Coverage | Div@10 | fit+tune+eval s |
|---|---|---|---|---|---|---|---|---|
| pop | 0.0670 | 0.1270 | 0.0296 | 0.0448 | 0.0670 | 0.462 | 0.525 | 43.5 |
| knn | 0.0700 | 0.1300 | 0.0306 | 0.0456 | 0.0700 | 0.490 | 0.529 | 43.5 |
| mf | 0.0950 | 0.1710 | 0.0492 | 0.0682 | 0.0950 | 0.974 | 0.594 | 43.5 |
| content | 0.1060 | 0.1740 | 0.0454 | 0.0622 | 0.1060 | 0.350 | 0.560 | 43.5 |
| kg | 0.0880 | 0.1510 | 0.0401 | 0.0560 | 0.0880 | 0.296 | 0.492 | 43.5 |
| content+kg | 0.1140 | 0.2100 | 0.0533 | 0.0773 | 0.1140 | 0.368 | 0.517 | 43.5 |
| hybrid-all | 0.1250 | 0.2110 | 0.0598 | 0.0813 | 0.1250 | 0.832 | 0.569 | 43.5 |

## games

| model | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | HitRate@10 | Coverage | Div@10 | fit+tune+eval s |
|---|---|---|---|---|---|---|---|---|
| pop | 0.0220 | 0.0530 | 0.0114 | 0.0189 | 0.0220 | 0.206 | 0.551 | 38.7 |
| knn | 0.0340 | 0.0650 | 0.0158 | 0.0235 | 0.0340 | 0.262 | 0.557 | 38.7 |
| mf | 0.0650 | 0.1230 | 0.0312 | 0.0457 | 0.0650 | 0.864 | 0.657 | 38.7 |
| content | 0.0340 | 0.0840 | 0.0136 | 0.0263 | 0.0340 | 0.290 | 0.461 | 38.7 |
| kg | 0.0180 | 0.0550 | 0.0084 | 0.0175 | 0.0180 | 0.178 | 0.488 | 38.7 |
| content+kg | 0.0200 | 0.0630 | 0.0067 | 0.0173 | 0.0200 | 0.240 | 0.445 | 38.7 |
| hybrid-all | 0.0480 | 0.1040 | 0.0210 | 0.0351 | 0.0480 | 0.696 | 0.517 | 38.7 |

## Hybrid weights - movie (validation-tuned)

| weight | popularity | itemknn | mf | content | kg |
|---|---|---|---|---|---|
| value | 0.00 | 0.00 | 0.50 | 0.50 | 0.00 |

## Hybrid weights - game (validation-tuned)

| weight | popularity | itemknn | mf | content | kg |
|---|---|---|---|---|---|
| value | 0.00 | 0.00 | 0.50 | 0.25 | 0.25 |
