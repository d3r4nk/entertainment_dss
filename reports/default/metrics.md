# Ablation metrics (per domain)

Generated 2026-10-30T14:20:36.079978+00:00 on profile `default`.

Split: last positive = test, second-last = validation; ranked against the
full sampled catalogue excluding train items. Diversity is mean pairwise
theme-Jaccard distance inside the top-10; coverage is over the catalogue.

## movies

| model | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | HitRate@10 | Coverage | Div@10 | fit+tune+eval s |
|---|---|---|---|---|---|---|---|---|
| pop | 0.0219 | 0.0374 | 0.0108 | 0.0147 | 0.0219 | 0.044 | 0.488 | 4035.0 |
| knn | 0.0268 | 0.0464 | 0.0135 | 0.0184 | 0.0268 | 0.084 | 0.504 | 4035.0 |
| mf | 0.0411 | 0.0688 | 0.0208 | 0.0278 | 0.0411 | 0.307 | 0.557 | 4035.0 |
| content | 0.0246 | 0.0426 | 0.0119 | 0.0164 | 0.0246 | 0.152 | 0.513 | 4035.0 |
| kg | 0.0110 | 0.0211 | 0.0055 | 0.0080 | 0.0110 | 0.081 | 0.303 | 4035.0 |
| content+kg | 0.0273 | 0.0441 | 0.0135 | 0.0178 | 0.0273 | 0.125 | 0.359 | 4035.0 |
| hybrid-all | 0.0425 | 0.0728 | 0.0212 | 0.0287 | 0.0425 | 0.264 | 0.519 | 4035.0 |

## games

| model | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | HitRate@10 | Coverage | Div@10 | fit+tune+eval s |
|---|---|---|---|---|---|---|---|---|
| pop | 0.0138 | 0.0266 | 0.0061 | 0.0094 | 0.0138 | 0.016 | 0.507 | 3031.3 |
| knn | 0.0216 | 0.0400 | 0.0100 | 0.0145 | 0.0216 | 0.251 | 0.569 | 3031.3 |
| mf | 0.0352 | 0.0607 | 0.0173 | 0.0237 | 0.0352 | 0.199 | 0.655 | 3031.3 |
| content | 0.0226 | 0.0419 | 0.0109 | 0.0157 | 0.0226 | 0.224 | 0.605 | 3031.3 |
| kg | 0.0041 | 0.0076 | 0.0018 | 0.0027 | 0.0041 | 0.121 | 0.492 | 3031.3 |
| content+kg | 0.0116 | 0.0214 | 0.0057 | 0.0082 | 0.0116 | 0.186 | 0.506 | 3031.3 |
| hybrid-all | 0.0352 | 0.0607 | 0.0173 | 0.0237 | 0.0352 | 0.199 | 0.655 | 3031.3 |

## Hybrid weights - movie (validation-tuned)

| weight | popularity | itemknn | mf | content | kg |
|---|---|---|---|---|---|
| value | 0.00 | 0.10 | 0.45 | 0.23 | 0.22 |

## Hybrid weights - game (validation-tuned)

| weight | popularity | itemknn | mf | content | kg |
|---|---|---|---|---|---|
| value | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 |
