# Cross-domain evaluation


Profile: `default`. Random seed: `42`. Bootstrap resamples: 1,000.
Gold pairs: 45; title-masked check: completed for 89 gold items.
Runtime: 53.1 s.

## Protocol and interpretation

MovieLens and Steam have no common user IDs, so cross-domain Recall/NDCG against user-held-out ground truth is not defined. The seed proxy draws half uniformly from the source catalogue and half from its top-20% positive-interaction items; the same seed rows are used for every method. Popularity-stratum draws avoid uniform-stratum items when enough top-20% candidates remain; draws use replacement only when a stratum is too small (as can happen on tiny). Each result list is scored against the seed's item themes and cached item embedding.

The `full_system` ranking is the deterministic `cross` command scorer: source theme profile, target-catalogue theme IDF (`ln(n_themed/df)`), and embedding cosine, min-maxed per seed and blended 0.6/0.4. The chatbot's cross-reference path transfers the same weighted theme vector through `extra_theme_weights`; the report omits its query-driven candidate retrieval, LLM intent, and answer generation. No Neo4j or LLM generation is used.

Seed-level proxy metrics use `idf(t)=ln((N+1)/(df(t)+1))+1`, with `N` all items in both domains and document frequency counted once per item. Weighted Jaccard is `sum(min(w_s,w_r)*idf)/sum(max(w_s,w_r)*idf)`; theme cosine is cosine after multiplying each vector weight by IDF. Generic themes are those with IDF at or below the median. Embedding cosine is computed from the cached item vectors.

The optional title mask removes each gold item's leading title/year clause and re-embeds only those gold documents with `qwen3-embedding:4b` at 1024 dimensions; document embeddings use no query-instruction prefix, matching the regular embedder.

Beyond-accuracy metrics aggregate all lists at each cutoff: catalogue coverage is distinct recommended target items / target catalogue size; popularity bias is the mean target-catalogue positive-interaction percentile; hubness is the share of recommendation slots occupied by the ten most-recommended items. Intervals are percentile 95% bootstrap CIs resampled over seed rows; baseline deltas use paired seed resamples.

The embedding-only baseline is favoured by embedding-cosine and theme-only by theme-overlap by construction (metric circularity). Read across all measures, not as independent evidence of user relevance.
Seed-level theme/embedding measures use percentile bootstrap means. The set-valued coverage and hubness measures use centered percentile-bootstrap deviations to reduce occupancy bias from duplicate seed rows in resamples.

## movie -> game: seed proxy results

Seeds: 1000 draws ({'top20_popularity': 500, 'uniform': 500}); catalogues: 5386 source / 5450 target; runtime: 6.0 s.

### Metrics @ 5

| method | IDF Jaccard | IDF theme cosine | shared themes/result | non-generic share | embedding cosine mean | embedding rank-1 | catalogue coverage | popularity bias | hubness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.0334 [0.0321, 0.0347] | 0.0566 [0.0543, 0.0592] | 0.9228 [0.8936, 0.9504] | 0.0044 [0.0028, 0.0062] | 0.2882 [0.2855, 0.2908] | 0.2875 [0.2836, 0.2920] | 0.605 [0.594, 0.616] | 0.494 [0.486, 0.502] | 0.011 [0.009, 0.013] |
| popularity | 0.0428 [0.0411, 0.0445] | 0.0735 [0.0701, 0.0771] | 1.2320 [1.1980, 1.2650] | 0.0000 [0.0000, 0.0000] | 0.3588 [0.3552, 0.3621] | 0.4016 [0.3972, 0.4060] | 0.001 [0.001, 0.001] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| genre_only | 0.0719 [0.0691, 0.0749] | 0.1440 [0.1373, 0.1509] | 1.6224 [1.5720, 1.6730] | 0.0360 [0.0268, 0.0462] | 0.3556 [0.3519, 0.3590] | 0.3599 [0.3558, 0.3641] | 0.055 [0.052, 0.058] | 0.700 [0.691, 0.709] | 0.271 [0.251, 0.292] |
| embedding_only | 0.0905 [0.0872, 0.0941] | 0.1892 [0.1815, 0.1973] | 1.7222 [1.6808, 1.7672] | 0.0910 [0.0776, 0.1062] | 0.4942 [0.4909, 0.4973] | 0.5195 [0.5159, 0.5233] | 0.210 [0.205, 0.215] | 0.615 [0.607, 0.624] | 0.157 [0.147, 0.169] |
| theme_only | 0.2240 [0.2200, 0.2281] | 0.4128 [0.4045, 0.4220] | 3.4690 [3.4292, 3.5140] | 0.1186 [0.1032, 0.1352] | 0.3481 [0.3452, 0.3514] | 0.3583 [0.3539, 0.3626] | 0.047 [0.046, 0.049] | 0.515 [0.507, 0.523] | 0.364 [0.349, 0.381] |
| full_system | 0.2195 [0.2157, 0.2236] | 0.4106 [0.4026, 0.4191] | 3.4230 [3.3834, 3.4660] | 0.1300 [0.1132, 0.1474] | 0.3885 [0.3854, 0.3921] | 0.4030 [0.3987, 0.4075] | 0.055 [0.053, 0.057] | 0.548 [0.540, 0.556] | 0.365 [0.350, 0.382] |

### Metrics @ 10

| method | IDF Jaccard | IDF theme cosine | shared themes/result | non-generic share | embedding cosine mean | embedding rank-1 | catalogue coverage | popularity bias | hubness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.0333 [0.0323, 0.0343] | 0.0569 [0.0549, 0.0589] | 0.9227 [0.9012, 0.9439] | 0.0046 [0.0034, 0.0060] | 0.2883 [0.2859, 0.2908] | 0.2875 [0.2836, 0.2920] | 0.841 [0.827, 0.853] | 0.496 [0.491, 0.502] | 0.008 [0.007, 0.009] |
| popularity | 0.0406 [0.0393, 0.0420] | 0.0662 [0.0635, 0.0690] | 1.2376 [1.2100, 1.2660] | 0.0000 [0.0000, 0.0000] | 0.3495 [0.3464, 0.3525] | 0.4016 [0.3972, 0.4060] | 0.002 [0.002, 0.002] | 0.999 [0.999, 0.999] | 1.000 [1.000, 1.000] |
| genre_only | 0.0728 [0.0701, 0.0756] | 0.1458 [0.1398, 0.1523] | 1.6387 [1.5942, 1.6859] | 0.0366 [0.0282, 0.0465] | 0.3538 [0.3504, 0.3570] | 0.3599 [0.3558, 0.3641] | 0.088 [0.084, 0.091] | 0.684 [0.675, 0.693] | 0.195 [0.184, 0.207] |
| embedding_only | 0.0865 [0.0836, 0.0894] | 0.1801 [0.1737, 0.1872] | 1.6789 [1.6400, 1.7182] | 0.0840 [0.0721, 0.0974] | 0.4813 [0.4782, 0.4844] | 0.5195 [0.5159, 0.5233] | 0.307 [0.300, 0.313] | 0.611 [0.604, 0.618] | 0.124 [0.117, 0.132] |
| theme_only | 0.2137 [0.2103, 0.2173] | 0.3977 [0.3906, 0.4054] | 3.3132 [3.2745, 3.3565] | 0.1054 [0.0911, 0.1200] | 0.3450 [0.3421, 0.3481] | 0.3583 [0.3539, 0.3626] | 0.070 [0.068, 0.073] | 0.523 [0.517, 0.530] | 0.254 [0.243, 0.264] |
| full_system | 0.2109 [0.2074, 0.2145] | 0.3977 [0.3904, 0.4055] | 3.2690 [3.2321, 3.3105] | 0.1160 [0.1004, 0.1322] | 0.3822 [0.3795, 0.3853] | 0.4030 [0.3987, 0.4075] | 0.087 [0.084, 0.090] | 0.552 [0.546, 0.558] | 0.253 [0.243, 0.264] |

### Metrics @ 20

| method | IDF Jaccard | IDF theme cosine | shared themes/result | non-generic share | embedding cosine mean | embedding rank-1 | catalogue coverage | popularity bias | hubness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.0327 [0.0318, 0.0336] | 0.0558 [0.0540, 0.0574] | 0.9099 [0.8908, 0.9284] | 0.0063 [0.0049, 0.0077] | 0.2883 [0.2860, 0.2906] | 0.2875 [0.2836, 0.2920] | 0.976 [0.967, 0.985] | 0.499 [0.496, 0.503] | 0.005 [0.005, 0.006] |
| popularity | 0.0372 [0.0359, 0.0385] | 0.0630 [0.0604, 0.0656] | 1.1433 [1.1152, 1.1715] | 0.0000 [0.0000, 0.0000] | 0.3469 [0.3436, 0.3500] | 0.4016 [0.3972, 0.4060] | 0.004 [0.004, 0.004] | 0.998 [0.998, 0.998] | 0.500 [0.500, 0.500] |
| genre_only | 0.0701 [0.0678, 0.0727] | 0.1402 [0.1349, 0.1462] | 1.6249 [1.5853, 1.6642] | 0.0358 [0.0283, 0.0454] | 0.3517 [0.3484, 0.3548] | 0.3599 [0.3558, 0.3641] | 0.138 [0.133, 0.142] | 0.674 [0.666, 0.681] | 0.124 [0.118, 0.131] |
| embedding_only | 0.0811 [0.0787, 0.0837] | 0.1680 [0.1621, 0.1743] | 1.6240 [1.5924, 1.6561] | 0.0731 [0.0629, 0.0843] | 0.4675 [0.4646, 0.4705] | 0.5195 [0.5159, 0.5233] | 0.424 [0.415, 0.432] | 0.612 [0.606, 0.618] | 0.095 [0.090, 0.099] |
| theme_only | 0.1989 [0.1960, 0.2022] | 0.3716 [0.3654, 0.3781] | 3.1421 [3.1086, 3.1822] | 0.0901 [0.0778, 0.1030] | 0.3395 [0.3368, 0.3424] | 0.3583 [0.3539, 0.3626] | 0.106 [0.102, 0.110] | 0.518 [0.513, 0.523] | 0.173 [0.167, 0.179] |
| full_system | 0.1969 [0.1940, 0.2002] | 0.3745 [0.3682, 0.3815] | 3.0894 [3.0550, 3.1278] | 0.1042 [0.0902, 0.1185] | 0.3732 [0.3705, 0.3762] | 0.4030 [0.3987, 0.4075] | 0.137 [0.133, 0.141] | 0.542 [0.537, 0.547] | 0.169 [0.163, 0.175] |

### Seed strata @ 10

| stratum | method | IDF Jaccard | IDF theme cosine | shared themes/result | non-generic share | embedding cosine mean | embedding rank-1 | catalogue coverage | popularity bias | hubness |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | random | 0.0342 [0.0327, 0.0357] | 0.0585 [0.0554, 0.0613] | 0.9478 [0.9172, 0.9792] | 0.0058 [0.0038, 0.0080] | 0.2866 [0.2831, 0.2901] | 0.2839 [0.2780, 0.2896] | 0.601 [0.586, 0.615] | 0.498 [0.490, 0.506] | 0.011 [0.009, 0.013] |
| uniform | popularity | 0.0409 [0.0392, 0.0429] | 0.0666 [0.0629, 0.0705] | 1.2472 [1.2128, 1.2856] | 0.0000 [0.0000, 0.0000] | 0.3426 [0.3382, 0.3470] | 0.3926 [0.3867, 0.3985] | 0.002 [0.002, 0.002] | 0.999 [0.999, 0.999] | 1.000 [1.000, 1.000] |
| uniform | genre_only | 0.0753 [0.0715, 0.0790] | 0.1495 [0.1410, 0.1582] | 1.6878 [1.6274, 1.7504] | 0.0496 [0.0348, 0.0666] | 0.3483 [0.3435, 0.3526] | 0.3560 [0.3500, 0.3621] | 0.079 [0.074, 0.083] | 0.684 [0.672, 0.697] | 0.200 [0.184, 0.217] |
| uniform | embedding_only | 0.0897 [0.0855, 0.0939] | 0.1867 [0.1768, 0.1966] | 1.7102 [1.6570, 1.7650] | 0.0980 [0.0792, 0.1156] | 0.4759 [0.4718, 0.4802] | 0.5134 [0.5083, 0.5184] | 0.242 [0.235, 0.248] | 0.598 [0.588, 0.608] | 0.112 [0.103, 0.122] |
| uniform | theme_only | 0.2200 [0.2148, 0.2251] | 0.4121 [0.4020, 0.4220] | 3.3672 [3.3096, 3.4224] | 0.1220 [0.0996, 0.1470] | 0.3421 [0.3375, 0.3460] | 0.3506 [0.3445, 0.3571] | 0.062 [0.060, 0.065] | 0.510 [0.501, 0.519] | 0.231 [0.218, 0.245] |
| uniform | full_system | 0.2166 [0.2115, 0.2218] | 0.4111 [0.4005, 0.4211] | 3.3212 [3.2606, 3.3776] | 0.1336 [0.1104, 0.1592] | 0.3800 [0.3761, 0.3836] | 0.3945 [0.3881, 0.4003] | 0.076 [0.073, 0.079] | 0.538 [0.529, 0.547] | 0.235 [0.222, 0.250] |
| top20_popularity | random | 0.0324 [0.0310, 0.0339] | 0.0553 [0.0524, 0.0582] | 0.8976 [0.8662, 0.9276] | 0.0034 [0.0018, 0.0052] | 0.2900 [0.2865, 0.2932] | 0.2911 [0.2855, 0.2968] | 0.603 [0.589, 0.618] | 0.495 [0.487, 0.503] | 0.010 [0.008, 0.012] |
| top20_popularity | popularity | 0.0403 [0.0384, 0.0423] | 0.0659 [0.0622, 0.0698] | 1.2280 [1.1906, 1.2678] | 0.0000 [0.0000, 0.0000] | 0.3564 [0.3524, 0.3608] | 0.4105 [0.4050, 0.4165] | 0.002 [0.002, 0.002] | 0.999 [0.999, 0.999] | 1.000 [1.000, 1.000] |
| top20_popularity | genre_only | 0.0703 [0.0667, 0.0741] | 0.1422 [0.1339, 0.1510] | 1.5896 [1.5312, 1.6532] | 0.0236 [0.0136, 0.0340] | 0.3594 [0.3551, 0.3637] | 0.3638 [0.3583, 0.3694] | 0.069 [0.066, 0.072] | 0.683 [0.671, 0.696] | 0.191 [0.177, 0.207] |
| top20_popularity | embedding_only | 0.0833 [0.0788, 0.0872] | 0.1735 [0.1633, 0.1826] | 1.6476 [1.5896, 1.7006] | 0.0700 [0.0536, 0.0878] | 0.4866 [0.4820, 0.4916] | 0.5256 [0.5207, 0.5313] | 0.212 [0.205, 0.218] | 0.624 [0.614, 0.634] | 0.137 [0.126, 0.149] |
| top20_popularity | theme_only | 0.2075 [0.2022, 0.2127] | 0.3833 [0.3721, 0.3943] | 3.2592 [3.2032, 3.3202] | 0.0888 [0.0702, 0.1088] | 0.3478 [0.3435, 0.3518] | 0.3659 [0.3595, 0.3719] | 0.056 [0.054, 0.059] | 0.536 [0.527, 0.546] | 0.281 [0.267, 0.297] |
| top20_popularity | full_system | 0.2051 [0.1997, 0.2107] | 0.3843 [0.3730, 0.3963] | 3.2168 [3.1610, 3.2740] | 0.0984 [0.0792, 0.1186] | 0.3844 [0.3802, 0.3886] | 0.4115 [0.4060, 0.4173] | 0.067 [0.064, 0.070] | 0.566 [0.557, 0.576] | 0.274 [0.260, 0.288] |

### Paired full-system differences @ 5

| baseline | metric | full minus baseline (95% CI) |
|---|---|---:|
| random | theme_jaccard | 0.1861 [0.1824, 0.1898] |
| random | theme_cosine | 0.3541 [0.3458, 0.3620] |
| random | shared_themes_per_result | 2.5002 [2.4536, 2.5494] |
| random | non_generic_share | 0.1256 [0.1088, 0.1428] |
| random | embedding_cosine_mean | 0.1003 [0.0975, 0.1033] |
| random | embedding_cosine_rank1 | 0.1155 [0.1102, 0.1208] |
| random | catalogue_coverage | -0.5503 [-0.5612, -0.5392] |
| random | popularity_bias | 0.0541 [0.0425, 0.0658] |
| random | hubness | 0.3546 [0.3393, 0.3723] |
| popularity | theme_jaccard | 0.1767 [0.1726, 0.1810] |
| popularity | theme_cosine | 0.3372 [0.3285, 0.3461] |
| popularity | shared_themes_per_result | 2.1910 [2.1432, 2.2438] |
| popularity | non_generic_share | 0.1300 [0.1132, 0.1474] |
| popularity | embedding_cosine_mean | 0.0298 [0.0267, 0.0332] |
| popularity | embedding_cosine_rank1 | 0.0015 [-0.0033, 0.0059] |
| popularity | catalogue_coverage | 0.0541 [0.0520, 0.0561] |
| popularity | popularity_bias | -0.4515 [-0.4597, -0.4435] |
| popularity | hubness | -0.6348 [-0.6504, -0.6176] |
| genre_only | theme_jaccard | 0.1476 [0.1441, 0.1510] |
| genre_only | theme_cosine | 0.2666 [0.2588, 0.2739] |
| genre_only | shared_themes_per_result | 1.8006 [1.7548, 1.8480] |
| genre_only | non_generic_share | 0.0940 [0.0790, 0.1098] |
| genre_only | embedding_cosine_mean | 0.0329 [0.0298, 0.0362] |
| genre_only | embedding_cosine_rank1 | 0.0431 [0.0378, 0.0483] |
| genre_only | catalogue_coverage | 0.0002 [-0.0030, 0.0036] |
| genre_only | popularity_bias | -0.1516 [-0.1629, -0.1396] |
| genre_only | hubness | 0.0943 [0.0691, 0.1175] |
| embedding_only | theme_jaccard | 0.1290 [0.1257, 0.1326] |
| embedding_only | theme_cosine | 0.2214 [0.2142, 0.2285] |
| embedding_only | shared_themes_per_result | 1.7008 [1.6586, 1.7424] |
| embedding_only | non_generic_share | 0.0390 [0.0282, 0.0506] |
| embedding_only | embedding_cosine_mean | -0.1056 [-0.1084, -0.1028] |
| embedding_only | embedding_cosine_rank1 | -0.1165 [-0.1208, -0.1125] |
| embedding_only | catalogue_coverage | -0.1550 [-0.1602, -0.1500] |
| embedding_only | popularity_bias | -0.0667 [-0.0779, -0.0556] |
| embedding_only | hubness | 0.2080 [0.1899, 0.2275] |
| theme_only | theme_jaccard | -0.0044 [-0.0057, -0.0032] |
| theme_only | theme_cosine | -0.0022 [-0.0048, 0.0004] |
| theme_only | shared_themes_per_result | -0.0460 [-0.0640, -0.0294] |
| theme_only | non_generic_share | 0.0114 [0.0066, 0.0164] |
| theme_only | embedding_cosine_mean | 0.0404 [0.0385, 0.0423] |
| theme_only | embedding_cosine_rank1 | 0.0447 [0.0408, 0.0490] |
| theme_only | catalogue_coverage | 0.0079 [0.0061, 0.0096] |
| theme_only | popularity_bias | 0.0329 [0.0263, 0.0396] |
| theme_only | hubness | 0.0008 [-0.0101, 0.0119] |

### Paired full-system differences @ 10

| baseline | metric | full minus baseline (95% CI) |
|---|---|---:|
| random | theme_jaccard | 0.1776 [0.1743, 0.1810] |
| random | theme_cosine | 0.3408 [0.3337, 0.3484] |
| random | shared_themes_per_result | 2.3463 [2.3070, 2.3889] |
| random | non_generic_share | 0.1114 [0.0958, 0.1274] |
| random | embedding_cosine_mean | 0.0939 [0.0916, 0.0962] |
| random | embedding_cosine_rank1 | 0.1155 [0.1102, 0.1208] |
| random | catalogue_coverage | -0.7539 [-0.7665, -0.7420] |
| random | popularity_bias | 0.0555 [0.0476, 0.0637] |
| random | hubness | 0.2458 [0.2355, 0.2569] |
| popularity | theme_jaccard | 0.1702 [0.1668, 0.1740] |
| popularity | theme_cosine | 0.3315 [0.3242, 0.3395] |
| popularity | shared_themes_per_result | 2.0314 [1.9889, 2.0770] |
| popularity | non_generic_share | 0.1160 [0.1004, 0.1322] |
| popularity | embedding_cosine_mean | 0.0327 [0.0300, 0.0355] |
| popularity | embedding_cosine_rank1 | 0.0015 [-0.0033, 0.0059] |
| popularity | catalogue_coverage | 0.0848 [0.0818, 0.0878] |
| popularity | popularity_bias | -0.4472 [-0.4535, -0.4409] |
| popularity | hubness | -0.7466 [-0.7569, -0.7358] |
| genre_only | theme_jaccard | 0.1381 [0.1350, 0.1413] |
| genre_only | theme_cosine | 0.2519 [0.2446, 0.2589] |
| genre_only | shared_themes_per_result | 1.6303 [1.5874, 1.6737] |
| genre_only | non_generic_share | 0.0794 [0.0653, 0.0936] |
| genre_only | embedding_cosine_mean | 0.0283 [0.0256, 0.0311] |
| genre_only | embedding_cosine_rank1 | 0.0431 [0.0378, 0.0483] |
| genre_only | catalogue_coverage | -0.0011 [-0.0057, 0.0041] |
| genre_only | popularity_bias | -0.1318 [-0.1419, -0.1217] |
| genre_only | hubness | 0.0580 [0.0430, 0.0724] |
| embedding_only | theme_jaccard | 0.1244 [0.1217, 0.1273] |
| embedding_only | theme_cosine | 0.2176 [0.2117, 0.2237] |
| embedding_only | shared_themes_per_result | 1.5901 [1.5561, 1.6280] |
| embedding_only | non_generic_share | 0.0320 [0.0230, 0.0413] |
| embedding_only | embedding_cosine_mean | -0.0991 [-0.1013, -0.0968] |
| embedding_only | embedding_cosine_rank1 | -0.1165 [-0.1208, -0.1125] |
| embedding_only | catalogue_coverage | -0.2200 [-0.2264, -0.2138] |
| embedding_only | popularity_bias | -0.0590 [-0.0676, -0.0504] |
| embedding_only | hubness | 0.1296 [0.1177, 0.1422] |
| theme_only | theme_jaccard | -0.0029 [-0.0037, -0.0021] |
| theme_only | theme_cosine | 0.0000 [-0.0017, 0.0017] |
| theme_only | shared_themes_per_result | -0.0442 [-0.0561, -0.0316] |
| theme_only | non_generic_share | 0.0106 [0.0069, 0.0148] |
| theme_only | embedding_cosine_mean | 0.0372 [0.0358, 0.0386] |
| theme_only | embedding_cosine_rank1 | 0.0447 [0.0408, 0.0490] |
| theme_only | catalogue_coverage | 0.0165 [0.0142, 0.0188] |
| theme_only | popularity_bias | 0.0288 [0.0245, 0.0330] |
| theme_only | hubness | -0.0006 [-0.0079, 0.0069] |

### Paired full-system differences @ 20

| baseline | metric | full minus baseline (95% CI) |
|---|---|---:|
| random | theme_jaccard | 0.1642 [0.1614, 0.1672] |
| random | theme_cosine | 0.3187 [0.3127, 0.3253] |
| random | shared_themes_per_result | 2.1795 [2.1451, 2.2158] |
| random | non_generic_share | 0.0980 [0.0847, 0.1120] |
| random | embedding_cosine_mean | 0.0849 [0.0828, 0.0869] |
| random | embedding_cosine_rank1 | 0.1155 [0.1102, 0.1208] |
| random | catalogue_coverage | -0.8389 [-0.8485, -0.8292] |
| random | popularity_bias | 0.0427 [0.0363, 0.0487] |
| random | hubness | 0.1636 [0.1576, 0.1698] |
| popularity | theme_jaccard | 0.1598 [0.1568, 0.1631] |
| popularity | theme_cosine | 0.3115 [0.3051, 0.3184] |
| popularity | shared_themes_per_result | 1.9461 [1.9069, 1.9883] |
| popularity | non_generic_share | 0.1042 [0.0902, 0.1185] |
| popularity | embedding_cosine_mean | 0.0263 [0.0240, 0.0289] |
| popularity | embedding_cosine_rank1 | 0.0015 [-0.0033, 0.0059] |
| popularity | catalogue_coverage | 0.1336 [0.1289, 0.1377] |
| popularity | popularity_bias | -0.4561 [-0.4617, -0.4509] |
| popularity | hubness | -0.3311 [-0.3369, -0.3250] |
| genre_only | theme_jaccard | 0.1268 [0.1239, 0.1297] |
| genre_only | theme_cosine | 0.2342 [0.2277, 0.2401] |
| genre_only | shared_themes_per_result | 1.4646 [1.4278, 1.5036] |
| genre_only | non_generic_share | 0.0684 [0.0560, 0.0815] |
| genre_only | embedding_cosine_mean | 0.0216 [0.0193, 0.0240] |
| genre_only | embedding_cosine_rank1 | 0.0431 [0.0378, 0.0483] |
| genre_only | catalogue_coverage | -0.0004 [-0.0064, 0.0059] |
| genre_only | popularity_bias | -0.1315 [-0.1396, -0.1229] |
| genre_only | hubness | 0.0446 [0.0361, 0.0529] |
| embedding_only | theme_jaccard | 0.1158 [0.1137, 0.1182] |
| embedding_only | theme_cosine | 0.2065 [0.2017, 0.2116] |
| embedding_only | shared_themes_per_result | 1.4655 [1.4376, 1.4953] |
| embedding_only | non_generic_share | 0.0311 [0.0231, 0.0397] |
| embedding_only | embedding_cosine_mean | -0.0943 [-0.0962, -0.0924] |
| embedding_only | embedding_cosine_rank1 | -0.1165 [-0.1208, -0.1125] |
| embedding_only | catalogue_coverage | -0.2864 [-0.2946, -0.2781] |
| embedding_only | popularity_bias | -0.0697 [-0.0765, -0.0635] |
| embedding_only | hubness | 0.0744 [0.0675, 0.0816] |
| theme_only | theme_jaccard | -0.0020 [-0.0026, -0.0014] |
| theme_only | theme_cosine | 0.0029 [0.0015, 0.0043] |
| theme_only | shared_themes_per_result | -0.0527 [-0.0621, -0.0438] |
| theme_only | non_generic_share | 0.0142 [0.0109, 0.0180] |
| theme_only | embedding_cosine_mean | 0.0337 [0.0326, 0.0348] |
| theme_only | embedding_cosine_rank1 | 0.0447 [0.0408, 0.0490] |
| theme_only | catalogue_coverage | 0.0312 [0.0276, 0.0344] |
| theme_only | popularity_bias | 0.0239 [0.0210, 0.0269] |
| theme_only | hubness | -0.0041 [-0.0089, -0.0002] |

## game -> movie: seed proxy results

Seeds: 1000 draws ({'top20_popularity': 500, 'uniform': 500}); catalogues: 5450 source / 5386 target; runtime: 5.7 s.

### Metrics @ 5

| method | IDF Jaccard | IDF theme cosine | shared themes/result | non-generic share | embedding cosine mean | embedding rank-1 | catalogue coverage | popularity bias | hubness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.0315 [0.0296, 0.0332] | 0.0550 [0.0516, 0.0581] | 0.8794 [0.8358, 0.9242] | 0.0072 [0.0048, 0.0100] | 0.2925 [0.2900, 0.2952] | 0.2937 [0.2895, 0.2979] | 0.605 [0.594, 0.616] | 0.499 [0.491, 0.508] | 0.010 [0.009, 0.012] |
| popularity | 0.0395 [0.0375, 0.0417] | 0.0644 [0.0610, 0.0678] | 1.0808 [1.0362, 1.1326] | 0.0000 [0.0000, 0.0000] | 0.3368 [0.3335, 0.3399] | 0.2953 [0.2922, 0.2984] | 0.001 [0.001, 0.001] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| genre_only | 0.0626 [0.0601, 0.0655] | 0.1251 [0.1199, 0.1310] | 1.3514 [1.3060, 1.4006] | 0.0136 [0.0094, 0.0184] | 0.3499 [0.3447, 0.3549] | 0.3567 [0.3511, 0.3618] | 0.046 [0.043, 0.049] | 0.678 [0.669, 0.687] | 0.393 [0.371, 0.415] |
| embedding_only | 0.0731 [0.0702, 0.0760] | 0.1564 [0.1490, 0.1634] | 1.4692 [1.4232, 1.5200] | 0.0610 [0.0502, 0.0716] | 0.5073 [0.5040, 0.5107] | 0.5322 [0.5285, 0.5363] | 0.192 [0.186, 0.197] | 0.507 [0.498, 0.516] | 0.237 [0.224, 0.250] |
| theme_only | 0.1630 [0.1589, 0.1674] | 0.3491 [0.3411, 0.3580] | 2.4786 [2.4326, 2.5306] | 0.1530 [0.1328, 0.1740] | 0.3600 [0.3558, 0.3642] | 0.3667 [0.3622, 0.3714] | 0.091 [0.087, 0.096] | 0.487 [0.482, 0.493] | 0.487 [0.460, 0.514] |
| full_system | 0.1616 [0.1573, 0.1657] | 0.3464 [0.3380, 0.3554] | 2.4276 [2.3806, 2.4794] | 0.1466 [0.1268, 0.1666] | 0.3967 [0.3929, 0.4008] | 0.3997 [0.3952, 0.4045] | 0.108 [0.103, 0.112] | 0.510 [0.503, 0.516] | 0.473 [0.446, 0.498] |

### Metrics @ 10

| method | IDF Jaccard | IDF theme cosine | shared themes/result | non-generic share | embedding cosine mean | embedding rank-1 | catalogue coverage | popularity bias | hubness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.0316 [0.0299, 0.0332] | 0.0554 [0.0524, 0.0583] | 0.8765 [0.8377, 0.9205] | 0.0075 [0.0059, 0.0093] | 0.2923 [0.2900, 0.2949] | 0.2937 [0.2895, 0.2979] | 0.847 [0.834, 0.861] | 0.499 [0.494, 0.505] | 0.007 [0.006, 0.009] |
| popularity | 0.0341 [0.0323, 0.0360] | 0.0545 [0.0517, 0.0574] | 0.9523 [0.9100, 0.9989] | 0.0000 [0.0000, 0.0000] | 0.3328 [0.3298, 0.3356] | 0.2953 [0.2922, 0.2984] | 0.002 [0.002, 0.002] | 0.999 [0.999, 0.999] | 1.000 [1.000, 1.000] |
| genre_only | 0.0614 [0.0590, 0.0641] | 0.1232 [0.1181, 0.1289] | 1.3467 [1.2998, 1.3960] | 0.0194 [0.0144, 0.0250] | 0.3473 [0.3423, 0.3521] | 0.3567 [0.3511, 0.3618] | 0.081 [0.077, 0.085] | 0.650 [0.644, 0.657] | 0.299 [0.282, 0.316] |
| embedding_only | 0.0719 [0.0690, 0.0748] | 0.1537 [0.1468, 0.1603] | 1.4479 [1.4054, 1.4970] | 0.0630 [0.0532, 0.0731] | 0.4941 [0.4910, 0.4974] | 0.5322 [0.5285, 0.5363] | 0.289 [0.281, 0.296] | 0.514 [0.507, 0.520] | 0.182 [0.173, 0.191] |
| theme_only | 0.1573 [0.1534, 0.1613] | 0.3371 [0.3293, 0.3453] | 2.4013 [2.3573, 2.4498] | 0.1387 [0.1203, 0.1572] | 0.3574 [0.3535, 0.3615] | 0.3667 [0.3622, 0.3714] | 0.149 [0.142, 0.156] | 0.515 [0.510, 0.520] | 0.374 [0.350, 0.397] |
| full_system | 0.1572 [0.1534, 0.1612] | 0.3382 [0.3303, 0.3466] | 2.3720 [2.3257, 2.4220] | 0.1396 [0.1219, 0.1595] | 0.3930 [0.3894, 0.3968] | 0.3997 [0.3952, 0.4045] | 0.174 [0.166, 0.181] | 0.530 [0.525, 0.535] | 0.359 [0.337, 0.380] |

### Metrics @ 20

| method | IDF Jaccard | IDF theme cosine | shared themes/result | non-generic share | embedding cosine mean | embedding rank-1 | catalogue coverage | popularity bias | hubness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.0318 [0.0302, 0.0334] | 0.0557 [0.0530, 0.0585] | 0.8798 [0.8420, 0.9226] | 0.0073 [0.0061, 0.0086] | 0.2926 [0.2903, 0.2948] | 0.2937 [0.2895, 0.2979] | 0.979 [0.969, 0.987] | 0.498 [0.494, 0.502] | 0.005 [0.005, 0.006] |
| popularity | 0.0329 [0.0312, 0.0347] | 0.0543 [0.0516, 0.0571] | 0.9281 [0.8883, 0.9749] | 0.0018 [0.0013, 0.0024] | 0.3306 [0.3276, 0.3333] | 0.2953 [0.2922, 0.2984] | 0.004 [0.004, 0.004] | 0.998 [0.998, 0.998] | 0.500 [0.500, 0.500] |
| genre_only | 0.0594 [0.0570, 0.0620] | 0.1189 [0.1139, 0.1243] | 1.3117 [1.2669, 1.3617] | 0.0219 [0.0171, 0.0273] | 0.3456 [0.3405, 0.3505] | 0.3567 [0.3511, 0.3618] | 0.134 [0.127, 0.141] | 0.648 [0.642, 0.655] | 0.210 [0.199, 0.221] |
| embedding_only | 0.0702 [0.0676, 0.0729] | 0.1490 [0.1428, 0.1553] | 1.4228 [1.3810, 1.4721] | 0.0579 [0.0493, 0.0666] | 0.4800 [0.4770, 0.4832] | 0.5322 [0.5285, 0.5363] | 0.412 [0.401, 0.422] | 0.525 [0.519, 0.530] | 0.137 [0.131, 0.142] |
| theme_only | 0.1501 [0.1465, 0.1540] | 0.3243 [0.3167, 0.3321] | 2.2982 [2.2525, 2.3464] | 0.1241 [0.1075, 0.1409] | 0.3525 [0.3486, 0.3563] | 0.3667 [0.3622, 0.3714] | 0.238 [0.227, 0.248] | 0.527 [0.523, 0.532] | 0.242 [0.228, 0.256] |
| full_system | 0.1507 [0.1471, 0.1545] | 0.3273 [0.3194, 0.3349] | 2.2856 [2.2421, 2.3338] | 0.1276 [0.1116, 0.1447] | 0.3868 [0.3834, 0.3905] | 0.3997 [0.3952, 0.4045] | 0.269 [0.259, 0.279] | 0.532 [0.528, 0.537] | 0.233 [0.219, 0.246] |

### Seed strata @ 10

| stratum | method | IDF Jaccard | IDF theme cosine | shared themes/result | non-generic share | embedding cosine mean | embedding rank-1 | catalogue coverage | popularity bias | hubness |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | random | 0.0331 [0.0308, 0.0356] | 0.0567 [0.0527, 0.0609] | 0.9024 [0.8406, 0.9620] | 0.0092 [0.0064, 0.0118] | 0.2838 [0.2803, 0.2875] | 0.2889 [0.2826, 0.2947] | 0.609 [0.594, 0.624] | 0.496 [0.489, 0.504] | 0.011 [0.009, 0.013] |
| uniform | popularity | 0.0357 [0.0332, 0.0383] | 0.0554 [0.0513, 0.0595] | 0.9714 [0.9119, 1.0380] | 0.0000 [0.0000, 0.0000] | 0.3178 [0.3140, 0.3219] | 0.2803 [0.2760, 0.2849] | 0.002 [0.002, 0.002] | 0.999 [0.999, 0.999] | 1.000 [1.000, 1.000] |
| uniform | genre_only | 0.0599 [0.0562, 0.0638] | 0.1179 [0.1100, 0.1260] | 1.3126 [1.2420, 1.3866] | 0.0234 [0.0158, 0.0320] | 0.3270 [0.3185, 0.3358] | 0.3372 [0.3279, 0.3462] | 0.072 [0.067, 0.077] | 0.641 [0.631, 0.651] | 0.287 [0.263, 0.312] |
| uniform | embedding_only | 0.0703 [0.0664, 0.0740] | 0.1461 [0.1376, 0.1547] | 1.4384 [1.3702, 1.5024] | 0.0728 [0.0578, 0.0880] | 0.4818 [0.4774, 0.4859] | 0.5191 [0.5143, 0.5240] | 0.220 [0.213, 0.228] | 0.503 [0.494, 0.513] | 0.168 [0.156, 0.180] |
| uniform | theme_only | 0.1580 [0.1524, 0.1638] | 0.3383 [0.3278, 0.3498] | 2.3784 [2.3104, 2.4538] | 0.1626 [0.1330, 0.1908] | 0.3467 [0.3417, 0.3524] | 0.3546 [0.3483, 0.3610] | 0.113 [0.108, 0.119] | 0.509 [0.501, 0.516] | 0.351 [0.318, 0.381] |
| uniform | full_system | 0.1580 [0.1521, 0.1636] | 0.3399 [0.3277, 0.3512] | 2.3502 [2.2766, 2.4234] | 0.1620 [0.1334, 0.1890] | 0.3834 [0.3784, 0.3885] | 0.3904 [0.3843, 0.3965] | 0.131 [0.125, 0.136] | 0.522 [0.514, 0.530] | 0.337 [0.307, 0.366] |
| top20_popularity | random | 0.0300 [0.0280, 0.0322] | 0.0540 [0.0503, 0.0578] | 0.8506 [0.7988, 0.9024] | 0.0058 [0.0036, 0.0082] | 0.3008 [0.2979, 0.3041] | 0.2986 [0.2933, 0.3045] | 0.610 [0.595, 0.624] | 0.502 [0.495, 0.510] | 0.010 [0.008, 0.012] |
| top20_popularity | popularity | 0.0325 [0.0300, 0.0350] | 0.0536 [0.0494, 0.0579] | 0.9332 [0.8726, 0.9954] | 0.0000 [0.0000, 0.0000] | 0.3477 [0.3438, 0.3518] | 0.3103 [0.3061, 0.3145] | 0.002 [0.002, 0.002] | 0.999 [0.999, 0.999] | 1.000 [1.000, 1.000] |
| top20_popularity | genre_only | 0.0629 [0.0597, 0.0664] | 0.1284 [0.1216, 0.1359] | 1.3808 [1.3194, 1.4490] | 0.0154 [0.0098, 0.0214] | 0.3677 [0.3627, 0.3732] | 0.3763 [0.3705, 0.3821] | 0.056 [0.052, 0.059] | 0.659 [0.651, 0.667] | 0.311 [0.290, 0.334] |
| top20_popularity | embedding_only | 0.0736 [0.0692, 0.0776] | 0.1614 [0.1513, 0.1711] | 1.4574 [1.3924, 1.5220] | 0.0532 [0.0404, 0.0678] | 0.5063 [0.5020, 0.5104] | 0.5452 [0.5402, 0.5503] | 0.198 [0.191, 0.205] | 0.524 [0.515, 0.533] | 0.208 [0.195, 0.222] |
| top20_popularity | theme_only | 0.1566 [0.1517, 0.1617] | 0.3358 [0.3255, 0.3468] | 2.4242 [2.3620, 2.4926] | 0.1148 [0.0926, 0.1382] | 0.3682 [0.3634, 0.3731] | 0.3788 [0.3725, 0.3851] | 0.109 [0.102, 0.115] | 0.522 [0.515, 0.528] | 0.404 [0.371, 0.435] |
| top20_popularity | full_system | 0.1564 [0.1516, 0.1615] | 0.3366 [0.3255, 0.3479] | 2.3938 [2.3344, 2.4585] | 0.1172 [0.0928, 0.1410] | 0.4025 [0.3977, 0.4076] | 0.4089 [0.4023, 0.4155] | 0.125 [0.117, 0.131] | 0.538 [0.531, 0.545] | 0.384 [0.355, 0.415] |

### Paired full-system differences @ 5

| baseline | metric | full minus baseline (95% CI) |
|---|---|---:|
| random | theme_jaccard | 0.1301 [0.1262, 0.1338] |
| random | theme_cosine | 0.2914 [0.2828, 0.2999] |
| random | shared_themes_per_result | 1.5482 [1.5078, 1.5832] |
| random | non_generic_share | 0.1394 [0.1198, 0.1596] |
| random | embedding_cosine_mean | 0.1042 [0.1005, 0.1079] |
| random | embedding_cosine_rank1 | 0.1059 [0.1009, 0.1114] |
| random | catalogue_coverage | -0.4974 [-0.5084, -0.4865] |
| random | popularity_bias | 0.0103 [0.0007, 0.0200] |
| random | hubness | 0.4622 [0.4359, 0.4879] |
| popularity | theme_jaccard | 0.1220 [0.1180, 0.1261] |
| popularity | theme_cosine | 0.2820 [0.2730, 0.2913] |
| popularity | shared_themes_per_result | 1.3468 [1.3068, 1.3834] |
| popularity | non_generic_share | 0.1466 [0.1268, 0.1666] |
| popularity | embedding_cosine_mean | 0.0599 [0.0566, 0.0634] |
| popularity | embedding_cosine_rank1 | 0.1044 [0.0999, 0.1088] |
| popularity | catalogue_coverage | 0.1068 [0.1018, 0.1111] |
| popularity | popularity_bias | -0.4901 [-0.4964, -0.4839] |
| popularity | hubness | -0.5274 [-0.5538, -0.5020] |
| genre_only | theme_jaccard | 0.0989 [0.0953, 0.1026] |
| genre_only | theme_cosine | 0.2213 [0.2133, 0.2297] |
| genre_only | shared_themes_per_result | 1.0762 [1.0382, 1.1094] |
| genre_only | non_generic_share | 0.1330 [0.1140, 0.1534] |
| genre_only | embedding_cosine_mean | 0.0468 [0.0425, 0.0519] |
| genre_only | embedding_cosine_rank1 | 0.0429 [0.0369, 0.0489] |
| genre_only | catalogue_coverage | 0.0615 [0.0564, 0.0666] |
| genre_only | popularity_bias | -0.1689 [-0.1802, -0.1579] |
| genre_only | hubness | 0.0797 [0.0485, 0.1104] |
| embedding_only | theme_jaccard | 0.0885 [0.0851, 0.0922] |
| embedding_only | theme_cosine | 0.1900 [0.1824, 0.1984] |
| embedding_only | shared_themes_per_result | 0.9584 [0.9210, 0.9938] |
| embedding_only | non_generic_share | 0.0856 [0.0674, 0.1058] |
| embedding_only | embedding_cosine_mean | -0.1106 [-0.1134, -0.1076] |
| embedding_only | embedding_cosine_rank1 | -0.1325 [-0.1367, -0.1288] |
| embedding_only | catalogue_coverage | -0.0841 [-0.0907, -0.0777] |
| embedding_only | popularity_bias | 0.0024 [-0.0083, 0.0130] |
| embedding_only | hubness | 0.2358 [0.2074, 0.2636] |
| theme_only | theme_jaccard | -0.0015 [-0.0023, -0.0007] |
| theme_only | theme_cosine | -0.0027 [-0.0047, -0.0010] |
| theme_only | shared_themes_per_result | -0.0510 [-0.0640, -0.0398] |
| theme_only | non_generic_share | -0.0064 [-0.0122, -0.0010] |
| theme_only | embedding_cosine_mean | 0.0367 [0.0342, 0.0396] |
| theme_only | embedding_cosine_rank1 | 0.0330 [0.0294, 0.0368] |
| theme_only | catalogue_coverage | 0.0167 [0.0133, 0.0198] |
| theme_only | popularity_bias | 0.0223 [0.0177, 0.0270] |
| theme_only | hubness | -0.0143 [-0.0227, -0.0061] |

### Paired full-system differences @ 10

| baseline | metric | full minus baseline (95% CI) |
|---|---|---:|
| random | theme_jaccard | 0.1256 [0.1222, 0.1290] |
| random | theme_cosine | 0.2829 [0.2750, 0.2907] |
| random | shared_themes_per_result | 1.4955 [1.4612, 1.5260] |
| random | non_generic_share | 0.1321 [0.1144, 0.1517] |
| random | embedding_cosine_mean | 0.1006 [0.0975, 0.1041] |
| random | embedding_cosine_rank1 | 0.1059 [0.1009, 0.1114] |
| random | catalogue_coverage | -0.6736 [-0.6866, -0.6608] |
| random | popularity_bias | 0.0307 [0.0234, 0.0380] |
| random | hubness | 0.3519 [0.3296, 0.3727] |
| popularity | theme_jaccard | 0.1231 [0.1196, 0.1267] |
| popularity | theme_cosine | 0.2837 [0.2755, 0.2922] |
| popularity | shared_themes_per_result | 1.4197 [1.3868, 1.4512] |
| popularity | non_generic_share | 0.1396 [0.1219, 0.1595] |
| popularity | embedding_cosine_mean | 0.0602 [0.0574, 0.0633] |
| popularity | embedding_cosine_rank1 | 0.1044 [0.0999, 0.1088] |
| popularity | catalogue_coverage | 0.1719 [0.1643, 0.1788] |
| popularity | popularity_bias | -0.4692 [-0.4743, -0.4639] |
| popularity | hubness | -0.6407 [-0.6632, -0.6196] |
| genre_only | theme_jaccard | 0.0958 [0.0927, 0.0989] |
| genre_only | theme_cosine | 0.2151 [0.2079, 0.2228] |
| genre_only | shared_themes_per_result | 1.0253 [0.9913, 1.0568] |
| genre_only | non_generic_share | 0.1202 [0.1021, 0.1405] |
| genre_only | embedding_cosine_mean | 0.0456 [0.0416, 0.0504] |
| genre_only | embedding_cosine_rank1 | 0.0429 [0.0369, 0.0489] |
| genre_only | catalogue_coverage | 0.0926 [0.0850, 0.1001] |
| genre_only | popularity_bias | -0.1204 [-0.1287, -0.1123] |
| genre_only | hubness | 0.0599 [0.0364, 0.0854] |
| embedding_only | theme_jaccard | 0.0852 [0.0821, 0.0883] |
| embedding_only | theme_cosine | 0.1845 [0.1773, 0.1919] |
| embedding_only | shared_themes_per_result | 0.9241 [0.8931, 0.9547] |
| embedding_only | non_generic_share | 0.0766 [0.0607, 0.0943] |
| embedding_only | embedding_cosine_mean | -0.1011 [-0.1035, -0.0986] |
| embedding_only | embedding_cosine_rank1 | -0.1325 [-0.1367, -0.1288] |
| embedding_only | catalogue_coverage | -0.1149 [-0.1238, -0.1058] |
| embedding_only | popularity_bias | 0.0162 [0.0083, 0.0247] |
| embedding_only | hubness | 0.1776 [0.1551, 0.2006] |
| theme_only | theme_jaccard | -0.0001 [-0.0007, 0.0005] |
| theme_only | theme_cosine | 0.0012 [-0.0004, 0.0026] |
| theme_only | shared_themes_per_result | -0.0293 [-0.0387, -0.0203] |
| theme_only | non_generic_share | 0.0009 [-0.0036, 0.0053] |
| theme_only | embedding_cosine_mean | 0.0355 [0.0333, 0.0380] |
| theme_only | embedding_cosine_rank1 | 0.0330 [0.0294, 0.0368] |
| theme_only | catalogue_coverage | 0.0249 [0.0204, 0.0296] |
| theme_only | popularity_bias | 0.0147 [0.0107, 0.0186] |
| theme_only | hubness | -0.0145 [-0.0205, -0.0091] |

### Paired full-system differences @ 20

| baseline | metric | full minus baseline (95% CI) |
|---|---|---:|
| random | theme_jaccard | 0.1188 [0.1155, 0.1221] |
| random | theme_cosine | 0.2715 [0.2643, 0.2788] |
| random | shared_themes_per_result | 1.4058 [1.3761, 1.4334] |
| random | non_generic_share | 0.1203 [0.1042, 0.1372] |
| random | embedding_cosine_mean | 0.0943 [0.0915, 0.0973] |
| random | embedding_cosine_rank1 | 0.1059 [0.1009, 0.1114] |
| random | catalogue_coverage | -0.7096 [-0.7219, -0.6976] |
| random | popularity_bias | 0.0346 [0.0282, 0.0410] |
| random | hubness | 0.2272 [0.2134, 0.2402] |
| popularity | theme_jaccard | 0.1178 [0.1145, 0.1211] |
| popularity | theme_cosine | 0.2730 [0.2655, 0.2810] |
| popularity | shared_themes_per_result | 1.3575 [1.3280, 1.3860] |
| popularity | non_generic_share | 0.1258 [0.1100, 0.1430] |
| popularity | embedding_cosine_mean | 0.0562 [0.0536, 0.0589] |
| popularity | embedding_cosine_rank1 | 0.1044 [0.0999, 0.1088] |
| popularity | catalogue_coverage | 0.2653 [0.2549, 0.2753] |
| popularity | popularity_bias | -0.4659 [-0.4704, -0.4613] |
| popularity | hubness | -0.2673 [-0.2808, -0.2543] |
| genre_only | theme_jaccard | 0.0913 [0.0885, 0.0943] |
| genre_only | theme_cosine | 0.2084 [0.2015, 0.2158] |
| genre_only | shared_themes_per_result | 0.9739 [0.9442, 1.0025] |
| genre_only | non_generic_share | 0.1057 [0.0895, 0.1234] |
| genre_only | embedding_cosine_mean | 0.0413 [0.0373, 0.0462] |
| genre_only | embedding_cosine_rank1 | 0.0429 [0.0369, 0.0489] |
| genre_only | catalogue_coverage | 0.1346 [0.1232, 0.1452] |
| genre_only | popularity_bias | -0.1161 [-0.1240, -0.1086] |
| genre_only | hubness | 0.0228 [0.0068, 0.0387] |
| embedding_only | theme_jaccard | 0.0804 [0.0778, 0.0830] |
| embedding_only | theme_cosine | 0.1783 [0.1720, 0.1845] |
| embedding_only | shared_themes_per_result | 0.8628 [0.8355, 0.8891] |
| embedding_only | non_generic_share | 0.0697 [0.0563, 0.0848] |
| embedding_only | embedding_cosine_mean | -0.0932 [-0.0953, -0.0910] |
| embedding_only | embedding_cosine_rank1 | -0.1325 [-0.1367, -0.1288] |
| embedding_only | catalogue_coverage | -0.1428 [-0.1554, -0.1307] |
| embedding_only | popularity_bias | 0.0078 [0.0011, 0.0146] |
| embedding_only | hubness | 0.0961 [0.0826, 0.1102] |
| theme_only | theme_jaccard | 0.0005 [-0.0000, 0.0011] |
| theme_only | theme_cosine | 0.0029 [0.0015, 0.0043] |
| theme_only | shared_themes_per_result | -0.0125 [-0.0206, -0.0046] |
| theme_only | non_generic_share | 0.0035 [-0.0009, 0.0077] |
| theme_only | embedding_cosine_mean | 0.0343 [0.0323, 0.0367] |
| theme_only | embedding_cosine_rank1 | 0.0330 [0.0294, 0.0368] |
| theme_only | catalogue_coverage | 0.0310 [0.0244, 0.0374] |
| theme_only | popularity_bias | 0.0051 [0.0018, 0.0083] |
| theme_only | hubness | -0.0093 [-0.0123, -0.0062] |

## Gold-pair all-catalogue ranks

45 curated pairs are unreviewed (`reviewed=false`) and both directions are ranked against every target item. Bootstrap CIs resample pairs. Title-mask status: completed for 89 gold items.

### movie -> game

| pair group | method | Hit@5 | Hit@10 | Hit@20 | MRR | median rank |
|---|---|---:|---:|---:|---:|---:|
| franchise_adaptation | random | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.001 [0.000, 0.002] | 3589.000 [1483.000, 3958.000] |
| franchise_adaptation | popularity | 0.105 [0.000, 0.263] | 0.105 [0.000, 0.263] | 0.105 [0.000, 0.263] | 0.087 [0.006, 0.215] | 402.000 [49.000, 1136.000] |
| franchise_adaptation | genre_only | 0.053 [0.000, 0.158] | 0.053 [0.000, 0.158] | 0.105 [0.000, 0.263] | 0.062 [0.005, 0.168] | 264.000 [84.000, 541.000] |
| franchise_adaptation | embedding_only | 0.789 [0.579, 0.947] | 0.842 [0.684, 1.000] | 0.895 [0.737, 1.000] | 0.684 [0.502, 0.845] | 1.000 [1.000, 3.000] |
| franchise_adaptation | theme_only | 0.000 [0.000, 0.000] | 0.053 [0.000, 0.158] | 0.053 [0.000, 0.158] | 0.007 [0.001, 0.018] | 785.000 [392.000, 1161.000] |
| franchise_adaptation | full_system | 0.053 [0.000, 0.158] | 0.053 [0.000, 0.158] | 0.158 [0.000, 0.316] | 0.044 [0.012, 0.100] | 78.000 [40.000, 220.000] |
| franchise_adaptation | embedding_only_title_masked | 0.368 [0.158, 0.579] | 0.421 [0.211, 0.632] | 0.632 [0.421, 0.842] | 0.293 [0.137, 0.482] | 14.000 [3.000, 27.000] |
| franchise_adaptation | full_system_title_masked | 0.053 [0.000, 0.158] | 0.053 [0.000, 0.158] | 0.105 [0.000, 0.263] | 0.024 [0.006, 0.055] | 126.000 [60.000, 606.000] |
| thematic | random | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.001 [0.001, 0.002] | 2627.000 [1410.000, 3220.000] |
| thematic | popularity | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.077 [0.000, 0.192] | 0.012 [0.006, 0.020] | 180.500 [116.000, 453.000] |
| thematic | genre_only | 0.000 [0.000, 0.000] | 0.038 [0.000, 0.115] | 0.115 [0.000, 0.231] | 0.013 [0.003, 0.026] | 781.500 [403.338, 2045.000] |
| thematic | embedding_only | 0.038 [0.000, 0.115] | 0.154 [0.038, 0.308] | 0.423 [0.231, 0.615] | 0.056 [0.036, 0.076] | 27.000 [13.000, 58.500] |
| thematic | theme_only | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.002 [0.001, 0.006] | 2015.000 [1208.000, 2997.000] |
| thematic | full_system | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.038 [0.000, 0.115] | 0.005 [0.002, 0.010] | 551.500 [379.000, 682.000] |
| thematic | embedding_only_title_masked | 0.115 [0.000, 0.269] | 0.269 [0.115, 0.423] | 0.346 [0.192, 0.538] | 0.108 [0.039, 0.202] | 55.000 [16.000, 126.000] |
| thematic | full_system_title_masked | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.038 [0.000, 0.115] | 0.005 [0.002, 0.010] | 669.500 [331.000, 909.000] |

### game -> movie

| pair group | method | Hit@5 | Hit@10 | Hit@20 | MRR | median rank |
|---|---|---:|---:|---:|---:|---:|
| franchise_adaptation | random | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.001 [0.001, 0.002] | 2039.000 [776.000, 3653.000] |
| franchise_adaptation | popularity | 0.053 [0.000, 0.158] | 0.105 [0.000, 0.263] | 0.211 [0.053, 0.421] | 0.033 [0.011, 0.061] | 149.000 [26.000, 1929.000] |
| franchise_adaptation | genre_only | 0.053 [0.000, 0.158] | 0.053 [0.000, 0.158] | 0.105 [0.000, 0.263] | 0.027 [0.006, 0.064] | 269.000 [101.000, 317.000] |
| franchise_adaptation | embedding_only | 0.895 [0.737, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.682 [0.526, 0.833] | 1.000 [1.000, 3.000] |
| franchise_adaptation | theme_only | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.053 [0.000, 0.158] | 0.009 [0.005, 0.015] | 132.000 [103.000, 498.000] |
| franchise_adaptation | full_system | 0.053 [0.000, 0.158] | 0.053 [0.000, 0.158] | 0.316 [0.105, 0.526] | 0.042 [0.025, 0.064] | 37.000 [20.000, 50.000] |
| franchise_adaptation | embedding_only_title_masked | 0.737 [0.526, 0.895] | 0.947 [0.842, 1.000] | 1.000 [1.000, 1.000] | 0.615 [0.446, 0.790] | 1.000 [1.000, 5.000] |
| franchise_adaptation | full_system_title_masked | 0.053 [0.000, 0.158] | 0.053 [0.000, 0.158] | 0.263 [0.104, 0.474] | 0.037 [0.021, 0.059] | 48.000 [24.000, 82.000] |
| thematic | random | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.001 [0.000, 0.001] | 2783.500 [1721.000, 3976.000] |
| thematic | popularity | 0.038 [0.000, 0.115] | 0.038 [0.000, 0.115] | 0.077 [0.000, 0.192] | 0.036 [0.013, 0.075] | 73.500 [44.500, 187.000] |
| thematic | genre_only | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.005 [0.002, 0.008] | 711.500 [301.000, 1069.000] |
| thematic | embedding_only | 0.115 [0.000, 0.231] | 0.115 [0.000, 0.231] | 0.308 [0.154, 0.500] | 0.107 [0.026, 0.224] | 53.000 [25.500, 80.000] |
| thematic | theme_only | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.002 [0.001, 0.004] | 1083.000 [459.500, 2184.500] |
| thematic | full_system | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.007 [0.004, 0.010] | 250.500 [165.875, 421.500] |
| thematic | embedding_only_title_masked | 0.462 [0.269, 0.654] | 0.538 [0.346, 0.731] | 0.731 [0.577, 0.885] | 0.367 [0.219, 0.532] | 8.000 [2.000, 16.000] |
| thematic | full_system_title_masked | 0.000 [0.000, 0.000] | 0.038 [0.000, 0.115] | 0.038 [0.000, 0.115] | 0.014 [0.006, 0.028] | 152.000 [108.000, 275.500] |

Per-pair ranks are saved in `cross_domain_gold_ranks.csv`; the unreviewed candidate-selection notes are in `cross_domain_gold_unresolved.md`.

## Limitations and rerun

These proxy scores establish similarity under the project's own learned representations, not shared-user preference, satisfaction, or causal lift. Genre-only may return short lists when no canonical genre overlaps. The gold set is small and not yet human-reviewed; franchise/adaptation labels and thematic judgments need independent review. Title masking diagnoses title leakage only for these gold items; it does not remove all franchise or metadata cues.

Rerun with `python -m entdec crosseval --profile default --seeds 1000 --k 10` (or `--gold-only`). Runtime: 53.1 s.
