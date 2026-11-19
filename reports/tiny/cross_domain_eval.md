# Cross-domain evaluation

> **DEVELOPMENT SAMPLE:** this report uses the tiny profile and is only a sanity check.
Profile: `tiny`. Random seed: `42`. Bootstrap resamples: 1,000.
Gold pairs: 21; title-masked check: completed for 41 gold items.
Runtime: 30.3 s.

## Protocol and interpretation

MovieLens and Steam have no common user IDs, so cross-domain Recall/NDCG against user-held-out ground truth is not defined. The seed proxy draws half uniformly from the source catalogue and half from its top-20% positive-interaction items; the same seed rows are used for every method. Popularity-stratum draws avoid uniform-stratum items when enough top-20% candidates remain; draws use replacement only when a stratum is too small (as can happen on tiny). Each result list is scored against the seed's item themes and cached item embedding.

The `full_system` ranking is the deterministic `cross` command scorer: source theme profile, target-catalogue theme IDF (`ln(n_themed/df)`), and embedding cosine, min-maxed per seed and blended 0.6/0.4. The chatbot's cross-reference path transfers the same weighted theme vector through `extra_theme_weights`; the report omits its query-driven candidate retrieval, LLM intent, and answer generation. No Neo4j or LLM generation is used.

Seed-level proxy metrics use `idf(t)=ln((N+1)/(df(t)+1))+1`, with `N` all items in both domains and document frequency counted once per item. Weighted Jaccard is `sum(min(w_s,w_r)*idf)/sum(max(w_s,w_r)*idf)`; theme cosine is cosine after multiplying each vector weight by IDF. Generic themes are those with IDF at or below the median. Embedding cosine is computed from the cached item vectors.

The optional title mask removes each gold item's leading title/year clause and re-embeds only those gold documents with `qwen3-embedding:4b` at 1024 dimensions; document embeddings use no query-instruction prefix, matching the regular embedder.

Beyond-accuracy metrics aggregate all lists at each cutoff: catalogue coverage is distinct recommended target items / target catalogue size; popularity bias is the mean target-catalogue positive-interaction percentile; hubness is the share of recommendation slots occupied by the ten most-recommended items. Intervals are percentile 95% bootstrap CIs resampled over seed rows; baseline deltas use paired seed resamples.

The embedding-only baseline is favoured by embedding-cosine and theme-only by theme-overlap by construction (metric circularity). Read across all measures, not as independent evidence of user relevance.
Seed-level theme/embedding measures use percentile bootstrap means. The set-valued coverage and hubness measures use centered percentile-bootstrap deviations to reduce occupancy bias from duplicate seed rows in resamples.

## movie -> game: seed proxy results

Seeds: 1000 draws ({'top20_popularity': 500, 'uniform': 500}); catalogues: 500 source / 500 target; runtime: 5.1 s.

### Metrics @ 5

| method | IDF Jaccard | IDF theme cosine | shared themes/result | non-generic share | embedding cosine mean | embedding rank-1 | catalogue coverage | popularity bias | hubness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.0282 [0.0270, 0.0295] | 0.0505 [0.0480, 0.0530] | 0.7700 [0.7426, 0.7980] | 0.0026 [0.0012, 0.0042] | 0.3313 [0.3288, 0.3340] | 0.3308 [0.3270, 0.3348] | 1.000 [0.996, 1.000] | 0.504 [0.496, 0.511] | 0.035 [0.032, 0.039] |
| popularity | 0.0289 [0.0278, 0.0301] | 0.0497 [0.0475, 0.0520] | 0.8710 [0.8428, 0.9010] | 0.0000 [0.0000, 0.0000] | 0.4049 [0.4021, 0.4077] | 0.4410 [0.4376, 0.4447] | 0.010 [0.010, 0.010] | 0.996 [0.996, 0.996] | 1.000 [1.000, 1.000] |
| genre_only | 0.0672 [0.0650, 0.0693] | 0.1323 [0.1277, 0.1370] | 1.5731 [1.5341, 1.6123] | 0.0156 [0.0108, 0.0210] | 0.3821 [0.3793, 0.3852] | 0.3874 [0.3835, 0.3918] | 0.162 [0.153, 0.169] | 0.729 [0.719, 0.740] | 0.470 [0.454, 0.488] |
| embedding_only | 0.0596 [0.0573, 0.0618] | 0.1346 [0.1288, 0.1406] | 1.2732 [1.2368, 1.3096] | 0.0474 [0.0372, 0.0576] | 0.4812 [0.4780, 0.4843] | 0.5076 [0.5039, 0.5116] | 0.544 [0.524, 0.562] | 0.686 [0.677, 0.694] | 0.272 [0.258, 0.285] |
| theme_only | 0.1203 [0.1182, 0.1224] | 0.2079 [0.2035, 0.2121] | 2.4244 [2.3980, 2.4518] | 0.0276 [0.0224, 0.0332] | 0.3196 [0.3170, 0.3222] | 0.3232 [0.3194, 0.3272] | 0.124 [0.115, 0.131] | 0.526 [0.518, 0.534] | 0.831 [0.820, 0.842] |
| full_system | 0.1261 [0.1238, 0.1282] | 0.2326 [0.2273, 0.2376] | 2.4210 [2.3946, 2.4468] | 0.0480 [0.0402, 0.0556] | 0.3509 [0.3480, 0.3539] | 0.3661 [0.3614, 0.3711] | 0.170 [0.158, 0.180] | 0.561 [0.553, 0.569] | 0.727 [0.715, 0.739] |

### Metrics @ 10

| method | IDF Jaccard | IDF theme cosine | shared themes/result | non-generic share | embedding cosine mean | embedding rank-1 | catalogue coverage | popularity bias | hubness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.0288 [0.0278, 0.0299] | 0.0519 [0.0498, 0.0539] | 0.7831 [0.7594, 0.8062] | 0.0027 [0.0016, 0.0040] | 0.3312 [0.3288, 0.3337] | 0.3308 [0.3270, 0.3348] | 1.000 [1.000, 1.000] | 0.504 [0.498, 0.509] | 0.031 [0.029, 0.033] |
| popularity | 0.0360 [0.0348, 0.0373] | 0.0601 [0.0578, 0.0625] | 1.0447 [1.0174, 1.0753] | 0.0000 [0.0000, 0.0000] | 0.3908 [0.3880, 0.3935] | 0.4410 [0.4376, 0.4447] | 0.020 [0.020, 0.020] | 0.991 [0.991, 0.991] | 1.000 [1.000, 1.000] |
| genre_only | 0.0656 [0.0636, 0.0675] | 0.1318 [0.1274, 0.1363] | 1.4963 [1.4597, 1.5328] | 0.0128 [0.0086, 0.0178] | 0.3826 [0.3795, 0.3861] | 0.3874 [0.3835, 0.3918] | 0.248 [0.240, 0.252] | 0.718 [0.709, 0.728] | 0.343 [0.331, 0.355] |
| embedding_only | 0.0552 [0.0533, 0.0570] | 0.1193 [0.1146, 0.1238] | 1.2298 [1.1955, 1.2631] | 0.0288 [0.0228, 0.0349] | 0.4669 [0.4639, 0.4699] | 0.5076 [0.5039, 0.5116] | 0.662 [0.642, 0.682] | 0.682 [0.676, 0.689] | 0.252 [0.242, 0.262] |
| theme_only | 0.1220 [0.1204, 0.1236] | 0.2253 [0.2219, 0.2287] | 2.3922 [2.3710, 2.4154] | 0.0394 [0.0338, 0.0452] | 0.3355 [0.3332, 0.3378] | 0.3232 [0.3194, 0.3272] | 0.186 [0.177, 0.191] | 0.504 [0.497, 0.511] | 0.632 [0.624, 0.641] |
| full_system | 0.1226 [0.1208, 0.1243] | 0.2342 [0.2304, 0.2380] | 2.3525 [2.3308, 2.3758] | 0.0451 [0.0389, 0.0518] | 0.3584 [0.3558, 0.3610] | 0.3661 [0.3614, 0.3711] | 0.266 [0.248, 0.280] | 0.538 [0.531, 0.546] | 0.555 [0.546, 0.564] |

### Metrics @ 20

| method | IDF Jaccard | IDF theme cosine | shared themes/result | non-generic share | embedding cosine mean | embedding rank-1 | catalogue coverage | popularity bias | hubness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.0288 [0.0279, 0.0298] | 0.0517 [0.0499, 0.0535] | 0.7871 [0.7661, 0.8083] | 0.0025 [0.0017, 0.0034] | 0.3313 [0.3291, 0.3337] | 0.3308 [0.3270, 0.3348] | 1.000 [1.000, 1.000] | 0.501 [0.497, 0.505] | 0.028 [0.027, 0.029] |
| popularity | 0.0345 [0.0335, 0.0356] | 0.0591 [0.0572, 0.0613] | 0.9693 [0.9472, 0.9943] | 0.0027 [0.0020, 0.0033] | 0.3743 [0.3716, 0.3771] | 0.4410 [0.4376, 0.4447] | 0.040 [0.040, 0.040] | 0.981 [0.981, 0.981] | 0.500 [0.500, 0.500] |
| genre_only | 0.0607 [0.0589, 0.0624] | 0.1205 [0.1165, 0.1246] | 1.4080 [1.3745, 1.4408] | 0.0123 [0.0082, 0.0174] | 0.3774 [0.3744, 0.3805] | 0.3874 [0.3835, 0.3918] | 0.372 [0.358, 0.378] | 0.699 [0.692, 0.706] | 0.235 [0.227, 0.242] |
| embedding_only | 0.0515 [0.0498, 0.0531] | 0.1082 [0.1042, 0.1124] | 1.1812 [1.1532, 1.2106] | 0.0181 [0.0144, 0.0218] | 0.4518 [0.4490, 0.4546] | 0.5076 [0.5039, 0.5116] | 0.784 [0.764, 0.802] | 0.675 [0.669, 0.680] | 0.215 [0.209, 0.221] |
| theme_only | 0.1121 [0.1107, 0.1136] | 0.2156 [0.2125, 0.2188] | 2.2276 [2.2066, 2.2520] | 0.0348 [0.0304, 0.0394] | 0.3430 [0.3410, 0.3452] | 0.3232 [0.3194, 0.3272] | 0.334 [0.321, 0.343] | 0.501 [0.496, 0.507] | 0.414 [0.409, 0.419] |
| full_system | 0.1129 [0.1114, 0.1144] | 0.2208 [0.2175, 0.2242] | 2.2366 [2.2128, 2.2625] | 0.0378 [0.0328, 0.0430] | 0.3692 [0.3668, 0.3715] | 0.3661 [0.3614, 0.3711] | 0.440 [0.421, 0.455] | 0.538 [0.532, 0.546] | 0.359 [0.354, 0.364] |

### Seed strata @ 10

| stratum | method | IDF Jaccard | IDF theme cosine | shared themes/result | non-generic share | embedding cosine mean | embedding rank-1 | catalogue coverage | popularity bias | hubness |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | random | 0.0285 [0.0270, 0.0301] | 0.0517 [0.0484, 0.0548] | 0.7662 [0.7316, 0.8002] | 0.0026 [0.0012, 0.0044] | 0.3250 [0.3213, 0.3287] | 0.3256 [0.3199, 0.3312] | 1.000 [0.995, 1.000] | 0.502 [0.494, 0.509] | 0.036 [0.033, 0.040] |
| uniform | popularity | 0.0365 [0.0345, 0.0383] | 0.0623 [0.0583, 0.0660] | 1.0406 [0.9944, 1.0828] | 0.0000 [0.0000, 0.0000] | 0.3828 [0.3785, 0.3869] | 0.4284 [0.4225, 0.4340] | 0.020 [0.020, 0.020] | 0.991 [0.991, 0.991] | 1.000 [1.000, 1.000] |
| uniform | genre_only | 0.0691 [0.0658, 0.0723] | 0.1396 [0.1319, 0.1466] | 1.5565 [1.4995, 1.6080] | 0.0182 [0.0124, 0.0246] | 0.3759 [0.3712, 0.3806] | 0.3848 [0.3793, 0.3901] | 0.248 [0.240, 0.254] | 0.715 [0.702, 0.728] | 0.351 [0.332, 0.371] |
| uniform | embedding_only | 0.0572 [0.0544, 0.0601] | 0.1237 [0.1168, 0.1312] | 1.2292 [1.1826, 1.2750] | 0.0258 [0.0180, 0.0342] | 0.4572 [0.4527, 0.4621] | 0.4945 [0.4888, 0.5005] | 0.662 [0.641, 0.681] | 0.675 [0.665, 0.684] | 0.232 [0.220, 0.246] |
| uniform | theme_only | 0.1260 [0.1237, 0.1287] | 0.2331 [0.2282, 0.2390] | 2.4214 [2.3896, 2.4576] | 0.0402 [0.0320, 0.0488] | 0.3310 [0.3275, 0.3344] | 0.3232 [0.3176, 0.3286] | 0.186 [0.176, 0.194] | 0.493 [0.484, 0.502] | 0.612 [0.600, 0.624] |
| uniform | full_system | 0.1277 [0.1247, 0.1304] | 0.2441 [0.2380, 0.2505] | 2.3954 [2.3574, 2.4292] | 0.0482 [0.0400, 0.0590] | 0.3537 [0.3499, 0.3573] | 0.3585 [0.3521, 0.3648] | 0.266 [0.249, 0.281] | 0.529 [0.519, 0.538] | 0.531 [0.518, 0.545] |
| top20_popularity | random | 0.0290 [0.0277, 0.0305] | 0.0520 [0.0493, 0.0548] | 0.8000 [0.7700, 0.8336] | 0.0028 [0.0012, 0.0048] | 0.3373 [0.3340, 0.3408] | 0.3360 [0.3303, 0.3421] | 1.000 [0.997, 1.000] | 0.506 [0.498, 0.514] | 0.034 [0.031, 0.038] |
| top20_popularity | popularity | 0.0355 [0.0338, 0.0373] | 0.0578 [0.0545, 0.0610] | 1.0488 [1.0054, 1.0906] | 0.0000 [0.0000, 0.0000] | 0.3988 [0.3950, 0.4024] | 0.4536 [0.4487, 0.4582] | 0.020 [0.020, 0.020] | 0.991 [0.991, 0.991] | 1.000 [1.000, 1.000] |
| top20_popularity | genre_only | 0.0621 [0.0597, 0.0645] | 0.1239 [0.1182, 0.1296] | 1.4360 [1.3919, 1.4850] | 0.0074 [0.0014, 0.0150] | 0.3894 [0.3844, 0.3940] | 0.3901 [0.3839, 0.3963] | 0.198 [0.194, 0.200] | 0.721 [0.707, 0.736] | 0.351 [0.335, 0.368] |
| top20_popularity | embedding_only | 0.0532 [0.0510, 0.0554] | 0.1148 [0.1087, 0.1205] | 1.2304 [1.1910, 1.2720] | 0.0318 [0.0222, 0.0418] | 0.4765 [0.4729, 0.4801] | 0.5207 [0.5155, 0.5258] | 0.406 [0.399, 0.409] | 0.690 [0.682, 0.699] | 0.272 [0.258, 0.287] |
| top20_popularity | theme_only | 0.1180 [0.1162, 0.1200] | 0.2176 [0.2134, 0.2216] | 2.3630 [2.3342, 2.3938] | 0.0386 [0.0300, 0.0476] | 0.3401 [0.3371, 0.3430] | 0.3232 [0.3178, 0.3289] | 0.138 [0.135, 0.139] | 0.515 [0.504, 0.524] | 0.652 [0.640, 0.664] |
| top20_popularity | full_system | 0.1175 [0.1157, 0.1194] | 0.2242 [0.2202, 0.2286] | 2.3096 [2.2800, 2.3366] | 0.0420 [0.0324, 0.0520] | 0.3631 [0.3597, 0.3665] | 0.3738 [0.3665, 0.3803] | 0.158 [0.155, 0.159] | 0.547 [0.537, 0.558] | 0.588 [0.577, 0.600] |

### Paired full-system differences @ 5

| baseline | metric | full minus baseline (95% CI) |
|---|---|---:|
| random | theme_jaccard | 0.0979 [0.0956, 0.1000] |
| random | theme_cosine | 0.1820 [0.1769, 0.1870] |
| random | shared_themes_per_result | 1.6510 [1.6174, 1.6834] |
| random | non_generic_share | 0.0454 [0.0380, 0.0530] |
| random | embedding_cosine_mean | 0.0196 [0.0169, 0.0225] |
| random | embedding_cosine_rank1 | 0.0354 [0.0296, 0.0407] |
| random | catalogue_coverage | -0.8300 [-0.8416, -0.8196] |
| random | popularity_bias | 0.0573 [0.0469, 0.0684] |
| random | hubness | 0.6920 [0.6794, 0.7042] |
| popularity | theme_jaccard | 0.0972 [0.0950, 0.0992] |
| popularity | theme_cosine | 0.1829 [0.1780, 0.1877] |
| popularity | shared_themes_per_result | 1.5500 [1.5140, 1.5818] |
| popularity | non_generic_share | 0.0480 [0.0402, 0.0556] |
| popularity | embedding_cosine_mean | -0.0540 [-0.0565, -0.0514] |
| popularity | embedding_cosine_rank1 | -0.0749 [-0.0796, -0.0695] |
| popularity | catalogue_coverage | 0.1600 [0.1485, 0.1705] |
| popularity | popularity_bias | -0.4348 [-0.4427, -0.4267] |
| popularity | hubness | -0.2728 [-0.2848, -0.2608] |
| genre_only | theme_jaccard | 0.0589 [0.0565, 0.0613] |
| genre_only | theme_cosine | 0.1003 [0.0944, 0.1061] |
| genre_only | shared_themes_per_result | 0.8479 [0.8117, 0.8858] |
| genre_only | non_generic_share | 0.0324 [0.0240, 0.0402] |
| genre_only | embedding_cosine_mean | -0.0312 [-0.0342, -0.0283] |
| genre_only | embedding_cosine_rank1 | -0.0213 [-0.0266, -0.0156] |
| genre_only | catalogue_coverage | 0.0080 [-0.0044, 0.0216] |
| genre_only | popularity_bias | -0.1682 [-0.1803, -0.1550] |
| genre_only | hubness | 0.2571 [0.2359, 0.2780] |
| embedding_only | theme_jaccard | 0.0665 [0.0644, 0.0687] |
| embedding_only | theme_cosine | 0.0980 [0.0923, 0.1035] |
| embedding_only | shared_themes_per_result | 1.1478 [1.1140, 1.1808] |
| embedding_only | non_generic_share | 0.0006 [-0.0072, 0.0080] |
| embedding_only | embedding_cosine_mean | -0.1303 [-0.1329, -0.1277] |
| embedding_only | embedding_cosine_rank1 | -0.1415 [-0.1460, -0.1365] |
| embedding_only | catalogue_coverage | -0.3740 [-0.3937, -0.3537] |
| embedding_only | popularity_bias | -0.1244 [-0.1347, -0.1146] |
| embedding_only | hubness | 0.4554 [0.4385, 0.4723] |
| theme_only | theme_jaccard | 0.0057 [0.0049, 0.0065] |
| theme_only | theme_cosine | 0.0247 [0.0223, 0.0269] |
| theme_only | shared_themes_per_result | -0.0034 [-0.0140, 0.0070] |
| theme_only | non_generic_share | 0.0204 [0.0160, 0.0252] |
| theme_only | embedding_cosine_mean | 0.0313 [0.0296, 0.0332] |
| theme_only | embedding_cosine_rank1 | 0.0429 [0.0386, 0.0477] |
| theme_only | catalogue_coverage | 0.0460 [0.0379, 0.0539] |
| theme_only | popularity_bias | 0.0347 [0.0298, 0.0396] |
| theme_only | hubness | -0.1040 [-0.1141, -0.0953] |

### Paired full-system differences @ 10

| baseline | metric | full minus baseline (95% CI) |
|---|---|---:|
| random | theme_jaccard | 0.0938 [0.0923, 0.0954] |
| random | theme_cosine | 0.1823 [0.1789, 0.1859] |
| random | shared_themes_per_result | 1.5694 [1.5444, 1.5937] |
| random | non_generic_share | 0.0424 [0.0362, 0.0491] |
| random | embedding_cosine_mean | 0.0272 [0.0254, 0.0291] |
| random | embedding_cosine_rank1 | 0.0354 [0.0296, 0.0407] |
| random | catalogue_coverage | -0.7340 [-0.7516, -0.7196] |
| random | popularity_bias | 0.0343 [0.0255, 0.0438] |
| random | hubness | 0.5240 [0.5151, 0.5334] |
| popularity | theme_jaccard | 0.0866 [0.0849, 0.0881] |
| popularity | theme_cosine | 0.1741 [0.1704, 0.1777] |
| popularity | shared_themes_per_result | 1.3078 [1.2796, 1.3334] |
| popularity | non_generic_share | 0.0451 [0.0389, 0.0518] |
| popularity | embedding_cosine_mean | -0.0324 [-0.0344, -0.0305] |
| popularity | embedding_cosine_rank1 | -0.0749 [-0.0796, -0.0695] |
| popularity | catalogue_coverage | 0.2460 [0.2284, 0.2604] |
| popularity | popularity_bias | -0.4530 [-0.4603, -0.4453] |
| popularity | hubness | -0.4449 [-0.4536, -0.4357] |
| genre_only | theme_jaccard | 0.0570 [0.0552, 0.0590] |
| genre_only | theme_cosine | 0.1024 [0.0977, 0.1071] |
| genre_only | shared_themes_per_result | 0.8562 [0.8266, 0.8876] |
| genre_only | non_generic_share | 0.0323 [0.0247, 0.0396] |
| genre_only | embedding_cosine_mean | -0.0243 [-0.0267, -0.0218] |
| genre_only | embedding_cosine_rank1 | -0.0213 [-0.0266, -0.0156] |
| genre_only | catalogue_coverage | 0.0180 [0.0001, 0.0321] |
| genre_only | popularity_bias | -0.1801 [-0.1909, -0.1694] |
| genre_only | hubness | 0.2119 [0.1980, 0.2271] |
| embedding_only | theme_jaccard | 0.0675 [0.0659, 0.0690] |
| embedding_only | theme_cosine | 0.1149 [0.1109, 0.1188] |
| embedding_only | shared_themes_per_result | 1.1227 [1.0946, 1.1516] |
| embedding_only | non_generic_share | 0.0163 [0.0114, 0.0214] |
| embedding_only | embedding_cosine_mean | -0.1085 [-0.1105, -0.1066] |
| embedding_only | embedding_cosine_rank1 | -0.1415 [-0.1460, -0.1365] |
| embedding_only | catalogue_coverage | -0.3960 [-0.4193, -0.3733] |
| embedding_only | popularity_bias | -0.1444 [-0.1520, -0.1369] |
| embedding_only | hubness | 0.3030 [0.2901, 0.3157] |
| theme_only | theme_jaccard | 0.0006 [0.0002, 0.0010] |
| theme_only | theme_cosine | 0.0089 [0.0076, 0.0102] |
| theme_only | shared_themes_per_result | -0.0397 [-0.0476, -0.0316] |
| theme_only | non_generic_share | 0.0057 [0.0031, 0.0087] |
| theme_only | embedding_cosine_mean | 0.0228 [0.0218, 0.0240] |
| theme_only | embedding_cosine_rank1 | 0.0429 [0.0386, 0.0477] |
| theme_only | catalogue_coverage | 0.0800 [0.0650, 0.0930] |
| theme_only | popularity_bias | 0.0339 [0.0310, 0.0372] |
| theme_only | hubness | -0.0772 [-0.0831, -0.0710] |

### Paired full-system differences @ 20

| baseline | metric | full minus baseline (95% CI) |
|---|---|---:|
| random | theme_jaccard | 0.0841 [0.0829, 0.0853] |
| random | theme_cosine | 0.1691 [0.1664, 0.1722] |
| random | shared_themes_per_result | 1.4495 [1.4318, 1.4677] |
| random | non_generic_share | 0.0353 [0.0307, 0.0403] |
| random | embedding_cosine_mean | 0.0379 [0.0366, 0.0392] |
| random | embedding_cosine_rank1 | 0.0354 [0.0296, 0.0407] |
| random | catalogue_coverage | -0.5600 [-0.5793, -0.5453] |
| random | popularity_bias | 0.0376 [0.0303, 0.0456] |
| random | hubness | 0.3311 [0.3259, 0.3365] |
| popularity | theme_jaccard | 0.0784 [0.0771, 0.0796] |
| popularity | theme_cosine | 0.1617 [0.1585, 0.1648] |
| popularity | shared_themes_per_result | 1.2673 [1.2489, 1.2838] |
| popularity | non_generic_share | 0.0352 [0.0306, 0.0400] |
| popularity | embedding_cosine_mean | -0.0051 [-0.0065, -0.0037] |
| popularity | embedding_cosine_rank1 | -0.0749 [-0.0796, -0.0695] |
| popularity | catalogue_coverage | 0.4000 [0.3807, 0.4147] |
| popularity | popularity_bias | -0.4425 [-0.4487, -0.4352] |
| popularity | hubness | -0.1411 [-0.1461, -0.1359] |
| genre_only | theme_jaccard | 0.0522 [0.0508, 0.0537] |
| genre_only | theme_cosine | 0.1004 [0.0968, 0.1041] |
| genre_only | shared_themes_per_result | 0.8286 [0.8032, 0.8555] |
| genre_only | non_generic_share | 0.0255 [0.0190, 0.0313] |
| genre_only | embedding_cosine_mean | -0.0082 [-0.0099, -0.0064] |
| genre_only | embedding_cosine_rank1 | -0.0213 [-0.0266, -0.0156] |
| genre_only | catalogue_coverage | 0.0680 [0.0466, 0.0866] |
| genre_only | popularity_bias | -0.1604 [-0.1691, -0.1522] |
| genre_only | hubness | 0.1244 [0.1153, 0.1335] |
| embedding_only | theme_jaccard | 0.0614 [0.0602, 0.0625] |
| embedding_only | theme_cosine | 0.1126 [0.1095, 0.1157] |
| embedding_only | shared_themes_per_result | 1.0554 [1.0361, 1.0749] |
| embedding_only | non_generic_share | 0.0197 [0.0164, 0.0230] |
| embedding_only | embedding_cosine_mean | -0.0827 [-0.0840, -0.0813] |
| embedding_only | embedding_cosine_rank1 | -0.1415 [-0.1460, -0.1365] |
| embedding_only | catalogue_coverage | -0.3440 [-0.3673, -0.3232] |
| embedding_only | popularity_bias | -0.1363 [-0.1421, -0.1306] |
| embedding_only | hubness | 0.1437 [0.1362, 0.1506] |
| theme_only | theme_jaccard | 0.0008 [0.0005, 0.0011] |
| theme_only | theme_cosine | 0.0052 [0.0044, 0.0060] |
| theme_only | shared_themes_per_result | 0.0090 [0.0029, 0.0148] |
| theme_only | non_generic_share | 0.0030 [0.0018, 0.0043] |
| theme_only | embedding_cosine_mean | 0.0262 [0.0253, 0.0270] |
| theme_only | embedding_cosine_rank1 | 0.0429 [0.0386, 0.0477] |
| theme_only | catalogue_coverage | 0.1060 [0.0877, 0.1217] |
| theme_only | popularity_bias | 0.0375 [0.0349, 0.0402] |
| theme_only | hubness | -0.0554 [-0.0593, -0.0519] |

## game -> movie: seed proxy results

Seeds: 1000 draws ({'top20_popularity': 500, 'uniform': 500}); catalogues: 500 source / 500 target; runtime: 4.6 s.

### Metrics @ 5

| method | IDF Jaccard | IDF theme cosine | shared themes/result | non-generic share | embedding cosine mean | embedding rank-1 | catalogue coverage | popularity bias | hubness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.0297 [0.0282, 0.0313] | 0.0537 [0.0508, 0.0566] | 0.8112 [0.7742, 0.8522] | 0.0028 [0.0014, 0.0044] | 0.3382 [0.3352, 0.3407] | 0.3357 [0.3316, 0.3397] | 1.000 [0.995, 1.000] | 0.497 [0.490, 0.506] | 0.038 [0.034, 0.042] |
| popularity | 0.0372 [0.0353, 0.0389] | 0.0608 [0.0578, 0.0639] | 1.0132 [0.9676, 1.0566] | 0.0000 [0.0000, 0.0000] | 0.3901 [0.3868, 0.3932] | 0.3768 [0.3729, 0.3805] | 0.010 [0.010, 0.010] | 0.996 [0.996, 0.996] | 1.000 [1.000, 1.000] |
| genre_only | 0.0597 [0.0576, 0.0618] | 0.1229 [0.1185, 0.1271] | 1.3426 [1.2970, 1.3881] | 0.0130 [0.0094, 0.0168] | 0.3880 [0.3835, 0.3917] | 0.4027 [0.3967, 0.4078] | 0.188 [0.174, 0.202] | 0.343 [0.334, 0.352] | 0.673 [0.652, 0.693] |
| embedding_only | 0.0665 [0.0639, 0.0690] | 0.1447 [0.1383, 0.1510] | 1.3826 [1.3378, 1.4220] | 0.0346 [0.0248, 0.0448] | 0.4977 [0.4939, 0.5012] | 0.5221 [0.5179, 0.5259] | 0.512 [0.492, 0.530] | 0.501 [0.493, 0.510] | 0.359 [0.342, 0.378] |
| theme_only | 0.1270 [0.1237, 0.1301] | 0.2653 [0.2580, 0.2725] | 2.1086 [2.0640, 2.1512] | 0.0292 [0.0226, 0.0370] | 0.3732 [0.3689, 0.3773] | 0.3969 [0.3917, 0.4017] | 0.256 [0.238, 0.270] | 0.313 [0.307, 0.320] | 0.693 [0.669, 0.716] |
| full_system | 0.1277 [0.1245, 0.1308] | 0.2686 [0.2616, 0.2755] | 2.1080 [2.0632, 2.1502] | 0.0332 [0.0264, 0.0410] | 0.4015 [0.3983, 0.4048] | 0.4195 [0.4155, 0.4234] | 0.358 [0.337, 0.379] | 0.344 [0.337, 0.352] | 0.648 [0.624, 0.672] |

### Metrics @ 10

| method | IDF Jaccard | IDF theme cosine | shared themes/result | non-generic share | embedding cosine mean | embedding rank-1 | catalogue coverage | popularity bias | hubness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.0296 [0.0282, 0.0311] | 0.0532 [0.0508, 0.0557] | 0.8125 [0.7762, 0.8512] | 0.0033 [0.0021, 0.0045] | 0.3385 [0.3357, 0.3410] | 0.3357 [0.3316, 0.3397] | 1.000 [1.000, 1.000] | 0.496 [0.490, 0.501] | 0.032 [0.030, 0.034] |
| popularity | 0.0345 [0.0326, 0.0361] | 0.0568 [0.0538, 0.0595] | 0.9631 [0.9157, 1.0065] | 0.0000 [0.0000, 0.0000] | 0.3827 [0.3795, 0.3856] | 0.3768 [0.3729, 0.3805] | 0.020 [0.020, 0.020] | 0.991 [0.991, 0.991] | 1.000 [1.000, 1.000] |
| genre_only | 0.0570 [0.0549, 0.0589] | 0.1134 [0.1093, 0.1172] | 1.3141 [1.2680, 1.3550] | 0.0095 [0.0069, 0.0122] | 0.3827 [0.3784, 0.3863] | 0.4027 [0.3967, 0.4078] | 0.296 [0.270, 0.318] | 0.412 [0.405, 0.418] | 0.560 [0.542, 0.577] |
| embedding_only | 0.0628 [0.0604, 0.0651] | 0.1344 [0.1287, 0.1400] | 1.3347 [1.2911, 1.3727] | 0.0260 [0.0186, 0.0337] | 0.4833 [0.4798, 0.4865] | 0.5221 [0.5179, 0.5259] | 0.672 [0.648, 0.694] | 0.514 [0.508, 0.520] | 0.303 [0.289, 0.316] |
| theme_only | 0.1183 [0.1152, 0.1214] | 0.2587 [0.2519, 0.2655] | 1.9923 [1.9473, 2.0340] | 0.0318 [0.0264, 0.0375] | 0.3743 [0.3701, 0.3780] | 0.3969 [0.3917, 0.4017] | 0.416 [0.392, 0.436] | 0.379 [0.374, 0.384] | 0.506 [0.485, 0.528] |
| full_system | 0.1195 [0.1165, 0.1226] | 0.2615 [0.2550, 0.2679] | 2.0152 [1.9712, 2.0553] | 0.0342 [0.0284, 0.0403] | 0.4039 [0.4007, 0.4070] | 0.4195 [0.4155, 0.4234] | 0.556 [0.526, 0.580] | 0.399 [0.393, 0.404] | 0.485 [0.465, 0.504] |

### Metrics @ 20

| method | IDF Jaccard | IDF theme cosine | shared themes/result | non-generic share | embedding cosine mean | embedding rank-1 | catalogue coverage | popularity bias | hubness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.0295 [0.0282, 0.0309] | 0.0531 [0.0508, 0.0555] | 0.8115 [0.7754, 0.8495] | 0.0035 [0.0024, 0.0045] | 0.3386 [0.3358, 0.3410] | 0.3357 [0.3316, 0.3397] | 1.000 [1.000, 1.000] | 0.499 [0.495, 0.503] | 0.027 [0.026, 0.029] |
| popularity | 0.0339 [0.0322, 0.0354] | 0.0599 [0.0573, 0.0625] | 0.9200 [0.8780, 0.9593] | 0.0011 [0.0006, 0.0015] | 0.3779 [0.3748, 0.3808] | 0.3768 [0.3729, 0.3805] | 0.040 [0.040, 0.040] | 0.981 [0.981, 0.981] | 0.500 [0.500, 0.500] |
| genre_only | 0.0556 [0.0536, 0.0574] | 0.1096 [0.1059, 0.1131] | 1.2913 [1.2467, 1.3312] | 0.0077 [0.0057, 0.0096] | 0.3803 [0.3761, 0.3840] | 0.4027 [0.3967, 0.4078] | 0.450 [0.415, 0.475] | 0.424 [0.419, 0.430] | 0.354 [0.344, 0.363] |
| embedding_only | 0.0591 [0.0566, 0.0612] | 0.1241 [0.1190, 0.1290] | 1.2843 [1.2407, 1.3232] | 0.0209 [0.0152, 0.0265] | 0.4671 [0.4638, 0.4702] | 0.5221 [0.5179, 0.5259] | 0.784 [0.763, 0.803] | 0.522 [0.518, 0.527] | 0.238 [0.230, 0.247] |
| theme_only | 0.1107 [0.1077, 0.1135] | 0.2506 [0.2442, 0.2567] | 1.9178 [1.8750, 1.9586] | 0.0344 [0.0274, 0.0418] | 0.3798 [0.3754, 0.3836] | 0.3969 [0.3917, 0.4017] | 0.638 [0.607, 0.663] | 0.422 [0.418, 0.427] | 0.287 [0.276, 0.297] |
| full_system | 0.1107 [0.1078, 0.1136] | 0.2487 [0.2423, 0.2549] | 1.9087 [1.8643, 1.9497] | 0.0362 [0.0291, 0.0439] | 0.4083 [0.4048, 0.4114] | 0.4195 [0.4155, 0.4234] | 0.750 [0.724, 0.774] | 0.442 [0.438, 0.446] | 0.281 [0.271, 0.291] |

### Seed strata @ 10

| stratum | method | IDF Jaccard | IDF theme cosine | shared themes/result | non-generic share | embedding cosine mean | embedding rank-1 | catalogue coverage | popularity bias | hubness |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | random | 0.0279 [0.0259, 0.0300] | 0.0504 [0.0468, 0.0540] | 0.7608 [0.7100, 0.8144] | 0.0038 [0.0020, 0.0058] | 0.3257 [0.3222, 0.3294] | 0.3217 [0.3158, 0.3277] | 1.000 [0.996, 1.000] | 0.498 [0.490, 0.506] | 0.039 [0.035, 0.043] |
| uniform | popularity | 0.0328 [0.0303, 0.0353] | 0.0538 [0.0495, 0.0580] | 0.9154 [0.8512, 0.9800] | 0.0000 [0.0000, 0.0000] | 0.3649 [0.3604, 0.3689] | 0.3557 [0.3503, 0.3604] | 0.020 [0.020, 0.020] | 0.991 [0.991, 0.991] | 1.000 [1.000, 1.000] |
| uniform | genre_only | 0.0537 [0.0505, 0.0568] | 0.1065 [0.1002, 0.1127] | 1.2360 [1.1690, 1.3006] | 0.0092 [0.0056, 0.0134] | 0.3618 [0.3550, 0.3680] | 0.3832 [0.3749, 0.3911] | 0.296 [0.269, 0.317] | 0.404 [0.394, 0.413] | 0.546 [0.519, 0.574] |
| uniform | embedding_only | 0.0574 [0.0540, 0.0608] | 0.1209 [0.1134, 0.1284] | 1.2338 [1.1730, 1.2992] | 0.0162 [0.0094, 0.0242] | 0.4642 [0.4599, 0.4684] | 0.5012 [0.4966, 0.5061] | 0.672 [0.645, 0.697] | 0.518 [0.511, 0.527] | 0.301 [0.283, 0.318] |
| uniform | theme_only | 0.1149 [0.1104, 0.1193] | 0.2543 [0.2445, 0.2640] | 1.9086 [1.8478, 1.9722] | 0.0378 [0.0298, 0.0474] | 0.3566 [0.3498, 0.3629] | 0.3820 [0.3734, 0.3896] | 0.416 [0.389, 0.439] | 0.386 [0.378, 0.396] | 0.422 [0.394, 0.452] |
| uniform | full_system | 0.1153 [0.1110, 0.1197] | 0.2552 [0.2455, 0.2649] | 1.9214 [1.8564, 1.9842] | 0.0388 [0.0310, 0.0478] | 0.3900 [0.3859, 0.3942] | 0.4080 [0.4026, 0.4138] | 0.556 [0.525, 0.583] | 0.413 [0.404, 0.421] | 0.405 [0.377, 0.433] |
| top20_popularity | random | 0.0312 [0.0293, 0.0332] | 0.0561 [0.0525, 0.0596] | 0.8642 [0.8106, 0.9162] | 0.0028 [0.0014, 0.0046] | 0.3513 [0.3484, 0.3543] | 0.3496 [0.3439, 0.3557] | 1.000 [0.996, 1.000] | 0.494 [0.486, 0.502] | 0.034 [0.031, 0.038] |
| top20_popularity | popularity | 0.0361 [0.0336, 0.0385] | 0.0597 [0.0554, 0.0639] | 1.0108 [0.9458, 1.0750] | 0.0000 [0.0000, 0.0000] | 0.4005 [0.3971, 0.4040] | 0.3979 [0.3929, 0.4029] | 0.020 [0.020, 0.020] | 0.991 [0.991, 0.991] | 1.000 [1.000, 1.000] |
| top20_popularity | genre_only | 0.0603 [0.0574, 0.0630] | 0.1203 [0.1151, 0.1257] | 1.3922 [1.3316, 1.4532] | 0.0098 [0.0066, 0.0134] | 0.4035 [0.3997, 0.4073] | 0.4221 [0.4152, 0.4295] | 0.144 [0.142, 0.146] | 0.420 [0.410, 0.430] | 0.573 [0.548, 0.599] |
| top20_popularity | embedding_only | 0.0682 [0.0650, 0.0716] | 0.1480 [0.1401, 0.1560] | 1.4356 [1.3802, 1.4936] | 0.0358 [0.0238, 0.0504] | 0.5024 [0.4982, 0.5065] | 0.5430 [0.5376, 0.5484] | 0.418 [0.407, 0.425] | 0.509 [0.501, 0.518] | 0.318 [0.299, 0.339] |
| top20_popularity | theme_only | 0.1217 [0.1172, 0.1261] | 0.2632 [0.2540, 0.2728] | 2.0760 [2.0170, 2.1314] | 0.0258 [0.0192, 0.0332] | 0.3919 [0.3879, 0.3959] | 0.4117 [0.4058, 0.4170] | 0.248 [0.239, 0.253] | 0.373 [0.367, 0.378] | 0.594 [0.567, 0.622] |
| top20_popularity | full_system | 0.1238 [0.1192, 0.1284] | 0.2678 [0.2587, 0.2771] | 2.1090 [2.0466, 2.1654] | 0.0296 [0.0216, 0.0382] | 0.4178 [0.4140, 0.4219] | 0.4310 [0.4261, 0.4364] | 0.292 [0.280, 0.298] | 0.384 [0.378, 0.391] | 0.565 [0.538, 0.591] |

### Paired full-system differences @ 5

| baseline | metric | full minus baseline (95% CI) |
|---|---|---:|
| random | theme_jaccard | 0.0980 [0.0953, 0.1008] |
| random | theme_cosine | 0.2149 [0.2086, 0.2214] |
| random | shared_themes_per_result | 1.2968 [1.2678, 1.3264] |
| random | non_generic_share | 0.0304 [0.0234, 0.0376] |
| random | embedding_cosine_mean | 0.0634 [0.0601, 0.0666] |
| random | embedding_cosine_rank1 | 0.0839 [0.0793, 0.0890] |
| random | catalogue_coverage | -0.6420 [-0.6648, -0.6208] |
| random | popularity_bias | -0.1535 [-0.1646, -0.1428] |
| random | hubness | 0.6106 [0.5841, 0.6357] |
| popularity | theme_jaccard | 0.0905 [0.0877, 0.0935] |
| popularity | theme_cosine | 0.2078 [0.2013, 0.2148] |
| popularity | shared_themes_per_result | 1.0948 [1.0674, 1.1258] |
| popularity | non_generic_share | 0.0332 [0.0264, 0.0410] |
| popularity | embedding_cosine_mean | 0.0114 [0.0082, 0.0146] |
| popularity | embedding_cosine_rank1 | 0.0427 [0.0388, 0.0468] |
| popularity | catalogue_coverage | 0.3480 [0.3266, 0.3686] |
| popularity | popularity_bias | -0.6521 [-0.6594, -0.6444] |
| popularity | hubness | -0.3516 [-0.3764, -0.3276] |
| genre_only | theme_jaccard | 0.0680 [0.0658, 0.0704] |
| genre_only | theme_cosine | 0.1457 [0.1403, 0.1516] |
| genre_only | shared_themes_per_result | 0.7654 [0.7406, 0.7916] |
| genre_only | non_generic_share | 0.0202 [0.0130, 0.0282] |
| genre_only | embedding_cosine_mean | 0.0136 [0.0100, 0.0174] |
| genre_only | embedding_cosine_rank1 | 0.0169 [0.0115, 0.0222] |
| genre_only | catalogue_coverage | 0.1700 [0.1449, 0.1949] |
| genre_only | popularity_bias | 0.0007 [-0.0094, 0.0117] |
| genre_only | hubness | -0.0244 [-0.0505, 0.0033] |
| embedding_only | theme_jaccard | 0.0613 [0.0588, 0.0637] |
| embedding_only | theme_cosine | 0.1239 [0.1176, 0.1304] |
| embedding_only | shared_themes_per_result | 0.7254 [0.6956, 0.7560] |
| embedding_only | non_generic_share | -0.0014 [-0.0120, 0.0094] |
| embedding_only | embedding_cosine_mean | -0.0962 [-0.0990, -0.0935] |
| embedding_only | embedding_cosine_rank1 | -0.1025 [-0.1062, -0.0991] |
| embedding_only | catalogue_coverage | -0.1540 [-0.1758, -0.1338] |
| embedding_only | popularity_bias | -0.1576 [-0.1680, -0.1468] |
| embedding_only | hubness | 0.2894 [0.2632, 0.3156] |
| theme_only | theme_jaccard | 0.0008 [0.0002, 0.0013] |
| theme_only | theme_cosine | 0.0034 [0.0018, 0.0049] |
| theme_only | shared_themes_per_result | -0.0006 [-0.0090, 0.0070] |
| theme_only | non_generic_share | 0.0040 [0.0012, 0.0070] |
| theme_only | embedding_cosine_mean | 0.0284 [0.0254, 0.0318] |
| theme_only | embedding_cosine_rank1 | 0.0227 [0.0188, 0.0271] |
| theme_only | catalogue_coverage | 0.1020 [0.0825, 0.1205] |
| theme_only | popularity_bias | 0.0307 [0.0265, 0.0353] |
| theme_only | hubness | -0.0442 [-0.0535, -0.0349] |

### Paired full-system differences @ 10

| baseline | metric | full minus baseline (95% CI) |
|---|---|---:|
| random | theme_jaccard | 0.0900 [0.0875, 0.0924] |
| random | theme_cosine | 0.2083 [0.2024, 0.2139] |
| random | shared_themes_per_result | 1.2027 [1.1789, 1.2258] |
| random | non_generic_share | 0.0309 [0.0254, 0.0368] |
| random | embedding_cosine_mean | 0.0654 [0.0630, 0.0680] |
| random | embedding_cosine_rank1 | 0.0839 [0.0793, 0.0890] |
| random | catalogue_coverage | -0.4440 [-0.4739, -0.4199] |
| random | popularity_bias | -0.0971 [-0.1050, -0.0890] |
| random | hubness | 0.4529 [0.4335, 0.4724] |
| popularity | theme_jaccard | 0.0851 [0.0826, 0.0876] |
| popularity | theme_cosine | 0.2047 [0.1990, 0.2109] |
| popularity | shared_themes_per_result | 1.0521 [1.0281, 1.0792] |
| popularity | non_generic_share | 0.0342 [0.0284, 0.0403] |
| popularity | embedding_cosine_mean | 0.0212 [0.0187, 0.0238] |
| popularity | embedding_cosine_rank1 | 0.0427 [0.0388, 0.0468] |
| popularity | catalogue_coverage | 0.5360 [0.5061, 0.5601] |
| popularity | popularity_bias | -0.5924 [-0.5979, -0.5870] |
| popularity | hubness | -0.5153 [-0.5347, -0.4960] |
| genre_only | theme_jaccard | 0.0625 [0.0607, 0.0646] |
| genre_only | theme_cosine | 0.1481 [0.1437, 0.1529] |
| genre_only | shared_themes_per_result | 0.7011 [0.6830, 0.7198] |
| genre_only | non_generic_share | 0.0247 [0.0195, 0.0300] |
| genre_only | embedding_cosine_mean | 0.0212 [0.0181, 0.0248] |
| genre_only | embedding_cosine_rank1 | 0.0169 [0.0115, 0.0222] |
| genre_only | catalogue_coverage | 0.2600 [0.2259, 0.2959] |
| genre_only | popularity_bias | -0.0133 [-0.0221, -0.0046] |
| genre_only | hubness | -0.0754 [-0.0970, -0.0536] |
| embedding_only | theme_jaccard | 0.0567 [0.0548, 0.0587] |
| embedding_only | theme_cosine | 0.1271 [0.1221, 0.1322] |
| embedding_only | shared_themes_per_result | 0.6805 [0.6596, 0.7019] |
| embedding_only | non_generic_share | 0.0082 [0.0026, 0.0146] |
| embedding_only | embedding_cosine_mean | -0.0794 [-0.0814, -0.0774] |
| embedding_only | embedding_cosine_rank1 | -0.1025 [-0.1062, -0.0991] |
| embedding_only | catalogue_coverage | -0.1160 [-0.1443, -0.0883] |
| embedding_only | popularity_bias | -0.1154 [-0.1232, -0.1076] |
| embedding_only | hubness | 0.1821 [0.1622, 0.2012] |
| theme_only | theme_jaccard | 0.0012 [0.0009, 0.0016] |
| theme_only | theme_cosine | 0.0027 [0.0017, 0.0038] |
| theme_only | shared_themes_per_result | 0.0229 [0.0170, 0.0295] |
| theme_only | non_generic_share | 0.0024 [0.0008, 0.0040] |
| theme_only | embedding_cosine_mean | 0.0296 [0.0269, 0.0327] |
| theme_only | embedding_cosine_rank1 | 0.0227 [0.0188, 0.0271] |
| theme_only | catalogue_coverage | 0.1400 [0.1144, 0.1644] |
| theme_only | popularity_bias | 0.0192 [0.0157, 0.0225] |
| theme_only | hubness | -0.0218 [-0.0294, -0.0144] |

### Paired full-system differences @ 20

| baseline | metric | full minus baseline (95% CI) |
|---|---|---:|
| random | theme_jaccard | 0.0811 [0.0789, 0.0833] |
| random | theme_cosine | 0.1956 [0.1903, 0.2008] |
| random | shared_themes_per_result | 1.0973 [1.0772, 1.1169] |
| random | non_generic_share | 0.0327 [0.0260, 0.0400] |
| random | embedding_cosine_mean | 0.0696 [0.0676, 0.0716] |
| random | embedding_cosine_rank1 | 0.0839 [0.0793, 0.0890] |
| random | catalogue_coverage | -0.2500 [-0.2764, -0.2264] |
| random | popularity_bias | -0.0569 [-0.0622, -0.0515] |
| random | hubness | 0.2541 [0.2438, 0.2640] |
| popularity | theme_jaccard | 0.0768 [0.0747, 0.0790] |
| popularity | theme_cosine | 0.1887 [0.1833, 0.1943] |
| popularity | shared_themes_per_result | 0.9887 [0.9689, 1.0083] |
| popularity | non_generic_share | 0.0350 [0.0280, 0.0426] |
| popularity | embedding_cosine_mean | 0.0304 [0.0285, 0.0323] |
| popularity | embedding_cosine_rank1 | 0.0427 [0.0388, 0.0468] |
| popularity | catalogue_coverage | 0.7100 [0.6836, 0.7336] |
| popularity | popularity_bias | -0.5392 [-0.5429, -0.5353] |
| popularity | hubness | -0.2185 [-0.2288, -0.2089] |
| genre_only | theme_jaccard | 0.0551 [0.0534, 0.0569] |
| genre_only | theme_cosine | 0.1390 [0.1348, 0.1434] |
| genre_only | shared_themes_per_result | 0.6175 [0.6018, 0.6349] |
| genre_only | non_generic_share | 0.0285 [0.0222, 0.0354] |
| genre_only | embedding_cosine_mean | 0.0280 [0.0251, 0.0311] |
| genre_only | embedding_cosine_rank1 | 0.0169 [0.0115, 0.0222] |
| genre_only | catalogue_coverage | 0.3000 [0.2650, 0.3390] |
| genre_only | popularity_bias | 0.0176 [0.0102, 0.0246] |
| genre_only | hubness | -0.0724 [-0.0842, -0.0603] |
| embedding_only | theme_jaccard | 0.0516 [0.0500, 0.0533] |
| embedding_only | theme_cosine | 0.1245 [0.1206, 0.1286] |
| embedding_only | shared_themes_per_result | 0.6244 [0.6083, 0.6405] |
| embedding_only | non_generic_share | 0.0153 [0.0114, 0.0192] |
| embedding_only | embedding_cosine_mean | -0.0589 [-0.0604, -0.0575] |
| embedding_only | embedding_cosine_rank1 | -0.1025 [-0.1062, -0.0991] |
| embedding_only | catalogue_coverage | -0.0340 [-0.0631, -0.0071] |
| embedding_only | popularity_bias | -0.0807 [-0.0865, -0.0752] |
| embedding_only | hubness | 0.0430 [0.0325, 0.0531] |
| theme_only | theme_jaccard | -0.0000 [-0.0003, 0.0002] |
| theme_only | theme_cosine | -0.0019 [-0.0027, -0.0012] |
| theme_only | shared_themes_per_result | -0.0091 [-0.0145, -0.0035] |
| theme_only | non_generic_share | 0.0017 [0.0002, 0.0036] |
| theme_only | embedding_cosine_mean | 0.0285 [0.0257, 0.0316] |
| theme_only | embedding_cosine_rank1 | 0.0227 [0.0188, 0.0271] |
| theme_only | catalogue_coverage | 0.1120 [0.0909, 0.1329] |
| theme_only | popularity_bias | 0.0194 [0.0167, 0.0221] |
| theme_only | hubness | -0.0053 [-0.0091, -0.0016] |

## Gold-pair all-catalogue ranks

21 curated pairs are unreviewed (`reviewed=false`) and both directions are ranked against every target item. Bootstrap CIs resample pairs. Title-mask status: completed for 41 gold items.

### movie -> game

| pair group | method | Hit@5 | Hit@10 | Hit@20 | MRR | median rank |
|---|---|---:|---:|---:|---:|---:|
| franchise_adaptation | random | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.006 [0.003, 0.011] | 291.000 [63.000, 468.000] |
| franchise_adaptation | popularity | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.200 [0.000, 0.600] | 0.031 [0.011, 0.055] | 36.000 [13.000, 334.000] |
| franchise_adaptation | genre_only | 0.000 [0.000, 0.000] | 0.400 [0.000, 0.800] | 0.400 [0.000, 0.800] | 0.061 [0.011, 0.114] | 46.000 [7.000, 388.000] |
| franchise_adaptation | embedding_only | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.900 [0.700, 1.000] | 1.000 [1.000, 2.000] |
| franchise_adaptation | theme_only | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.200 [0.000, 0.600] | 0.029 [0.015, 0.049] | 43.000 [15.000, 111.000] |
| franchise_adaptation | full_system | 0.200 [0.000, 0.600] | 0.400 [0.000, 0.800] | 1.000 [1.000, 1.000] | 0.170 [0.078, 0.336] | 11.000 [2.000, 15.000] |
| franchise_adaptation | embedding_only_title_masked | 0.800 [0.400, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.633 [0.367, 0.900] | 2.000 [1.000, 6.000] |
| franchise_adaptation | full_system_title_masked | 0.200 [0.000, 0.600] | 0.200 [0.000, 0.600] | 0.600 [0.200, 1.000] | 0.145 [0.044, 0.325] | 17.000 [2.000, 30.000] |
| thematic | random | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.011 [0.005, 0.019] | 257.500 [109.000, 399.000] |
| thematic | popularity | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.062 [0.000, 0.188] | 0.015 [0.008, 0.024] | 113.000 [50.000, 220.000] |
| thematic | genre_only | 0.062 [0.000, 0.188] | 0.125 [0.000, 0.312] | 0.125 [0.000, 0.312] | 0.036 [0.005, 0.084] | 268.000 [90.000, 347.000] |
| thematic | embedding_only | 0.500 [0.250, 0.750] | 0.625 [0.375, 0.812] | 0.688 [0.438, 0.875] | 0.173 [0.106, 0.237] | 5.500 [4.000, 38.000] |
| thematic | theme_only | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.006 [0.004, 0.009] | 249.000 [159.000, 378.000] |
| thematic | full_system | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.012 [0.009, 0.017] | 109.500 [78.000, 138.000] |
| thematic | embedding_only_title_masked | 0.250 [0.062, 0.500] | 0.375 [0.125, 0.625] | 0.562 [0.312, 0.812] | 0.232 [0.060, 0.422] | 19.500 [8.963, 71.000] |
| thematic | full_system_title_masked | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.011 [0.008, 0.014] | 123.000 [78.000, 156.000] |

### game -> movie

| pair group | method | Hit@5 | Hit@10 | Hit@20 | MRR | median rank |
|---|---|---:|---:|---:|---:|---:|
| franchise_adaptation | random | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.004 [0.003, 0.007] | 314.000 [107.000, 435.000] |
| franchise_adaptation | popularity | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.200 [0.000, 0.600] | 0.024 [0.008, 0.042] | 60.000 [19.000, 344.000] |
| franchise_adaptation | genre_only | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.012 [0.008, 0.016] | 81.000 [56.000, 216.000] |
| franchise_adaptation | embedding_only | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.900 [0.700, 1.000] | 1.000 [1.000, 2.000] |
| franchise_adaptation | theme_only | 0.000 [0.000, 0.000] | 0.400 [0.000, 0.800] | 0.400 [0.000, 0.800] | 0.078 [0.039, 0.119] | 21.000 [7.000, 31.000] |
| franchise_adaptation | full_system | 0.400 [0.000, 0.800] | 0.800 [0.400, 1.000] | 1.000 [1.000, 1.000] | 0.177 [0.099, 0.270] | 9.000 [3.000, 11.000] |
| franchise_adaptation | embedding_only_title_masked | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| franchise_adaptation | full_system_title_masked | 0.400 [0.000, 0.800] | 0.800 [0.400, 1.000] | 1.000 [1.000, 1.000] | 0.179 [0.103, 0.272] | 9.000 [3.000, 11.000] |
| thematic | random | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.011 [0.006, 0.019] | 150.000 [101.000, 297.000] |
| thematic | popularity | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.062 [0.000, 0.188] | 0.016 [0.009, 0.025] | 91.000 [64.000, 191.000] |
| thematic | genre_only | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.062 [0.000, 0.188] | 0.014 [0.007, 0.024] | 139.500 [85.000, 189.000] |
| thematic | embedding_only | 0.375 [0.125, 0.625] | 0.500 [0.311, 0.750] | 0.688 [0.438, 0.875] | 0.292 [0.131, 0.470] | 11.000 [3.500, 25.000] |
| thematic | theme_only | 0.000 [0.000, 0.000] | 0.062 [0.000, 0.188] | 0.125 [0.000, 0.312] | 0.024 [0.010, 0.044] | 149.000 [42.000, 342.000] |
| thematic | full_system | 0.062 [0.000, 0.188] | 0.125 [0.000, 0.312] | 0.375 [0.125, 0.625] | 0.071 [0.032, 0.133] | 23.500 [17.000, 59.000] |
| thematic | embedding_only_title_masked | 0.750 [0.500, 0.938] | 0.812 [0.625, 1.000] | 0.875 [0.688, 1.000] | 0.704 [0.491, 0.884] | 1.000 [1.000, 5.000] |
| thematic | full_system_title_masked | 0.125 [0.000, 0.312] | 0.188 [0.000, 0.375] | 0.500 [0.250, 0.750] | 0.114 [0.040, 0.242] | 22.500 [14.488, 39.000] |

Per-pair ranks are saved in `cross_domain_gold_ranks.csv`; the unreviewed candidate-selection notes are in `cross_domain_gold_unresolved.md`.

## Limitations and rerun

These proxy scores establish similarity under the project's own learned representations, not shared-user preference, satisfaction, or causal lift. Genre-only may return short lists when no canonical genre overlaps. The gold set is small and not yet human-reviewed; franchise/adaptation labels and thematic judgments need independent review. Title masking diagnoses title leakage only for these gold items; it does not remove all franchise or metadata cues.

Rerun with `python -m entdec crosseval --profile tiny --seeds 1000 --k 10` (or `--gold-only`). Runtime: 30.3 s.
