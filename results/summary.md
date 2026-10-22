# Results

## Exp 1A: accuracy vs prompt length, gold passage always present

| condition         |   acc |    em |   context_tokens |
|:------------------|------:|------:|-----------------:|
| closed_book       |  0.22 | 0.18  |             0    |
| gold_only         |  0.82 | 0.7   |           125.15 |
| nogold_related_2K |  0.09 | 0.085 |          1897.6  |

Accuracy, related distractors (rows: length, columns: gold position):

| length   |   start |   middle |   end |
|:---------|--------:|---------:|------:|
| 2K       |   0.5   |    0.49  | 0.56  |
| 8K       |   0.465 |    0.42  | 0.515 |
| 16K      |   0.47  |    0.425 | 0.57  |
| 30K      |   0.515 |    0.395 | 0.65  |

Accuracy, unrelated distractors (rows: length, columns: gold position):

| length   |   start |   middle |   end |
|:---------|--------:|---------:|------:|
| 2K       |   0.825 |    0.8   | 0.795 |
| 8K       |   0.795 |    0.76  | 0.8   |
| 16K      |   0.825 |    0.735 | 0.815 |
| 30K      |   0.83  |    0.78  | 0.88  |

**H1 (primary).** logit coefficient on log2(context tokens) = 0.007 (SE 0.020; each doubling multiplies the odds of a correct answer by 1.01), one-sided p = 0.641

- length × distractor type: 0.025 (p = 0.452); > 0 means unrelated text hurts less than related text
- gold only − related 30K (mean over positions): 0.3000 [0.2367, 0.3633], p<1e-4
- gold only − unrelated 30K (mean over positions): -0.0100 [-0.0567, 0.0350], p=0.691

Format compliance (secondary): share of answers over 8 words, and accuracy when long answers are counted as wrong (a lower bound on accuracy):

|                      |   acc |    em |   acc_short_only |   long_answers |   mean_words |
|:---------------------|------:|------:|-----------------:|---------------:|-------------:|
| ('related', '2K')    | 0.517 | 0.46  |            0.502 |          0.057 |        2.658 |
| ('related', '8K')    | 0.467 | 0.418 |            0.453 |          0.037 |        2.64  |
| ('related', '16K')   | 0.488 | 0.418 |            0.465 |          0.063 |        3.158 |
| ('related', '30K')   | 0.52  | 0.128 |            0.187 |          0.643 |       16.19  |
| ('unrelated', '2K')  | 0.807 | 0.702 |            0.77  |          0.045 |        2.613 |
| ('unrelated', '8K')  | 0.785 | 0.693 |            0.76  |          0.037 |        2.542 |
| ('unrelated', '16K') | 0.792 | 0.655 |            0.732 |          0.09  |        3.965 |
| ('unrelated', '30K') | 0.83  | 0.092 |            0.143 |          0.84  |       21.21  |

- H1 refit on the lower bound: log2(tokens) coefficient -0.418 (SE 0.030); not a primary test

## Exp 1B: how long should the RAG prompt be?

**Best k** (shortest prompt within 1 accuracy point of the best predicted): **k = 161** (~24449 tokens), 95% bootstrap range [7, 181]

| k | tokens | retriever recall | predicted accuracy | observed accuracy | accuracy per 1K tokens (predicted) |
|---|---|---|---|---|---|
| 1 | 152 | 0.380 | 0.359 | 0.395 [0.330, 0.465] | 2.365 |
| 2 | 304 | 0.515 | 0.415 |  | 1.366 |
| 3 | 456 | 0.590 | 0.436 |  | 0.956 |
| 5 | 759 | 0.705 | 0.463 | 0.470 [0.405, 0.540] | 0.609 |
| 10 | 1519 | 0.850 | 0.474 |  | 0.312 |
| 20 | 3037 | 0.960 | 0.484 | 0.515 [0.445, 0.585] | 0.159 |
| 50 | 7593 | 0.990 | 0.464 | 0.490 [0.420, 0.555] | 0.061 |
| 100 | 15186 | 0.995 | 0.485 |  | 0.032 |
| 161 | 24449 | 0.995 | 0.508 |  | 0.021 |
| 200 | 30371 | 0.995 | 0.518 |  | 0.017 |

The prediction is checked against real top-k RAG (observed accuracy, 95% CI) at k = 1, 5, 20, 50.

## Exp 1: attention on the gold passage

**H2 (primary).** Condition-stratified AUROC of gold attention share for correctness (2400 rows, 100 questions): 0.767 [0.705, 0.828], one-sided p = < 1e-3 (no bootstrap resample as extreme)

|                      |   gold_share |   gold_length_share |    acc |
|:---------------------|-------------:|--------------------:|-------:|
| ('related', '16K')   |       0.1847 |              0.0076 | 0.53   |
| ('related', '2K')    |       0.3024 |              0.0621 | 0.5333 |
| ('related', '30K')   |       0.1614 |              0.004  | 0.56   |
| ('related', '8K')    |       0.2299 |              0.0151 | 0.5033 |
| ('unrelated', '16K') |       0.4089 |              0.0075 | 0.7867 |
| ('unrelated', '2K')  |       0.665  |              0.0621 | 0.7833 |
| ('unrelated', '30K') |       0.3264 |              0.004  | 0.8267 |
| ('unrelated', '8K')  |       0.5352 |              0.0151 | 0.77   |

- log(gold share) per doubling of length: -0.233 (p = 4.8e-56); a slope of -1 would mean pure 1/n dilution
- robustness, length-adjusted share (log lift): stratified AUROC 0.7866 [0.7261, 0.8450], p<1e-4

## Exp 1C: tiny GPT, multi-query associative recall (trained on N = 8–64 facts)

Excluded (final training accuracy < 0.95): learned_s0, learned_s1, learned_s2, rope_ssmax_s1, rope_ssmax_s2

Lookup layer (largest needle share at N = 16): none s0: 4, none s1: 4, none s2: 4, rope s0: 2, rope s1: 2, rope s2: 2, rope_ssmax s0: 2

acc, related distractors, standard attention (mean ± SD over seeds):

| variant    | 16            | 32            | 64            | 128           | 256           | 512           |
|:-----------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| none       | 0.997 ± 0.004 | 0.994 ± 0.005 | 0.967 ± 0.029 | 0.680 ± 0.038 | 0.152 ± 0.122 | 0.040 ± 0.020 |
| rope       | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.869 ± 0.017 | 0.363 ± 0.062 | 0.087 ± 0.012 |
| rope_ssmax | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.891 ± 0.000 | 0.377 ± 0.000 | 0.178 ± 0.000 |

needle_share_lookup, related distractors, standard attention (mean ± SD over seeds):

| variant    | 16            | 32            | 64            | 128           | 256           | 512           |
|:-----------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| none       | 0.821 ± 0.107 | 0.760 ± 0.120 | 0.604 ± 0.151 | 0.248 ± 0.059 | 0.039 ± 0.037 | 0.006 ± 0.004 |
| rope       | 0.896 ± 0.052 | 0.840 ± 0.067 | 0.735 ± 0.084 | 0.450 ± 0.064 | 0.116 ± 0.032 | 0.015 ± 0.003 |
| rope_ssmax | 0.917 ± 0.000 | 0.892 ± 0.000 | 0.842 ± 0.000 | 0.626 ± 0.000 | 0.201 ± 0.000 | 0.084 ± 0.000 |

acc, unrelated distractors, standard attention (mean ± SD over seeds):

| variant    | 16            | 32            | 64            | 128           | 256           | 512           |
|:-----------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| none       | 0.998 ± 0.002 | 0.997 ± 0.003 | 0.989 ± 0.010 | 0.742 ± 0.033 | 0.202 ± 0.164 | 0.025 ± 0.016 |
| rope       | 1.000 ± 0.000 | 0.999 ± 0.001 | 1.000 ± 0.000 | 0.906 ± 0.036 | 0.426 ± 0.090 | 0.070 ± 0.038 |
| rope_ssmax | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.902 ± 0.000 | 0.391 ± 0.000 | 0.178 ± 0.000 |

needle_share_lookup, unrelated distractors, standard attention (mean ± SD over seeds):

| variant    | 16            | 32            | 64            | 128           | 256           | 512           |
|:-----------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| none       | 0.841 ± 0.102 | 0.786 ± 0.118 | 0.666 ± 0.155 | 0.298 ± 0.065 | 0.041 ± 0.036 | 0.005 ± 0.003 |
| rope       | 0.933 ± 0.030 | 0.892 ± 0.046 | 0.818 ± 0.058 | 0.565 ± 0.057 | 0.185 ± 0.063 | 0.021 ± 0.011 |
| rope_ssmax | 0.951 ± 0.000 | 0.932 ± 0.000 | 0.908 ± 0.000 | 0.689 ± 0.000 | 0.264 ± 0.000 | 0.094 ± 0.000 |

Inference-time log-n scaling, accuracy gain (same weights), mean over seeds [min, max]:

|                             | 128                     | 256                     | 512                     |
|:----------------------------|:------------------------|:------------------------|:------------------------|
| ('none', 'related')         | +0.003 [-0.016, +0.025] | +0.092 [+0.086, +0.105] | +0.038 [+0.008, +0.084] |
| ('none', 'unrelated')       | -0.015 [-0.033, -0.002] | +0.079 [+0.045, +0.102] | +0.049 [+0.000, +0.096] |
| ('rope', 'related')         | +0.020 [+0.006, +0.027] | +0.081 [+0.080, +0.084] | +0.082 [+0.059, +0.105] |
| ('rope', 'unrelated')       | +0.011 [+0.002, +0.018] | +0.059 [+0.029, +0.094] | +0.078 [+0.062, +0.100] |
| ('rope_ssmax', 'related')   | +0.002 [+0.002, +0.002] | +0.062 [+0.062, +0.062] | +0.025 [+0.025, +0.025] |
| ('rope_ssmax', 'unrelated') | +0.000 [+0.000, +0.000] | +0.033 [+0.033, +0.033] | +0.023 [+0.023, +0.023] |

H3 is descriptive (3 seeds per variant) and is not part of the Holm family.

### Exp 1C fine-tuning for longer inputs (500 steps; long: N = 8–256, same: N = 8–64)

acc, related distractors (mean over seeds):

|                                 |    16 |    32 |    64 |   128 |   256 |   512 |
|:--------------------------------|------:|------:|------:|------:|------:|------:|
| ('none', 'base_standard')       | 0.997 | 0.994 | 0.967 | 0.68  | 0.152 | 0.04  |
| ('none', 'base_logn')           | 0.997 | 0.994 | 0.967 | 0.683 | 0.244 | 0.079 |
| ('none', 'ft_same')             | 0.995 | 0.995 | 0.958 | 0.656 | 0.158 | 0.031 |
| ('none', 'ft_long')             | 0.986 | 0.979 | 0.954 | 0.89  | 0.628 | 0.167 |
| ('rope', 'base_standard')       | 1     | 1     | 1     | 0.869 | 0.363 | 0.087 |
| ('rope', 'base_logn')           | 1     | 1     | 1     | 0.889 | 0.445 | 0.169 |
| ('rope', 'ft_same')             | 1     | 1     | 0.999 | 0.854 | 0.355 | 0.076 |
| ('rope', 'ft_long')             | 1     | 1     | 1     | 1     | 0.957 | 0.364 |
| ('rope_ssmax', 'base_standard') | 1     | 1     | 1     | 0.891 | 0.377 | 0.178 |
| ('rope_ssmax', 'base_logn')     | 1     | 1     | 1     | 0.893 | 0.439 | 0.203 |
| ('rope_ssmax', 'ft_same')       | 1     | 1     | 0.998 | 0.877 | 0.379 | 0.143 |
| ('rope_ssmax', 'ft_long')       | 1     | 1     | 1     | 1     | 0.977 | 0.551 |

needle_share_lookup, related distractors (mean over seeds):

|                                 |    16 |    32 |    64 |   128 |   256 |   512 |
|:--------------------------------|------:|------:|------:|------:|------:|------:|
| ('none', 'base_standard')       | 0.821 | 0.76  | 0.604 | 0.248 | 0.039 | 0.006 |
| ('none', 'base_logn')           | 0.821 | 0.76  | 0.604 | 0.314 | 0.074 | 0.017 |
| ('none', 'ft_same')             | 0.817 | 0.753 | 0.589 | 0.233 | 0.039 | 0.005 |
| ('none', 'ft_long')             | 0.747 | 0.698 | 0.581 | 0.421 | 0.171 | 0.028 |
| ('rope', 'base_standard')       | 0.896 | 0.84  | 0.735 | 0.45  | 0.116 | 0.015 |
| ('rope', 'base_logn')           | 0.896 | 0.84  | 0.735 | 0.54  | 0.226 | 0.061 |
| ('rope', 'ft_same')             | 0.899 | 0.84  | 0.734 | 0.45  | 0.121 | 0.016 |
| ('rope', 'ft_long')             | 0.888 | 0.824 | 0.732 | 0.571 | 0.307 | 0.071 |
| ('rope_ssmax', 'base_standard') | 0.917 | 0.892 | 0.842 | 0.626 | 0.201 | 0.084 |
| ('rope_ssmax', 'base_logn')     | 0.917 | 0.892 | 0.842 | 0.672 | 0.271 | 0.124 |
| ('rope_ssmax', 'ft_same')       | 0.919 | 0.888 | 0.84  | 0.617 | 0.218 | 0.082 |
| ('rope_ssmax', 'ft_long')       | 0.857 | 0.82  | 0.784 | 0.692 | 0.474 | 0.212 |

acc, unrelated distractors (mean over seeds):

|                                 |    16 |    32 |    64 |   128 |   256 |   512 |
|:--------------------------------|------:|------:|------:|------:|------:|------:|
| ('none', 'base_standard')       | 0.998 | 0.997 | 0.989 | 0.742 | 0.202 | 0.025 |
| ('none', 'base_logn')           | 0.998 | 0.997 | 0.989 | 0.727 | 0.281 | 0.074 |
| ('none', 'ft_same')             | 0.997 | 0.995 | 0.986 | 0.716 | 0.184 | 0.028 |
| ('none', 'ft_long')             | 0.993 | 0.991 | 0.984 | 0.94  | 0.726 | 0.24  |
| ('rope', 'base_standard')       | 1     | 0.999 | 1     | 0.906 | 0.426 | 0.07  |
| ('rope', 'base_logn')           | 1     | 0.999 | 1     | 0.917 | 0.485 | 0.148 |
| ('rope', 'ft_same')             | 1     | 1     | 0.999 | 0.889 | 0.405 | 0.085 |
| ('rope', 'ft_long')             | 1     | 1     | 0.999 | 0.999 | 0.94  | 0.389 |
| ('rope_ssmax', 'base_standard') | 1     | 1     | 1     | 0.902 | 0.391 | 0.178 |
| ('rope_ssmax', 'base_logn')     | 1     | 1     | 1     | 0.902 | 0.424 | 0.201 |
| ('rope_ssmax', 'ft_same')       | 1     | 1     | 1     | 0.895 | 0.412 | 0.168 |
| ('rope_ssmax', 'ft_long')       | 1     | 1     | 1     | 0.994 | 0.973 | 0.521 |

needle_share_lookup, unrelated distractors (mean over seeds):

|                                 |    16 |    32 |    64 |   128 |   256 |   512 |
|:--------------------------------|------:|------:|------:|------:|------:|------:|
| ('none', 'base_standard')       | 0.841 | 0.786 | 0.666 | 0.298 | 0.041 | 0.005 |
| ('none', 'base_logn')           | 0.841 | 0.786 | 0.666 | 0.363 | 0.08  | 0.012 |
| ('none', 'ft_same')             | 0.84  | 0.781 | 0.656 | 0.274 | 0.039 | 0.005 |
| ('none', 'ft_long')             | 0.779 | 0.751 | 0.658 | 0.503 | 0.228 | 0.037 |
| ('rope', 'base_standard')       | 0.933 | 0.892 | 0.818 | 0.565 | 0.185 | 0.021 |
| ('rope', 'base_logn')           | 0.933 | 0.892 | 0.818 | 0.63  | 0.292 | 0.07  |
| ('rope', 'ft_same')             | 0.938 | 0.897 | 0.824 | 0.568 | 0.19  | 0.023 |
| ('rope', 'ft_long')             | 0.912 | 0.853 | 0.773 | 0.611 | 0.347 | 0.087 |
| ('rope_ssmax', 'base_standard') | 0.951 | 0.932 | 0.908 | 0.689 | 0.264 | 0.094 |
| ('rope_ssmax', 'base_logn')     | 0.951 | 0.932 | 0.908 | 0.714 | 0.319 | 0.133 |
| ('rope_ssmax', 'ft_same')       | 0.954 | 0.936 | 0.91  | 0.691 | 0.281 | 0.1   |
| ('rope_ssmax', 'ft_long')       | 0.894 | 0.868 | 0.828 | 0.719 | 0.493 | 0.21  |

Effect of training on longer inputs (ft_long − ft_same accuracy), mean [min, max] over seeds:

|                             | 16                      | 32                      | 64                      | 128                     | 256                     | 512                     |
|:----------------------------|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|:------------------------|
| ('none', 'related')         | -0.009 [-0.016, +0.000] | -0.016 [-0.023, +0.000] | -0.005 [-0.006, -0.002] | +0.234 [+0.168, +0.334] | +0.469 [+0.322, +0.764] | +0.135 [+0.031, +0.201] |
| ('none', 'unrelated')       | -0.004 [-0.008, -0.002] | -0.004 [-0.008, +0.000] | -0.001 [-0.002, +0.000] | +0.224 [+0.176, +0.301] | +0.542 [+0.391, +0.809] | +0.212 [+0.033, +0.322] |
| ('rope', 'related')         | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.001 [+0.000, +0.002] | +0.146 [+0.133, +0.164] | +0.602 [+0.537, +0.711] | +0.288 [+0.238, +0.324] |
| ('rope', 'unrelated')       | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [-0.002, +0.002] | +0.110 [+0.070, +0.158] | +0.535 [+0.463, +0.650] | +0.305 [+0.301, +0.309] |
| ('rope_ssmax', 'related')   | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.002 [+0.002, +0.002] | +0.123 [+0.123, +0.123] | +0.598 [+0.598, +0.598] | +0.408 [+0.408, +0.408] |
| ('rope_ssmax', 'unrelated') | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.100 [+0.100, +0.100] | +0.561 [+0.561, +0.561] | +0.354 [+0.354, +0.354] |

Descriptive (3 seeds per variant); not part of the Holm family.

## Exp 2: single-shot vs chain of thought vs iterative retrieval

|                           |   acc |    em |    f1 |   all_golds |   llm_calls |   prompt_tokens |   fallback |
|:--------------------------|------:|------:|------:|------------:|------------:|----------------:|-----------:|
| ('musique', 'ircot')      | 0.315 | 0.265 | 0.34  |       0.55  |       2.365 |        2621.08  |      0.01  |
| ('musique', 'oracle')     | 0.625 | 0.57  | 0.688 |       1     |       1     |         346.485 |    nan     |
| ('musique', 'single')     | 0.215 | 0.19  | 0.274 |       0.275 |       1     |         735.99  |    nan     |
| ('musique', 'single_cot') | 0.27  | 0.18  | 0.261 |       0.275 |       1.005 |         737.51  |      0.005 |
| ('nq', 'ircot')           | 0.535 | 0.365 | 0.484 |       0.86  |       2.325 |        2807.86  |      0.02  |
| ('nq', 'oracle')          | 0.815 | 0.69  | 0.803 |       1     |       1     |         203.77  |    nan     |
| ('nq', 'single')          | 0.47  | 0.405 | 0.52  |       0.705 |       1     |         825.755 |    nan     |
| ('nq', 'single_cot')      | 0.45  | 0.24  | 0.382 |       0.705 |       1.005 |         826.03  |      0.005 |

**H4 (primary).** MuSiQue, IRCoT − single-shot = +0.100 (IRCoT-only right 32, single-only right 12), one-sided exact McNemar p = 0.00183

- MuSiQue single_cot − single: +0.055 (two-sided p = 0.0614)
- MuSiQue ircot − single_cot: +0.045 (two-sided p = 0.188)
- H5, single-shot already had both golds (n = 55): IRCoT − single = 0.1273 [-0.0182, 0.2727], p=0.103
- H5, single-shot missed a gold (n = 145): IRCoT − single = 0.0897 [0.0207, 0.1586], p=0.0137
- MuSiQue IRCoT, share of questions with both golds in context after step 0..5: 0.28, 0.52, 0.54, 0.55, 0.55, 0.55

**H6 (primary).** method × dataset interaction (IRCoT gain smaller on NQ): -0.258 (SE 0.199), one-sided p = 0.0976
- NQ IRCoT − single = 0.0650 [0.0150, 0.1150], p=0.0129; 90% CI [0.025, 0.105] not inside ±0.03 (equivalence)
- mean LLM calls, IRCoT vs single: MuSiQue 2.37, NQ 2.33

## Primary tests, Holm–Bonferroni across H1, H2, H4, H6

| test | p | Holm p | reject at 0.05 |
|---|---|---|---|
| H1 | 0.641 | 0.641 | False |
| H2 | < 1e-3 (no bootstrap resample as extreme) | < 1e-3 (no bootstrap resample as extreme) | True |
| H4 | 0.00183 | 0.00549 | True |
| H6 | 0.0976 | 0.195 | False |
