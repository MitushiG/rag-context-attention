# Pre-registration: Hops, Length and Attention

This file is committed **before any experiment is run**. Its git commit timestamp is the
pre-registration record. Any later change to a hypothesis, metric, sample or test goes in
the **Amendments** section at the bottom, with a date and a reason. Earlier text is never
edited silently.

## Common design

**Data**

| Dataset | Role | Sample |
|---|---|---|
| NQ-Open (Lost-in-the-Middle release, 20 docs/question, 1 gold) | single-hop control | 500 questions |
| HotpotQA (distractor setting, dev) | 2-hop | 500 questions (bridge + comparison, type recorded) |
| MuSiQue-Ans v1.0 (dev) | 2–4 hop | 500 questions: 200 two-hop, 150 three-hop, 150 four-hop |
| ViDoRe (v1 subsets) | multimodal (H6 only) | all queries in the chosen subsets |

- Questions are sampled once with `seed = 13` (`scripts/prepare_data.py`). The sampled IDs are saved to `data/samples/` and committed.
- **Retrieval corpus:** the union of every sampled question's candidate passages (gold + distractors), deduplicated. This is a closed-corpus setting, which keeps indexing cheap. It is a stated limitation: results may not transfer to open Wikipedia-scale retrieval.

**Models**

| Role | Model |
|---|---|
| Generator | `Qwen/Qwen2.5-7B-Instruct`, greedy decoding (temperature 0), max 32 new tokens for answers |
| Dense retriever | `intfloat/e5-base-v2` |
| Sparse retriever | BM25 (`bm25s`) |
| Hybrid | Reciprocal rank fusion of BM25 + dense (k = 60) |
| Visual retriever | `vidore/colqwen2.5-v0.2` |

**Metrics**
- QA: exact match (EM) and token F1 after SQuAD answer normalization, against all gold aliases.
- Primary outcome for binary tests: **EM**.
- Retrieval: Recall@k per gold passage (k ∈ {2, 5, 10, 20}; primary k = 10) and nDCG@5 for ViDoRe.

**Statistics**
- Every condition is run on the same questions, so all comparisons are **paired**.
- Binary paired outcomes: **McNemar's exact test**.
- Continuous paired differences: **paired bootstrap**, 10,000 resamples over questions, percentile 95% confidence interval.
- Trend and interaction hypotheses: **logistic regression** with cluster-robust standard errors, clustered by question ID, because each question contributes several observations.
- **Primary family: H1, H2, H3, H7.** Their primary p-values are corrected together with **Holm–Bonferroni** at α = 0.05.
- H4, H5 and H6 are secondary. They are reported with uncorrected p-values and labeled exploratory.
- Every claim is reported with an effect size and a 95% CI. Null results are reported.

---

## H1: Single-shot retrieval misses later hops *(primary)*

**Claim:** For multi-hop questions, a single retrieval with the original question recovers
hop-1 gold passages more often than later-hop gold passages, and recall declines as hop
index increases.

- **Data:** MuSiQue, where hop order comes from `question_decomposition`; HotpotQA bridge questions, where hop 1 is the gold passage whose title overlaps the question more.
- **Unit:** one row per (question, gold passage, retriever).
- **Outcome:** gold passage retrieved in the top 10 (0/1).
- **Primary test:** logistic regression `retrieved ~ hop_index + C(retriever) + C(dataset)`. One-sided test that the `hop_index` coefficient is < 0.
- **Effect size:** hop-1 minus hop-2 Recall@10 with a paired bootstrap CI.
- **Secondary:** the same analysis per retriever, and whether hybrid > dense > BM25 on hop-2 recall.

## H2: Iteration helps only when hops exist *(primary)*

**Claim:** IRCoT beats single-shot RAG on multi-hop datasets, the gain grows with hop
count, and there is no meaningful gain on NQ.

- **Conditions:** single-shot RAG (top 5, hybrid retriever) vs IRCoT (at most 4 reasoning steps, 3 new passages per step, same retriever).
- **Primary test:** McNemar's test on EM for MuSiQue (all hops pooled), testing that IRCoT is better.
- **Secondary tests:**
  - McNemar's test per dataset.
  - Logistic regression `correct ~ method * hop_count` on MuSiQue. A positive interaction supports "gain grows with hops".
  - On NQ: the 95% CI of the EM difference is reported. Equivalence is claimed only if the CI lies within ±3 points.
- **Cost:** mean LLM calls and wall-clock latency per question are reported for each condition.

## H3: Long context loses to RAG past a crossover length *(primary)*

**Claim:** When the model gets every candidate passage, accuracy falls as distractor-padded
context grows, and top-5 RAG overtakes long-context prompting at some length.

- **Data:** 300 questions, 100 from each dataset.
- **Conditions:**
  - **long-context:** all gold passages + distractors, padded to {2K, 8K, 16K, 32K} tokens of passage text, with gold passages at random positions (seed fixed). Nominal 32K = 31,000 passage tokens, leaving room for the instruction, question and answer within the model's 32,768-token window;
  - **RAG:** top-5 from the hybrid retriever over the same padded pool.
- **Primary test:** logistic regression `correct ~ log2(length) * condition + C(dataset)`. The interaction term tests whether the long-context condition degrades faster with length.
- **Effect size:** the crossover length (where predicted accuracies are equal), with a bootstrap CI over questions. If the curves don't cross inside 2K–32K, this is reported as "no crossover in range".

## H4: Position effects *(secondary)*

**Claim:** (a) Accuracy is U-shaped in the position of the gold passage. (b) For two-hop
questions, accuracy falls as the distance between the two gold passages grows.

- **Design (a):** NQ, 20-passage contexts, gold placed at positions {1, 5, 10, 15, 20}.
- **Design (b):** HotpotQA, 20 passages, the two gold passages separated by {0, 5, 10, 18} positions.
- **Tests:**
  - (a) logistic regression with linear + quadratic position terms; a U-shape is supported if the quadratic coefficient is > 0 and the minimum falls strictly inside the range.
  - (b) slope on distance < 0.

## H5: Attention tracks evidence use *(secondary)*

**Claim:** (a) The share of attention the answer tokens place on the gold passage predicts
whether the answer is correct. (b) That share follows the same position profile as
accuracy in H4.

- **Design:** H4(a) contexts (about 3K tokens). Run a teacher-forced forward pass over the prompt + the model's own greedy answer, with eager attention.
- **Measure:** for each layer, average over heads and answer tokens the attention mass on each passage's token span, normalized by the total mass on all passages.
- **Primary statistic:** the mean over the upper half of the layers. This choice is fixed here, before looking at any data.
- **Tests:**
  - (a) AUROC of gold attention share for predicting correctness, with a bootstrap CI; supported if the CI lower bound is > 0.5.
  - (b) the same quadratic regression as H4(a), with attention share as the outcome.
- **Caveat stated in advance:** attention weights are not a faithful explanation of model behavior (Jain & Wallace, 2019). H5 is correlational.

## H6: Looking beats reading on visual pages *(secondary)*

**Claim:** Page-image retrieval (ColQwen2.5) beats OCR-text retrieval on chart, table and
infographic pages, with no significant difference on text-heavy pages.

- **Data:** ViDoRe v1 subsets, grouped *a priori*:
  - **visual:** InfoVQA, ArxivQA, TAT-DQA, TabFQuAD;
  - **text-heavy:** ShiftProject and the SyntheticDocQA subsets.
- **Conditions:** OCR (Tesseract) + BM25, OCR + e5 dense, ColQwen2.5.
- **Outcome:** per-query nDCG@5.
- **Tests:**
  - Paired Wilcoxon signed-rank test, ColQwen vs best OCR baseline, within each group.
  - Interaction: bootstrap CI on (visual-group gain − text-group gain), which is supported if it is > 0.

## H7: Confidence routing is cost-efficient *(primary)*

**Claim:** (a) The single-shot RAG answer's mean token log-probability discriminates
correct from incorrect answers, but less well on multi-hop questions than on NQ. (b) A
router that escalates to IRCoT only below a confidence threshold matches always-IRCoT
accuracy at lower cost.

- **Tests:**
  - (a) AUROC per dataset with bootstrap CIs. **Primary:** bootstrap test that AUROC(NQ) > AUROC(MuSiQue). Expected calibration error (ECE, 10 bins) is also reported.
  - (b) The threshold is chosen on a **held-out half** of the questions (random split, seed 13) as the lowest-cost threshold whose EM is within 1 point of always-IRCoT. On the other half, report EM and mean LLM calls vs always-single-shot and always-IRCoT, with bootstrap CIs.

---

## Amendments

_None yet._
