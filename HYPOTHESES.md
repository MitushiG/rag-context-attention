# Pre-registration: How Long Should a RAG Prompt Be, and Does Iterative Retrieval Help?

The design below is kept current: changes are made in place in the relevant section,
and git history records every change and its date. The Amendments section at the bottom
records the earlier redesigns.

This version replaces two earlier designs (see git history and Amendments). It was
written **before any data was downloaded or any experiment was run**.

## Two experiments

| | Question |
|---|---|
| **Exp 1** | How does prompt length affect a model that *always* has the answer in its context, why, and what prompt length is best for RAG once retrieval is taken into account? |
| **Exp 2** | Does iterative retrieval (IRCoT) beat single-shot retrieval on multi-hop questions, is the gain from re-retrieving rather than from reasoning, and does single-hop stay the same? |

**Primary tests:** H1, H2, H4 and H6, corrected together with Holm–Bonferroni at
family-wise α = 0.05. H3 (tiny models, 3 seeds) is descriptive; H5 and everything
marked *secondary* are reported as such.

## Shared setup

| | |
|---|---|
| Single-hop data | NQ-Open, "Lost in the Middle" 20-document release: one gold passage per question, 19 retrieved passages without the answer |
| Multi-hop data | MuSiQue-Ans v1.0 dev, 2-hop, answerable, one hop-1 and one hop-2 gold passage |
| Sample | 200 NQ and 200 MuSiQue questions, drawn once with seed 13 (`scripts/prepare_data.py`); the first 100 NQ questions are the attention subset. No pilot or power analysis: the sample size is fixed by the compute budget. |
| Corpus | every passage of every NQ (release) and MuSiQue 2-hop dev question, de-duplicated (closed corpus) |
| Generator | `Qwen/Qwen2.5-7B-Instruct`, greedy decoding, vLLM, 32K-token window |
| Retriever | `intfloat/e5-base-v2` dense retrieval (the only retriever) |
| Metric | exact match (EM) after SQuAD normalization, best over answer aliases; F1 secondary |

Every question appears in several conditions, so all regressions use question-clustered
robust standard errors and every bootstrap resamples whole questions.

---

## Exp 1A — Accuracy vs prompt length, with the gold passage always present

The gold passage is always in the prompt, so retrieval plays no part.

| Factor | Levels |
|---|---|
| Length | 2K, 8K, 16K, 30K tokens |
| Distractors | **related**: the retriever's ranking for the question, most similar first; **unrelated**: random corpus passages |
| Gold position | start / middle / end |

Distractors never include a passage that contains an answer string. For each question
and distractor type, a longer budget only *adds* passages. Reference conditions:
closed-book, gold passage only, and related distractors at 2K **without** the gold.

**H1 (primary).** Accuracy falls as the prompt gets longer.

    logit P(EM) = β₀ + β₁ · log2(context tokens) + distractor type + gold position

over the 24 length × type × position conditions. One-sided test of **β₁ < 0**.

*Secondary:* length × distractor-type interaction (expected: related text hurts more);
gold only vs 30K per distractor type (paired bootstrap); EM per 1K tokens.

## Exp 1B — The best RAG prompt length

Combines the two curves measured separately:

- **retriever curve** R(k): share of questions whose gold passage is in the top k;
- **reader curve**: EM vs log2(tokens) from Exp 1A (gold only, then related
  distractors averaged over positions), linearly interpolated;
- **gold-absent accuracy** a₀: EM of the "related distractors at 2K, no gold" condition.

Predicted RAG EM(k) = R(k) · reader(tokens of k passages) + (1 − R(k)) · a₀, for
k = 1 … 200. **Best length = the smallest k whose predicted EM is within 1 EM point of
the maximum** (the most accuracy per token without giving up accuracy). 95% interval
for the best k by question-level bootstrap. Predicted EM per 1K tokens is reported for
every k.

*Check:* real top-k RAG at k = 1, 5, 20, 50 (rank order). The prediction is supported
if it falls inside the observed 95% CI at most of these k. This is reported, not tested.

## Exp 1 — Attention on the gold passage (100 questions)

For every Exp 1A condition with the gold passage present (and gold only): teacher-force
the model's own answer and read attention from the positions that predict the answer
tokens. Per layer, average over heads and answer tokens, sum the mass on each passage,
and normalize over passages. **Gold share** = the gold passage's share, averaged over
the upper half of layers (14–27 of 28). Implementation: `hla.attention`, checked against
eager attention.

**H2 (primary).** Gold share predicts correctness: the **condition-stratified AUROC**
(computed within each condition, averaged with weights = number of correct–incorrect
pairs) is > 0.5. One-sided, question-level bootstrap (2,000 resamples). Stratifying
stops length, which lowers both share and accuracy, from producing the AUROC by itself.

*Secondary:* log(gold share) vs log2(length) by distractor type (slope −1 = pure 1/n
dilution); robustness with the length-adjusted share log(share / length share).

**Limit:** on Qwen this is an association, not a cause (Jain & Wallace, 2019). Exp 1C
supplies the interventions.

## Exp 1C — Why: a tiny GPT on a synthetic lookup task

`foundations/tiny_recall.py`. N facts "k k k v", then 8 queries (distinct keys from the
facts), each to be answered with that key's value; earlier queries are followed by
their answer, so one sequence gives 8 training signals. Accuracy and attention are
measured on the final query. Related distractors share the query key's first symbol;
unrelated keys are random. Train on N = 8–64 (drawn per batch; half related, half
unrelated); test on N = 16, 32, 64, 128, 256, 512. 4 layers, 4 heads, width 128;
AdamW (weight decay 0.1, gradient clipping 1.0), one-cycle learning-rate schedule, batch
64; seeds 0, 1, 2. A run counts only if its final training accuracy is ≥ 0.95.

**Training recipe.** The first recipe tried (8,000 steps, learning rate 1e-3, all lengths
from the start) left training accuracy at 0.11, so the recipe is chosen by a training-only
tuning run on seed 99, which is not a study seed: the first of learning rate {1e-3, 3e-3}
× {no curriculum, curriculum} at 20,000 steps to reach ≥ 0.95 training accuracy. The
curriculum raises the largest N linearly from 8 to 64 over the first half of training.
Tuning looks only at training accuracy, never at the length evaluation.
(Design note: a CPU check before any experiment showed single-query training at N = 64
stays at chance, hence multi-query training and mixed lengths.)

| Variant | Change |
|---|---|
| rope | rotary positions (baseline) |
| learned | learned absolute positions; positions beyond training are never trained |
| none | no position encoding |
| rope_ssmax | rope + learned length-aware attention scaling, s · log n (scalable softmax) |

Inference-time intervention on every trained model: multiply attention logits by
max(1, log n / log n_train), which is identical inside the training length.

**Fine-tuning for longer inputs.** Each trained model is then trained for a further 500
steps (learning rate 3e-4, one-cycle) in two ways, each starting from the same weights:
*long*, on N = 8–256 facts, and *same*, on the original N = 8–64, which controls for
the extra training alone. Both are evaluated on the same test sets as above. The effect
of training on longer inputs is *long* − *same*; it is set beside log-n scaling, which
needs no training. This is how context length is extended in practice.

**H3 (descriptive).** (a) *learned* collapses beyond N = 64 while *rope* degrades
gradually; (b) the share of attention on the needle falls as N grows; (c) log-n scaling
(at inference) and *rope_ssmax* (in training) reduce both the attention-share loss and
the accuracy loss beyond the training length; (d) fine-tuning on longer inputs
(*long* − *same*) recovers accuracy at N = 128–256, which it saw in fine-tuning, more
than at N = 512, which it did not, and helps *learned* most, whose positions beyond the
original training length were never trained. Reported as means with the range over
seeds; not part of the Holm family.

---

## Exp 2 — Single-shot vs chain of thought vs iterative retrieval

| Method | |
|---|---|
| single | top 5 passages, direct answer |
| single_cot | the same top 5, reason step by step, then "So the answer is: …" |
| ircot | start from the top 5; each reasoning sentence becomes the next query, adding up to 3 new passages (max 15 passages, max 5 steps) until an answer is stated |
| oracle | gold passages only (reference) |

If a chain of thought never states an answer, one direct-answer call over the same
passages is made (recorded as `fallback`). For every method we record EM, which gold
passages ended up in context (for IRCoT, after every step), LLM calls and tokens.

**H4 (primary).** On MuSiQue, IRCoT is more accurate than single-shot. One-sided exact
McNemar test.

**H5 (secondary).** The gain comes from re-retrieval: it is concentrated in questions
where single-shot missed a gold passage, and near zero where single-shot already had
both. Also reported: single_cot vs single and ircot vs single_cot on MuSiQue (exact
McNemar, two-sided), and the share of questions with both gold passages after each step.

**H6 (primary).** The IRCoT gain is smaller on NQ than on MuSiQue:

    logit P(EM) = β₀ + β₁ · ircot + β₂ · nq + β₃ · ircot × nq   (single and ircot rows)

One-sided test of **β₃ < 0**. *Secondary:* equivalence on NQ, 90% CI of (IRCoT −
single) inside ±3 EM points; with 200 questions this is expected to be inconclusive
unless the difference is very close to 0. Cost per method is reported.

---

## What the results can and cannot say

- Exp 1A isolates the reader: it says nothing about retrieval quality on its own. Exp 1B
  combines it with one retriever on a closed corpus; the best k is specific to them.
- Exp 1C explains mechanisms in a model we control. It does not predict Qwen's numbers.
- Qwen attention results are correlational.
- One generator, one retriever, closed corpus; IRCoT is a simplified, zero-shot version.

---

## Amendments

- **2026-10-11:** Study renamed from "Hops, Length and Attention". Title only.
- **2026-10-12:** Design narrowed from seven hypotheses (H1–H7: per-hop retrieval,
  iterative retrieval, long context vs RAG, position and gap, attention, multimodal
  retrieval, confidence routing) on NQ, HotpotQA, MuSiQue and ViDoRe, to the three
  experiments above on MuSiQue 2-hop only. Reason: scope and compute; the three questions
  share one dataset, model, retriever and evaluation pipeline. Additions: a pilot and a
  power-based sample size instead of fixed sample sizes, the nested-prefix Exp 1 design,
  condition-stratified AUROC for Exp 2, and the midpoint-matched Exp 3 layout.
  **No data had been downloaded or run when this change was made.**
- **2026-10-12 (second change):** Redesigned again, still before any data was downloaded or run, to
  the two experiments above: Exp 1 separates the model's ability to read long prompts
  (single-hop NQ, gold always present, related vs unrelated distractors) from retrieval,
  combines the two to choose a prompt length by accuracy per token, and adds a tiny GPT
  trained from scratch to test mechanisms by intervention; Exp 2 restores the
  single-shot vs CoT vs IRCoT comparison on MuSiQue and NQ. The gold-pair separation
  experiment, the pilot and the power analysis were dropped; samples fixed at 200 + 200.
