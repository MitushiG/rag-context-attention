# Pre-registration: Retrieval vs. Long Context — How LLMs Use Evidence

Any later change to a question, metric, sample or test goes in the **Amendments**
section at the bottom, with a date and a reason. Earlier text is never edited.

This version replaces an earlier seven-hypothesis design (see git history). The study
was narrowed to three experiments **before any data was collected**; see Amendments.

## Three questions

| | Question | Level |
|---|---|---|
| **Exp 1** | As more lower-ranked text is added to the context, does accuracy fall, and does top-k retrieval beat long context? | system behaviour |
| **Exp 2** | Is the model more likely to answer correctly when its attention is concentrated on the gold passages? | internal behaviour |
| **Exp 3** | Does separating the two required passages hurt, even when both are present and the context length is fixed? | evidence geometry |

Each experiment has **one primary test** (P1, P2, P3). The three primary p-values are
corrected together with Holm–Bonferroni at family-wise α = 0.05. Everything else is
secondary and is reported as such.

## Shared setup

| | |
|---|---|
| Dataset | MuSiQue-Ans v1.0 dev, **2-hop questions only** (answerable; exactly 2 gold passages with hop order 1, 2; at least 18 distractors) |
| Corpus | every passage of every 2-hop dev question, de-duplicated (closed corpus) |
| Generator | `Qwen/Qwen2.5-7B-Instruct`, greedy decoding, ≤ 32 new tokens, vLLM |
| Retriever | `intfloat/e5-base-v2` dense retrieval, the study's only retriever |
| Prompt | one fixed QA prompt for every condition (`hla.pipelines.qa_body`) |
| Primary metric | exact match (EM) after SQuAD normalization, best over answer aliases; F1 is secondary |

MuSiQue 2-hop is used because its questions were built so a single passage is not
enough. Every 2-hop question has the same structure (two gold passages and 18
distractors), which Exp 3 needs.

**Sample.** `scripts/prepare_data.py` shuffles the eligible questions once (seed 13).
The first 50 are the **pilot**; the rest, in that fixed order, are the main pool.

**Pilot (50 questions).** Run all three experiments on the pilot to check the pipeline
and to measure base rates. Pilot questions are never part of the main analysis. After
the pilot, only two things may change, and both are logged here as amendments before
the main run:
1. feasibility settings (the token budgets, if a budget fails on the available GPU);
2. the prompt, only if answers are systematically unparseable (e.g. the model writes
   sentences instead of short answers).

**Sample size.** `scripts/power.py` simulates P1 and P3 with the pilot's base rates, a
per-question random intercept (SD 1.5 on the logit scale) and the same clustered
regressions used in the analysis. The smallest effects of interest are:
- P1: an 8-point EM drop from RAG to the 28K budget among retrieval-complete questions;
- P3: a 5-point EM drop from close to far gold passages.

N is the smallest value in {200, 300, 400, 500, 600} at which both P1 and P3 reach 80%
power at one-sided α = 0.05/3 (the strictest Holm step). If none does, N = 600, the
largest sample the compute budget allows, and the achieved power is reported. The main
sample is the **first N questions of the main pool**, so it does not depend on what the
pilot showed. `data/samples/main_ids.txt` and `results/pilot/power.md` are committed
before the main run.

**Clustering.** Every question appears in several conditions, so every regression uses
question-clustered robust standard errors and every bootstrap resamples whole questions.

---

## Exp 1 — Retrieval vs. long context

**Design.** For each question, rank the whole corpus once with the retriever. Each
condition gives the model a **prefix of that ranking**:

| Condition | Context |
|---|---|
| RAG | top 5 passages (~0.7K tokens) |
| 4K / 12K / 28K | as many top-ranked passages as fit in 4,000 / 12,000 / 28,000 tokens |
| closed-book, oracle | no passages / the two gold passages only (reference points, not tested) |

Because contexts are nested prefixes, a longer budget only ever *adds* lower-ranked
passages. For every condition we record how many gold passages the context contains and
the rank of each gold passage, so retrieval failures (gold absent) can be separated from
evidence-use failures (gold present, answer wrong).

**P1 (primary).** Among **retrieval-complete** questions (both gold passages are in the
RAG top 5, hence in every condition), every passage added by a longer budget is a
non-gold passage. We test whether this extra text lowers accuracy:

    logit P(EM) = β₀ + β₁ · log2(context tokens),   rows = RAG, 4K, 12K, 28K

with question-clustered SEs. **One-sided test of β₁ < 0.**

**Secondary.**
- All questions: RAG vs each budget, exact McNemar (two-sided), Holm-corrected within
  these three comparisons, with paired-bootstrap CIs on the EM difference. This is the
  system-level answer to "does retrieval beat long context?", where longer contexts can
  also *gain* gold passages that the top 5 missed.
- Interaction: `em ~ log2(tokens) × both-golds-present`, to see whether the length
  penalty differs when the evidence is complete.
- EM by condition × number of gold passages present.
- The best condition, with a question-level bootstrap distribution. A "crossover" is
  reported only if this distribution supports one.

---

## Exp 2 — Does attention to gold evidence predict correctness?

**Design.** No new generations. For every Exp 1 row (RAG, 4K, 12K, 28K) **with both
gold passages in context**, and every Exp 3 row, re-run the model on the same prompt
with its own answer appended (teacher forcing) and read attention from the positions
that predict the answer tokens.

**Attention share (fixed before any data).** For each layer: average the attention
rows of the answer-predicting positions over heads and answer tokens; sum the mass on
each passage's tokens; normalize so the passages sum to 1. The **gold share** is the
sum over the two gold passages, averaged over the upper half of layers (layers 14–27 of
28). Implementation: `hla.attention`, checked against eager attention.

**P2 (primary).** Using the Exp 1 rows: the **condition-stratified AUROC** of gold share
for EM (AUROC computed within each of RAG / 4K / 12K / 28K and averaged, weighted by the
number of correct–incorrect pairs). Stratifying stops context length, which lowers both
attention share and accuracy, from producing a high AUROC on its own. 95% CI by
question-level bootstrap (2,000 resamples). **One-sided test of AUROC > 0.5.**

**Secondary.**
- AUROC within each condition; mean gold share vs the share the gold passages would get
  if attention were proportional to their length.
- `em ~ share_z × log2(tokens)`: does the association change with context length?
- Robustness: stratified AUROC of the length-adjusted share, log(gold share / length share).
- Exp 3 rows: does gold share change with location and separation? (OLS, clustered.)

**Interpretation limit.** This is an association. A high AUROC does not show that
attention *causes* correct answers: attention weights are not a faithful explanation
(Jain & Wallace, 2019), and no intervention is made.

---

## Exp 3 — Does the relative position of the two gold passages matter?

**Design.** 20 passages per question: its 18 own distractors in a fixed (seeded) order,
plus its two gold passages. Both gold passages are **always present**. Only their slots
change; the passages, their order otherwise, and the total length are identical across
conditions. Close and far pairs share the same midpoint, so location and separation
vary independently:

| Location (pair midpoint) | Close (0 passages between) | Far (8 passages between) |
|---|---|---|
| Early (5.5) | slots 5, 6 | slots 1, 10 |
| Middle (10.5) | slots 10, 11 | slots 6, 15 |
| Late (15.5) | slots 15, 16 | slots 11, 20 |

Which hop comes first is drawn once per question (seeded), kept across all six
conditions, and recorded.

**P3 (primary).**

    logit P(EM) = β₀ + location effects + β_far · far

with question-clustered SEs. **One-sided test of β_far < 0** (far apart is worse, after
accounting for location).

**Secondary.**
- Location × separation interaction (Wald test, 2 df).
- Close vs far within each location (exact McNemar).
- Whether hop-1-first order matters, and whether it changes the separation effect.

---

## What the three results can and cannot say

- Exp 1 isolates the effect of extra *lower-ranked* text on one retriever's rankings over
  a closed corpus. It does not cover random or adversarial filler, or open-web retrieval.
- Exp 2 is correlational (see above).
- Exp 3 holds length and content fixed, so a separation effect cannot be explained by
  missing evidence or longer inputs.
- One model, one dataset, one retriever: effects may differ for other models and sizes.

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
