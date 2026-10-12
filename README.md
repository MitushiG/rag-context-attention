# Retrieval vs. Long Context: How LLMs Use Evidence

A pre-registered experimental study of how an LLM uses retrieved evidence on multi-step
questions, built from three experiments that share one dataset, model, retriever and
evaluation pipeline:

1. **Retrieval vs. long context.** As more lower-ranked text is added, does accuracy
   fall, and does top-k RAG beat a long context?
2. **Attention to evidence.** Is the model more likely to be right when its attention is
   concentrated on the gold passages?
3. **Evidence geometry.** Does separating the two required passages hurt, even when both
   are present and the length is fixed?

> **Status: in progress.** The framework, the pre-registration and the analysis code are
> written and unit-tested on CPU. **No experiments have been run yet**, so there are no
> results in this repository. `results/main/summary.md` will appear once they are.

## Design

The full pre-registration, with designs, metrics and tests, is in
[`HYPOTHESES.md`](HYPOTHESES.md).

| | Conditions | Primary test |
|---|---|---|
| **Exp 1** | RAG top-5 vs. the top-ranked passages that fit in 4K / 12K / 28K tokens (nested prefixes of one ranking) | among questions whose gold passages are in the top 5: EM falls with log2(context tokens), clustered logistic regression |
| **Exp 2** | attention share on the gold passages, read from the answer tokens of Exp 1 and Exp 3 runs | condition-stratified AUROC for correctness > 0.5 |
| **Exp 3** | 2 gold passages placed early / middle / late × close / far, among 20 passages of fixed length | far vs close, adjusting for location, clustered logistic regression |

The three primary tests are Holm–Bonferroni corrected together. The sample size comes
from a 50-question pilot and a simulation-based power analysis (`scripts/power.py`).

| | |
|---|---|
| Data | MuSiQue-Ans v1.0 dev, 2-hop questions (2 gold + 18 distractor passages each) |
| Corpus | closed pool of every 2-hop dev question's passages |
| Retriever | dense, `e5-base-v2` |
| Generator | `Qwen2.5-7B-Instruct` via vLLM, greedy decoding |
| Attention | SDPA forward pass with per-layer pre-hooks that recompute only the answer rows of attention, so 28K-token contexts fit in memory; verified against eager attention |

## Layout

```
HYPOTHESES.md             pre-registration (amend only in its Amendments section)
src/hla/
  data.py                 MuSiQue 2-hop loader, pilot / main-pool split
  corpus.py               pooled closed corpus
  retrieval.py            dense retriever
  generation.py           vLLM wrapper (greedy)
  pipelines.py            the single QA prompt, batched answering
  contexts.py             ranked-prefix contexts (Exp 1), gold-pair layouts (Exp 3)
  attention.py            memory-efficient answer-row attention probe (Exp 2)
  stats.py                McNemar, Holm, AUROC, stratified AUROC, bootstraps, clustered logit
scripts/
  prepare_data.py         sample + corpus
  run_exp1_context.py     Exp 1
  run_exp3_position.py    Exp 3
  run_exp2_attention.py   Exp 2 (after Exp 1 and Exp 3)
  power.py                pilot -> sample size
  analyze.py              all tests -> results/<split>/summary.md
foundations/minigpt.py    a small GPT from scratch (learning exercise)
tests/                    CPU unit tests
```

## Running

Everything except `power.py`, `analyze.py` and the tests needs a GPU. Colab with an L4
should do; an A100 is safer for the 28K-token runs.

```bash
pip install vllm                        # first: it pins torch
pip install -r requirements.txt && pip install -e .

# MuSiQue needs a manual download: run download_data.sh from
# https://github.com/StonyBrookNLP/musique and copy musique_ans_v1.0_dev.jsonl to data/raw/
python scripts/prepare_data.py          # commit data/samples/*.txt

# 1. pilot (50 questions)
python scripts/run_exp1_context.py --split pilot
python scripts/run_exp3_position.py --split pilot
python scripts/run_exp2_attention.py --split pilot --check
python scripts/analyze.py --split pilot
python scripts/power.py                 # writes data/samples/main_ids.txt; commit it

# 2. main study
python scripts/run_exp1_context.py --split main
python scripts/run_exp3_position.py --split main
python scripts/run_exp2_attention.py --split main
python scripts/analyze.py --split main

pytest                                  # CPU tests
```

Use `--limit 5` on any GPU script for a quick debug run.

## Known limitations

- **Closed corpus.** Retrieval runs over the benchmark's own passages, not full
  Wikipedia, so absolute recall is optimistic.
- **One model, one dataset, one retriever.** Effects may differ elsewhere.
- **Attention is not explanation** (Jain & Wallace, 2019). Exp 2 is correlational.

## References

Lewis et al. 2020 (RAG) · Karpukhin et al. 2020 (DPR) · Trivedi et al. 2022 (MuSiQue) ·
Liu et al. 2023 (Lost in the Middle) · Jain & Wallace 2019 (Attention is not Explanation) ·
Vaswani et al. 2017 (Attention Is All You Need)
