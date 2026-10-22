# How Long Should a RAG Prompt Be? Long Context, Attention and Iterative Retrieval

A pre-registered experimental study of two practical questions about retrieval-augmented
LLMs:

1. **How long should the prompt be, and why?** Measure how a model's accuracy changes
   with prompt length when the answer is *always* in the prompt (so retrieval plays no
   part), combine that with how often the retriever finds the answer, and pick the
   prompt length with the best accuracy per token. Then explain the drop with a tiny GPT
   trained from scratch, where each architectural choice can be changed one at a time.
2. **When does iterative retrieval help?** Single-shot RAG vs chain of thought vs IRCoT
   on multi-hop and single-hop questions, separating the gain from *reasoning* from the
   gain from *retrieving again*.

## Results

All experiments were run on one RTX 4090 (Qwen2.5-7B-Instruct, bf16). Full tables and
every test: [`results/summary.md`](results/summary.md). Primary tests are Holm–Bonferroni
corrected together.

| | Finding |
|---|---|
| **H1** (not supported, p = 0.64) | Length alone barely matters: 30K tokens of unrelated text left accuracy at 83% (gold passage alone: 82%). **Similar** distractors are what hurt: −30 points at 30K [−36, −24], worst with the gold passage in the middle (39% vs 65% at the end). |
| **H2** (supported, p < 0.001) | Attention on the gold passage predicts a correct answer: condition-stratified AUROC **0.77** [0.71, 0.83]. |
| **1B** | Real top-k RAG peaks at k = 20 (52%); the predicted best length is not well determined (bootstrap range k = 7–181) because the reader curve is flat. |
| **1C** | In the tiny GPT, attention at the lookup layer collapses from 0.90 (16 facts) to 0.02 (512); fine-tuning on longer inputs raises accuracy at 4× the training length from 36% (equal-compute control) to 96%; log-n scaling recovers part of it with no training. Learned-position and scalable-softmax variants mostly failed to train under the shared recipe (0/3 and 1/3 seeds). |
| **H4** (supported, Holm p = 0.005) | IRCoT beats single-shot RAG on MuSiQue 2-hop: 31.5% vs 21.5%, and has both evidence passages in context twice as often (55% vs 28%). |
| **H6** (not supported, p = 0.20) | IRCoT also helped on single-hop NQ (53.5% vs 47.0%). |

Accuracy is the share of answers that contain an accepted answer (Liu et al., 2023). At
30K tokens the model starts answering in full sentences (65–84% of answers over 8 words),
so exact match falls to 9–13% there; `results/summary.md` reports a lower bound that
counts every long answer as wrong.

## Design

The full pre-registration is in [`HYPOTHESES.md`](HYPOTHESES.md).

| | What | Main test |
|---|---|---|
| **1A** | 200 NQ questions, gold passage always present; length 2K–30K × related vs unrelated distractors × gold at start / middle / end | H1: accuracy falls with log2(length), clustered logistic regression |
| **1B** | retriever recall@k × reader accuracy at the length of k passages → predicted RAG accuracy(k); checked with real top-k RAG | best k = shortest prompt within 1 accuracy point of the best |
| **1 attn** | attention share on the gold passage (100 questions, all 1A conditions) | H2: condition-stratified AUROC for correctness > 0.5 |
| **1C** | tiny GPT on a synthetic key→value lookup; RoPE vs learned vs no positions vs learned length-aware scaling; inference-time log-n scaling on the same weights; fine-tuning on longer inputs vs an equal-length control | H3: descriptive, 3 seeds |
| **2** | 200 MuSiQue 2-hop + 200 NQ; single-shot / single-shot + CoT / IRCoT / oracle | H4: IRCoT > single-shot on MuSiQue (McNemar); H6: gain smaller on NQ (interaction) |

H1, H2, H4 and H6 are primary and Holm–Bonferroni corrected together.

| | |
|---|---|
| Generator | `Qwen2.5-7B-Instruct` via vLLM, greedy decoding |
| Retriever | dense, `e5-base-v2`, over a closed pool of the benchmarks' passages |
| Attention | SDPA forward pass with per-layer pre-hooks that recompute only the answer rows of attention, so 30K-token prompts fit in memory; verified against eager attention |
| Tiny GPT | 4 layers, 4 heads, width 128, written from scratch in PyTorch |

## Layout

```
HYPOTHESES.md                pre-registration (amend only in its Amendments section)
src/hla/
  data.py                    NQ (Lost in the Middle release) and MuSiQue 2-hop loaders
  corpus.py                  pooled closed corpus
  retrieval.py               dense retriever
  generation.py              vLLM wrapper (greedy)
  pipelines.py               direct QA, single-shot chain of thought, IRCoT
  contexts.py                length x distractor type x position contexts, top-k contexts
  attention.py               memory-efficient answer-row attention probe
  stats.py                   McNemar, Holm, AUROC, stratified AUROC, bootstraps, clustered logit
scripts/
  prepare_data.py            samples + corpus
  run_exp1_length.py         Exp 1A and the 1B check runs
  run_exp1_attention.py      Exp 1 attention (after run_exp1_length.py)
  run_exp2_iterative.py      Exp 2
  analyze.py                 all tests -> results/summary.md
foundations/
  minigpt.py                 GPT from scratch on Shakespeare (learning exercise, TODOs)
  tiny_recall.py             Exp 1C: tiny GPT on synthetic lookup, variants + interventions
tests/                       CPU unit tests
```

## Running

On Colab, open [`notebooks/run_on_colab.ipynb`](notebooks/run_on_colab.ipynb): it runs everything below in order,
saves to Google Drive and resumes after a disconnect.

`analyze.py` and the tests run on CPU. `tiny_recall.py` needs only a T4. Everything else
needs a large GPU: an A100 is recommended for the 30K-token runs.

```bash
pip install vllm                        # first: it pins torch
pip install -r requirements.txt && pip install -e .

# MuSiQue needs a manual download: run download_data.sh from
# https://github.com/StonyBrookNLP/musique and copy musique_ans_v1.0_dev.jsonl to data/raw/
python scripts/prepare_data.py          # then commit data/samples/*.txt

python scripts/run_exp1_length.py --limit 5          # debug first
python scripts/run_exp1_length.py
python scripts/run_exp1_attention.py --check --limit 2
python scripts/run_exp1_attention.py
python scripts/run_exp2_iterative.py --limit 20      # read the reasoning chains
python scripts/run_exp2_iterative.py
python foundations/tiny_recall.py --sweep            # T4, ~1 hour
python foundations/tiny_recall.py --finetune         # after --sweep, minutes
python scripts/analyze.py

pytest
```

## Known limitations

- **Closed corpus.** Retrieval runs over the benchmarks' own passages, not full
  Wikipedia, so recall is optimistic and the best k is specific to this setup.
- **One generator, one retriever.** Effects may differ for other models and sizes.
- **IRCoT is a simplified re-implementation** (zero-shot prompt, no demonstrations).
- **Attention is not explanation** (Jain & Wallace, 2019): the Qwen attention results are
  correlational; the tiny-GPT interventions are the causal part, in a much smaller model.

## References

Lewis et al. 2020 (RAG) · Karpukhin et al. 2020 (DPR) · Liu et al. 2023 (Lost in the
Middle) · Trivedi et al. 2022 (MuSiQue) · Trivedi et al. 2023 (IRCoT) · Su et al. 2021
(RoPE) · Nakanishi 2025 (Scalable-Softmax) · Arora et al. 2023 (Zoology, associative
recall) · Jain & Wallace 2019 (Attention is not Explanation) · Vaswani et al. 2017
