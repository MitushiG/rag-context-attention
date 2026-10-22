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

> **Status: in progress.** The framework, the pre-registration and the analysis code are
> written and unit-tested on CPU. **No experiments have been run yet**, so there are no
> results in this repository. `results/summary.md` will appear once they are.

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
