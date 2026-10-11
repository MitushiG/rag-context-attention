# Context, Position and Attention in Retrieval-Augmented LLMs

A pre-registered empirical study of how retrieval-augmented LLMs use evidence:
- when long-context prompting beats RAG, and when it loses;
- whether evidence position and the model's attention explain the difference;
- when single-shot retrieval breaks, and whether iterative retrieval fixes it;
- whether "looking at the page" beats OCR for visual documents.

> **Status: in progress.** The framework, the pre-registration and the analysis code are
> written and unit-tested on CPU. **No experiments have been run yet**, so there are no
> results in this repository. `results/summary.md` will appear once they are.

## Hypotheses

The full pre-registration, with designs, metrics and statistical tests, is in
[`HYPOTHESES.md`](HYPOTHESES.md). It was committed before any experiment was run.

| | Hypothesis | Main test |
|---|---|---|
| H1 | Single-shot retrieval finds hop-1 evidence far more often than later-hop evidence | clustered logistic regression on hop index |
| H2 | IRCoT beats single-shot RAG on multi-hop questions, with gains growing with hop count, and not on NQ | McNemar's test; method × hops interaction |
| H3 | Long-context accuracy decays with distractor length, and RAG overtakes it at a crossover length | logistic regression: log(length) × condition |
| H4 | Accuracy is U-shaped in gold-passage position; multi-hop accuracy falls as gold passages move apart | quadratic / slope terms |
| H5 | Attention share on the gold passage predicts correctness and mirrors the position curve | AUROC; quadratic regression |
| H6 | Page-image retrieval (ColQwen2.5) beats OCR text on charts and tables, but not on text pages | Wilcoxon test per group + interaction CI |
| H7 | Answer log-probability is a usable escalation signal; a router cuts LLM calls vs always-IRCoT | AUROC difference; held-out router evaluation |

H1, H2, H3 and H7 are primary and are Holm–Bonferroni corrected together. H4–H6 are
secondary.

## Setup

| | |
|---|---|
| Data | 500 NQ-Open (single-hop control), 500 HotpotQA (2-hop), 500 MuSiQue (200/150/150 at 2/3/4 hops), ViDoRe v1 subsets |
| Corpus | closed pool of every sampled question's gold + distractor passages |
| Retrieval | BM25 (`bm25s`), dense (`e5-base-v2`), hybrid (reciprocal rank fusion), optional cross-encoder rerank |
| Generator | `Qwen2.5-7B-Instruct` via vLLM, greedy decoding |
| Multi-hop | IRCoT (each reasoning sentence becomes the next retrieval query) |
| Attention | eager-attention forward hooks, answer-token attention mass per passage per layer |
| Multimodal | `colqwen2.5-v0.2` with a from-scratch MaxSim, checked against `colpali-engine` |
| Serving | FastAPI endpoint that answers single-shot and escalates to IRCoT when confidence is low |

## Layout

```
HYPOTHESES.md            pre-registration (amend only in its Amendments section)
src/hla/
  data.py                NQ / HotpotQA / MuSiQue loaders -> one schema, with hop order
  corpus.py              pooled closed corpus
  retrieval.py           BM25, dense, hybrid (RRF), cross-encoder reranker
  generation.py          vLLM wrapper: greedy answers + mean token log-prob
  pipelines.py           closed-book, fixed-context QA, single-shot RAG, IRCoT
  contexts.py            context builders for length (H3), position and gap (H4)
  attention.py           per-passage attention probe (H5)
  multimodal.py          ViDoRe loading, OCR, ColQwen2.5, MaxSim from scratch (H6)
  router.py              confidence router + threshold selection (H7)
  stats.py               bootstrap, McNemar, Holm, AUROC, ECE, clustered logit
scripts/                 prepare_data, run_h1 ... run_h7, analyze
app/server.py            FastAPI router service
foundations/minigpt.py   a small GPT from scratch (exercise skeleton)
tests/                   CPU unit tests
```

## Running

Everything except `analyze.py` and the tests needs a GPU. Colab with an L4 or A100 is enough.

```bash
pip install vllm                        # first: it pins torch
pip install -r requirements.txt && pip install -e .
apt-get install -y tesseract-ocr        # for H6 OCR

# MuSiQue needs a manual download: run download_data.sh from
# https://github.com/StonyBrookNLP/musique and copy musique_ans_v1.0_dev.jsonl to data/raw/
python scripts/prepare_data.py

python scripts/run_h1_retrieval.py
python scripts/run_h2_iteration.py              # closed-book, oracle, single-shot, IRCoT
python scripts/run_h3_h4_context.py --part h3   # A100 recommended (32K contexts)
python scripts/run_h3_h4_context.py --part h4a
python scripts/run_h3_h4_context.py --part h4b
python scripts/run_h5_attention.py              # after h4a
python scripts/run_h6_multimodal.py
python scripts/run_h7_router.py                 # CPU, uses H2 outputs
python scripts/analyze.py                       # CPU -> results/summary.md

uvicorn app.server:app --port 8000              # after H7
pytest                                          # CPU tests
```

Debug any GPU script on a few questions first, e.g. `run_h2_iteration.py --limit 20`.

## Known limitations

- **Closed corpus.** Retrieval runs over the benchmarks' own candidate pools, not full
  Wikipedia, so absolute recall is optimistic. The comparisons between methods are what
  the study relies on.
- **One generator** (Qwen2.5-7B). Effects may differ for other model families and sizes.
- **IRCoT is a simplified re-implementation** (zero-shot prompt, no few-shot demonstrations).
- **Router cost.** The router pays for the single-shot call *plus* IRCoT when it
  escalates, so it only saves calls if it escalates on fewer than about
  (IRCoT calls − 1) / IRCoT calls of the questions.
- **Attention is not explanation** (Jain & Wallace, 2019). H5 is correlational.

## References

Lewis et al. 2020 (RAG) · Karpukhin et al. 2020 (DPR) · Khattab & Zaharia 2020 (ColBERT) ·
Yang et al. 2018 (HotpotQA) · Trivedi et al. 2022 (MuSiQue) · Trivedi et al. 2023 (IRCoT) ·
Jiang et al. 2023 (FLARE) · Jeong et al. 2024 (Adaptive-RAG) · Liu et al. 2023 (Lost in the
Middle) · Faysse et al. 2024 (ColPali) · Jain & Wallace 2019 (Attention is not Explanation)
