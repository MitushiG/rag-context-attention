"""Exp 1: top-k RAG vs ranked expanded context at three token budgets.

  python scripts/run_exp1_context.py --split pilot
  python scripts/run_exp1_context.py --split main

One dense ranking per question over the pooled corpus. Every context is a prefix of
that ranking (RAG = top 5; 4K/12K/28K = as many ranked passages as fit), so a longer
context only ever adds lower-ranked passages. Closed-book and oracle (the two gold
passages only) are reference conditions, not part of any test.
"""
import argparse
import gc

from hla.contexts import golds, ranked_context
from hla.generation import Generator
from hla.pipelines import answer_with_contexts, format_passages
from hla.retrieval import DenseRetriever
from hla.runtime import SPLITS, load_corpus, load_examples, save, score

RAG_K = 5
# 28K leaves room for the instruction, the question and the answer in Qwen2.5-7B's
# 32,768-token window.
BUDGETS = {"4K": 4_000, "12K": 12_000, "28K": 28_000}
RANK_DEPTH = 600  # more than enough passages to fill 28K tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=SPLITS, required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N questions (debug)")
    args = ap.parse_args()

    examples = load_examples(args.split)[: args.limit]
    corpus = load_corpus()
    docs = {d["doc_id"]: d for d in corpus}

    # Rank first, then free the encoder before vLLM takes the GPU.
    retriever = DenseRetriever(corpus)
    ranked = retriever.search([e["question"] for e in examples], k=RANK_DEPTH)
    del retriever
    gc.collect()
    import torch
    torch.cuda.empty_cache()

    gen = Generator(max_model_len=32768, gpu_memory_utilization=0.85)
    conditions = {
        "closed_book": [None] * len(examples),
        "oracle": [golds(e) for e in examples],
        "rag": [ranked_context(e, r, docs, None, gen.count_tokens, k=RAG_K) for e, r in zip(examples, ranked)],
    }
    for name, budget in BUDGETS.items():
        conditions[name] = [ranked_context(e, r, docs, budget, gen.count_tokens) for e, r in zip(examples, ranked)]

    rows = []
    for cond, ctxs in conditions.items():
        recs = score(answer_with_contexts(gen, examples, ctxs), examples)
        for ex, rank, ctx, r in zip(examples, ranked, ctxs, recs):
            gold_ranks = [rank.index(g) + 1 if g in rank else None for g in ex["gold_ids"]]
            rows.append({
                **r, "condition": cond, "budget_tokens": BUDGETS.get(cond),
                "context_tokens": gen.count_tokens(format_passages(ctx)) if ctx else 0,
                "n_passages": len(ctx or []),
                "n_gold": sum(p["is_gold"] for p in (ctx or [])),
                "gold_ranks": gold_ranks,  # rank of the hop-1 and hop-2 passage
                # Pre-registered subgroup: both gold passages are in the RAG top 5, so
                # every passage added by a longer budget is a non-gold passage.
                "retrieval_complete": all(g is not None and g <= RAG_K for g in gold_ranks),
            })
        em = sum(r["em"] for r in recs) / len(recs)
        print(f"{cond:12s} EM {em:.3f}")
    save(rows, args.split, "exp1.jsonl")


if __name__ == "__main__":
    main()
