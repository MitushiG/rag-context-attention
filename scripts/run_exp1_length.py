"""Exp 1: how does prompt length affect a model that always has the answer in context?

  python scripts/run_exp1_length.py --limit 5     # debug
  python scripts/run_exp1_length.py

200 NQ questions. Conditions (one batch each):
  closed_book            no passages
  gold_only              the gold passage alone
  nogold_related_2K      related distractors, no gold (accuracy when retrieval misses)
  {related,unrelated}_{2K,8K,16K,30K}_{start,middle,end}
                         gold + distractors up to the budget, gold at that position
  rag_k{1,5,20,50}       the retriever's real top-k, in rank order (checks Exp 1B)

Also saves each question's ranking (results/exp1/rankings.jsonl) for the retriever curve.
"""
import argparse
import gc
import random

from hla.contexts import POSITIONS, TokenCounter, distractors, length_context, topk_context
from hla.generation import Generator
from hla.pipelines import answer_with_contexts, format_passages
from hla.retrieval import DenseRetriever
from hla.data import read_jsonl, write_jsonl
from hla.runtime import RESULTS, load_corpus, load_examples, save, score

# 30K leaves room for the instruction, question and answer in the 32,768-token window.
LENGTHS = {"2K": 2_000, "8K": 8_000, "16K": 16_000, "30K": 30_000}
KINDS = ("related", "unrelated")
RAG_K = (1, 5, 20, 50)
RANK_DEPTH = 1500  # enough related passages to fill 30K tokens after exclusions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="first N questions (debug)")
    ap.add_argument("--gpu-mem", type=float, default=0.85,
                    help="vLLM GPU memory share; lower it if another job shares the GPU")
    args = ap.parse_args()

    examples = load_examples("nq_ids.txt")[: args.limit]
    corpus = load_corpus()
    docs = {d["doc_id"]: d for d in corpus}
    all_ids = list(docs)

    # Rank first, then free the encoder before vLLM takes the GPU.
    retriever = DenseRetriever(corpus)
    ranked = retriever.search([e["question"] for e in examples], k=RANK_DEPTH)
    del retriever
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    save([{"qid": e["qid"], "gold_rank": (r.index(e["gold_ids"][0]) + 1) if e["gold_ids"][0] in r else None,
           "top": r[:200]} for e, r in zip(examples, ranked)], "exp1", "rankings.jsonl")

    gen = Generator(max_model_len=32768, gpu_memory_utilization=args.gpu_mem)
    cost = TokenCounter(gen.count_tokens)

    conds: dict[str, tuple[dict, list]] = {}
    conds["closed_book"] = ({}, [None] * len(examples))
    conds["gold_only"] = ({}, [length_context(e, [], None, "start", cost) for e in examples])
    nogold = []
    for e, r in zip(examples, ranked):
        ctx, total = [], 0
        for p in distractors(e, "related", r, all_ids, docs):
            if total + cost(p) > LENGTHS["2K"]:
                break
            ctx.append(p)
            total += cost(p)
        random.Random(f"13-{e['qid']}-nogold").shuffle(ctx)
        nogold.append(ctx)
    conds["nogold_related_2K"] = ({"kind": "related", "length": "2K"}, nogold)
    for kind in KINDS:
        for name, budget in LENGTHS.items():
            for pos in POSITIONS:
                conds[f"{kind}_{name}_{pos}"] = (
                    {"kind": kind, "length": name, "budget": budget, "position": pos},
                    [length_context(e, distractors(e, kind, r, all_ids, docs), budget, pos, cost)
                     for e, r in zip(examples, ranked)],
                )
    for k in RAG_K:
        conds[f"rag_k{k}"] = ({"k": k}, [topk_context(e, r, docs, k) for e, r in zip(examples, ranked)])

    # Each condition is saved as it finishes, so a disconnected Colab session resumes
    # where it stopped. --limit runs write to a separate folder.
    part_dir = RESULTS / "exp1" / ("parts" if args.limit is None else f"parts_limit{args.limit}")
    rows = []
    for cond, (meta, ctxs) in conds.items():
        part = part_dir / f"{cond}.jsonl"
        if part.exists():
            rows += read_jsonl(part)
            print(f"{cond:24s} (done, loaded)")
            continue
        recs = score(answer_with_contexts(gen, examples, ctxs), examples)
        cond_rows = []
        for ctx, r in zip(ctxs, recs):
            gold_at = [i for i, p in enumerate(ctx or [], start=1) if p["is_gold"]]
            cond_rows.append({
                **r, "condition": cond, **meta,
                "context_tokens": gen.count_tokens(format_passages(ctx)) if ctx else 0,
                "n_passages": len(ctx or []),
                "gold_in_context": bool(gold_at), "gold_index": gold_at[0] if gold_at else None,
            })
        write_jsonl(cond_rows, part)
        rows += cond_rows
        print(f"{cond:24s} acc {sum(r['acc'] for r in recs) / len(recs):.3f}  EM {sum(r['em'] for r in recs) / len(recs):.3f}")
    save(rows, "exp1", "exp1.jsonl")


if __name__ == "__main__":
    main()
