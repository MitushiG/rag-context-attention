"""Exp 2: single-shot RAG vs single-shot + chain of thought vs IRCoT, on multi-hop
(MuSiQue 2-hop) and single-hop (NQ) questions.

  python scripts/run_exp2_iterative.py --limit 20   # debug: read the reasoning chains!
  python scripts/run_exp2_iterative.py
"""
import argparse

from hla.contexts import golds, topk_context
from hla.generation import Generator
from hla.pipelines import answer_with_contexts, ircot, single_shot_cot
from hla.retrieval import DenseRetriever
from hla.data import read_jsonl, write_jsonl
from hla.runtime import RESULTS, load_corpus, load_examples, save, score

K = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="first N questions per dataset (debug)")
    args = ap.parse_args()

    examples = load_examples("musique_ids.txt")[: args.limit] + load_examples("nq_ids.txt")[: args.limit]
    corpus = load_corpus()
    docs = {d["doc_id"]: d for d in corpus}
    retriever = DenseRetriever(corpus)  # stays loaded: IRCoT retrieves during reasoning
    gen = Generator(max_model_len=8192, gpu_memory_utilization=0.75)

    top = [topk_context(e, r, docs, K) for e, r in zip(examples, retriever.search([e["question"] for e in examples], k=K))]
    methods = {
        "single": lambda: answer_with_contexts(gen, examples, top),
        "single_cot": lambda: single_shot_cot(gen, examples, top),
        "ircot": lambda: ircot(gen, retriever, docs, examples, k_init=K),
        "oracle": lambda: answer_with_contexts(gen, examples, [golds(e) for e in examples]),
    }
    # Each method's raw outputs are saved as they finish, so a disconnected session resumes.
    part_dir = RESULTS / "exp2" / ("parts" if args.limit is None else f"parts_limit{args.limit}")
    runs = {}
    for method, fn in methods.items():
        part = part_dir / f"{method}.jsonl"
        if part.exists():
            runs[method] = read_jsonl(part)
            print(f"{method}: done, loaded")
        else:
            runs[method] = fn()
            write_jsonl(runs[method], part)
    rows = []
    by_id = {e["qid"]: e for e in examples}
    for method, recs in runs.items():
        for r in score(recs, examples):
            gold = by_id[r["qid"]]["gold_ids"]
            rows.append({**r, "method": method,
                         "golds_found": [g in r["context_ids"] for g in gold]})
        for ds in ("musique", "nq"):
            sub = [r for r in rows if r["method"] == method and r["dataset"] == ds]
            print(f"{method:10s} {ds:8s} EM {sum(r['em'] for r in sub) / max(1, len(sub)):.3f}  "
                  f"calls {sum(r['llm_calls'] for r in sub) / max(1, len(sub)):.2f}")
    save(rows, "exp2", "exp2.jsonl")


if __name__ == "__main__":
    main()
