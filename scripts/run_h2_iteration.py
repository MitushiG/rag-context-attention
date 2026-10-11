"""H2 (and inputs for H7): closed-book, oracle, single-shot RAG and IRCoT on all questions.

closed-book and oracle are not part of a hypothesis test; they show how much the model
knows without retrieval and the ceiling with perfect retrieval.
"""
import argparse
import time

from hla.generation import Generator
from hla.pipelines import answer_with_contexts, ircot, single_shot_rag
from hla.runtime import build_retriever, load_corpus, load_examples, save, score

CONDITIONS = ("closed_book", "oracle", "single_shot", "ircot")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--limit", type=int, default=None, help="debug: first N questions")
    args = ap.parse_args()

    examples = load_examples()[: args.limit]
    corpus = load_corpus()
    docs = {d["doc_id"]: d for d in corpus}
    retriever = build_retriever("hybrid", corpus)
    gen = Generator(max_model_len=8192, gpu_memory_utilization=0.75)  # leave room for the e5 encoder

    for cond in args.conditions:
        gen.n_calls, gen.seconds = 0, 0.0
        t0 = time.perf_counter()
        if cond == "closed_book":
            recs = answer_with_contexts(gen, examples, [None] * len(examples))
        elif cond == "oracle":
            ctxs = [[docs[g] for g in e["gold_ids"]] for e in examples]
            recs = answer_with_contexts(gen, examples, ctxs)
        elif cond == "single_shot":
            recs = single_shot_rag(gen, retriever, docs, examples, k=5)
        elif cond == "ircot":
            recs = ircot(gen, retriever, docs, examples)
        else:
            raise ValueError(cond)
        wall = time.perf_counter() - t0
        rows = score(recs, examples)
        for r in rows:
            r["condition"] = cond
        save(rows, "h2", f"{cond}.jsonl")
        print(f"{cond}: EM={sum(r['em'] for r in rows) / len(rows):.3f}  "
              f"LLM calls={gen.n_calls}  wall={wall:.0f}s  ({wall / len(rows):.2f}s/question, batched)")


if __name__ == "__main__":
    main()
