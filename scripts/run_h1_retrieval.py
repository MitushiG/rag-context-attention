"""H1: per-hop recall of single-shot retrieval.

One output row per (question, gold passage, retriever) with hit@k flags, plus the full
rankings so later scripts can reuse them.
"""
import argparse

from hla.metrics import gold_hits_at_k
from hla.retrieval import BM25Retriever, CrossEncoderReranker, DenseRetriever, HybridRetriever
from hla.runtime import load_corpus, load_examples, save

KS = (2, 5, 10, 20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank", action="store_true", help="also rerank hybrid top-50 with a cross-encoder")
    args = ap.parse_args()

    examples = load_examples()
    corpus = load_corpus()
    queries = [e["question"] for e in examples]

    bm25, dense = BM25Retriever(corpus), DenseRetriever(corpus)
    retrievers = {"bm25": bm25, "dense": dense, "hybrid": HybridRetriever([bm25, dense])}
    rankings = {name: r.search(queries, k=50) for name, r in retrievers.items()}
    if args.rerank:
        rankings["hybrid+rerank"] = CrossEncoderReranker(corpus).rerank(queries, rankings["hybrid"], k=50)

    rows = []
    for name, ranked in rankings.items():
        save([{"qid": e["qid"], "ranked": r[:50]} for e, r in zip(examples, ranked)], "h1", f"rankings_{name}.jsonl")
        for e, r in zip(examples, ranked):
            hits = {k: gold_hits_at_k(r, e["gold_ids"], k) for k in KS}
            for g in e["gold_ids"]:
                rows.append({
                    "qid": e["qid"], "dataset": e["dataset"], "qtype": e["qtype"], "hops": e["hops"],
                    "retriever": name, "doc_id": g, "hop_index": e["gold_hops"][g],
                    **{f"hit@{k}": hits[k][g] for k in KS},
                })
    save(rows, "h1", "hits.jsonl")


if __name__ == "__main__":
    main()
