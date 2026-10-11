"""Shared helpers for the experiment scripts."""
from __future__ import annotations

from pathlib import Path

from .data import read_jsonl, write_jsonl
from .metrics import exact_match, f1_score

PROCESSED = Path("data/processed")
SAMPLES = Path("data/samples")
RESULTS = Path("results")


def load_examples(ids_file: str | None = None) -> list[dict]:
    exs = read_jsonl(PROCESSED / "examples.jsonl")
    if ids_file:
        keep = set((SAMPLES / ids_file).read_text().split())
        exs = [e for e in exs if e["qid"] in keep]
    return exs


def load_corpus() -> list[dict]:
    return read_jsonl(PROCESSED / "corpus.jsonl")


def build_retriever(name: str, corpus: list[dict]):
    from .retrieval import BM25Retriever, DenseRetriever, HybridRetriever

    if name == "bm25":
        return BM25Retriever(corpus)
    if name == "dense":
        return DenseRetriever(corpus)
    if name == "hybrid":
        return HybridRetriever([BM25Retriever(corpus), DenseRetriever(corpus)])
    raise ValueError(name)


def score(records: list[dict], examples: list[dict]) -> list[dict]:
    """Attach EM/F1 and question metadata to pipeline outputs."""
    by_id = {e["qid"]: e for e in examples}
    out = []
    for r in records:
        ex = by_id[r["qid"]]
        out.append({
            **r,
            "dataset": ex["dataset"], "hops": ex["hops"], "qtype": ex["qtype"],
            "em": exact_match(r["pred"], ex["answers"]),
            "f1": f1_score(r["pred"], ex["answers"]),
        })
    return out


def save(rows: list[dict], *parts: str) -> Path:
    path = RESULTS.joinpath(*parts)
    write_jsonl(rows, path)
    print(f"wrote {len(rows)} rows -> {path}")
    return path
