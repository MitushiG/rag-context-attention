"""Pooled closed corpus: the union of every sampled question's candidate passages.

Indexing 21M Wikipedia passages is out of budget. Pooling the benchmarks' own gold and
distractor passages gives a ~15-25K passage corpus where distractors are hard (they were
chosen to look relevant). Limitation: absolute recall numbers are higher than they would
be on full Wikipedia; comparisons between methods are what the study relies on.
"""
from __future__ import annotations

import hashlib


def _key(title: str, text: str) -> str:
    norm = " ".join((title + " || " + text).lower().split())
    return hashlib.md5(norm.encode()).hexdigest()[:16]


def build_corpus(examples: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (corpus, examples_with_ids).

    corpus: [{"doc_id", "title", "text"}]. Each passage in each example gets a
    ``doc_id``; examples also get ``gold_ids`` (ordered by hop where defined) and
    ``gold_hops`` {doc_id: hop}.
    """
    corpus: dict[str, dict] = {}
    out = []
    for ex in examples:
        ex = {**ex, "passages": [dict(p) for p in ex["passages"]]}
        for p in ex["passages"]:
            did = _key(p["title"], p["text"])
            p["doc_id"] = did
            corpus.setdefault(did, {"doc_id": did, "title": p["title"], "text": p["text"]})
        golds = [p for p in ex["passages"] if p["is_gold"]]
        golds.sort(key=lambda p: (p["hop"] is None, p["hop"] or 0))
        ex["gold_ids"] = [p["doc_id"] for p in golds]
        ex["gold_hops"] = {p["doc_id"]: p["hop"] for p in golds}
        out.append(ex)
    return list(corpus.values()), out


def doc_text(doc: dict) -> str:
    return f"{doc['title']}. {doc['text']}"
