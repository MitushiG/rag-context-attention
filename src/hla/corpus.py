"""Pooled closed corpus: the union of the candidate passages of every NQ (Lost in the
Middle release) and MuSiQue 2-hop dev question (~70K passages after de-duplication).

Indexing 21M Wikipedia passages is out of budget. Pooling the benchmark's own gold and
distractor passages gives a corpus where many distractors are hard (they were chosen to
look relevant). Limitation: absolute recall is higher than it would be on full Wikipedia.
"""
from __future__ import annotations

import hashlib


def _key(title: str, text: str) -> str:
    norm = " ".join((title + " || " + text).lower().split())
    return hashlib.md5(norm.encode()).hexdigest()[:16]


def build_corpus(examples: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (corpus, examples_with_ids).

    corpus: [{"doc_id", "title", "text"}]. Each passage in each example gets a
    ``doc_id``; examples also get ``gold_ids`` (ordered by hop) and ``gold_hops``
    {doc_id: hop}.
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


def labelled(ex: dict, doc: dict) -> dict:
    """A corpus passage with gold labels relative to question ``ex``."""
    return {**doc, "is_gold": doc["doc_id"] in ex["gold_hops"], "hop": ex["gold_hops"].get(doc["doc_id"])}
