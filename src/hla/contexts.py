"""Context builders for Exp 1 (length x distractor type x gold position) and for top-k RAG.

All builders are deterministic, so the attention probe can rebuild exactly the contexts
the model saw (it actually re-reads them from the saved ``context_ids``).
"""
from __future__ import annotations

import random
from typing import Callable, Iterable, Iterator

from .corpus import labelled
from .metrics import answer_in_text
from .pipelines import format_passage

POSITIONS = ("start", "middle", "end")


def golds(ex: dict) -> list[dict]:
    g = [p for p in ex["passages"] if p["is_gold"]]
    return sorted(g, key=lambda p: p["hop"])


class TokenCounter:
    """Caches the token cost of each passage as one formatted document."""

    def __init__(self, count_tokens: Callable[[str], int]):
        self.count_tokens = count_tokens
        self._cache: dict[str, int] = {}

    def __call__(self, p: dict) -> int:
        if p["doc_id"] not in self._cache:
            # "Document [10]" stands in for any index; + 2 for the blank line between documents
            self._cache[p["doc_id"]] = self.count_tokens(format_passage(10, p)) + 2
        return self._cache[p["doc_id"]]


# --------------------------------------------------------------------------- Exp 1

def distractors(ex: dict, kind: str, ranked_ids: list[str], all_ids: list[str], docs: dict,
                seed: int = 13) -> Iterator[dict]:
    """Distractors for question ``ex`` in the order they are added to the context.

    related:   the retriever's ranking for this question, most similar first.
    unrelated: the whole corpus in a random (seeded, per-question) order.

    Both skip the gold passage and any passage that contains an answer string, so the
    gold passage is the only place the answer appears.
    """
    gold = set(ex["gold_ids"])
    if kind == "related":
        order: Iterable[str] = ranked_ids
    elif kind == "unrelated":
        order = list(all_ids)
        random.Random(f"{seed}-{ex['qid']}-unrelated").shuffle(order)
    else:
        raise ValueError(kind)
    for d in order:
        if d in gold:
            continue
        p = docs[d]
        if answer_in_text(p["title"] + " " + p["text"], ex["answers"]):
            continue
        yield labelled(ex, p)


def length_context(ex: dict, distractor_iter: Iterable[dict], budget_tokens: int | None,
                   position: str, cost: TokenCounter, seed: int = 13) -> list[dict]:
    """The gold passage plus distractors, taken in order while the total fits in
    ``budget_tokens`` (None = gold only). The chosen distractors are shuffled (seeded),
    then the gold passage is put at the start, the middle or the end.

    For a fixed question and distractor type, the distractor *set* at a shorter budget is
    a subset of the set at a longer budget, so a longer budget only adds passages.
    """
    gold = golds(ex)
    assert len(gold) == 1, "length_context is for single-gold (NQ) questions"
    gold = gold[0]
    chosen: list[dict] = []
    if budget_tokens is not None:
        total = cost(gold)
        for p in distractor_iter:
            c = cost(p)
            if total + c > budget_tokens:
                break
            chosen.append(p)
            total += c
        else:
            raise ValueError(f"{ex['qid']}: ran out of distractors before {budget_tokens} tokens")
        random.Random(f"{seed}-{ex['qid']}-{budget_tokens}").shuffle(chosen)
    at = {"start": 0, "middle": len(chosen) // 2, "end": len(chosen)}[position]
    return chosen[:at] + [gold] + chosen[at:]


# --------------------------------------------------------------------------- RAG

def topk_context(ex: dict, ranked_ids: list[str], docs: dict, k: int) -> list[dict]:
    """The retriever's top ``k`` passages, in rank order."""
    return [labelled(ex, docs[d]) for d in ranked_ids[:k]]
