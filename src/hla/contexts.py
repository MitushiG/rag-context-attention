"""Context builders for Exp 1 (ranked expansion) and Exp 3 (evidence-pair placement).

All builders are deterministic, so Exp 2 can rebuild exactly the contexts the model saw.
"""
from __future__ import annotations

import random
from typing import Callable

from .corpus import labelled
from .pipelines import format_passage


def golds(ex: dict) -> list[dict]:
    g = [p for p in ex["passages"] if p["is_gold"]]
    return sorted(g, key=lambda p: p["hop"])


def own_distractors(ex: dict) -> list[dict]:
    return [p for p in ex["passages"] if not p["is_gold"]]


# --------------------------------------------------------------------------- Exp 1

def ranked_context(ex: dict, ranked_ids: list[str], docs: dict, budget_tokens: int | None,
                   count_tokens: Callable[[str], int], k: int | None = None) -> list[dict]:
    """The retriever's ranked list, cut either at ``k`` passages (RAG) or at the longest
    prefix that fits in ``budget_tokens`` (expanded context).

    Because every context is a *prefix* of the same ranking, a longer budget only ever
    adds lower-ranked passages: RAG ⊂ 4K ⊂ 12K ⊂ 28K for every question.
    """
    if k is not None:
        return [labelled(ex, docs[d]) for d in ranked_ids[:k]]
    ctx, total = [], 0
    for i, d in enumerate(ranked_ids, start=1):
        p = labelled(ex, docs[d])
        cost = count_tokens(format_passage(i, p)) + 2  # + the blank line between documents
        if total + cost > budget_tokens:
            break
        ctx.append(p)
        total += cost
    else:
        raise ValueError(f"ranking too short to fill {budget_tokens} tokens; retrieve more")
    return ctx


# --------------------------------------------------------------------------- Exp 3

N_PASSAGES = 20
# Midpoint of the gold pair (1-indexed slots) and number of distractors between the two
# gold passages. Close and far pairs share a midpoint, so location and separation vary
# independently:
#   early:  close (5, 6)    far (1, 10)
#   middle: close (10, 11)  far (6, 15)
#   late:   close (15, 16)  far (11, 20)
LOCATIONS = {"early": 5.5, "middle": 10.5, "late": 15.5}
SEPARATIONS = {"close": 0, "far": 8}


def pair_slots(location: str, separation: str) -> tuple[int, int]:
    mid, between = LOCATIONS[location], SEPARATIONS[separation]
    half = (between + 1) / 2
    return int(mid - half), int(mid + half)


def pair_context(ex: dict, location: str, separation: str, seed: int = 13) -> list[dict]:
    """20 passages: the question's 18 own distractors in a fixed (seeded) order, with the
    two gold passages inserted at ``pair_slots(location, separation)``.

    Across the six conditions only the two gold slots change: same passages, same
    distractor order, same total length. Which hop comes first is drawn once per
    question (seeded) and kept across conditions; it is recorded as a covariate.
    """
    rng = random.Random(f"{seed}-{ex['qid']}-pair")
    g = golds(ex)
    assert len(g) == 2, "pair_context needs exactly two gold passages"
    dis = own_distractors(ex)
    rng.shuffle(dis)
    if len(dis) < N_PASSAGES - 2:
        raise ValueError(f"{ex['qid']}: needs {N_PASSAGES - 2} distractors, has {len(dis)}")
    dis = dis[: N_PASSAGES - 2]
    first, second = g if rng.random() < 0.5 else g[::-1]
    a, b = pair_slots(location, separation)
    ctx = list(dis)
    ctx.insert(a - 1, first)
    ctx.insert(b - 1, second)
    return ctx
