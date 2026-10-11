"""Context builders for H3 (length) and H4 (position / gap).

All builders are deterministic given ``seed`` so every condition sees identical inputs.
"""
from __future__ import annotations

import random
from typing import Callable


def golds(ex: dict) -> list[dict]:
    g = [p for p in ex["passages"] if p["is_gold"]]
    return sorted(g, key=lambda p: (p["hop"] is None, p["hop"] or 0))


def own_distractors(ex: dict) -> list[dict]:
    return [p for p in ex["passages"] if not p["is_gold"]]


def _as_distractor(p: dict) -> dict:
    """Corpus passages carry no gold labels; padding taken from other questions is a
    distractor for this one."""
    return {**p, "is_gold": False, "hop": None}


def _fillers(pool: list[dict], exclude: set[str], rng: random.Random, n: int) -> list[dict]:
    cands = [p for p in pool if p["doc_id"] not in exclude]
    return [_as_distractor(p) for p in rng.sample(cands, min(n, len(cands)))]


def position_context(ex: dict, gold_position: int, n_total: int = 20, pool: list[dict] | None = None,
                     seed: int = 13) -> list[dict]:
    """H4(a): one gold passage at 1-indexed ``gold_position`` among ``n_total`` passages.

    Distractors are the question's own (hard) distractors, topped up from ``pool``.
    The distractor order is fixed per question so only the gold passage moves.
    """
    rng = random.Random(f"{seed}-{ex['qid']}")
    gold = golds(ex)[0]
    dis = own_distractors(ex)
    rng.shuffle(dis)
    if len(dis) < n_total - 1:
        if pool is None:
            raise ValueError("not enough distractors; pass a pool")
        dis += _fillers(pool, {p["doc_id"] for p in ex["passages"]}, rng, n_total - 1 - len(dis))
    dis = dis[: n_total - 1]
    return dis[: gold_position - 1] + [gold] + dis[gold_position - 1:]


def gap_context(ex: dict, distance: int, n_total: int = 20, pool: list[dict] | None = None,
                seed: int = 13) -> list[dict]:
    """H4(b): two gold passages with ``distance`` passages between them (hop-1 first).

    The start position is drawn at random (seeded) from the positions that fit.
    """
    rng = random.Random(f"{seed}-{ex['qid']}-gap")
    g = golds(ex)
    assert len(g) == 2, "gap_context needs exactly two gold passages"
    dis = own_distractors(ex)
    rng.shuffle(dis)
    need = n_total - 2
    if len(dis) < need:
        if pool is None:
            raise ValueError("not enough distractors; pass a pool")
        dis += _fillers(pool, {p["doc_id"] for p in ex["passages"]}, rng, need - len(dis))
    dis = dis[:need]
    max_start = n_total - (distance + 2)  # 0-indexed
    if max_start < 0:
        raise ValueError(f"distance {distance} does not fit in {n_total} passages")
    start = rng.randint(0, max_start)
    ctx = list(dis)
    ctx.insert(start, g[0])
    ctx.insert(start + distance + 1, g[1])
    return ctx


def length_context(ex: dict, target_tokens: int, count_tokens: Callable[[str], int],
                   pool: list[dict], seed: int = 13) -> list[dict]:
    """H3: all gold passages, then the question's own (hard) distractors, then random
    corpus passages, added while they fit, until the formatted context reaches
    ``target_tokens``. Positions are shuffled (seeded).

    Gold passages are always kept, so at very short targets the context may slightly
    exceed the budget; the actual token count is recorded by the caller.
    """
    rng = random.Random(f"{seed}-{ex['qid']}-len")
    ctx = golds(ex)
    used = {p["doc_id"] for p in ex["passages"]}
    total = sum(count_tokens(p["title"] + " " + p["text"]) + 8 for p in ctx)
    own = own_distractors(ex)
    rng.shuffle(own)
    fill = [_as_distractor(p) for p in pool if p["doc_id"] not in used]
    rng.shuffle(fill)
    for p in own + fill:
        if total >= target_tokens:
            break
        cost = count_tokens(p["title"] + " " + p["text"]) + 8
        if total + cost > target_tokens * 1.02:
            continue
        ctx.append(p)
        total += cost
    rng.shuffle(ctx)
    return ctx
