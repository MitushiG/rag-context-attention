"""QA and retrieval metrics.

QA metrics follow the official SQuAD / HotpotQA normalization: lowercase, strip
punctuation and articles, collapse whitespace. A prediction is scored against every
gold alias and the best score is kept.
"""
from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Iterable, Sequence


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds: Sequence[str]) -> float:
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds))


def _f1(pred: str, gold: str) -> float:
    p_toks = normalize_answer(pred).split()
    g_toks = normalize_answer(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = Counter(p_toks) & Counter(g_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(p_toks)
    recall = n_same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def f1_score(pred: str, golds: Sequence[str]) -> float:
    return max((_f1(pred, g) for g in golds), default=0.0)


def answer_accuracy(pred: str, golds: Sequence[str]) -> float:
    """Primary metric: 1 if any accepted answer appears in the normalized prediction
    (the accuracy of Liu et al., 2023, on this NQ release). Unlike EM it does not count
    a correct answer wrapped in a sentence as wrong."""
    return float(answer_in_text(pred, golds))


def answer_in_text(text: str, golds: Sequence[str]) -> bool:
    """Loose match used for 'does this passage contain the answer' checks."""
    t = normalize_answer(text)
    return any(normalize_answer(g) and normalize_answer(g) in t for g in golds)


# --------------------------------------------------------------------------- retrieval

def gold_hits_at_k(ranked_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> dict[str, int]:
    """For each gold doc id, 1 if it appears in the top-k ranked ids, else 0."""
    top = set(ranked_ids[:k])
    return {g: int(g in top) for g in gold_ids}


def recall_at_k(ranked_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    if not gold_ids:
        return float("nan")
    hits = gold_hits_at_k(ranked_ids, gold_ids, k)
    return sum(hits.values()) / len(gold_ids)


def all_gold_at_k(ranked_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> int:
    """1 if every gold passage is in the top k (needed to answer a multi-hop question)."""
    return int(all(v for v in gold_hits_at_k(ranked_ids, gold_ids, k).values()))


def reciprocal_rank(ranked_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    gold = set(gold_ids)
    for i, d in enumerate(ranked_ids, start=1):
        if d in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevant: dict[str, float], k: int) -> float:
    """nDCG@k with graded relevance (binary relevance -> pass {id: 1.0})."""
    dcg = sum(relevant.get(d, 0.0) / math.log2(i + 2) for i, d in enumerate(ranked_ids[:k]))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0
