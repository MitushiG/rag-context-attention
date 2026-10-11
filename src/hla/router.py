"""H7(b): confidence router. Answer single-shot first; escalate to IRCoT below a threshold.

Inputs are per-question arrays aligned on qid:
    conf         single-shot answer confidence (exp mean token logprob)
    single_ok    single-shot EM (0/1)
    multi_ok     IRCoT EM (0/1)
    multi_calls  LLM calls IRCoT used on that question
Cost of routing = 1 single-shot call + IRCoT calls when escalated.
"""
from __future__ import annotations

import numpy as np


def route(conf, single_ok, multi_ok, multi_calls, threshold: float) -> dict:
    conf, s, m, c = (np.asarray(x, float) for x in (conf, single_ok, multi_ok, multi_calls))
    esc = conf < threshold
    correct = np.where(esc, m, s)
    calls = 1 + esc * c
    return {
        "threshold": float(threshold),
        "em": float(correct.mean()),
        "mean_calls": float(calls.mean()),
        "escalation_rate": float(esc.mean()),
        "correct": correct,
        "calls": calls,
    }


def sweep(conf, single_ok, multi_ok, multi_calls, n: int = 101) -> list[dict]:
    """Trade-off curve. Thresholds over the observed confidence quantiles, plus the two
    extremes (never / always escalate)."""
    qs = np.unique(np.quantile(np.asarray(conf, float), np.linspace(0, 1, n)))
    ths = np.concatenate([[-np.inf], qs, [np.inf]])
    return [{k: v for k, v in route(conf, single_ok, multi_ok, multi_calls, t).items()
             if k not in ("correct", "calls")} for t in ths]


def choose_threshold(conf, single_ok, multi_ok, multi_calls, max_em_drop: float = 0.01) -> float:
    """Pre-registered rule: cheapest threshold whose EM is within ``max_em_drop`` of
    always-IRCoT on the calibration split."""
    target = float(np.mean(multi_ok)) - max_em_drop
    curve = sweep(conf, single_ok, multi_ok, multi_calls)
    ok = [r for r in curve if r["em"] >= target]
    best = min(ok, key=lambda r: (r["mean_calls"], -r["em"]))
    return best["threshold"]


def split_halves(qids: list[str], seed: int = 13) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(qids))
    h = len(qids) // 2
    return np.sort(idx[:h]), np.sort(idx[h:])
