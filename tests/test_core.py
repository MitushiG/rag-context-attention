"""CPU-only tests for metrics, statistics, context builders and the router."""
import math
import random

import numpy as np
import pandas as pd
import pytest

from hla import stats
from hla.contexts import gap_context, length_context, position_context
from hla.corpus import build_corpus
from hla.metrics import exact_match, f1_score, ndcg_at_k, recall_at_k, reciprocal_rank
from hla.router import choose_threshold, route


# --------------------------------------------------------------------------- metrics

def test_em_f1_normalization():
    assert exact_match("The Eiffel Tower.", ["eiffel tower"]) == 1.0
    assert exact_match("Paris", ["London", "paris"]) == 1.0
    assert f1_score("Barack Obama", ["Obama"]) == pytest.approx(2 / 3)
    assert f1_score("", ["x"]) == 0.0


def test_retrieval_metrics():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, ["c", "z"], 3) == 0.5
    assert reciprocal_rank(ranked, ["c"]) == pytest.approx(1 / 3)
    assert ndcg_at_k(ranked, {"a": 1.0}, 5) == 1.0
    assert ndcg_at_k(ranked, {"b": 1.0}, 5) == pytest.approx(1 / math.log2(3))


# --------------------------------------------------------------------------- stats

def test_mcnemar_counts_and_direction():
    a = [1] * 30 + [0] * 5 + [1] * 50
    b = [0] * 30 + [1] * 5 + [1] * 50
    r = stats.mcnemar_exact(a, b, alternative="greater")
    assert (r["a_only"], r["b_only"]) == (30, 5)
    assert r["p_value"] < 0.001


def test_holm_matches_definition():
    out = stats.holm({"A": 0.01, "B": 0.04, "C": 0.03})
    assert out["A"]["p_holm"] == pytest.approx(0.03)
    assert out["C"]["p_holm"] == pytest.approx(0.06)
    assert out["B"]["p_holm"] == pytest.approx(0.06)  # monotone
    assert out["A"]["reject"] and not out["C"]["reject"] and not out["B"]["reject"]


def test_auroc_matches_sklearn():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = y + rng.normal(0, 1.5, 300)
    s[:20] = 0.0  # ties
    assert stats.auroc(s, y) == pytest.approx(roc_auc_score(y, s))


def test_paired_bootstrap_ci_covers_truth():
    rng = np.random.default_rng(1)
    a = rng.binomial(1, 0.6, 500)
    b = rng.binomial(1, 0.5, 500)
    est = stats.paired_bootstrap_diff(a, b, n_boot=2000)
    assert est.ci_low < est.value < est.ci_high
    assert est.ci_low < 0.1 < est.ci_high


def test_ece_perfectly_calibrated_is_small():
    rng = np.random.default_rng(2)
    c = rng.uniform(0, 1, 20000)
    y = rng.uniform(0, 1, 20000) < c
    assert stats.expected_calibration_error(c, y) < 0.02


def test_clustered_logit_recovers_sign():
    rng = np.random.default_rng(3)
    rows = []
    for q in range(300):
        u = rng.normal(0, 0.5)
        for hop in (1, 2, 3):
            p = 1 / (1 + np.exp(-(2.0 - 1.0 * hop + u)))
            rows.append({"qid": q, "hop_index": hop, "hit": int(rng.uniform() < p)})
    r = stats.logit_clustered("hit ~ hop_index", pd.DataFrame(rows), "qid")
    assert r.params["hop_index"] < 0
    assert stats.one_sided_p(r, "hop_index", "<") < 0.001


# --------------------------------------------------------------------------- contexts

def _example(qid, n_gold, n_dis, hops=None):
    ps = [{"title": f"{qid}-g{i}", "text": "gold " * 20, "is_gold": True, "hop": (hops or [i + 1] * n_gold)[i]}
          for i in range(n_gold)]
    ps += [{"title": f"{qid}-d{i}", "text": "dis " * 20, "is_gold": False, "hop": None} for i in range(n_dis)]
    return {"qid": qid, "dataset": "x", "question": "q?", "answers": ["a"], "hops": n_gold, "qtype": None,
            "passages": ps}


@pytest.fixture
def built():
    exs = [_example("q1", 1, 19), _example("q2", 2, 8), _example("q3", 2, 30)]
    corpus, exs = build_corpus(exs)
    return corpus, {e["qid"]: e for e in exs}


def test_corpus_ids_and_gold_order(built):
    corpus, exs = built
    assert len(corpus) == 20 + 10 + 32
    assert exs["q2"]["gold_hops"][exs["q2"]["gold_ids"][0]] == 1


@pytest.mark.parametrize("pos", [1, 5, 10, 20])
def test_position_context(built, pos):
    _, exs = built
    ctx = position_context(exs["q1"], pos)
    assert len(ctx) == 20
    assert ctx[pos - 1]["is_gold"] and sum(p["is_gold"] for p in ctx) == 1
    # distractor order does not depend on the gold position
    others = [p["doc_id"] for p in ctx if not p["is_gold"]]
    assert others == [p["doc_id"] for p in position_context(exs["q1"], 1) if not p["is_gold"]]


@pytest.mark.parametrize("d", [0, 5, 18])
def test_gap_context(built, d):
    corpus, exs = built
    ctx = gap_context(exs["q2"], d, pool=corpus)
    idx = [i for i, p in enumerate(ctx) if p["is_gold"]]
    assert len(ctx) == 20 and idx[1] - idx[0] - 1 == d
    assert ctx[idx[0]]["hop"] == 1


def test_length_context_keeps_gold_and_hits_budget(built):
    corpus, exs = built
    count = lambda s: len(s.split())
    for target in (60, 400, 1000):
        ctx = length_context(exs["q3"], target, count, corpus)
        assert sum(p["is_gold"] for p in ctx) == 2
        total = sum(count(p["title"] + " " + p["text"]) + 8 for p in ctx)
        assert total <= max(target * 1.02, 2 * 29)


# --------------------------------------------------------------------------- multimodal

def test_maxsim_from_scratch():
    torch = pytest.importorskip("torch")
    from hla.multimodal import maxsim

    rng = torch.Generator().manual_seed(0)
    qs = [torch.randn(n, 8, generator=rng) for n in (3, 5)]
    ds = [torch.randn(n, 8, generator=rng) for n in (4, 7, 2)]
    got = maxsim(qs, ds, device="cpu", q_chunk=1)
    for i, q in enumerate(qs):
        for j, d in enumerate(ds):
            assert got[i, j].item() == pytest.approx((q @ d.T).max(dim=1).values.sum().item(), abs=1e-5)


# --------------------------------------------------------------------------- router

def test_router_extremes_and_choice():
    rng = random.Random(0)
    n = 400
    conf = np.array([rng.random() for _ in range(n)])
    single = (conf > 0.5).astype(int)  # confident answers are right
    multi = np.ones(n, int)
    calls = np.full(n, 3)
    assert route(conf, single, multi, calls, -np.inf)["mean_calls"] == 1
    assert route(conf, single, multi, calls, np.inf)["em"] == 1
    t = choose_threshold(conf, single, multi, calls, max_em_drop=0.0)
    r = route(conf, single, multi, calls, t)
    assert r["em"] == 1 and r["mean_calls"] < 1 + 3
