"""CPU-only tests for metrics, statistics, context builders and the attention probe."""
import numpy as np
import pandas as pd
import pytest

from hla import stats
from hla.contexts import LOCATIONS, SEPARATIONS, pair_context, pair_slots, ranked_context
from hla.corpus import build_corpus
from hla.metrics import exact_match, f1_score


# --------------------------------------------------------------------------- metrics

def test_em_f1_normalization():
    assert exact_match("The Eiffel Tower.", ["eiffel tower"]) == 1.0
    assert exact_match("Paris", ["London", "paris"]) == 1.0
    assert f1_score("Barack Obama", ["Obama"]) == pytest.approx(2 / 3)
    assert f1_score("", ["x"]) == 0.0


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


def test_stratified_auroc_removes_confounding():
    # Score and accuracy both depend on the stratum only: pooled AUROC is high, but
    # within each stratum the score carries no information.
    rng = np.random.default_rng(1)
    g = np.repeat([0, 1], 2000)
    y = (rng.uniform(size=4000) < np.where(g == 0, 0.8, 0.2)).astype(int)
    s = np.where(g == 0, 1.0, 0.0) + rng.normal(0, 0.1, 4000)
    assert stats.auroc(s, y) > 0.75
    assert stats.stratified_auroc(s, y, g) == pytest.approx(0.5, abs=0.05)
    assert stats.stratified_auroc(s, y, np.zeros(4000)) == pytest.approx(stats.auroc(s, y))


def test_cluster_bootstrap_ci_covers_truth():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"qid": np.repeat(np.arange(300), 3), "x": rng.normal(0.2, 1, 900)})
    est = stats.cluster_bootstrap(df, "qid", lambda d: d.x.mean(), n_boot=500)
    assert est.ci_low < 0.2 < est.ci_high
    assert stats.one_sided_from_two(est, 0.0, ">") < 0.01


def test_paired_bootstrap_ci_covers_truth():
    rng = np.random.default_rng(1)
    a = rng.binomial(1, 0.6, 500)
    b = rng.binomial(1, 0.5, 500)
    est = stats.paired_bootstrap_diff(a, b, n_boot=2000)
    assert est.ci_low < est.value < est.ci_high
    assert est.ci_low < 0.1 < est.ci_high


def test_clustered_logit_recovers_sign():
    rng = np.random.default_rng(3)
    rows = []
    for q in range(300):
        u = rng.normal(0, 0.5)
        for x in (1, 2, 3):
            p = 1 / (1 + np.exp(-(2.0 - 1.0 * x + u)))
            rows.append({"qid": q, "x": x, "y": int(rng.uniform() < p)})
    r = stats.logit_clustered("y ~ x", pd.DataFrame(rows), "qid")
    assert r.params["x"] < 0
    assert stats.one_sided_p(r, "x", "<") < 0.001


# --------------------------------------------------------------------------- contexts

def _example(qid, n_dis=18):
    ps = [{"title": f"{qid}-g{h}", "text": "gold " * 20, "is_gold": True, "hop": h} for h in (1, 2)]
    ps += [{"title": f"{qid}-d{i}", "text": "dis " * 20, "is_gold": False, "hop": None} for i in range(n_dis)]
    return {"qid": qid, "question": "q?", "answers": ["a"], "passages": ps}


@pytest.fixture
def built():
    corpus, exs = build_corpus([_example(f"q{i}") for i in range(30)])
    return corpus, {e["qid"]: e for e in exs}


def test_corpus_ids_and_gold_order(built):
    corpus, exs = built
    assert len(corpus) == 30 * 20
    ex = exs["q1"]
    assert [ex["gold_hops"][d] for d in ex["gold_ids"]] == [1, 2]


def test_ranked_context_prefixes_are_nested(built):
    corpus, exs = built
    docs = {d["doc_id"]: d for d in corpus}
    ex = exs["q3"]
    ranked = [d["doc_id"] for d in corpus]
    count = lambda s: len(s.split())
    rag = ranked_context(ex, ranked, docs, None, count, k=5)
    ctxs = [ranked_context(ex, ranked, docs, b, count) for b in (300, 900, 2700)]
    assert [p["doc_id"] for p in rag] == ranked[:5]
    for short, long in zip([rag] + ctxs, ctxs):
        assert [p["doc_id"] for p in long[: len(short)]] == [p["doc_id"] for p in short]
    for b, c in zip((300, 900, 2700), ctxs):
        assert sum(count(f"Document [{i}] (Title: {p['title']}) {p['text']}") + 2
                   for i, p in enumerate(c, start=1)) <= b
    # gold labels are relative to the question
    assert sum(p["is_gold"] for p in ctxs[-1]) == sum(d in ex["gold_hops"] for d in ranked[: len(ctxs[-1])])
    with pytest.raises(ValueError):
        ranked_context(ex, ranked[:10], docs, 10**6, count)


def test_pair_slots_share_midpoints():
    for loc, mid in LOCATIONS.items():
        for sep, between in SEPARATIONS.items():
            a, b = pair_slots(loc, sep)
            assert (a + b) / 2 == mid and b - a - 1 == between and 1 <= a < b <= 20


def test_pair_context_only_moves_the_golds(built):
    _, exs = built
    ex = exs["q7"]
    seen = []
    for loc in LOCATIONS:
        for sep in SEPARATIONS:
            ctx = pair_context(ex, loc, sep)
            a, b = pair_slots(loc, sep)
            assert len(ctx) == 20 and [i + 1 for i, p in enumerate(ctx) if p["is_gold"]] == [a, b]
            seen.append(([p["doc_id"] for p in ctx if not p["is_gold"]],
                         [p["hop"] for p in ctx if p["is_gold"]]))
    assert all(s == seen[0] for s in seen)  # same distractor order and hop order everywhere


# --------------------------------------------------------------------------- attention probe

def test_attention_probe_matches_eager():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from hla.attention import AttentionProbe

    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    model = transformers.Qwen2ForCausalLM._from_config(cfg, attn_implementation="sdpa")
    probe = AttentionProbe(model, None, device="cpu")
    ids = torch.randint(0, 100, (60,)).tolist()
    assert probe.check_against_eager(ids, (50, 57), atol=1e-4) < 1e-4
    att = probe.answer_attention(ids, (50, 57))
    assert att.shape == (3, 60)
    assert np.allclose(att.sum(axis=1), 1, atol=1e-5)
    assert np.all(att[:, 57:] == 0)  # causal: answer rows never see later positions
