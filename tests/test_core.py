"""CPU-only tests for metrics, statistics, context builders, pipelines, the attention probe
and the tiny recall model."""
import numpy as np
import pandas as pd
import pytest

from hla import stats
from hla.contexts import TokenCounter, distractors, length_context
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

def _example(qid, n_dis=19, answer="zebra"):
    ps = [{"title": f"{qid}-gold", "text": f"the answer is {answer} " * 5, "is_gold": True, "hop": 1}]
    ps += [{"title": f"{qid}-d{i}", "text": f"filler text number {i} " * 5, "is_gold": False, "hop": None}
           for i in range(n_dis)]
    return {"qid": qid, "dataset": "nq", "question": "q?", "answers": [answer], "passages": ps}


@pytest.fixture
def built():
    exs = [_example(f"q{i}") for i in range(30)]
    exs[5]["passages"][3]["text"] = "this distractor mentions zebra"  # contains the answer
    corpus, exs = build_corpus(exs)
    return corpus, {e["qid"]: e for e in exs}


def test_corpus_ids_and_gold(built):
    corpus, exs = built
    assert len(corpus) == 30 * 20  # titles differ, so no passages are merged
    assert len(exs["q1"]["gold_ids"]) == 1


def _setup(built, qid="q1"):
    corpus, exs = built
    docs = {d["doc_id"]: d for d in corpus}
    return exs[qid], docs, list(docs), TokenCounter(lambda s: len(s.split()))


@pytest.mark.parametrize("kind", ["related", "unrelated"])
def test_distractors_exclude_gold_and_answer(built, kind):
    ex, docs, all_ids, _ = _setup(built)
    ds = list(distractors(ex, kind, all_ids, all_ids, docs))
    assert all(not p["is_gold"] for p in ds)
    assert all("zebra" not in p["text"] for p in ds)
    assert len(ds) < len(all_ids)


@pytest.mark.parametrize("kind", ["related", "unrelated"])
def test_length_context_nested_and_positioned(built, kind):
    ex, docs, all_ids, cost = _setup(built)
    sets = []
    for budget in (200, 800, 3000):
        for pos, at in (("start", 0), ("middle", None), ("end", -1)):
            ctx = length_context(ex, distractors(ex, kind, all_ids, all_ids, docs), budget, pos, cost)
            gi = [i for i, p in enumerate(ctx) if p["is_gold"]]
            assert len(gi) == 1
            assert gi[0] == ((len(ctx) - 1) // 2 if pos == "middle" else range(len(ctx))[at])
            assert sum(cost(p) for p in ctx) <= budget
        sets.append({p["doc_id"] for p in ctx})
    assert sets[0] <= sets[1] <= sets[2]  # a longer budget only adds passages
    assert len(length_context(ex, [], None, "start", cost)) == 1


def test_length_context_runs_out(built):
    ex, docs, all_ids, cost = _setup(built)
    with pytest.raises(ValueError):
        length_context(ex, distractors(ex, "related", all_ids[:5], all_ids, docs), 10**6, "start", cost)


# --------------------------------------------------------------------------- pipelines

class FakeGen:
    """Stands in for vLLM: scripted outputs, counts calls."""

    def __init__(self, script):
        self.script, self.calls = script, 0

    def chat_prompt(self, body):
        return body

    def count_tokens(self, text):
        return len(text.split())

    def generate(self, prompts, max_tokens=32, stop=None):
        from hla.generation import Generation
        self.calls += len(prompts)
        return [Generation(text=self.script(p), mean_logprob=-0.1, n_tokens=3) for p in prompts]


class FakeRetriever:
    def __init__(self, table):
        self.table = table

    def search(self, queries, k=10):
        return [self.table.get(q, self.table["default"])[:k] for q in queries]


def test_ircot_retrieves_with_reasoning_and_tracks_golds():
    docs = {d: {"doc_id": d, "title": d, "text": d} for d in ["a", "b", "c", "hop2", "x", "y"]}
    ex = {"qid": "m1", "question": "Who is the spouse of the director of Titanic?", "answers": ["Suzy Amis"],
          "gold_ids": ["a", "hop2"], "dataset": "musique"}
    retr = FakeRetriever({ex["question"]: ["a", "b", "c"], "Titanic was directed by James Cameron.": ["hop2", "x"],
                          "default": ["y"]})

    def script(prompt):
        if "Reasoning so far: (none)" in prompt:
            return "Titanic was directed by James Cameron."
        return "Cameron is married to Suzy Amis. So the answer is: Suzy Amis."

    from hla.pipelines import ircot
    r = ircot(FakeGen(script), retr, docs, [ex], k_init=3, k_step=1)[0]
    assert r["pred"] == "Suzy Amis" and not r["fallback"]
    assert r["context_ids"] == ["a", "b", "c", "hop2"]
    assert r["found_by_step"] == [[True, False], [True, True], [True, True]]
    assert r["llm_calls"] == 2


def test_cot_falls_back_when_no_answer_stated():
    from hla.pipelines import single_shot_cot
    ctx = [{"doc_id": "a", "title": "a", "text": "a"}]
    gen = FakeGen(lambda p: "Paris" if p.startswith("Answer the question. Respond") else "I am not sure.")
    r = single_shot_cot(gen, [{"qid": "n1", "question": "q?"}], [ctx])[0]
    assert r["pred"] == "Paris" and r["fallback"] and r["llm_calls"] == 2


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


# --------------------------------------------------------------------------- tiny recall model

def test_tiny_recall_batch_and_model():
    torch = pytest.importorskip("torch")
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location(
        "tiny_recall", pathlib.Path(__file__).parents[1] / "foundations" / "tiny_recall.py")
    t = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t)

    gen = torch.Generator().manual_seed(0)
    nf = 20
    for kind in ("related", "unrelated"):
        x, y, needle, qpos = t.make_batch(8, nf, kind, gen)
        assert x.shape == (8, nf * t.FACT + t.QUERY_BLOCK)
        assert qpos[-1].item() == x.shape[1] - 1                    # final query ends the sequence
        for i in range(8):
            facts = x[i, : nf * t.FACT].view(nf, t.FACT)
            assert len(set(needle[i].tolist())) == t.N_QUERIES       # distinct queried keys
            for j in range(t.N_QUERIES):
                p = qpos[j].item()
                assert x[i, p - t.KEY_LEN].item() == t.QUERY
                q = x[i, p - t.KEY_LEN + 1: p + 1]
                match = (facts[:, : t.KEY_LEN] == q).all(1).nonzero().flatten().tolist()
                assert match == [needle[i, j].item()]                # query key appears once
                assert facts[needle[i, j], -1].item() == y[i, j].item()  # answer = its value
                if j < t.N_QUERIES - 1:
                    assert x[i, p + 1].item() == y[i, j].item()          # earlier answers are shown
            if kind == "related":
                assert (facts[:, 0] == facts[0, 0]).all()            # shared first symbol

    for variant, (pos, ssmax) in t.VARIANTS.items():
        m = t.TinyGPT(pos=pos, ssmax=ssmax, d=32, n_layer=2)
        logits, rows = m(x, keep_attention=True)
        assert logits.shape == (8, x.shape[1], t.VOCAB) and len(rows) == 2
        assert torch.allclose(rows[0].sum(-1), torch.ones(8), atol=1e-5)

    # log-n scaling is the identity inside the training length
    m = t.TinyGPT(pos="rope", d=32, n_layer=2).eval()
    x = t.make_batch(4, t.N_TRAIN, "unrelated", gen)[0]
    assert torch.allclose(m(x)[0], m(x, logn=True)[0], atol=1e-5)
