"""Run every pre-registered test in HYPOTHESES.md and write results/summary.md.

Hypotheses whose result files are missing are skipped, so this can be re-run as
experiments finish. CPU only.
"""
import json

import numpy as np
import pandas as pd

from hla import stats
from hla.data import read_jsonl
from hla.runtime import RESULTS

lines: list[str] = []
primary_p: dict[str, float] = {}


def out(s: str = "") -> None:
    print(s)
    lines.append(s)


def load(*parts) -> pd.DataFrame | None:
    p = RESULTS.joinpath(*parts)
    return pd.DataFrame(read_jsonl(p)) if p.exists() else None


# --------------------------------------------------------------------------- H1

def h1():
    df = load("h1", "hits.jsonl")
    if df is None:
        return out("## H1: skipped (no results/h1/hits.jsonl)\n")
    df = df[(df.dataset == "musique") | ((df.dataset == "hotpotqa") & (df.qtype == "bridge"))]
    df = df[df.hop_index.notna()].copy()
    df["hit10"] = df["hit@10"].astype(int)
    df = df[df.retriever.isin(["bm25", "dense", "hybrid"])]
    res = stats.logit_clustered("hit10 ~ hop_index + C(retriever) + C(dataset)", df, "qid")
    p = stats.one_sided_p(res, "hop_index", "<")
    primary_p["H1"] = p
    out("## H1: single-shot retrieval misses later hops (primary)")
    out(f"- logit coef on hop_index: {res.params['hop_index']:.3f} (SE {res.bse['hop_index']:.3f}), one-sided p = {p:.3g}")
    for name, g in df.groupby("retriever"):
        w = g.pivot_table(index="qid", columns="hop_index", values="hit10", aggfunc="mean")
        if {1.0, 2.0} <= set(w.columns):
            w = w[[1.0, 2.0]].dropna()
            est = stats.paired_bootstrap_diff(w[1.0], w[2.0])
            out(f"- {name}: Recall@10 hop1 - hop2 = {est} (n={len(w)})")
    rec = df.groupby(["retriever", "hop_index"]).hit10.mean().unstack()
    out("\nRecall@10 by retriever x hop index:\n")
    out(rec.round(3).to_markdown())
    out()


# --------------------------------------------------------------------------- H2

def h2():
    s, m = load("h2", "single_shot.jsonl"), load("h2", "ircot.jsonl")
    if s is None or m is None:
        return out("## H2: skipped (need single_shot and ircot)\n")
    j = s.merge(m, on="qid", suffixes=("_s", "_m"))
    out("## H2: iteration helps only when hops exist (primary)")
    mus = j[j.dataset_s == "musique"]
    t = stats.mcnemar_exact(mus.em_m, mus.em_s, alternative="greater")
    primary_p["H2"] = t["p_value"]
    out(f"- MuSiQue: IRCoT EM {mus.em_m.mean():.3f} vs single-shot {mus.em_s.mean():.3f}; "
        f"McNemar one-sided p = {t['p_value']:.3g} (IRCoT-only {t['a_only']}, single-only {t['b_only']})")
    for ds, g in j.groupby("dataset_s"):
        t = stats.mcnemar_exact(g.em_m, g.em_s)
        est = stats.paired_bootstrap_diff(g.em_m, g.em_s)
        calls = g.llm_calls_m.mean()
        out(f"- {ds}: EM diff (IRCoT - single) = {est}; McNemar two-sided p = {t['p_value']:.3g}; IRCoT mean LLM calls = {calls:.2f}")
        if ds == "nq":
            eq = -0.03 < est.ci_low and est.ci_high < 0.03
            out(f"  - NQ equivalence within ±3 pts: {'yes' if eq else 'not shown'}")
    long = pd.concat([
        mus[["qid", "hops_s", "em_s"]].rename(columns={"hops_s": "hops", "em_s": "em"}).assign(method="single"),
        mus[["qid", "hops_s", "em_m"]].rename(columns={"hops_s": "hops", "em_m": "em"}).assign(method="ircot"),
    ])
    long["em"] = long.em.astype(int)
    res = stats.logit_clustered("em ~ C(method, Treatment('single')) * hops", long, "qid")
    term = "C(method, Treatment('single'))[T.ircot]:hops"
    out(f"- MuSiQue method x hops interaction: {res.params[term]:.3f} (SE {res.bse[term]:.3f}), "
        f"one-sided p = {stats.one_sided_p(res, term, '>'):.3g}")
    for name in ("closed_book", "oracle"):
        d = load("h2", f"{name}.jsonl")
        if d is not None:
            out(f"- {name} EM by dataset: " + ", ".join(f"{k}={v:.3f}" for k, v in d.groupby("dataset").em.mean().items()))
    out()


# --------------------------------------------------------------------------- H3

def _crossover(df) -> float:
    r = stats.logit_clustered("em ~ np.log2(target_tokens) * C(condition) + C(dataset)", df, "qid")
    b_rag = r.params["C(condition)[T.rag]"]
    b_int = r.params["np.log2(target_tokens):C(condition)[T.rag]"]
    return float(2 ** (-b_rag / b_int)) if b_int != 0 else float("nan")


def h3():
    df = load("h3", "length.jsonl")
    if df is None:
        return out("## H3: skipped\n")
    df["em"] = df.em.astype(int)
    out("## H3: long context loses to RAG past a crossover length (primary)")
    res = stats.logit_clustered("em ~ np.log2(target_tokens) * C(condition) + C(dataset)", df, "qid")
    term = "np.log2(target_tokens):C(condition)[T.rag]"
    p = stats.one_sided_p(res, term, ">")
    primary_p["H3"] = p
    out(f"- interaction (RAG degrades less per doubling of length): {res.params[term]:.3f} "
        f"(SE {res.bse[term]:.3f}), one-sided p = {p:.3g}")
    out("\nEM by condition x length:\n")
    out(df.pivot_table(index="length_name", columns="condition", values="em", aggfunc="mean")
        .reindex(["2K", "8K", "16K", "32K"]).round(3).to_markdown())
    x = _crossover(df)
    by_q = {q: g for q, g in df.groupby("qid")}
    qids = list(by_q)
    rng = np.random.default_rng(13)
    boots = []
    for _ in range(1000):  # cluster bootstrap: resample whole questions
        pick = rng.choice(qids, size=len(qids), replace=True)
        parts = [by_q[q].assign(qid=f"{q}#{i}") for i, q in enumerate(pick)]
        try:
            boots.append(_crossover(pd.concat(parts)))
        except Exception:
            pass
    boots = np.array([b for b in boots if np.isfinite(b)])
    lo, hi = np.quantile(boots, [0.025, 0.975]) if len(boots) else (np.nan, np.nan)
    inside = 2_000 <= x <= 31_000
    out(f"\n- crossover length: {x:,.0f} tokens [{lo:,.0f}, {hi:,.0f}] "
        f"({'inside' if inside else 'outside'} the tested 2K-32K range)")
    out()


# --------------------------------------------------------------------------- H4, H5

def _quadratic(df, y, cluster, logistic=True):
    if logistic:
        r = stats.logit_clustered(f"{y} ~ gold_position + I(gold_position**2)", df, cluster)
    else:
        import statsmodels.formula.api as smf

        groups = df[cluster].astype("category").cat.codes.to_numpy()
        r = smf.ols(f"{y} ~ gold_position + I(gold_position**2)", df).fit(cov_type="cluster", cov_kwds={"groups": groups})
    b1, b2 = r.params["gold_position"], r.params["I(gold_position ** 2)"]
    vertex = -b1 / (2 * b2) if b2 != 0 else float("nan")
    p = stats.one_sided_p(r, "I(gold_position ** 2)", ">")
    return b2, p, vertex


def h4():
    pos, gap = load("h4", "position.jsonl"), load("h4", "gap.jsonl")
    out("## H4: position effects (secondary, uncorrected)")
    if pos is not None:
        pos["em"] = pos.em.astype(int)
        b2, p, v = _quadratic(pos, "em", "qid")
        u = b2 > 0 and 1 < v < 20
        out(f"- (a) quadratic term {b2:.4f}, one-sided p = {p:.3g}, vertex at position {v:.1f} -> U-shape {'supported' if u and p < 0.05 else 'not supported'}")
        out("  EM by position: " + ", ".join(f"{k}: {v:.3f}" for k, v in pos.groupby("gold_position").em.mean().items()))
    if gap is not None:
        gap["em"] = gap.em.astype(int)
        r = stats.logit_clustered("em ~ gap", gap, "qid")
        out(f"- (b) slope on gap: {r.params['gap']:.4f}, one-sided p = {stats.one_sided_p(r, 'gap', '<'):.3g}")
        out("  EM by gap: " + ", ".join(f"{k}: {v:.3f}" for k, v in gap.groupby("gap").em.mean().items()))
    out()


def h5():
    df = load("h5", "attention.jsonl")
    if df is None:
        return out("## H5: skipped\n")
    out("## H5: attention tracks evidence use (secondary, correlational)")
    est = stats.auroc_ci(df.gold_share, df.em.astype(int))
    out(f"- (a) AUROC of gold attention share for correctness: {est} -> "
        f"{'supported' if est.ci_low > 0.5 else 'not supported'}")
    b2, p, v = _quadratic(df, "gold_share", "qid", logistic=False)
    out(f"- (b) quadratic term on attention share {b2:.5f}, one-sided p = {p:.3g}, vertex {v:.1f}")
    out("  mean gold share by position: " + ", ".join(
        f"{k}: {x:.3f}" for k, x in df.groupby("gold_position").gold_share.mean().items())
        + f" (uniform = {df.uniform_share.iloc[0]:.3f})")
    out()


# --------------------------------------------------------------------------- H6

def h6():
    df = load("h6", "vidore.jsonl")
    if df is None:
        return out("## H6: skipped\n")
    out("## H6: looking beats reading on visual pages (secondary)")
    gains = {}
    for grp, g in df.groupby("group"):
        w = g.pivot_table(index=["subset", "query"], columns="method", values="ndcg@5", aggfunc="first").dropna()
        best_ocr = max(("ocr_bm25", "ocr_dense"), key=lambda m: w[m].mean())
        t = stats.wilcoxon_paired(w["colqwen"], w[best_ocr], alternative="greater")
        gains[grp] = (w["colqwen"] - w[best_ocr]).to_numpy()
        out(f"- {grp}: nDCG@5 ColQwen {w.colqwen.mean():.3f} vs {best_ocr} {w[best_ocr].mean():.3f}; "
            f"Wilcoxon one-sided p = {t['p_value']:.3g} (n={len(w)})")
    if {"visual", "text"} <= set(gains):
        rng = np.random.default_rng(13)
        v, t_ = gains["visual"], gains["text"]
        boots = [rng.choice(v, len(v)).mean() - rng.choice(t_, len(t_)).mean() for _ in range(10_000)]
        lo, hi = np.quantile(boots, [0.025, 0.975])
        out(f"- interaction (visual gain - text gain): {v.mean() - t_.mean():.3f} [{lo:.3f}, {hi:.3f}]")
    out()


# --------------------------------------------------------------------------- H7

def h7():
    s = load("h2", "single_shot.jsonl")
    if s is None:
        return out("## H7: skipped\n")
    out("## H7: confidence routing (primary)")
    au = {}
    for ds, g in s.groupby("dataset"):
        est = stats.auroc_ci(g.confidence, g.em.astype(int))
        ece = stats.expected_calibration_error(g.confidence, g.em.astype(int))
        au[ds] = g
        out(f"- (a) {ds}: AUROC {est}, ECE {ece:.3f}")
    if {"nq", "musique"} <= set(au):
        a, b = au["nq"], au["musique"]
        rng = np.random.default_rng(13)
        diffs = []
        for _ in range(10_000):
            ia, ib = rng.integers(0, len(a), len(a)), rng.integers(0, len(b), len(b))
            diffs.append(stats.auroc(a.confidence.values[ia], a.em.values[ia])
                         - stats.auroc(b.confidence.values[ib], b.em.values[ib]))
        diffs = np.array(diffs)
        point = stats.auroc(a.confidence, a.em) - stats.auroc(b.confidence, b.em)
        p = float(np.mean(diffs - point >= point))  # shift method, one-sided H1: diff > 0
        primary_p["H7"] = p
        lo, hi = np.nanquantile(diffs, [0.025, 0.975])
        out(f"- (a) AUROC(NQ) - AUROC(MuSiQue) = {point:.3f} [{lo:.3f}, {hi:.3f}], one-sided p = {p:.3g}")
    rpath = RESULTS / "h7" / "router.json"
    if rpath.exists():
        r = json.loads(rpath.read_text())
        o = np.load(RESULTS / "h7" / "test_outcomes.npz")
        em = stats.paired_bootstrap_diff(o["router"], o["ircot"])
        calls = stats.paired_bootstrap_diff(o["router_calls"], o["ircot_calls"])
        out(f"- (b) held-out half (n={r['test_n']}), threshold {r['threshold']:.3f}:")
        out(f"  - router EM {r['router']['em']:.3f}, calls {r['router']['mean_calls']:.2f}, escalation {r['router']['escalation_rate']:.2%}")
        out(f"  - always single EM {r['always_single']['em']:.3f} (1 call); always IRCoT EM {r['always_ircot']['em']:.3f}, calls {r['always_ircot']['mean_calls']:.2f}")
        out(f"  - router - IRCoT: EM {em}; calls {calls}")
    out()


def main():
    out("# Results summary\n")
    out("_Generated by scripts/analyze.py. Tests as pre-registered in HYPOTHESES.md._\n")
    for f in (h1, h2, h3, h4, h5, h6, h7):
        f()
    if primary_p:
        out("## Holm–Bonferroni across primary hypotheses\n")
        out("| Hypothesis | p | Holm-adjusted p | Reject H0 at 0.05 |\n|---|---|---|---|")
        for k, v in stats.holm(primary_p).items():
            out(f"| {k} | {v['p']:.3g} | {v['p_holm']:.3g} | {'yes' if v['reject'] else 'no'} |")
        missing = {"H1", "H2", "H3", "H7"} - set(primary_p)
        if missing:
            out(f"\n_Missing primary hypotheses (correction will change once they are added): {sorted(missing)}_")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
