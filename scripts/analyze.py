"""Run every pre-registered test in HYPOTHESES.md and write results/summary.md.

  python scripts/analyze.py

Parts whose result files are missing are skipped, so this can be re-run as experiments
finish. CPU only.
"""
import json

import numpy as np
import pandas as pd

from hla import stats
from hla.data import read_jsonl
from hla.runtime import RESULTS

lines: list[str] = []
primary_p: dict[str, float] = {}
LENGTH_ORDER = ["2K", "8K", "16K", "30K"]
MARGIN = 0.01  # "best length" = shortest prompt within 1 EM point of the best


def out(s: str = "") -> None:
    print(s)
    lines.append(s)


def fmt_p(p: float) -> str:
    return "< 1e-3 (no bootstrap resample as extreme)" if p == 0 else f"{p:.3g}"


def load(*parts) -> pd.DataFrame | None:
    p = RESULTS.joinpath(*parts)
    return pd.DataFrame(read_jsonl(p)) if p.exists() else None


# --------------------------------------------------------------------------- Exp 1A

def exp1a(df):
    out("## Exp 1A: accuracy vs prompt length, gold passage always present\n")
    ref = df[df.condition.isin(["closed_book", "gold_only", "nogold_related_2K"])]
    out(ref.groupby("condition")[["acc", "em", "context_tokens"]].mean().round(3).to_markdown())
    out()
    grid = df[df.length.isin(LENGTH_ORDER) & df.gold_in_context].copy()
    grid["log2_tokens"] = np.log2(grid.context_tokens)
    for kind, g in grid.groupby("kind"):
        out(f"Accuracy, {kind} distractors (rows: length, columns: gold position):\n")
        out(g.pivot_table(index="length", columns="position", values="acc")
            .reindex(index=LENGTH_ORDER, columns=["start", "middle", "end"]).round(3).to_markdown())
        out()

    res = stats.logit_clustered("acc ~ log2_tokens + C(kind) + C(position)", grid, "qid")
    p = stats.one_sided_p(res, "log2_tokens", "<")
    primary_p["H1"] = p
    out(f"**H1 (primary).** logit coefficient on log2(context tokens) = {res.params['log2_tokens']:.3f} "
        f"(SE {res.bse['log2_tokens']:.3f}; each doubling multiplies the odds of a correct answer by "
        f"{np.exp(res.params['log2_tokens']):.2f}), one-sided p = {p:.3g}\n")

    r2 = stats.logit_clustered("acc ~ log2_tokens * C(kind) + C(position)", grid, "qid")
    t = "log2_tokens:C(kind)[T.unrelated]"
    out(f"- length × distractor type: {r2.params[t]:.3f} (p = {r2.pvalues[t]:.3g}); "
        f"> 0 means unrelated text hurts less than related text")
    wide = df.pivot_table(index="qid", columns="condition", values="acc")
    for kind in ("related", "unrelated"):
        long_ = wide[[f"{kind}_30K_{p}" for p in ("start", "middle", "end")]].mean(axis=1)
        d = stats.paired_bootstrap_diff(wide["gold_only"], long_)
        out(f"- gold only − {kind} 30K (mean over positions): {d}")
    out()
    exp1a_format(grid)


SHORT_WORDS = 8  # an answer longer than this was not a short answer


def exp1a_format(grid):
    """Secondary: format compliance by length, and whether accuracy holds up when only
    short answers count. A long answer that restates part of the prompt can contain the
    gold answer by chance, which would inflate accuracy where answers get long."""
    g = grid.copy()
    g["words"] = g.pred.fillna("").str.split().str.len()
    g["long_answer"] = g.words > SHORT_WORDS
    g["acc_short_only"] = g.acc.where(~g.long_answer, 0.0)  # long answers count as wrong
    t = g.groupby(["kind", "length"]).agg(acc=("acc", "mean"), em=("em", "mean"),
                                          acc_short_only=("acc_short_only", "mean"),
                                          long_answers=("long_answer", "mean"),
                                          mean_words=("words", "mean"))
    t = t.reindex([(k, l) for k in ("related", "unrelated") for l in LENGTH_ORDER if (k, l) in t.index])
    out(f"Format compliance (secondary): share of answers over {SHORT_WORDS} words, and accuracy "
        "when long answers are counted as wrong (a lower bound on accuracy):\n")
    out(t.round(3).to_markdown())
    out()
    r = stats.logit_clustered("acc_short_only ~ log2_tokens + C(kind) + C(position)", g, "qid")
    out(f"- H1 refit on the lower bound: log2(tokens) coefficient {r.params['log2_tokens']:.3f} "
        f"(SE {r.bse['log2_tokens']:.3f}); not a primary test\n")


# --------------------------------------------------------------------------- Exp 1B

def _reader_curve(df):
    """EM vs log2(tokens): gold only, then related distractors averaged over positions."""
    g = df[df.condition == "gold_only"]
    pts = [(np.log2(max(g.context_tokens.mean(), 1)), g.acc.mean())]
    rel = df[(df.kind == "related") & df.length.isin(LENGTH_ORDER) & df.gold_in_context]
    for L in LENGTH_ORDER:
        s = rel[rel.length == L]
        pts.append((np.log2(s.context_tokens.mean()), s.acc.mean()))
    x, y = map(np.array, zip(*sorted(pts)))
    return x, y


def predicted_curve(df, rk, ks):
    """Predicted RAG EM(k) = R(k) * reader(L(k)) + (1 - R(k)) * EM without the gold."""
    x, y = _reader_curve(df)
    rag = df[df.condition.str.startswith("rag")]
    tok_per_passage = rag.context_tokens.sum() / rag.n_passages.sum()
    a0 = df[df.condition == "nogold_related_2K"].acc.mean()
    ranks = rk.gold_rank.to_numpy(dtype=float)
    R = np.array([np.mean(np.nan_to_num(ranks, nan=np.inf) <= k) for k in ks])
    L = ks * tok_per_passage
    reader = np.interp(np.log2(L), x, y)  # flat beyond the measured range
    return R * reader + (1 - R) * a0, R, L


def best_k(pred, ks):
    return int(ks[np.argmax(pred >= pred.max() - MARGIN)])


def exp1b(df, rk):
    out("## Exp 1B: how long should the RAG prompt be?\n")
    ks = np.arange(1, 201)
    pred, R, L = predicted_curve(df, rk, ks)
    k_star = best_k(pred, ks)

    qids = rk.qid.to_numpy()
    by_q = {q: g for q, g in df.groupby("qid")}
    rk_by_q = rk.set_index("qid")
    rng = np.random.default_rng(13)
    boots = []
    for _ in range(500):
        pick = rng.choice(qids, size=len(qids))
        d = pd.concat([by_q[q] for q in pick])
        boots.append(best_k(predicted_curve(d, rk_by_q.loc[pick].reset_index(), ks)[0], ks))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    out(f"**Best k** (shortest prompt within {MARGIN * 100:.0f} EM point of the best predicted): "
        f"**k = {k_star}** (~{L[k_star - 1]:.0f} tokens), 95% bootstrap range [{lo:.0f}, {hi:.0f}]\n")

    out("| k | tokens | retriever recall | predicted EM | observed EM | EM per 1K tokens (predicted) |")
    out("|---|---|---|---|---|---|")
    obs = {int(c[5:]): g.acc for c, g in df[df.condition.str.startswith("rag_k")].groupby("condition")}
    for k in sorted(set([1, 2, 3, 5, 10, 20, 50, 100, 200, k_star])):
        o = obs.get(k)
        o_s = f"{o.mean():.3f}" if o is not None else ""
        if o is not None:
            e = stats.bootstrap(lambda idx, a=o.to_numpy(): float(a[idx].mean()), len(o), n_boot=2000)
            o_s += f" [{e.ci_low:.3f}, {e.ci_high:.3f}]"
        out(f"| {k} | {L[k - 1]:.0f} | {R[k - 1]:.3f} | {pred[k - 1]:.3f} | {o_s} | "
            f"{1000 * pred[k - 1] / L[k - 1]:.3f} |")
    out("\nThe prediction is checked against real top-k RAG (observed EM, 95% CI) at k = 1, 5, 20, 50.\n")


# --------------------------------------------------------------------------- Exp 1 attention

def exp1_attention(att):
    out("## Exp 1: attention on the gold passage\n")
    # gold only has a single passage, so its share is 1 by construction: it carries no
    # information about correctness and would pull the stratified AUROC towards 0.5
    att = att[att.n_passages > 1]
    est = stats.cluster_bootstrap(att, "qid", lambda d: stats.stratified_auroc(d.gold_share, d.acc, d.condition), null=0.5)
    p = stats.one_sided_from_two(est, 0.5, ">")
    primary_p["H2"] = p
    out(f"**H2 (primary).** Condition-stratified AUROC of gold attention share for correctness "
        f"({len(att)} rows, {att.qid.nunique()} questions): {est.value:.3f} "
        f"[{est.ci_low:.3f}, {est.ci_high:.3f}], one-sided p = {fmt_p(p)}\n")
    g = att.groupby(["kind", "length"])[["gold_share", "gold_length_share", "acc"]].mean()
    out(g.round(4).to_markdown())
    out()
    a = att[att.length.isin(LENGTH_ORDER)].copy()
    a["log2_tokens"] = np.log2(a.context_tokens)
    import statsmodels.formula.api as smf
    ols = smf.ols("np.log(gold_share) ~ log2_tokens * C(kind) + C(position)", a).fit(
        cov_type="cluster", cov_kwds={"groups": a.qid.astype("category").cat.codes})
    out(f"- log(gold share) per doubling of length: {ols.params['log2_tokens']:.3f} "
        f"(p = {ols.pvalues['log2_tokens']:.3g}); a slope of -1 would mean pure 1/n dilution")
    a["log_lift"] = np.log(a.gold_share / a.gold_length_share)
    lift = stats.cluster_bootstrap(a, "qid", lambda d: stats.stratified_auroc(d.log_lift, d.acc, d.condition), null=0.5)
    out(f"- robustness, length-adjusted share (log lift): stratified AUROC {lift}")
    out()


# --------------------------------------------------------------------------- Exp 1C

def lookup_layer(r) -> int:
    """The layer that does the lookup: the one with the largest needle share at the
    shortest test length (N = 16, inside the training range), standard attention. The
    last layer is not it for every model (rope looks up in layer 2, none in layer 4)."""
    shares = r["eval"]["related"][str(min(int(n) for n in r["eval"]["related"]))]["standard"]["needle_share"]
    return int(np.argmax(shares))


def exp1c():
    files = sorted((RESULTS / "tiny").glob("*.json"))
    if not files:
        return out("## Exp 1C: skipped (no results/tiny/*.json)\n")
    out("## Exp 1C: tiny GPT, multi-query associative recall (trained on N = 8–64 facts)\n")
    rows, dropped = [], []
    for f in files:
        r = json.loads(f.read_text())
        if r.get("train_curve") and r["train_curve"][-1]["acc"] < 0.95:  # pre-registered exclusion
            dropped.append(f.stem)
            continue
        L = lookup_layer(r)
        for kind, by_n in r["eval"].items():
            for n, modes in by_n.items():
                for mode, v in modes.items():
                    rows.append({"variant": r["variant"], "seed": r["seed"], "kind": kind, "N": int(n),
                                 "mode": mode, "acc": v["acc"], "lookup_layer": L + 1,
                                 "needle_share_lookup": v["needle_share"][L]})
    if dropped:
        out(f"Excluded (final training accuracy < 0.95): {', '.join(dropped)}\n")
    if rows:
        ll = pd.DataFrame(rows).groupby(["variant", "seed"]).lookup_layer.first()
        out("Lookup layer (largest needle share at N = 16): "
            + ", ".join(f"{v} s{sd}: {L}" for (v, sd), L in ll.items()) + "\n")
    if not rows:
        return
    d = pd.DataFrame(rows)
    for kind in ("related", "unrelated"):
        for metric in ("acc", "needle_share_lookup"):
            t = d[(d.kind == kind) & (d["mode"] == "standard")].groupby(["variant", "N"])[metric] \
                .agg(lambda s: f"{s.mean():.3f} ± {s.std(ddof=1) if len(s) > 1 else 0:.3f}").unstack("N")
            out(f"{metric}, {kind} distractors, standard attention (mean ± SD over seeds):\n")
            out(t.to_markdown())
            out()
    eff = d.pivot_table(index=["variant", "seed", "kind", "N"], columns="mode", values="acc").reset_index()
    eff["logn_gain"] = eff["logn"] - eff["standard"]
    out("Inference-time log-n scaling, accuracy gain (same weights), mean over seeds [min, max]:\n")
    out(eff[eff.N > 64].groupby(["variant", "kind", "N"]).logn_gain
        .agg(lambda s: f"{s.mean():+.3f} [{s.min():+.3f}, {s.max():+.3f}]").unstack("N").to_markdown())
    out("\nH3 is descriptive (3 seeds per variant) and is not part of the Holm family.\n")
    exp1c_finetune(d, set(dropped))


def exp1c_finetune(base: pd.DataFrame, dropped: set[str]):
    """Fine-tuning on longer inputs (long, N = 8-256) vs the same amount of extra training on
    the original lengths (same, N = 8-64), next to the base model with and without log-n."""
    files = sorted((RESULTS / "tiny" / "finetune").glob("*.json"))
    if not files:
        return out("### Exp 1C fine-tuning: skipped (no results/tiny/finetune/*.json)\n")
    rows = []
    for f in files:
        r = json.loads(f.read_text())
        if f"{r['variant']}_s{r['seed']}" in dropped:  # base model excluded, so is its fine-tune
            continue
        L = lookup_layer(r)
        for kind, by_n in r["eval"].items():
            for n, modes in by_n.items():
                v = modes["standard"]
                rows.append({"variant": r["variant"], "seed": r["seed"], "kind": kind, "N": int(n),
                             "model": f"ft_{r['mode']}", "acc": v["acc"],
                             "needle_share_lookup": v["needle_share"][L]})
    if not rows:
        return
    b = base.copy()
    b["model"] = "base_" + b["mode"]  # base_standard, base_logn
    d = pd.concat([b.drop(columns=["mode", "lookup_layer"]), pd.DataFrame(rows)])
    order = ["base_standard", "base_logn", "ft_same", "ft_long"]
    out("### Exp 1C fine-tuning for longer inputs (500 steps; long: N = 8–256, same: N = 8–64)\n")
    for kind in ("related", "unrelated"):
        for metric in ("acc", "needle_share_lookup"):
            t = d[d.kind == kind].groupby(["variant", "model", "N"])[metric].mean().unstack("N")
            t = t.reindex([(v, m) for v in sorted(d.variant.unique()) for m in order if (v, m) in t.index])
            out(f"{metric}, {kind} distractors (mean over seeds):\n")
            out(t.round(3).to_markdown())
            out()
    w = d[d.model.isin(["ft_long", "ft_same"])].pivot_table(
        index=["variant", "seed", "kind", "N"], columns="model", values="acc").reset_index()
    w["gain"] = w["ft_long"] - w["ft_same"]
    out("Effect of training on longer inputs (ft_long − ft_same accuracy), mean [min, max] over seeds:\n")
    out(w.groupby(["variant", "kind", "N"]).gain
        .agg(lambda s: f"{s.mean():+.3f} [{s.min():+.3f}, {s.max():+.3f}]").unstack("N").to_markdown())
    out("\nDescriptive (3 seeds per variant); not part of the Holm family.\n")


# --------------------------------------------------------------------------- Exp 2

def exp2(df):
    out("## Exp 2: single-shot vs chain of thought vs iterative retrieval\n")
    df["all_golds"] = df.golds_found.apply(all)
    out(df.groupby(["dataset", "method"])[["acc", "em", "f1", "all_golds", "llm_calls", "prompt_tokens", "fallback"]]
        .mean().round(3).to_markdown())
    out()
    mq = df[df.dataset == "musique"].pivot_table(index="qid", columns="method", values="acc")
    t = stats.mcnemar_exact(mq["ircot"], mq["single"], alternative="greater")
    primary_p["H4"] = t["p_value"]
    out(f"**H4 (primary).** MuSiQue, IRCoT − single-shot = {t['diff']:+.3f} (IRCoT-only right "
        f"{t['a_only']}, single-only right {t['b_only']}), one-sided exact McNemar p = {t['p_value']:.3g}\n")
    for a, b in (("single_cot", "single"), ("ircot", "single_cot")):
        tt = stats.mcnemar_exact(mq[a], mq[b])
        out(f"- MuSiQue {a} − {b}: {tt['diff']:+.3f} (two-sided p = {tt['p_value']:.3g})")

    s = df[(df.dataset == "musique") & (df.method == "single")].set_index("qid").all_golds
    for had, label in ((True, "single-shot already had both golds"), (False, "single-shot missed a gold")):
        q = s[s == had].index
        if len(q) > 1:
            d = stats.paired_bootstrap_diff(mq.loc[q, "ircot"], mq.loc[q, "single"])
            out(f"- H5, {label} (n = {len(q)}): IRCoT − single = {d}")
    ir = df[(df.dataset == "musique") & (df.method == "ircot")]
    steps = ir.found_by_step.apply(lambda f: [all(x) for x in f])
    mx = steps.apply(len).max()
    curve = [np.mean([f[min(i, len(f) - 1)] for f in steps]) for i in range(mx)]
    out(f"- MuSiQue IRCoT, share of questions with both golds in context after step 0..{mx - 1}: "
        + ", ".join(f"{c:.2f}" for c in curve))

    two = df[df.method.isin(["single", "ircot"])].copy()
    two["ircot"] = (two.method == "ircot").astype(int)
    two["nq"] = (two.dataset == "nq").astype(int)
    r = stats.logit_clustered("acc ~ ircot * nq", two, "qid")
    p = stats.one_sided_p(r, "ircot:nq", "<")
    primary_p["H6"] = p
    out(f"\n**H6 (primary).** method × dataset interaction (IRCoT gain smaller on NQ): "
        f"{r.params['ircot:nq']:.3f} (SE {r.bse['ircot:nq']:.3f}), one-sided p = {p:.3g}")
    nq = df[df.dataset == "nq"].pivot_table(index="qid", columns="method", values="acc")
    d = stats.paired_bootstrap_diff(nq["ircot"], nq["single"])
    e90 = stats.bootstrap(lambda idx, a=(nq["ircot"] - nq["single"]).to_numpy(): float(a[idx].mean()),
                          len(nq), alpha=0.10)
    equiv = -0.03 < e90.ci_low and e90.ci_high < 0.03
    out(f"- NQ IRCoT − single = {d}; 90% CI [{e90.ci_low:.3f}, {e90.ci_high:.3f}] "
        f"{'inside' if equiv else 'not inside'} ±0.03 (equivalence)")
    c = df.groupby(["dataset", "method"]).llm_calls.mean().unstack()
    out(f"- mean LLM calls, IRCoT vs single: MuSiQue {c.loc['musique', 'ircot']:.2f}, NQ {c.loc['nq', 'ircot']:.2f}")
    out()


def main():
    out("# Results\n")
    e1, rk, att, e2 = load("exp1", "exp1.jsonl"), load("exp1", "rankings.jsonl"), \
        load("exp1", "attention.jsonl"), load("exp2", "exp2.jsonl")
    if e1 is not None:
        exp1a(e1)
        if rk is not None:
            exp1b(e1, rk)
    else:
        out("## Exp 1A/1B: skipped (no results/exp1/exp1.jsonl)\n")
    exp1_attention(att) if att is not None else out("## Exp 1 attention: skipped\n")
    exp1c()
    exp2(e2) if e2 is not None else out("## Exp 2: skipped (no results/exp2/exp2.jsonl)\n")
    if primary_p:
        out("## Primary tests, Holm–Bonferroni across H1, H2, H4, H6\n")
        out("| test | p | Holm p | reject at 0.05 |\n|---|---|---|---|")
        for k, v in sorted(stats.holm(primary_p).items()):
            out(f"| {k} | {fmt_p(v['p'])} | {fmt_p(v['p_holm'])} | {v['reject']} |")
        if len(primary_p) < 4:
            out("\n(Holm applied only to the tests available so far; rerun when all four are done.)")
    path = RESULTS / "summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
