"""Run every pre-registered test in HYPOTHESES.md and write results/<split>/summary.md.

  python scripts/analyze.py --split main     # the confirmatory analysis
  python scripts/analyze.py --split pilot    # sanity check only; not reported as results

Experiments whose result files are missing are skipped. CPU only.
"""
import argparse

import numpy as np
import pandas as pd

from hla import stats
from hla.data import read_jsonl
from hla.runtime import RESULTS, SPLITS

lines: list[str] = []
primary_p: dict[str, float] = {}
EXP1_ORDER = ["closed_book", "rag", "4K", "12K", "28K", "oracle"]


def out(s: str = "") -> None:
    print(s)
    lines.append(s)


def fmt_p(p: float) -> str:
    return "< 1e-3 (no bootstrap resample as extreme)" if p == 0 else f"{p:.3g}"


def load(split, name) -> pd.DataFrame | None:
    p = RESULTS / split / name
    return pd.DataFrame(read_jsonl(p)) if p.exists() else None


# --------------------------------------------------------------------------- Exp 1

def exp1(split):
    df = load(split, "exp1.jsonl")
    if df is None:
        return out("## Exp 1: skipped (no exp1.jsonl)\n")
    out("## Exp 1: retrieval vs. long context\n")
    df["log2_tokens"] = np.log2(df.context_tokens.clip(lower=1))
    tab = df.pivot_table(index="condition", values=["em", "f1", "context_tokens", "n_gold"], aggfunc="mean")
    out(tab.reindex([c for c in EXP1_ORDER if c in tab.index]).round(3).to_markdown())
    out()

    ranked = df[df.condition.isin(["rag", "4K", "12K", "28K"])].copy()
    comp = ranked[ranked.retrieval_complete]
    n_comp = comp.qid.nunique()
    res = stats.logit_clustered("em ~ log2_tokens", comp, "qid")
    p = stats.one_sided_p(res, "log2_tokens", "<")
    primary_p["P1"] = p
    out(f"**P1 (primary).** Retrieval-complete questions (n = {n_comp}): logit coefficient on "
        f"log2(context tokens) = {res.params['log2_tokens']:.3f} (SE {res.bse['log2_tokens']:.3f}), "
        f"one-sided p = {p:.3g}\n")

    out("EM by condition and by how many gold passages the context contains:\n")
    out(ranked.pivot_table(index="condition", columns="n_gold", values="em", aggfunc="mean")
        .reindex(["rag", "4K", "12K", "28K"]).round(3).to_markdown())
    out("\n(n_gold = 2 rows are evidence-*use* failures when wrong; n_gold < 2 rows may be retrieval failures.)\n")

    out("Secondary: RAG vs each budget, all questions (exact McNemar, two-sided, Holm within this family)\n")
    wide = ranked.pivot_table(index="qid", columns="condition", values="em")
    tests = {b: stats.mcnemar_exact(wide["rag"], wide[b]) for b in ("4K", "12K", "28K")}
    adj = stats.holm({b: t["p_value"] for b, t in tests.items()})
    for b, t in tests.items():
        d = stats.paired_bootstrap_diff(wide["rag"], wide[b])
        out(f"- RAG − {b}: {d}  (RAG-only right {t['a_only']}, {b}-only right {t['b_only']}, "
            f"p = {t['p_value']:.3g}, Holm p = {adj[b]['p_holm']:.3g})")

    ranked["both_gold"] = (ranked.n_gold == 2).astype(int)
    try:
        r2 = stats.logit_clustered("em ~ log2_tokens * both_gold", ranked, "qid")
        out(f"- length × both-gold-present interaction: {r2.params['log2_tokens:both_gold']:.3f} "
            f"(SE {r2.bse['log2_tokens:both_gold']:.3f}, p = {r2.pvalues['log2_tokens:both_gold']:.3g})")
    except Exception as e:  # sparse cells in small pilots
        out(f"- interaction model not estimable: {e}")

    best = stats.cluster_bootstrap(ranked, "qid", lambda d: float(np.argmax(
        d.groupby("condition").em.mean().reindex(["rag", "4K", "12K", "28K"]).to_numpy())))
    out(f"- best condition index (0 = RAG ... 3 = 28K): {best.value:.0f}, 95% bootstrap range "
        f"[{best.ci_low:.0f}, {best.ci_high:.0f}]")
    out()


# --------------------------------------------------------------------------- Exp 2

def exp2(split):
    df = load(split, "exp2.jsonl")
    if df is None:
        return out("## Exp 2: skipped (no exp2.jsonl)\n")
    out("## Exp 2: attention to gold evidence\n")
    e1 = df[df.source == "exp1"].copy()
    est = stats.cluster_bootstrap(e1, "qid", lambda d: stats.stratified_auroc(d.gold_share, d.em, d.condition), null=0.5)
    p = stats.one_sided_from_two(est, 0.5, ">")
    primary_p["P2"] = p
    out(f"**P2 (primary).** Condition-stratified AUROC of gold attention share for correctness "
        f"(Exp 1 rows with both golds in context, n = {len(e1)} rows / {e1.qid.nunique()} questions): "
        f"{est.value:.3f} [{est.ci_low:.3f}, {est.ci_high:.3f}], one-sided p = {fmt_p(p)}\n")

    per = []
    for c, g in e1.groupby("condition"):
        per.append({"condition": c, "n": len(g), "EM": g.em.mean(), "gold share": g.gold_share.mean(),
                    "length-proportional share": g.gold_length_share.mean(),
                    "AUROC": stats.auroc(g.gold_share, g.em)})
    out(pd.DataFrame(per).set_index("condition").reindex(["rag", "4K", "12K", "28K"]).dropna(how="all").round(3).to_markdown())
    out()

    e1["log2_tokens"] = np.log2(e1.context_tokens)
    e1["share_z"] = (e1.gold_share - e1.gold_share.mean()) / e1.gold_share.std()
    e1["log_lift"] = np.log(e1.gold_share / e1.gold_length_share)
    r = stats.logit_clustered("em ~ share_z * log2_tokens", e1, "qid")
    out(f"- logit em ~ share_z × log2(tokens): share_z {r.params['share_z']:.3f} (p = {r.pvalues['share_z']:.3g}), "
        f"interaction {r.params['share_z:log2_tokens']:.3f} (p = {r.pvalues['share_z:log2_tokens']:.3g})")
    lift = stats.cluster_bootstrap(e1, "qid", lambda d: stats.stratified_auroc(d.log_lift, d.em, d.condition), null=0.5)
    out(f"- robustness, length-adjusted share (log lift): stratified AUROC {lift}")

    e3 = df[df.source == "exp3"]
    if len(e3):
        import statsmodels.formula.api as smf
        g = e3.groupby(["location", "separation"])[["gold_share", "hop1_share", "hop2_share", "em"]].mean()
        out("\nExp 3 contexts: mean shares by layout\n")
        out(g.round(3).to_markdown())
        ols = smf.ols("gold_share ~ C(location) * C(separation)", e3).fit(
            cov_type="cluster", cov_kwds={"groups": e3.qid.astype("category").cat.codes})
        out(f"\n- OLS gold_share ~ location × separation (clustered): far main effect "
            f"{ols.params['C(separation)[T.far]']:.4f} (p = {ols.pvalues['C(separation)[T.far]']:.3g})")
    out()


# --------------------------------------------------------------------------- Exp 3

def exp3(split):
    df = load(split, "exp3.jsonl")
    if df is None:
        return out("## Exp 3: skipped (no exp3.jsonl)\n")
    out("## Exp 3: relative position of the two gold passages\n")
    out(df.pivot_table(index="location", columns="separation", values="em", aggfunc="mean")
        .reindex(["early", "middle", "late"]).round(3).to_markdown())
    out()
    res = stats.logit_clustered("em ~ C(location) + far", df, "qid")
    p = stats.one_sided_p(res, "far", "<")
    primary_p["P3"] = p
    out(f"**P3 (primary).** logit coefficient on far (adjusting for location) = {res.params['far']:.3f} "
        f"(SE {res.bse['far']:.3f}, odds ratio {np.exp(res.params['far']):.2f}), one-sided p = {p:.3g}\n")

    full = stats.logit_clustered("em ~ C(location) * far", df, "qid")
    inter = [t for t in full.params.index if ":" in t]
    w = full.wald_test(" , ".join(f"{t} = 0" for t in inter), scalar=True)
    out(f"- location × separation interaction (Wald, {len(inter)} df): p = {float(w.pvalue):.3g}")
    for loc, g in df.groupby("location"):
        wide = g.pivot_table(index="qid", columns="separation", values="em")
        t = stats.mcnemar_exact(wide["close"], wide["far"])
        out(f"- {loc}: close − far = {t['diff']:+.3f} (exact McNemar p = {t['p_value']:.3g})")
    df["hop1_first"] = df.hop1_first.astype(int)
    r = stats.logit_clustered("em ~ C(location) + far * hop1_first", df, "qid")
    out(f"- hop-1 passage first: {r.params['hop1_first']:.3f} (p = {r.pvalues['hop1_first']:.3g}); "
        f"far × hop1_first {r.params['far:hop1_first']:.3f} (p = {r.pvalues['far:hop1_first']:.3g})")
    out()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=SPLITS, default="main")
    args = ap.parse_args()
    out(f"# Results ({args.split})\n")
    if args.split == "pilot":
        out("> Pilot run: implementation check only, not confirmatory.\n")
    for f in (exp1, exp2, exp3):
        f(args.split)
    if primary_p:
        out("## Primary tests, Holm–Bonferroni across P1–P3\n")
        out("| test | p | Holm p | reject at 0.05 |\n|---|---|---|---|")
        for k, v in sorted(stats.holm(primary_p).items()):
            out(f"| {k} | {fmt_p(v['p'])} | {fmt_p(v['p_holm'])} | {v['reject']} |")
        if len(primary_p) < 3:
            out("\n(Holm applied only to the tests available so far; rerun when all three are done.)")
    path = RESULTS / args.split / "summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
