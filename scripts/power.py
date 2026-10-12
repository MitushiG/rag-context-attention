"""Choose the main sample size by simulation, using pilot base rates (CPU, a few minutes).

  python scripts/power.py

Pre-registered rule (HYPOTHESES.md): N is the smallest value in N_GRID at which both
P1 and P3 reach 80% power for the smallest effects of interest below, at the strictest
Holm threshold (one-sided alpha = 0.05 / 3). If none does, N = max(N_GRID). The main
sample is the first N questions of data/samples/main_pool_ids.txt.

Simulation model: logistic, with a per-question random intercept (SD = SD_U, the
between-question difficulty spread) that makes a question's rows correlated, analysed
with the same clustered regressions as analyze.py.

Writes data/samples/main_ids.txt and results/pilot/power.md. Commit both.
"""
import argparse

import numpy as np
import pandas as pd

from hla import stats
from hla.data import read_jsonl
from hla.runtime import RESULTS, SAMPLES

N_GRID = (200, 300, 400, 500, 600)
P1_DROP = 0.08   # EM drop from RAG to the 28K budget, retrieval-complete questions
P3_DROP = 0.05   # EM drop from close to far gold passages
SD_U = 1.5
ALPHA = 0.05 / 3
TARGET = 0.80


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _solve(target, offset, u):
    """Intercept a such that mean(sigmoid(a + offset + u)) == target (bisection)."""
    lo, hi = -15.0, 15.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _sigmoid(mid + offset + u).mean() < target else (lo, mid)
    return (lo + hi) / 2


def calibrate(p_base, drop, u):
    a = _solve(p_base, 0.0, u)
    b = _solve(p_base - drop, a, u)  # b = shift on the logit scale at the largest change
    return a, b


def sim_p1(n_q, x, p_base, rng, u_cal):
    a, b_max = calibrate(p_base, P1_DROP, u_cal)
    slope = b_max / (x[-1] - x[0])
    u = rng.normal(0, SD_U, n_q)
    rows = [(q, xi, int(rng.uniform() < _sigmoid(a + u[q] + slope * (xi - x[0]))))
            for q in range(n_q) for xi in x]
    df = pd.DataFrame(rows, columns=["qid", "log2_tokens", "em"])
    return stats.one_sided_p(stats.logit_clustered("em ~ log2_tokens", df, "qid"), "log2_tokens", "<")


def sim_p3(n_q, p_base, rng, u_cal):
    a, c = calibrate(p_base, P3_DROP, u_cal)
    u = rng.normal(0, SD_U, n_q)
    rows = [(q, loc, far, int(rng.uniform() < _sigmoid(a + u[q] + c * far)))
            for q in range(n_q) for loc in ("early", "middle", "late") for far in (0, 1)]
    df = pd.DataFrame(rows, columns=["qid", "location", "far", "em"])
    return stats.one_sided_p(stats.logit_clustered("em ~ C(location) + far", df, "qid"), "far", "<")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=300)
    args = ap.parse_args()
    rng = np.random.default_rng(13)
    u_cal = np.random.default_rng(0).normal(0, SD_U, 20_000)

    e1 = pd.DataFrame(read_jsonl(RESULTS / "pilot" / "exp1.jsonl"))
    e3 = pd.DataFrame(read_jsonl(RESULTS / "pilot" / "exp3.jsonl"))
    rag = e1[e1.condition == "rag"]
    frac_complete = rag.retrieval_complete.mean()
    p1_base = rag[rag.retrieval_complete].em.mean()
    x = np.log2([e1[e1.condition == c].context_tokens.median() for c in ("rag", "4K", "12K", "28K")])
    p3_base = e3[e3.separation == "close"].em.mean()
    # keep the simulated drop possible when the pilot base rate is very low
    p1_base, p3_base = max(p1_base, P1_DROP + 0.05), max(p3_base, P3_DROP + 0.05)

    pool = (SAMPLES / "main_pool_ids.txt").read_text().split()
    lines = [
        "# Power analysis (from the pilot)", "",
        f"- pilot: {rag.qid.nunique()} questions; retrieval-complete fraction {frac_complete:.2f}",
        f"- base EM: RAG, retrieval-complete {p1_base:.3f}; Exp 3 close {p3_base:.3f}",
        f"- effects: P1 drop {P1_DROP}, P3 drop {P3_DROP}; SD_U {SD_U}; alpha {ALPHA:.4f}; {args.sims} sims", "",
        "| N | P1 power | P3 power |", "|---|---|---|",
    ]
    chosen = None
    for n in N_GRID:
        n = min(n, len(pool))
        n1 = max(10, int(round(n * frac_complete)))
        pw1 = np.mean([sim_p1(n1, x, p1_base, rng, u_cal) < ALPHA for _ in range(args.sims)])
        pw3 = np.mean([sim_p3(n, p3_base, rng, u_cal) < ALPHA for _ in range(args.sims)])
        lines.append(f"| {n} | {pw1:.2f} | {pw3:.2f} |")
        print(lines[-1])
        if chosen is None and pw1 >= TARGET and pw3 >= TARGET:
            chosen = n
    note = "" if chosen else " (no N in the grid reached the target; using the maximum)"
    chosen = chosen or min(max(N_GRID), len(pool))
    lines += ["", f"**Chosen N = {chosen}**{note}"]

    (SAMPLES / "main_ids.txt").write_text("\n".join(pool[:chosen]) + "\n")
    out = RESULTS / "pilot" / "power.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"N = {chosen}; wrote {SAMPLES / 'main_ids.txt'} and {out}")


if __name__ == "__main__":
    main()
