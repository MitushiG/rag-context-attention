"""Statistical tests used in HYPOTHESES.md.

All comparisons are paired: index i of every array refers to the same question.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import stats as sps


@dataclass
class Estimate:
    value: float
    ci_low: float
    ci_high: float
    p_value: float | None = None

    def __str__(self) -> str:
        if self.p_value is None:
            p = ""
        elif self.p_value == 0:
            p = ", p<1e-4"  # no bootstrap resample as extreme as observed
        else:
            p = f", p={self.p_value:.3g}"
        return f"{self.value:.4f} [{self.ci_low:.4f}, {self.ci_high:.4f}]{p}"


# --------------------------------------------------------------------------- bootstrap

def bootstrap(
    stat: Callable[[np.ndarray], float],
    n: int,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 13,
) -> Estimate:
    """Percentile bootstrap over question indices.

    ``stat`` takes an index array (a resample of range(n)) and returns a scalar. This
    keeps pairing intact: every array the statistic touches is indexed the same way.
    """
    rng = np.random.default_rng(seed)
    point = stat(np.arange(n))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = stat(rng.integers(0, n, size=n))
    boots = boots[~np.isnan(boots)]
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    # Two-sided p for H0: stat == 0. Centre the bootstrap distribution on 0 (shift method)
    # and ask how often it is at least as extreme as the observed value.
    p = float(np.mean(np.abs(boots - point) >= abs(point)))
    return Estimate(float(point), float(lo), float(hi), p)


def paired_bootstrap_diff(
    a: Sequence[float], b: Sequence[float], n_boot: int = 10_000, seed: int = 13
) -> Estimate:
    """Mean(a) - mean(b) with a paired bootstrap CI."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, "paired arrays must have the same length"
    d = a - b
    return bootstrap(lambda idx: float(np.mean(d[idx])), len(d), n_boot=n_boot, seed=seed)


# --------------------------------------------------------------------------- binary paired

def mcnemar_exact(
    a_correct: Sequence[int], b_correct: Sequence[int], alternative: str = "two-sided"
) -> dict:
    """Exact McNemar test on paired 0/1 outcomes.

    ``alternative='greater'`` tests that A is right more often than B among discordant
    pairs.
    """
    a = np.asarray(a_correct).astype(bool)
    b = np.asarray(b_correct).astype(bool)
    a_only = int(np.sum(a & ~b))
    b_only = int(np.sum(~a & b))
    n_disc = a_only + b_only
    if n_disc == 0:
        p = 1.0
    else:
        p = sps.binomtest(a_only, n_disc, 0.5, alternative=alternative).pvalue
    return {
        "a_only": a_only,
        "b_only": b_only,
        "diff": float(a.mean() - b.mean()),
        "p_value": float(p),
    }


# --------------------------------------------------------------------------- multiplicity

def holm(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm–Bonferroni step-down adjustment. Returns adjusted p and reject flag per key."""
    keys = sorted(p_values, key=p_values.get)
    m = len(keys)
    out, running_max, still_rejecting = {}, 0.0, True
    for i, k in enumerate(keys):
        adj = min(1.0, (m - i) * p_values[k])
        running_max = max(running_max, adj)
        still_rejecting = still_rejecting and p_values[k] <= alpha / (m - i)
        out[k] = {"p": p_values[k], "p_holm": running_max, "reject": still_rejecting}
    return out


# --------------------------------------------------------------------------- calibration

def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """AUROC via the Mann–Whitney U statistic (ties count half)."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels).astype(bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = sps.rankdata(s)
    u = ranks[y].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def auroc_ci(scores, labels, n_boot: int = 10_000, seed: int = 13) -> Estimate:
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    est = bootstrap(lambda idx: auroc(s[idx], y[idx]), len(s), n_boot=n_boot, seed=seed)
    # p-value here is for H0: AUROC == 0.5
    est_shift = bootstrap(lambda idx: auroc(s[idx], y[idx]) - 0.5, len(s), n_boot=n_boot, seed=seed)
    est.p_value = est_shift.p_value
    return est


def stratified_auroc(scores, labels, strata) -> float:
    """AUROC computed within each stratum and averaged, weighted by the number of
    (positive, negative) pairs. Equivalent to only comparing correct vs incorrect answers
    *within* the same condition, so a condition that lowers both attention share and
    accuracy cannot by itself produce a high AUROC."""
    s, y, g = np.asarray(scores, float), np.asarray(labels, int), np.asarray(strata)
    num = den = 0.0
    for k in np.unique(g):
        m = g == k
        n_pairs = y[m].sum() * (1 - y[m]).sum()
        if n_pairs:
            num += n_pairs * auroc(s[m], y[m])
            den += n_pairs
    return float(num / den) if den else float("nan")


def cluster_bootstrap(df, cluster_col: str, stat: Callable, null: float = 0.0,
                      n_boot: int = 2000, alpha: float = 0.05, seed: int = 13) -> Estimate:
    """Percentile bootstrap that resamples whole clusters (questions), so a question that
    contributes several rows (one per condition) stays intact. ``stat(df) -> float``.
    The p-value is two-sided for H0: stat == ``null`` (shift method)."""
    rng = np.random.default_rng(seed)
    groups = [np.asarray(g) for g in df.groupby(cluster_col, sort=False).indices.values()]
    point = stat(df)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        boots[b] = stat(df.iloc[np.concatenate([groups[i] for i in pick])])
    boots = boots[~np.isnan(boots)]
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    p = float(np.mean(np.abs(boots - point) >= abs(point - null)))
    return Estimate(float(point), float(lo), float(hi), p)


def one_sided_from_two(est: Estimate, null: float, direction: str) -> float:
    """Convert a two-sided bootstrap p into a one-sided one ('>' or '<' than ``null``)."""
    agrees = est.value > null if direction == ">" else est.value < null
    return est.p_value / 2 if agrees else 1 - est.p_value / 2


# --------------------------------------------------------------------------- regression

def logit_clustered(formula: str, df, cluster_col: str):
    """Logistic regression with cluster-robust SEs (clusters = questions).

    Needed when one question contributes several rows (one per condition), which makes
    rows non-independent.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    groups = df[cluster_col].astype("category").cat.codes.to_numpy()
    model = smf.glm(formula, data=df, family=sm.families.Binomial())
    return model.fit(cov_type="cluster", cov_kwds={"groups": groups})


def one_sided_p(result, term: str, direction: str) -> float:
    """One-sided p-value for a regression coefficient ('<' or '>' 0)."""
    z = result.params[term] / result.bse[term]
    return float(sps.norm.cdf(z) if direction == "<" else sps.norm.sf(z))
