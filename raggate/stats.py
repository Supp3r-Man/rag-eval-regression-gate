"""Statistics for the gate: paired bootstrap, Holm correction, Cohen's kappa. Stdlib only.

The unit of resampling is the QUESTION, and the same resampled question sets are reused for every
metric and slice in one gate run, so the baseline/candidate pairing is never broken and every test
sees the same bootstrap worlds.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from operator import itemgetter


def make_resamples(n: int, boot: int, seed: int) -> list:
    """`boot` resamples (with replacement) of `n` positions, as fast itemgetters."""
    if n < 2:
        raise ValueError("need at least 2 questions to bootstrap")
    rng = random.Random(seed)
    return [itemgetter(*[rng.randrange(n) for _ in range(n)]) for _ in range(boot)]


@dataclass(frozen=True)
class BootResult:
    diff: float          # observed candidate - baseline
    lo: float            # 95% percentile interval
    hi: float
    p_worse: float       # one-sided: P(bootstrap diff >= 0), small when the candidate is clearly worse
    n: int               # questions the metric is defined for


def paired_bootstrap(diffs: list[float | None], resamples: list) -> BootResult | None:
    """Mean of per-question (candidate - baseline) differences; None entries are 'not defined'.

    The statistic of a resample is sum(diff)/count over the resampled questions where the metric is
    defined, so metrics that only exist for answerable questions are handled without a second pass.
    """
    vals = [0.0 if d is None else d for d in diffs]
    mask = [0 if d is None else 1 for d in diffs]
    n_def = sum(mask)
    if n_def == 0:
        return None
    obs = sum(vals) / n_def
    boots = []
    for g in resamples:
        den = sum(g(mask))
        if den:
            boots.append(sum(g(vals)) / den)
    boots.sort()
    m = len(boots)
    lo = boots[int(0.025 * m)]
    hi = boots[min(m - 1, int(0.975 * m))]
    p = (1 + sum(1 for b in boots if b >= 0)) / (m + 1)
    return BootResult(obs, lo, hi, p, n_def)


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, returned in the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def cohens_kappa(a: list[bool], b: list[bool]) -> float:
    """Agreement beyond chance for two binary raters. Raises when chance agreement is total."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label lists")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe >= 1.0:
        raise ValueError("kappa is undefined when both raters use a single label")
    return (po - pe) / (1 - pe)


def kappa_interval(a: list[bool], b: list[bool], boot: int = 1000, seed: int = 5) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(a)
    ks = []
    for _ in range(boot):
        idx = [rng.randrange(n) for _ in range(n)]
        try:
            ks.append(cohens_kappa([a[i] for i in idx], [b[i] for i in idx]))
        except ValueError:
            continue
    if not ks:
        raise ValueError("kappa interval undefined")
    ks.sort()
    return ks[int(0.025 * len(ks))], ks[min(len(ks) - 1, int(0.975 * len(ks)))]
