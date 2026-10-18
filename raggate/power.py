"""How big a drop can this gate see? Simulation, not guesswork.

A gate that never fires is not safe, it is blind. This module simulates paired baseline/candidate
outcomes and asks how often the gate's statistical rule fires:

  * `churn`  per-direction rate at which a question flips between two runs for reasons unrelated to
             the change (sampling noise, retrieval ties). 0 means a fully deterministic pipeline.
  * `drop`   a real regression: the candidate loses `drop` of the questions net.

Each simulated question is "worse" (pass -> fail) with probability churn + drop, "better" with
probability churn, else unchanged, which is all a paired test depends on. Power is the fraction of
simulated gate runs that fail the gate. Two thresholds are reported: a lone test at alpha, and the
conservative case of Holm's first step in a family of `family` tests (alpha / family), which is what
the weakest metric faces in the real gate. At drop = 0 the same numbers are the FALSE-ALARM rate.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class PowerCell:
    n: int
    drop: float
    churn: float
    single: float      # fraction of simulated runs failing with the lone-test threshold
    family: float      # ... with the alpha/family threshold


def _fires(diffs: list[int], rng: random.Random, boot: int, alpha: float, family: int,
           min_drop: float) -> tuple[bool, bool]:
    n = len(diffs)
    obs = sum(diffs) / n
    if -obs < min_drop:
        return False, False
    ge0 = 0
    for _ in range(boot):
        if sum(rng.choices(diffs, k=n)) >= 0:
            ge0 += 1
    p = (1 + ge0) / (boot + 1)
    return p < alpha, p < alpha / family


def simulate_power(n: int, drop: float, churn: float, trials: int = 200, boot: int = 400,
                   alpha: float = 0.05, family: int = 15, min_drop: float = 0.02, seed: int = 1) -> PowerCell:
    if 1 / (boot + 1) >= alpha / family:
        raise ValueError(f"boot={boot} is too coarse for alpha/family={alpha / family:.4f}: the test could never fire")
    rng = random.Random(f"{seed}:{n}:{drop}:{churn}")
    worse, better = churn + drop, churn
    s = f = 0
    for _ in range(trials):
        diffs = []
        for _ in range(n):
            u = rng.random()
            diffs.append(-1 if u < worse else (1 if u < worse + better else 0))
        a, b = _fires(diffs, rng, boot, alpha, family, min_drop)
        s += a
        f += b
    return PowerCell(n, drop, churn, s / trials, f / trials)


def power_table(ns=(50, 100, 200, 400), drops=(0.0, 0.02, 0.04, 0.06, 0.10, 0.15), churn: float = 0.04,
                **kw) -> list[PowerCell]:
    return [simulate_power(n, d, churn, **kw) for n in ns for d in drops]


def min_detectable_drop(cells: list[PowerCell], n: int, target: float = 0.8, family: bool = True) -> float | None:
    """Smallest simulated drop with power >= target at this n, or None if none on the grid reaches it."""
    ok = [c.drop for c in cells if c.n == n and (c.family if family else c.single) >= target]
    return min(ok) if ok else None


def critical_false_alarm(m: int, churn: float, min_net_drop: int = 1) -> float:
    """Exact P(net loss >= min_net_drop) in a critical slice of m questions when NOTHING changed.

    Each question independently goes pass->fail with prob `churn`, fail->pass with prob `churn`.
    Computed by dynamic programming over the net count, so there is no sampling error.
    """
    dist = {0: 1.0}
    for _ in range(m):
        nxt: dict[int, float] = {}
        for net, pr in dist.items():
            for step, w in ((1, churn), (-1, churn), (0, 1 - 2 * churn)):
                nxt[net + step] = nxt.get(net + step, 0.0) + pr * w
        dist = nxt
    return sum(pr for net, pr in dist.items() if net >= min_net_drop)
