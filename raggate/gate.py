"""The regression gate: baseline run vs candidate run on one split of the golden set.

Verdicts
  PASS          no significant drop and no critical-slice regression      -> exit 0
  FAIL          a significant drop, or a critical slice got worse         -> exit 1
  INCONCLUSIVE  the gate could not judge (broken set, errored or mismatched runs, too few
                questions, an unguarded critical slice)                   -> exit 2

Two different rules, on purpose:
  * Statistical rule, for everything that is not critical. A metric or slice FAILS the gate only if
    its paired-bootstrap one-sided p-value, Holm-adjusted over the whole family of tests, is below
    alpha AND the observed drop is at least `min_drop`. Holm keeps the family-wise false-alarm rate at
    alpha even though we test the 6 metrics plus every qtype/area slice; `min_drop` stops a
    statistically detectable but practically meaningless wiggle from blocking a release.
  * Critical-slice rule. In a critical slice (default: unanswerable questions, where a regression means
    hallucinating, and the auth area) a net loss of one question fails the gate with no significance
    test: the slice is small, a test would have no power, and you do not ship that regression anyway.
    The price is noise-sensitivity when the pipeline is non-deterministic: see `raggate power`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .attribution import CLASSES, breakdown
from .golden import GoldenSet
from .hygiene import check_golden, errors
from .metrics import METRICS, QScore, aggregate, metric_value, score_question
from .pipelines import Run
from .stats import holm, make_resamples, paired_bootstrap

PASS, FAIL, INCONCLUSIVE = "PASS", "FAIL", "INCONCLUSIVE"
EXIT = {PASS: 0, FAIL: 1, INCONCLUSIVE: 2}


@dataclass(frozen=True)
class GateConfig:
    split: str = "heldout"
    alpha: float = 0.05
    min_drop: float = 0.02
    boot: int = 2000
    seed: int = 17
    min_questions: int = 60
    min_slice_n: int = 8
    critical_slices: tuple[str, ...] = ("qtype:unanswerable", "area:auth")
    critical_min_net_drop: int = 1


@dataclass
class Row:
    name: str
    kind: str                 # metric | slice
    base: float
    cand: float
    diff: float
    lo: float
    hi: float
    p: float
    p_adj: float = 1.0
    n: int = 0
    significant_drop: bool = False


@dataclass
class CriticalRow:
    name: str
    n: int
    down: int                 # pass -> fail
    up: int                   # fail -> pass
    failed: bool


@dataclass
class GateReport:
    decision: str
    reasons: list[str]
    split: str
    n: int = 0
    baseline: str = ""
    candidate: str = ""
    config: GateConfig = field(default_factory=GateConfig)
    rows: list[Row] = field(default_factory=list)
    critical: list[CriticalRow] = field(default_factory=list)
    base_metrics: dict = field(default_factory=dict)
    cand_metrics: dict = field(default_factory=dict)
    base_failures: dict = field(default_factory=dict)
    cand_failures: dict = field(default_factory=dict)
    regressed: list[str] = field(default_factory=list)
    improved: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return EXIT[self.decision]


def _inconclusive(reasons, cfg, base, cand, warnings=()):
    return GateReport(INCONCLUSIVE, list(reasons), cfg.split, 0, base.pipeline, cand.pipeline, cfg,
                      warnings=list(warnings))


def _validate_run(run: Run, role: str, expected: list[str], cfg: GateConfig) -> list[str]:
    out = []
    if run.split != cfg.split:
        out.append(f"{role} run is for split {run.split!r}, gate is set to {cfg.split!r}")
    qids = [r.qid for r in run.results]
    if len(set(qids)) != len(qids):
        out.append(f"{role} run has duplicate question ids")
    errored = [r for r in run.results if r.error]
    if errored:
        out.append(f"{role} run has {len(errored)} errored question(s), first: {errored[0].qid}: {errored[0].error}")
    missing, extra = set(expected) - set(qids), set(qids) - set(expected)
    if missing:
        out.append(f"{role} run is missing {len(missing)} question(s) of the {cfg.split} split")
    if extra:
        out.append(f"{role} run has {len(extra)} question(s) not in the {cfg.split} split")
    return out


def run_gate(gs: GoldenSet, base: Run, cand: Run, cfg: GateConfig = GateConfig()) -> GateReport:
    items = gs.split(cfg.split)
    expected = [i.qid for i in items]

    reasons: list[str] = []
    findings = check_golden(gs, cfg.min_slice_n)
    for f in errors(findings):
        reasons.append(f"golden set: {f.code} ({f.ref}): {f.message}")
    warnings = [f"{f.code} ({f.ref}): {f.message}" for f in findings if f.severity == "warn"]
    reasons += _validate_run(base, "baseline", expected, cfg)
    reasons += _validate_run(cand, "candidate", expected, cfg)
    if len(items) < cfg.min_questions:
        reasons.append(f"only {len(items)} questions in {cfg.split}; need at least {cfg.min_questions}")
    present = {s for i in items for s in i.slices()}
    for s in cfg.critical_slices:
        if s not in present:
            reasons.append(f"critical slice {s} has no questions in {cfg.split}: nothing guards it")
    if reasons:
        return _inconclusive(reasons, cfg, base, cand, warnings)

    def score(run: Run) -> list[QScore]:
        by = {r.qid: r for r in run.results}
        k = int(run.config.get("k", 4))
        return [score_question(i, by[i.qid], k) for i in items]

    bs, cs = score(base), score(cand)
    resamples = make_resamples(len(items), cfg.boot, cfg.seed)
    critical_set = set(cfg.critical_slices)

    rows: list[Row] = []
    for m in METRICS:
        bv = [metric_value(s, m) for s in bs]
        cv = [metric_value(s, m) for s in cs]
        diffs = [None if b is None or c is None else c - b for b, c in zip(bv, cv)]
        res = paired_bootstrap(diffs, resamples)
        if res is None:
            continue
        defined = [(b, c) for b, c in zip(bv, cv) if b is not None and c is not None]
        rows.append(Row(f"metric:{m}", "metric", sum(b for b, _ in defined) / len(defined),
                        sum(c for _, c in defined) / len(defined), res.diff, res.lo, res.hi, res.p_worse,
                        n=res.n))
    slice_names = sorted(present - critical_set)
    for name in slice_names:
        member = [name in i.slices() for i in items]
        if sum(member) < cfg.min_slice_n:
            continue
        bv = [1.0 if s.passed else 0.0 for s in bs]
        cv = [1.0 if s.passed else 0.0 for s in cs]
        diffs = [c - b if mem else None for b, c, mem in zip(bv, cv, member)]
        res = paired_bootstrap(diffs, resamples)
        sel = [(b, c) for b, c, mem in zip(bv, cv, member) if mem]
        rows.append(Row(f"slice:{name}", "slice", sum(b for b, _ in sel) / len(sel),
                        sum(c for _, c in sel) / len(sel), res.diff, res.lo, res.hi, res.p_worse, n=res.n))
    for row, adj in zip(rows, holm([r.p for r in rows])):
        row.p_adj = adj
        row.significant_drop = adj < cfg.alpha and -row.diff >= cfg.min_drop

    critical: list[CriticalRow] = []
    for name in cfg.critical_slices:
        down = up = n = 0
        for i, b, c in zip(items, bs, cs):
            if name in i.slices():
                n += 1
                down += b.passed and not c.passed
                up += (not b.passed) and c.passed
        critical.append(CriticalRow(name, n, down, up, down - up >= cfg.critical_min_net_drop))

    out_reasons: list[str] = []
    for r in rows:
        if r.significant_drop:
            out_reasons.append(
                f"{r.name} fell {abs(r.diff) * 100:.1f} pts ({r.base * 100:.1f}% -> {r.cand * 100:.1f}%), "
                f"95% CI [{r.lo * 100:+.1f}, {r.hi * 100:+.1f}], Holm p={r.p_adj:.4f}")
    for c in critical:
        if c.failed:
            out_reasons.append(f"critical slice {c.name}: {c.down} question(s) newly fail, {c.up} newly pass "
                               f"(net loss {c.down - c.up} >= {cfg.critical_min_net_drop}, no test needed)")
    report = GateReport(
        FAIL if out_reasons else PASS, out_reasons, cfg.split, len(items), base.pipeline, cand.pipeline, cfg,
        rows=rows, critical=critical,
        base_metrics=aggregate(bs), cand_metrics=aggregate(cs),
        base_failures=breakdown(bs), cand_failures=breakdown(cs),
        regressed=[i.qid for i, b, c in zip(items, bs, cs) if b.passed and not c.passed],
        improved=[i.qid for i, b, c in zip(items, bs, cs) if (not b.passed) and c.passed],
        warnings=warnings)
    return report


def failure_shift(report: GateReport) -> dict[str, int]:
    """Candidate failures minus baseline failures per attribution class."""
    return {c: report.cand_failures.get(c, 0) - report.base_failures.get(c, 0) for c in CLASSES}
