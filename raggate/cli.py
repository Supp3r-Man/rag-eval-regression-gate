"""Command line. Exit codes: 0 pass, 1 regression, 2 could not judge (including any crash)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .attribution import breakdown
from .corpus import build_golden
from .gate import GateConfig, run_gate
from .golden import GoldenSet, load_golden, save_golden
from .hygiene import check_golden
from .judge import (AlwaysPassJudge, CouldNotCalibrate, LenientJudge, RuleJudge, build_judge_labels, calibrate)
from .metrics import aggregate, score_question
from .pipelines import load_run, named_config, pipeline_names, run_pipeline, save_run
from .power import critical_false_alarm, min_detectable_drop, power_table
from .report import failures_table, metrics_table, render_gate, report_to_dict


def _golden(path: str | None) -> GoldenSet:
    return load_golden(path) if path else build_golden()


def _summaries(gs: GoldenSet, runs: dict) -> tuple[dict, dict]:
    by = gs.by_qid()
    metrics, fails = {}, {}
    for name, run in runs.items():
        sc = [score_question(by[r.qid], r, int(run.config.get("k", 4))) for r in run.results]
        metrics[name], fails[name] = aggregate(sc), breakdown(sc)
    return metrics, fails


def cmd_demo(args) -> int:
    gs = _golden(args.golden)
    findings = check_golden(gs)
    errs = [f for f in findings if f.severity == "error"]
    print(f"golden set: {len(gs.passages)} passages, {len(gs.items)} questions "
          f"({len(gs.split('dev'))} dev / {len(gs.split('heldout'))} held-out); "
          f"hygiene: {len(errs)} errors, {len(findings) - len(errs)} warnings")
    if errs:
        print("hygiene errors, the gate would refuse to judge:")
        for f in errs[:5]:
            print(f"  {f.code} {f.ref}: {f.message}")
        return 2
    runs = {n: run_pipeline(gs, named_config(n), "heldout") for n in pipeline_names()}
    metrics, fails = _summaries(gs, runs)
    print("\n== held-out scores of the five toy pipelines ==")
    print(metrics_table(metrics))
    print("\n== failed questions by the stage that lost them ==")
    print(failures_table(fails))
    base = runs["baseline"]
    print("\n== the gate: baseline vs each candidate ==")
    for name in pipeline_names()[1:]:
        r = run_gate(gs, base, runs[name])
        print(f"\n{name}: {r.decision} (exit {r.exit_code})")
        for reason in r.reasons[:4]:
            print(f"  - {reason}")
        if len(r.reasons) > 4:
            print(f"  ... and {len(r.reasons) - 4} more")
    aa = run_gate(gs, base, base)
    n1, n2 = runs["noisy-s1"], run_pipeline(gs, named_config("noisy-s2"), "heldout")
    churn = run_gate(gs, n1, n2)
    print(f"\nA/A: baseline vs itself -> {aa.decision}")
    print(f"A/A with churn: noisy-s1 vs noisy-s2 (same pipeline, different noise seed) -> {churn.decision}; "
          f"{len(churn.regressed)} newly fail / {len(churn.improved)} newly pass")
    for reason in churn.reasons[:3]:
        print(f"  - {reason}")
    print("\n== judge calibration (synthetic labels, true by construction) ==")
    labels = build_judge_labels(gs)
    for jname, judge in [("rule", RuleJudge()), ("rule+15% noise", RuleJudge(0.15, 1)),
                         ("lenient", LenientJudge()), ("always-pass", AlwaysPassJudge())]:
        c = calibrate(judge, labels)
        print(f"  {jname:15s} accuracy {c.accuracy:.2f}  kappa {c.kappa:.2f} "
              f"[{c.kappa_lo:.2f}, {c.kappa_hi:.2f}]  trustworthy(>=0.7)={c.trustworthy}")
    print("\nrun `python -m raggate power` for what drop this gate can detect at a given n.")
    return 0


def cmd_run(args) -> int:
    gs = _golden(args.golden)
    run = run_pipeline(gs, named_config(args.pipeline), args.split)
    save_run(run, args.out)
    errs = sum(1 for r in run.results if r.error)
    print(f"{args.pipeline} on {args.split}: {len(run.results)} questions, {errs} errored -> {args.out}")
    return 2 if errs else 0


def cmd_gate(args) -> int:
    gs = _golden(args.golden)
    cfg = GateConfig(split=args.split, boot=args.boot)
    report = run_gate(gs, load_run(args.baseline), load_run(args.candidate), cfg)
    print(json.dumps(report_to_dict(report), indent=1) if args.json else render_gate(report))
    return report.exit_code


def cmd_hygiene(args) -> int:
    gs = _golden(args.golden)
    findings = check_golden(gs)
    for f in findings:
        print(f"{f.severity:5s} {f.code:24s} {f.ref}: {f.message}")
    errs = sum(f.severity == "error" for f in findings)
    print(f"{errs} errors, {len(findings) - errs} warnings")
    return 2 if errs else 0


def cmd_judge(args) -> int:
    gs = _golden(args.golden)
    labels = build_judge_labels(gs)
    judges = {"rule": RuleJudge(), "lenient": LenientJudge(), "always-pass": AlwaysPassJudge(),
              "rule+15%noise": RuleJudge(0.15, 1), "rule+40%noise": RuleJudge(0.4, 1)}
    try:
        for name, judge in judges.items():
            c = calibrate(judge, labels)
            blind = ", ".join(f"{k} {w}/{n}" for k, (n, w) in sorted(c.by_kind.items()) if w)
            print(f"{name:14s} acc {c.accuracy:.2f}  kappa {c.kappa:.2f} [{c.kappa_lo:.2f}, {c.kappa_hi:.2f}]  "
                  f"trustworthy={c.trustworthy}  wrong on: {blind or 'nothing'}")
    except CouldNotCalibrate as exc:
        print(f"could not calibrate: {exc}", file=sys.stderr)
        return 2
    return 0


def cmd_power(args) -> int:
    ns = tuple(int(x) for x in args.n.split(","))
    cells = power_table(ns=ns, churn=args.churn, trials=args.trials)
    print(f"power of the statistical rule (churn {args.churn:.0%} per direction, {args.trials} trials per cell, "
          "min_drop 2 pts)")
    print("  'single' = lone test at alpha 0.05; 'family' = alpha/15, the weakest test in the real gate\n")
    print("     n   true drop   single   family")
    for c in cells:
        print(f"  {c.n:4d}   {c.drop * 100:5.1f} pts   {c.single * 100:5.1f}%   {c.family * 100:5.1f}%"
              + ("   <- false-alarm rate" if c.drop == 0 else ""))
    print("\nsmallest simulated drop with >= 80% power (family threshold):")
    for n in ns:
        mdd = min_detectable_drop(cells, n)
        print(f"  n={n:4d}: " + (f"{mdd * 100:.0f} pts" if mdd is not None else "none on the grid"))
    print("\ncritical-slice rule on an UNCHANGED pipeline (exact): P(net loss >= 1)")
    for m in (10, 20, 40):
        print(f"  slice of {m:2d} questions: " + "   ".join(
            f"churn {c:.0%} -> {critical_false_alarm(m, c) * 100:4.1f}%" for c in (0.0, 0.02, 0.05)))
    return 0


def cmd_tune(args) -> int:
    """Dev-split sweep of the abstain threshold, the only thing the baseline was tuned on."""
    from dataclasses import replace
    gs = _golden(args.golden)
    by = gs.by_qid()
    print("dev split only (held-out is never used to choose a setting)")
    print("  tau    pass_rate  unanswerable refused")
    for tau in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        run = run_pipeline(gs, replace(named_config("baseline"), tau=tau), "dev")
        sc = [score_question(by[r.qid], r) for r in run.results]
        un = [s for s in sc if not s.answerable]
        print(f"  {tau:.1f}    {aggregate(sc)['pass_rate'] * 100:5.1f}%     "
              f"{sum(s.passed for s in un)}/{len(un)}")
    return 0


def cmd_export(args) -> int:
    save_golden(build_golden(), args.out)
    print(f"wrote {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="raggate", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = argparse.ArgumentParser(add_help=False)
    g.add_argument("--golden", help="golden set JSON (default: the generated one)")
    sub.add_parser("demo", parents=[g]).set_defaults(fn=cmd_demo)
    r = sub.add_parser("run", parents=[g])
    r.add_argument("pipeline")
    r.add_argument("--split", default="heldout")
    r.add_argument("--out", required=True)
    r.set_defaults(fn=cmd_run)
    ga = sub.add_parser("gate", parents=[g])
    ga.add_argument("baseline")
    ga.add_argument("candidate")
    ga.add_argument("--split", default="heldout")
    ga.add_argument("--boot", type=int, default=2000)
    ga.add_argument("--json", action="store_true")
    ga.set_defaults(fn=cmd_gate)
    sub.add_parser("hygiene", parents=[g]).set_defaults(fn=cmd_hygiene)
    sub.add_parser("judge", parents=[g]).set_defaults(fn=cmd_judge)
    sub.add_parser("tune", parents=[g]).set_defaults(fn=cmd_tune)
    pw = sub.add_parser("power")
    pw.add_argument("--n", default="50,100,200,400")
    pw.add_argument("--churn", type=float, default=0.04)
    pw.add_argument("--trials", type=int, default=200)
    pw.set_defaults(fn=cmd_power)
    ex = sub.add_parser("export")
    ex.add_argument("--out", required=True)
    ex.set_defaults(fn=cmd_export)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except Exception as exc:  # noqa: BLE001 - a crash must never look like "regression found" (exit 1)
        print(f"raggate: could not judge: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
