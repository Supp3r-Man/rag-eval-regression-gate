"""Plain-text and JSON rendering of gate reports and pipeline summaries."""
from __future__ import annotations

from dataclasses import asdict

from .attribution import CLASSES
from .gate import GateReport, failure_shift
from .metrics import METRICS


def pct(x: float | None) -> str:
    return "   -  " if x is None else f"{x * 100:5.1f}%"


def table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(str(c)) for c in col) for col in zip(headers, *rows)]
    line = lambda r: "  ".join(str(c).ljust(w) for c, w in zip(r, widths)).rstrip()  # noqa: E731
    return "\n".join([line(headers), line(["-" * w for w in widths])] + [line(r) for r in rows])


def render_gate(r: GateReport) -> str:
    out = [f"GATE {r.decision}   {r.baseline} -> {r.candidate}   ({r.split}, n={r.n})"]
    if r.decision == "INCONCLUSIVE":
        out.append("The gate could not judge, so it will not say pass or fail:")
        out += [f"  - {x}" for x in r.reasons]
        return "\n".join(out)
    if r.reasons:
        out.append("Why it failed:")
        out += [f"  - {x}" for x in r.reasons]
    out.append("")
    out.append(table(
        ["test", "n", "baseline", "candidate", "diff", "95% CI", "Holm p", "verdict"],
        [[x.name, x.n, pct(x.base), pct(x.cand), f"{x.diff * 100:+6.1f}",
          f"[{x.lo * 100:+6.1f},{x.hi * 100:+6.1f}]", f"{x.p_adj:.4f}",
          "DROP" if x.significant_drop else ("better" if x.lo > 0 else "ok")] for x in r.rows]))
    out.append("")
    out.append("critical slices (any net loss fails, no test):")
    out.append(table(["slice", "n", "newly fail", "newly pass", "verdict"],
                     [[c.name, c.n, c.down, c.up, "FAIL" if c.failed else "ok"] for c in r.critical]))
    out.append("")
    out.append("where the failures are (count of failed questions by the stage that lost them):")
    shift = failure_shift(r)
    out.append(table(["stage", "baseline", "candidate", "shift"],
                     [[c, r.base_failures[c], r.cand_failures[c], f"{shift[c]:+d}"] for c in CLASSES]))
    out.append("")
    out.append(f"newly failing: {len(r.regressed)}   newly passing: {len(r.improved)}")
    if r.regressed:
        out.append("  e.g. " + ", ".join(r.regressed[:6]))
    if r.warnings:
        out.append("")
        out.append(f"golden-set warnings ({len(r.warnings)}):")
        out += [f"  - {w}" for w in r.warnings[:5]]
    return "\n".join(out)


def metrics_table(summaries: dict[str, dict]) -> str:
    return table(["pipeline", *METRICS],
                 [[name, *[pct(m[k]) for k in METRICS]] for name, m in summaries.items()])


def failures_table(summaries: dict[str, dict]) -> str:
    return table(["pipeline", *CLASSES], [[name, *[f[c] for c in CLASSES]] for name, f in summaries.items()])


def report_to_dict(r: GateReport) -> dict:
    d = asdict(r)
    d["exit_code"] = r.exit_code
    return d
