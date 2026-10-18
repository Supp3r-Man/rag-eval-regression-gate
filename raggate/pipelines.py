"""Toy RAG pipelines with known behaviour, so the harness has something to measure.

These are NOT good RAG systems and make no claim to be. Each is a few lines of lexical code with one
knob turned, built so that the gate's verdicts can be checked against what we know we changed:

  baseline              BM25 top-4, 1400-char context, extractive answer, abstains below 0.9 coverage (chosen on dev only: `raggate tune`)
  regressed_retriever   same, but an ingestion change stopped indexing the "-add" addendum passages
  regressed_generator   same retriever, but a lower abstain threshold and numeric drift in answers
  tight_context         same, but the context budget shrank to 520 chars (truncates gold passages)
  noisy-s<seed>         baseline with seeded, random retrieval corruption on ~12% of questions
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable

from .golden import GoldenSet, GoldItem
from .retrieval import BM25
from .text import content_tokens, sentences

ABSTAIN_TEXT = "I could not find that in the documentation."


@dataclass
class RagResult:
    qid: str
    retrieved: list[str]                       # ranked passage ids, depth N
    context: list[tuple[str, str]]             # (passage id, text as placed in the prompt)
    answer: str
    abstained: bool
    error: str = ""


@dataclass
class Run:
    pipeline: str
    split: str
    config: dict
    results: list[RagResult] = field(default_factory=list)


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    k: int = 4
    depth: int = 10
    budget_chars: int = 1400
    skip_suffix: str = ""        # ingestion bug: passages whose id ends with this are never indexed
    tau: float = 0.9              # abstain unless a context sentence covers this much of the question
    drift: float = 0.0            # probability the answer's number is silently changed
    noise: float = 0.0            # probability retrieval is corrupted for a question
    seed: int = 0


BASE = PipelineConfig("baseline")


def named_config(name: str) -> PipelineConfig:
    if name == "baseline":
        return BASE
    if name == "regressed_retriever":
        return PipelineConfig(name, skip_suffix="-add")
    if name == "regressed_generator":
        return PipelineConfig(name, tau=0.55, drift=0.3)
    if name == "tight_context":
        return PipelineConfig(name, budget_chars=520)
    m = re.fullmatch(r"noisy-s(\d+)", name)
    if m:
        return PipelineConfig(name, noise=0.12, seed=int(m.group(1)))
    raise KeyError(f"unknown pipeline {name!r}")


def pipeline_names() -> list[str]:
    return ["baseline", "regressed_retriever", "regressed_generator", "tight_context", "noisy-s1"]


def _rng(cfg: PipelineConfig, qid: str, salt: str) -> random.Random:
    return random.Random(f"{cfg.seed}:{salt}:{qid}")


def build_context(passages: dict, ranked: list[str], k: int, budget: int) -> list[tuple[str, str]]:
    """Whole passages in rank order; the last one that does not fit is cut at the budget."""
    ctx, remaining = [], budget
    for pid in ranked[:k]:
        text = passages[pid].title + ". " + passages[pid].text
        if remaining >= len(text):
            ctx.append((pid, text))
            remaining -= len(text)
        elif remaining >= 40:
            ctx.append((pid, text[:remaining]))
            break
        else:
            break
    return ctx


_MULTI = re.compile(r"^what are the (.+?) and the (.+?) of the (.+?)\?$", re.I)


def decompose(question: str) -> list[str]:
    """Query decomposition: 'What are the A and the B of the X?' -> two single-attribute questions."""
    m = _MULTI.match(question.strip())
    if not m:
        return [question]
    a, b, e = m.groups()
    return [f"What is the {a} of the {e}?", f"What is the {b} of the {e}?"]


def _best_sentence(sub: str, ctx_sentences: list[str]) -> tuple[float, str]:
    q = set(content_tokens(sub))
    best = (-1.0, "")
    if not q:
        return best
    for s in ctx_sentences:
        cov = len(q & set(content_tokens(s))) / len(q)
        if cov > best[0]:
            best = (cov, s)
    return best


def _drift_number(sentence: str) -> str:
    return re.sub(r"\d+", lambda m: str(int(m.group()) * 2), sentence, count=1)


def generate(question: str, ctx: list[tuple[str, str]], cfg: PipelineConfig, qid: str) -> tuple[str, bool]:
    ctx_sents = [s for _, text in ctx for s in sentences(text)]
    chosen: list[str] = []
    for sub in decompose(question):
        cov, sent = _best_sentence(sub, ctx_sents)
        if cov < cfg.tau:
            return ABSTAIN_TEXT, True
        if sent not in chosen:
            chosen.append(sent)
    if cfg.drift and _rng(cfg, qid, "drift").random() < cfg.drift:
        chosen = [_drift_number(chosen[0])] + chosen[1:]
    return " ".join(chosen), False


def run_one(item: GoldItem, retriever, passages: dict, cfg: PipelineConfig) -> RagResult:
    hits = [pid for pid, _ in retriever.search(item.question, cfg.depth)]
    if cfg.noise and _rng(cfg, item.qid, "noise").random() < cfg.noise and hits:
        hits = hits[1:] + hits[:1]                       # demote the top hit to the bottom
    ctx = build_context(passages, hits, cfg.k, cfg.budget_chars)
    answer, abstained = generate(item.question, ctx, cfg, item.qid)
    return RagResult(item.qid, hits, ctx, answer, abstained)


def run_pipeline(gs: GoldenSet, cfg: PipelineConfig, split: str,
                 hook: Callable[[GoldItem], None] | None = None) -> Run:
    """Run one pipeline over a split. A per-question crash is recorded as an error, not swallowed."""
    passages = gs.passage_map()
    indexed = [p for p in gs.passages if not (cfg.skip_suffix and p.pid.endswith(cfg.skip_suffix))]
    retriever = BM25(indexed)
    run = Run(cfg.name, split, asdict(cfg))
    for item in gs.split(split):
        try:
            if hook:
                hook(item)
            run.results.append(run_one(item, retriever, passages, cfg))
        except Exception as exc:  # noqa: BLE001 - recorded, and the gate refuses to judge errored runs
            run.results.append(RagResult(item.qid, [], [], "", False, error=f"{type(exc).__name__}: {exc}"))
    return run


def save_run(run: Run, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(run), indent=1), encoding="utf-8")


def load_run(path: str | Path) -> Run:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    results = [RagResult(r["qid"], r["retrieved"], [tuple(c) for c in r["context"]], r["answer"],
                         r["abstained"], r.get("error", "")) for r in d["results"]]
    return Run(d["pipeline"], d["split"], d["config"], results)
