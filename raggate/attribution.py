"""Failure attribution: for a failed question, which stage lost it?

Stages are checked in pipeline order and the EARLIEST stage that lost any piece of gold evidence is
blamed, because a later stage cannot recover what an earlier one dropped:

  retrieval_miss          a gold passage is not in the retrieved list at all
  ranking_miss            every gold passage was retrieved, but one is below the context cut (rank > k)
  context_truncation      gold passages made the cut, but the budget chopped the fact out of the prompt
  generation_error        the gold fact was in the prompt and the answer is still wrong, unsupported
                          or an abstention
  unanswerable_mishandled the question has no answer in the corpus and the pipeline answered anyway

This is what turns "pass rate fell 9 points" into "the retriever lost 14 questions".
"""
from __future__ import annotations

from .golden import GoldItem
from .pipelines import RagResult
from .text import contains_phrase

CLASSES = ("retrieval_miss", "ranking_miss", "context_truncation", "generation_error",
           "unanswerable_mishandled")


def classify_failure(item: GoldItem, result: RagResult, sc, k: int) -> tuple[str, str]:
    if not item.answerable:
        return "unanswerable_mishandled", "answered a question the corpus cannot answer"
    retrieved = result.retrieved
    missing = [g for g in item.gold if g.passage not in retrieved]
    if missing:
        return "retrieval_miss", f"{missing[0].passage} not in top {len(retrieved)}"
    below = [g for g in item.gold if g.passage not in retrieved[:k]]
    if below:
        rank = retrieved.index(below[0].passage) + 1
        return "ranking_miss", f"{below[0].passage} at rank {rank} (> k={k})"
    ctx = dict(result.context)
    for g in item.gold:
        if g.passage not in ctx or not contains_phrase(ctx[g.passage], g.fact):
            return "context_truncation", f"fact {g.fact!r} not in the prompt text of {g.passage}"
    if result.abstained:
        return "generation_error", "abstained although the gold fact was in the prompt"
    if sc.fact_coverage is not None and sc.fact_coverage < 1.0:
        return "generation_error", "gold fact missing or stated for the wrong component"
    return "generation_error", "answer contains a claim the context does not support"


def breakdown(scores) -> dict[str, int]:
    out = {c: 0 for c in CLASSES}
    for s in scores:
        if s.failure:
            out[s.failure] += 1
    return out
