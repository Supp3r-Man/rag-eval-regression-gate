"""Per-question scoring: retrieval metrics, generation metrics and the pass/fail verdict.

A question PASSES when
  * answerable:   every gold fact is stated in an answer sentence that also names the component,
                  AND every answer sentence is supported by one sentence of the retrieved context;
  * unanswerable: the pipeline abstained.
Scoring is recomputed from the saved run plus the golden set, so the scorer is the single source of
truth and a run file cannot carry its own verdict.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .golden import GoldItem
from .pipelines import RagResult
from .text import content_tokens, contains_phrase, is_number, sentences, tokenize

METRICS = ("pass_rate", "recall", "mrr", "ndcg", "fact_coverage", "faithfulness")
SUPPORT_THRESHOLD = 0.9


@dataclass
class QScore:
    qid: str
    answerable: bool
    passed: bool
    recall: float | None
    mrr: float | None
    ndcg: float | None
    fact_coverage: float | None
    faithfulness: float
    abstained: bool
    failure: str | None = None
    detail: str = ""


def recall_at_k(gold: set[str], retrieved: list[str], k: int) -> float:
    return len(gold & set(retrieved[:k])) / len(gold)


def reciprocal_rank(gold: set[str], retrieved: list[str]) -> float:
    for i, pid in enumerate(retrieved, 1):
        if pid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(gold: set[str], retrieved: list[str], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, pid in enumerate(retrieved[:k], 1) if pid in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal


def fact_coverage(item: GoldItem, answer: str) -> float:
    """A fact only counts if it sits in an answer sentence that also names the component."""
    ent = set(content_tokens(item.entity))
    sents = [s for s in sentences(answer) if ent <= set(tokenize(s))]
    hit = sum(1 for g in item.gold if any(contains_phrase(s, g.fact) for s in sents))
    return hit / len(item.gold)


def sentence_supported(sentence: str, ctx_sentences: list[str]) -> bool:
    toks = tokenize(sentence)
    if not toks:
        return True
    for c in ctx_sentences:
        ctoks = set(tokenize(c))
        present = sum(1 for t in toks if t in ctoks)
        nums_ok = all(t in ctoks for t in toks if is_number(t))
        if nums_ok and present / len(toks) >= SUPPORT_THRESHOLD:
            return True
    return False


def faithfulness(result: RagResult) -> float:
    if result.abstained:
        return 1.0                    # no claims, nothing to be unfaithful about
    ans = sentences(result.answer)
    if not ans:
        return 0.0
    ctx = [s for _, text in result.context for s in sentences(text)]
    return sum(sentence_supported(s, ctx) for s in ans) / len(ans)


def score_question(item: GoldItem, result: RagResult, k: int = 4) -> QScore:
    from .attribution import classify_failure
    faith = faithfulness(result)
    if not item.answerable:
        passed = result.abstained
        sc = QScore(item.qid, False, passed, None, None, None, None, faith, result.abstained)
    else:
        gold = {g.passage for g in item.gold}
        cov = 0.0 if result.abstained else fact_coverage(item, result.answer)
        passed = (not result.abstained) and cov == 1.0 and faith == 1.0
        sc = QScore(item.qid, True, passed,
                    recall_at_k(gold, result.retrieved, k),
                    reciprocal_rank(gold, result.retrieved),
                    ndcg_at_k(gold, result.retrieved, 10),
                    cov, faith, result.abstained)
    if not sc.passed:
        sc.failure, sc.detail = classify_failure(item, result, sc, k)
    return sc


def metric_value(sc: QScore, metric: str) -> float | None:
    if metric == "pass_rate":
        return 1.0 if sc.passed else 0.0
    return getattr(sc, metric)


def aggregate(scores: list[QScore]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for m in METRICS:
        vals = [v for v in (metric_value(s, m) for s in scores) if v is not None]
        out[m] = sum(vals) / len(vals) if vals else None
    return out
