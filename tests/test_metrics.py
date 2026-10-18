import math

import pytest

from raggate.metrics import (aggregate, fact_coverage, faithfulness, ndcg_at_k, recall_at_k, reciprocal_rank,
                             score_question)
from raggate.pipelines import ABSTAIN_TEXT
from raggate.text import contains_phrase, stem, tokenize

from conftest import result, tiny_golden


def test_stemmer_and_phrase_matching_respect_token_boundaries():
    assert stem("retries") == "retry" and stem("requests") == "request" and stem("access") == "access"
    assert tokenize("Retries failed requests") == ["retry", "failed", "request"]
    assert contains_phrase("The limit is 5 MB.", "5 MB")
    assert not contains_phrase("The limit is 15 MB.", "5 MB")          # not a substring match
    assert not contains_phrase("The limit is 25 MB.", "5 MB")


def test_recall_mrr_ndcg_match_hand_computation():
    gold = {"a", "b"}
    retrieved = ["x", "a", "y", "b", "z"]
    assert recall_at_k(gold, retrieved, 2) == 0.5
    assert recall_at_k(gold, retrieved, 4) == 1.0
    assert reciprocal_rank(gold, retrieved) == 0.5
    assert reciprocal_rank(gold, ["x", "y"]) == 0.0
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(gold, retrieved, 10) == pytest.approx(dcg / ideal)
    assert ndcg_at_k({"a"}, ["a"], 10) == 1.0


def test_fact_coverage_needs_the_component_in_the_same_sentence():
    item = tiny_golden().items[0]                                   # foo service, 30 seconds
    assert fact_coverage(item, "The default timeout of the foo service is 30 seconds.") == 1.0
    # right number, wrong component: must NOT count
    assert fact_coverage(item, "The default timeout of the baz worker is 30 seconds.") == 0.0
    # fact and component in different sentences: must NOT count
    assert fact_coverage(item, "The foo service is documented here. Another thing is 30 seconds.") == 0.0


def test_fact_coverage_is_fractional_for_multi_fact_questions():
    gs = tiny_golden()
    item = gs.items[1]
    two = type(item)(item.qid, item.question, "multi", item.area, item.split, item.entity,
                     item.gold + (type(item.gold[0])("a-02", "99 attempts"),))
    assert fact_coverage(two, "The retry limit of the bar queue is 5 attempts.") == 0.5


def test_faithfulness_catches_a_changed_number():
    ctx = [("a-01", "Foo Service. The default timeout of the foo service is 30 seconds.")]
    ok = result("q1", ["a-01"], ctx, "The default timeout of the foo service is 30 seconds.")
    drift = result("q1", ["a-01"], ctx, "The default timeout of the foo service is 60 seconds.")
    assert faithfulness(ok) == 1.0
    assert faithfulness(drift) == 0.0


def test_faithfulness_of_abstention_is_vacuously_one_and_empty_answer_is_zero():
    assert faithfulness(result("q", [], [], ABSTAIN_TEXT, True)) == 1.0
    assert faithfulness(result("q", [], [], "", False)) == 0.0


def test_correct_answer_passes_and_is_scored_on_every_metric():
    gs = tiny_golden()
    item = gs.items[1]
    ctx = [("a-02", gs.passages[1].text)]
    s = score_question(item, result("q2", ["a-02", "a-01"], ctx, "The retry limit of the bar queue is 5 attempts."), 4)
    assert s.passed and s.recall == 1.0 and s.mrr == 1.0 and s.fact_coverage == 1.0 and s.failure is None


def test_unanswerable_passes_only_by_abstaining():
    gs = tiny_golden()
    item = gs.items[3]
    assert score_question(item, result("q4", [], [], ABSTAIN_TEXT, True), 4).passed
    s = score_question(item, result("q4", ["b-01"], [("b-01", gs.passages[2].text)],
                                     "The default timeout of the baz worker is 15 seconds."), 4)
    assert not s.passed and s.failure == "unanswerable_mishandled"
    assert s.recall is None and s.mrr is None                        # retrieval metrics undefined without gold


def test_abstaining_on_an_answerable_question_fails():
    gs = tiny_golden()
    s = score_question(gs.items[0], result("q1", ["a-01"], [("a-01", gs.passages[0].text)], ABSTAIN_TEXT, True), 4)
    assert not s.passed and s.fact_coverage == 0.0 and s.failure == "generation_error"


def test_aggregate_skips_undefined_metrics():
    gs = tiny_golden()
    good = score_question(gs.items[1], result("q2", ["a-02"], [("a-02", gs.passages[1].text)],
                                              "The retry limit of the bar queue is 5 attempts."), 4)
    una = score_question(gs.items[3], result("q4", [], [], ABSTAIN_TEXT, True), 4)
    agg = aggregate([good, una])
    assert agg["pass_rate"] == 1.0 and agg["recall"] == 1.0         # recall averaged over the one answerable question
    assert aggregate([una])["recall"] is None
