from raggate.attribution import CLASSES, breakdown
from raggate.golden import GoldItem, GoldRef
from raggate.metrics import score_question
from raggate.pipelines import ABSTAIN_TEXT, build_context

from conftest import result, tiny_golden

GOOD = "The default timeout of the foo service is 30 seconds."


def _one(item, res, k=4):
    return score_question(item, res, k)


def test_retrieval_miss_when_gold_not_retrieved_at_all():
    gs = tiny_golden()
    s = _one(gs.items[0], result("q1", ["b-01", "a-02"], [("b-01", gs.passages[2].text)], ABSTAIN_TEXT, True))
    assert s.failure == "retrieval_miss"


def test_ranking_miss_when_gold_retrieved_but_below_the_cut():
    gs = tiny_golden()
    retrieved = ["b-01", "a-02", "a-01"]
    ctx = [("b-01", gs.passages[2].text)]
    s = _one(gs.items[0], result("q1", retrieved, ctx, ABSTAIN_TEXT, True), k=1)
    assert s.failure == "ranking_miss" and "rank 3" in s.detail


def test_context_truncation_when_the_fact_was_cut_out_of_the_prompt():
    gs = tiny_golden()
    cut = gs.passages[0].text[:20]                                   # the fact sentence is gone
    s = _one(gs.items[0], result("q1", ["a-01"], [("a-01", cut)], ABSTAIN_TEXT, True))
    assert s.failure == "context_truncation"


def test_context_truncation_when_the_passage_never_made_it_into_the_prompt():
    gs = tiny_golden()
    s = _one(gs.items[0], result("q1", ["a-02", "a-01"], [("a-02", gs.passages[1].text)], ABSTAIN_TEXT, True), k=2)
    assert s.failure == "context_truncation"


def test_generation_error_variants_have_distinct_details():
    gs = tiny_golden()
    ctx = [("a-01", gs.passages[0].text)]
    abst = _one(gs.items[0], result("q1", ["a-01"], ctx, ABSTAIN_TEXT, True))
    wrong = _one(gs.items[0], result("q1", ["a-01"], ctx, "The default timeout of the baz worker is 30 seconds."))
    assert abst.failure == wrong.failure == "generation_error"
    assert "abstained" in abst.detail and "wrong component" in wrong.detail
    # right fact and component, plus a sentence the context does not support
    extra = _one(gs.items[0], result("q1", ["a-01"], ctx, GOOD + " Operators must restart it nightly."))
    assert extra.failure == "generation_error" and "does not support" in extra.detail


def test_earliest_stage_is_blamed_for_multi_gold_questions():
    gs = tiny_golden()
    multi = GoldItem("m", "What are the default timeout and the retry limit of the foo service?", "multi", "alpha",
                     "heldout", "foo service", (GoldRef("a-01", "30 seconds"), GoldRef("a-02", "5 attempts")))
    # a-01 is in context, a-02 is not retrieved at all -> retrieval_miss (not a later stage)
    s = _one(multi, result("m", ["a-01"], [("a-01", gs.passages[0].text)], GOOD))
    assert s.failure == "retrieval_miss"
    # both retrieved, a-02 below the cut -> ranking_miss
    s = _one(multi, result("m", ["a-01", "b-01", "a-02"], [("a-01", gs.passages[0].text)], GOOD), k=2)
    assert s.failure == "ranking_miss"


def test_passing_questions_have_no_failure_class():
    gs = tiny_golden()
    s = _one(gs.items[0], result("q1", ["a-01"], [("a-01", gs.passages[0].text)], GOOD))
    assert s.passed and s.failure is None


def test_breakdown_counts_every_class_and_totals_the_failures():
    gs = tiny_golden()
    scores = [
        _one(gs.items[0], result("q1", ["b-01"], [], ABSTAIN_TEXT, True)),
        _one(gs.items[1], result("q2", [], [], ABSTAIN_TEXT, True)),
        _one(gs.items[3], result("q4", [], [], "Something.", False)),
    ]
    b = breakdown(scores)
    assert set(b) == set(CLASSES)
    assert b["retrieval_miss"] == 2 and b["unanswerable_mishandled"] == 1 and sum(b.values()) == 3


def test_build_context_cuts_the_last_passage_at_the_budget_and_drops_passages_that_do_not_fit():
    gs = tiny_golden()
    pm = gs.passage_map()
    full = len(pm["a-01"].title + ". " + pm["a-01"].text)
    ctx = build_context(pm, ["a-01", "a-02"], k=2, budget=full + 60)
    assert [p for p, _ in ctx] == ["a-01", "a-02"] and len(ctx[1][1]) == 60
    assert build_context(pm, ["a-01", "a-02"], k=2, budget=full + 10) == [ctx[0]]    # < 40 chars left: stop
