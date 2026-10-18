import dataclasses

from raggate.golden import GoldItem, GoldRef, GoldenSet, Passage
from raggate.hygiene import check_golden, errors

from conftest import tiny_golden


def codes(findings, severity=None):
    return sorted(f.code for f in findings if severity in (None, f.severity))


def with_items(gs, *extra, replace=None):
    items = [replace.get(i.qid, i) if replace else i for i in gs.items] + list(extra)
    return GoldenSet(gs.passages, items)


def test_the_generated_golden_set_is_clean(gs):
    assert errors(check_golden(gs)) == []


def test_hand_made_set_is_clean_apart_from_size_warnings():
    f = check_golden(tiny_golden())
    assert errors(f) == [] and set(codes(f)) == {"small_slice"}


def test_duplicate_qid_is_an_error():
    gs = tiny_golden()
    dup = dataclasses.replace(gs.items[1], question="What is the retry limit of the bar queue now?")
    assert "duplicate_qid" in codes(check_golden(with_items(gs, dup)), "error")


def test_gold_passage_missing_from_corpus_is_an_error():
    gs = tiny_golden()
    bad = dataclasses.replace(gs.items[1], gold=(GoldRef("zz-99", "5 attempts"),))
    f = check_golden(with_items(gs, replace={"q2": bad}))
    assert "gold_passage_missing" in codes(f, "error")


def test_fact_not_present_in_its_gold_passage_is_an_error():
    gs = tiny_golden()
    bad = dataclasses.replace(gs.items[1], gold=(GoldRef("a-02", "9 attempts"),))
    assert "fact_not_in_passage" in codes(check_golden(with_items(gs, replace={"q2": bad})), "error")


def test_a_fact_that_is_only_a_substring_of_the_passage_value_does_not_count():
    gs = tiny_golden()
    bad = dataclasses.replace(gs.items[2], gold=(GoldRef("b-01", "5 seconds"),))      # passage says 15 seconds
    assert "fact_not_in_passage" in codes(check_golden(with_items(gs, replace={"q3": bad})), "error")


def test_answer_leaked_in_the_question_is_an_error():
    gs = tiny_golden()
    bad = dataclasses.replace(gs.items[1], question="Is the retry limit of the bar queue 5 attempts?")
    assert "answer_in_question" in codes(check_golden(with_items(gs, replace={"q2": bad})), "error")


def test_missing_entity_unknown_split_and_empty_split():
    gs = tiny_golden()
    f = check_golden(with_items(gs, replace={"q2": dataclasses.replace(gs.items[1], entity="", split="test")}))
    assert {"missing_entity", "unknown_split"} <= set(codes(f, "error"))
    only_dev = GoldenSet(gs.passages, [dataclasses.replace(i, split="dev") for i in gs.items])
    assert "empty_split" in codes(check_golden(only_dev), "error")


def test_same_question_twice_in_one_split_is_an_error():
    gs = tiny_golden()
    exact = dataclasses.replace(gs.items[2], qid="q9", gold=(GoldRef("b-01", "15 seconds"),))
    assert "duplicate_question" in codes(check_golden(with_items(gs, exact)), "error")
    # punctuation and case do not make it a different question
    shouty = dataclasses.replace(exact, question="WHAT is the default timeout of the baz worker??")
    assert "duplicate_question" in codes(check_golden(with_items(gs, shouty)), "error")


def test_near_duplicate_within_a_split_is_only_a_warning():
    gs = tiny_golden()
    base = "What is the default timeout of the baz worker in the production cluster of the primary region today"
    a = GoldItem("n1", base, "unanswerable", "beta", "heldout", "baz worker", ())
    b = GoldItem("n2", base + " now", "unanswerable", "beta", "heldout", "baz worker", ())
    f = check_golden(with_items(gs, a, b))
    assert "near_duplicate_question" in codes(f, "warn") and errors(f) == []


def test_dev_heldout_leakage_by_question_text_is_an_error():
    gs = tiny_golden()
    leak = dataclasses.replace(gs.items[2], qid="q9", split="dev", gold=())
    leak = dataclasses.replace(leak, qtype="unanswerable")
    f = check_golden(with_items(gs, leak))
    assert "leak_question" in codes(f, "error")


def test_dev_heldout_leakage_by_shared_gold_fact_is_an_error():
    gs = tiny_golden()
    leak = GoldItem("q9", "How long before the baz worker times out?", "paraphrase", "beta", "dev", "baz worker",
                    (GoldRef("b-01", "15 seconds"),))
    f = check_golden(with_items(gs, leak))
    assert "leak_gold_fact" in codes(f, "error")


def test_two_questions_sharing_a_gold_fact_in_one_split_warn():
    gs = tiny_golden()
    again = GoldItem("q9", "How long before the baz worker times out?", "paraphrase", "beta", "heldout", "baz worker",
                     (GoldRef("b-01", "15 seconds"),))
    f = check_golden(with_items(gs, again))
    assert "shared_gold_fact" in codes(f, "warn") and errors(f) == []


def test_small_slice_warns_with_the_threshold():
    f = check_golden(tiny_golden(), min_slice_n=3)
    assert any(x.code == "small_slice" and x.ref == "area:alpha" for x in f)
    assert not any(x.ref == "qtype:numeric" for x in check_golden(tiny_golden(), min_slice_n=2))
