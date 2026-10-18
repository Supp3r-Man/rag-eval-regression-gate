from dataclasses import replace

import pytest

from raggate.attribution import breakdown
from raggate.metrics import score_question
from raggate.pipelines import (ABSTAIN_TEXT, decompose, load_run, named_config, run_pipeline, save_run)


def _scores(gs, run):
    by = gs.by_qid()
    return [score_question(by[r.qid], r, int(run.config["k"])) for r in run.results]


def test_pipelines_are_deterministic(gs):
    a = run_pipeline(gs, named_config("noisy-s1"), "dev")
    b = run_pipeline(gs, named_config("noisy-s1"), "dev")
    assert a.results == b.results


def test_noise_seed_changes_which_questions_are_corrupted(gs):
    a = run_pipeline(gs, named_config("noisy-s1"), "dev")
    b = run_pipeline(gs, named_config("noisy-s2"), "dev")
    assert [r.retrieved for r in a.results] != [r.retrieved for r in b.results]


def test_unknown_pipeline_is_an_error():
    with pytest.raises(KeyError):
        named_config("nope")


def test_decompose_splits_two_attribute_questions_only():
    q = "What are the default timeout and the retry limit of the event delivery worker?"
    assert decompose(q) == ["What is the default timeout of the event delivery worker?",
                            "What is the retry limit of the event delivery worker?"]
    single = "What is the retry limit of the retry scheduler?"
    assert decompose(single) == [single]


def test_baseline_is_not_a_strawman(gs, runs):
    """The baseline must be good enough that its failures are interesting, and imperfect enough to be real."""
    sc = _scores(gs, runs["baseline"])
    rate = sum(s.passed for s in sc) / len(sc)
    assert 0.75 < rate < 0.95
    assert sum(1 for s in sc if s.failure == "generation_error") > 5           # real, diagnosable failures


def test_each_regression_leaves_its_own_fingerprint_in_the_attribution(gs, runs):
    base = breakdown(_scores(gs, runs["baseline"]))
    retr = breakdown(_scores(gs, runs["regressed_retriever"]))
    gen = breakdown(_scores(gs, runs["regressed_generator"]))
    ctx = breakdown(_scores(gs, runs["tight_context"]))
    noisy = breakdown(_scores(gs, runs["noisy-s1"]))
    assert base["retrieval_miss"] == base["ranking_miss"] == base["context_truncation"] == 0
    assert retr["retrieval_miss"] > 20 and gen["retrieval_miss"] == 0
    assert gen["unanswerable_mishandled"] > base["unanswerable_mishandled"] + 10
    assert ctx["context_truncation"] > 5
    assert noisy["ranking_miss"] > 5 and noisy["context_truncation"] == 0


def test_regressed_retriever_really_does_not_index_addenda(gs):
    run = run_pipeline(gs, named_config("regressed_retriever"), "heldout")
    assert not any(pid.endswith("-add") for r in run.results for pid in r.retrieved)


def test_regressed_generator_drifts_numbers_and_the_scorer_notices(gs, runs):
    drifted = [r for r in runs["regressed_generator"].results if any(c.isdigit() for c in r.answer)]
    faith = [s.faithfulness for s in _scores(gs, runs["regressed_generator"])]
    assert any(f < 1.0 for f in faith) and all(f == 1.0 for f in
                                               [s.faithfulness for s in _scores(gs, runs["baseline"])])
    assert drifted


def test_abstention_is_marked_and_uses_the_standard_text(runs):
    ab = [r for r in runs["baseline"].results if r.abstained]
    assert ab and all(r.answer == ABSTAIN_TEXT for r in ab)


def test_a_crash_is_recorded_as_an_error_not_swallowed(gs):
    def boom(item):
        if item.qid == gs.split("dev")[0].qid:
            raise RuntimeError("index offline")
    run = run_pipeline(gs, named_config("baseline"), "dev", hook=boom)
    errored = [r for r in run.results if r.error]
    assert len(errored) == 1 and "RuntimeError: index offline" in errored[0].error
    assert len(run.results) == len(gs.split("dev"))                            # the rest still ran


def test_run_round_trips_through_json(tmp_path, runs):
    p = tmp_path / "r.json"
    save_run(runs["noisy-s1"], p)
    back = load_run(p)
    assert back.results == runs["noisy-s1"].results and back.config == runs["noisy-s1"].config


def test_config_knob_changes_only_what_it_claims(gs):
    base = run_pipeline(gs, named_config("baseline"), "dev")
    lower_k = run_pipeline(gs, replace(named_config("baseline"), k=1), "dev")
    assert [r.retrieved for r in base.results] == [r.retrieved for r in lower_k.results]   # retrieval untouched
    assert any(len(a.context) > len(b.context) for a, b in zip(base.results, lower_k.results))
