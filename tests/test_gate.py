import dataclasses
import json

import pytest

from raggate.gate import FAIL, INCONCLUSIVE, PASS, GateConfig, failure_shift, run_gate
from raggate.golden import GoldenSet
from raggate.metrics import score_question
from raggate.pipelines import Run
from raggate.report import render_gate, report_to_dict

from conftest import break_answer, with_result


def passing(gs, run, **where):
    """Held-out ANSWERABLE qids that this run passes, filtered by item attributes (area=..., qtype=...).

    Unanswerable questions are skipped: they pass by abstaining, so 'breaking' them with an abstention
    would change nothing."""
    by = gs.by_qid()
    out = []
    for r in run.results:
        it = by[r.qid]
        matches = all(getattr(it, k) == v for k, v in where.items())
        if it.answerable and matches and score_question(it, r, 4).passed:
            out.append(r.qid)
    return out


def rows(report):
    return {r.name: r for r in report.rows}


def test_a_run_gated_against_itself_passes(gs, runs):
    r = run_gate(gs, runs["baseline"], runs["baseline"])
    assert (r.decision, r.exit_code, r.reasons) == (PASS, 0, [])
    assert r.regressed == [] and r.improved == []
    assert all(x.diff == 0 and not x.significant_drop for x in r.rows)


def test_regressed_retriever_fails_and_the_attribution_points_at_retrieval(gs, runs):
    r = run_gate(gs, runs["baseline"], runs["regressed_retriever"])
    assert (r.decision, r.exit_code) == (FAIL, 1)
    assert rows(r)["metric:recall"].significant_drop and rows(r)["metric:pass_rate"].significant_drop
    assert not rows(r)["metric:faithfulness"].significant_drop            # generation did not get less faithful
    shift = failure_shift(r)
    assert shift["retrieval_miss"] > 20 and shift["unanswerable_mishandled"] <= 0


def test_regressed_generator_fails_on_faithfulness_and_the_critical_unanswerable_slice(gs, runs):
    r = run_gate(gs, runs["baseline"], runs["regressed_generator"])
    assert r.decision == FAIL
    assert rows(r)["metric:faithfulness"].significant_drop
    assert not rows(r)["metric:recall"].significant_drop                 # retrieval untouched
    crit = {c.name: c for c in r.critical}
    assert crit["qtype:unanswerable"].failed and crit["qtype:unanswerable"].down > 10
    assert failure_shift(r)["unanswerable_mishandled"] > 10


def test_tight_context_is_caught_by_the_multi_slice_that_the_overall_rate_hides(gs, runs):
    r = run_gate(gs, runs["baseline"], runs["tight_context"])
    assert r.decision == FAIL
    multi = rows(r)["slice:qtype:multi"]
    assert multi.significant_drop and multi.diff < -0.3
    assert abs(rows(r)["metric:pass_rate"].diff) < abs(multi.diff) / 4    # the aggregate diluted it
    assert failure_shift(r)["context_truncation"] > 5


def test_an_improvement_is_not_a_failure(gs, runs):
    r = run_gate(gs, runs["regressed_retriever"], runs["baseline"])
    assert r.decision == PASS and not any(x.significant_drop for x in r.rows)
    assert any(x.lo > 0 for x in r.rows)                                  # and the report says "better"


def test_lowering_the_abstain_threshold_trades_one_slice_for_another(gs, runs):
    """regressed_generator answers paraphrased questions the baseline refuses, at the cost of hallucinating on
    unanswerable ones. Going back to the baseline is better overall and on the critical slice, yet the gate
    still flags the one slice that gets worse: it guards slices, not just the average."""
    r = run_gate(gs, runs["regressed_generator"], runs["baseline"])
    assert r.cand_metrics["pass_rate"] > r.base_metrics["pass_rate"] + 0.15
    assert r.decision == FAIL
    assert [x.name for x in r.rows if x.significant_drop] == ["slice:qtype:paraphrase"]
    assert not any(c.failed for c in r.critical)


def test_one_lost_question_in_a_critical_slice_fails_without_any_significance(gs, runs):
    base = runs["baseline"]
    victim = passing(gs, base, area="auth")[0]
    r = run_gate(gs, base, break_answer(base, victim))
    assert r.decision == FAIL
    assert not any(x.significant_drop for x in r.rows)                    # the statistics saw nothing
    assert any("critical slice area:auth" in x for x in r.reasons)
    assert r.regressed == [victim]


def test_the_same_single_loss_in_a_non_critical_slice_passes(gs, runs):
    base = runs["baseline"]
    victim = passing(gs, base, area="edge", qtype="numeric")[0]
    r = run_gate(gs, base, break_answer(base, victim))
    assert r.decision == PASS and r.regressed == [victim]


def test_a_drop_below_min_drop_is_not_a_failure_even_when_a_slice_test_would_be_close(gs, runs):
    base = runs["baseline"]
    cand = base
    for q in passing(gs, base, area="storage", qtype="numeric")[:2]:
        cand = break_answer(cand, q)
    assert run_gate(gs, base, cand).decision == PASS


def test_min_drop_separates_statistically_detectable_from_practically_relevant(gs, runs):
    only_unanswerable_critical = dict(critical_slices=("qtype:unanswerable",))
    default = run_gate(gs, runs["baseline"], runs["noisy-s1"], GateConfig(**only_unanswerable_critical))
    assert default.decision == FAIL and rows(default)["metric:pass_rate"].p_adj < 0.05
    lenient = run_gate(gs, runs["baseline"], runs["noisy-s1"], GateConfig(min_drop=0.5, **only_unanswerable_critical))
    assert rows(lenient)["metric:pass_rate"].p_adj < 0.05                  # still detectable...
    assert lenient.decision == PASS and not any(x.significant_drop for x in lenient.rows)   # ...but too small to block


def test_holm_adjustment_is_applied_across_the_whole_family(gs, runs):
    r = run_gate(gs, runs["baseline"], runs["regressed_generator"])
    assert len(r.rows) >= 12
    assert all(x.p_adj >= x.p - 1e-12 for x in r.rows)
    assert any(x.p_adj > x.p for x in r.rows)                             # something was actually adjusted
    assert all(x.name != "slice:qtype:unanswerable" and x.name != "slice:area:auth" for x in r.rows)   # critical ones use their own rule


def test_gate_is_deterministic_for_a_seed(gs, runs):
    a = run_gate(gs, runs["baseline"], runs["noisy-s1"])
    b = run_gate(gs, runs["baseline"], runs["noisy-s1"])
    assert report_to_dict(a) == report_to_dict(b)


def test_two_noisy_runs_of_the_same_pipeline_do_not_fail_on_statistics(gs, runs):
    """A/A with churn: same pipeline, different noise seed. Pinned to these seeds: the statistical rule is quiet."""
    r = run_gate(gs, runs["noisy-s1"], runs["noisy-s2"])
    assert not any(x.significant_drop for x in r.rows)
    assert len(r.regressed) > 0 and len(r.improved) > 0                   # there WAS churn, in both directions


# ------------------------------------------------------------------- could-not-judge (exit 2)

def test_errored_candidate_run_is_inconclusive_not_a_regression(gs, runs):
    base = runs["baseline"]
    bad = with_result(base, base.results[0].qid, error="TimeoutError: retriever")
    r = run_gate(gs, base, bad)
    assert (r.decision, r.exit_code) == (INCONCLUSIVE, 2)
    assert any("errored" in x and "TimeoutError" in x for x in r.reasons)


def test_errored_baseline_run_is_also_inconclusive(gs, runs):
    base = runs["baseline"]
    r = run_gate(gs, with_result(base, base.results[0].qid, error="boom"), base)
    assert r.decision == INCONCLUSIVE and any(x.startswith("baseline run has 1 errored") for x in r.reasons)


def test_candidate_missing_questions_is_inconclusive(gs, runs):
    base = runs["baseline"]
    short = Run(base.pipeline, base.split, base.config, base.results[:-5])
    r = run_gate(gs, base, short)
    assert r.decision == INCONCLUSIVE and any("missing 5 question" in x for x in r.reasons)


def test_candidate_with_unknown_or_duplicate_questions_is_inconclusive(gs, runs):
    base = runs["baseline"]
    dup = Run(base.pipeline, base.split, base.config, base.results + [base.results[0]])
    assert any("duplicate" in x for x in run_gate(gs, base, dup).reasons)
    alien = dataclasses.replace(base.results[0], qid="q9999")
    extra = Run(base.pipeline, base.split, base.config, base.results + [alien])
    assert any("not in the heldout split" in x for x in run_gate(gs, base, extra).reasons)


def test_run_for_the_wrong_split_is_inconclusive(gs, runs):
    r = run_gate(gs, runs["baseline"], runs["baseline"], GateConfig(split="dev"))
    assert r.decision == INCONCLUSIVE and any("is for split 'heldout'" in x for x in r.reasons)


def test_broken_golden_set_makes_the_gate_refuse(gs, runs):
    gold_pid = next(i for i in gs.split("heldout") if i.gold).gold[0].passage
    broken = GoldenSet([p for p in gs.passages if p.pid != gold_pid], gs.items)
    r = run_gate(broken, runs["baseline"], runs["baseline"])
    assert r.decision == INCONCLUSIVE and any(x.startswith("golden set: gold_passage_missing") for x in r.reasons)


def test_too_few_questions_is_inconclusive(gs, runs):
    r = run_gate(gs, runs["baseline"], runs["baseline"], GateConfig(min_questions=10_000))
    assert r.decision == INCONCLUSIVE and any("need at least 10000" in x for x in r.reasons)


def test_a_critical_slice_with_no_questions_is_inconclusive_because_nothing_guards_it(gs, runs):
    r = run_gate(gs, runs["baseline"], runs["baseline"], GateConfig(critical_slices=("qtype:imaginary",)))
    assert r.decision == INCONCLUSIVE and any("nothing guards it" in x for x in r.reasons)


# ------------------------------------------------------------------------------------ reporting

def test_text_report_names_the_reasons_the_stage_and_the_critical_slices(gs, runs):
    text = render_gate(run_gate(gs, runs["baseline"], runs["regressed_retriever"]))
    assert text.startswith("GATE FAIL   baseline -> regressed_retriever")
    for needle in ("Holm p", "retrieval_miss", "critical slices", "newly failing"):
        assert needle in text


def test_inconclusive_report_says_it_will_not_say_pass_or_fail(gs, runs):
    text = render_gate(run_gate(gs, runs["baseline"], runs["baseline"], GateConfig(min_questions=10_000)))
    assert text.startswith("GATE INCONCLUSIVE") and "will not say pass or fail" in text


def test_json_report_is_serialisable_and_carries_the_exit_code(gs, runs):
    d = json.loads(json.dumps(report_to_dict(run_gate(gs, runs["baseline"], runs["tight_context"]))))
    assert d["decision"] == "FAIL" and d["exit_code"] == 1 and d["rows"]
