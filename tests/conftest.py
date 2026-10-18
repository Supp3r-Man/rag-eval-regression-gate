import dataclasses

import pytest

from raggate.corpus import build_golden
from raggate.golden import GoldenSet, GoldItem, GoldRef, Passage
from raggate.pipelines import ABSTAIN_TEXT, RagResult, Run, named_config, pipeline_names, run_pipeline


@pytest.fixture(scope="session")
def gs() -> GoldenSet:
    return build_golden()


@pytest.fixture(scope="session")
def runs(gs):
    out = {n: run_pipeline(gs, named_config(n), "heldout") for n in pipeline_names()}
    out["noisy-s2"] = run_pipeline(gs, named_config("noisy-s2"), "heldout")
    return out


def with_result(run: Run, qid: str, **changes) -> Run:
    """A copy of `run` where one question's result is replaced."""
    results = [dataclasses.replace(r, **changes) if r.qid == qid else r for r in run.results]
    return Run(run.pipeline + "*", run.split, run.config, results)


def break_answer(run: Run, qid: str) -> Run:
    """Make one question fail by turning its answer into an abstention."""
    return with_result(run, qid, answer=ABSTAIN_TEXT, abstained=True)


def tiny_golden() -> GoldenSet:
    """A hand-made set small enough to reason about; passes hygiene except for size warnings."""
    passages = [
        Passage("a-01", "alpha", "Foo Service", "Foo Service. The default timeout of the foo service is 30 seconds."),
        Passage("a-02", "alpha", "Bar Queue", "Bar Queue. The retry limit of the bar queue is 5 attempts."),
        Passage("b-01", "beta", "Baz Worker", "Baz Worker. The default timeout of the baz worker is 15 seconds."),
    ]
    items = [
        GoldItem("q1", "What is the default timeout of the foo service?", "numeric", "alpha", "dev", "foo service",
                 (GoldRef("a-01", "30 seconds"),)),
        GoldItem("q2", "What is the retry limit of the bar queue?", "numeric", "alpha", "heldout", "bar queue",
                 (GoldRef("a-02", "5 attempts"),)),
        GoldItem("q3", "What is the default timeout of the baz worker?", "numeric", "beta", "heldout", "baz worker",
                 (GoldRef("b-01", "15 seconds"),)),
        GoldItem("q4", "What is the retry limit of the baz worker?", "unanswerable", "beta", "heldout", "baz worker", ()),
    ]
    return GoldenSet(passages, items)


def result(qid, retrieved, context, answer, abstained=False, error=""):
    return RagResult(qid, retrieved, context, answer, abstained, error)
