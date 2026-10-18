"""Judges and judge calibration.

Faithfulness and answer correctness are sometimes graded by an LLM judge. A judge is a rater, and an
uncalibrated rater is a source of noise you cannot see, so before its scores are allowed to gate
anything it is compared with labelled answers using Cohen's kappa (agreement beyond chance) with a
bootstrap interval, and its mistakes are broken down by KIND of answer, which says what it is blind to.

What is and is not here:
  * `RuleJudge`, `LenientJudge`, `AlwaysPassJudge` are deterministic judges used to exercise the code.
  * `TransportJudge` wraps any `prompt -> text` function (a real model). It is UNTESTED AGAINST A LIVE
    MODEL: no API key was used. It is tested here with fake transports, including garbage replies.
  * The labels are SYNTHETIC: each labelled answer is built by a known operation (a correct sentence, a
    changed number, the right value for the wrong component, ...), so its label is true by
    construction. They are not human annotations; a real deployment needs a few hundred of those.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable, Protocol

from .corpus import ATTRS, REGIONS, TEAMS
from .golden import GoldenSet
from .pipelines import ABSTAIN_TEXT
from .stats import cohens_kappa, kappa_interval
from .text import content_tokens, contains_phrase, sentences, tokenize


class JudgeError(Exception):
    """The judge returned something that cannot be read as a verdict."""


class CouldNotCalibrate(Exception):
    """Not enough usable labels or judgements to say anything: the CLI exits 2."""


@dataclass(frozen=True)
class JudgeInput:
    question: str
    answer: str
    context: str
    entity: str
    gold_facts: tuple[str, ...]
    answerable: bool
    abstained: bool


@dataclass(frozen=True)
class LabelledExample:
    kind: str
    label: bool            # True = an acceptable answer
    input: JudgeInput


class Judge(Protocol):
    def __call__(self, inp: JudgeInput) -> bool: ...


def _states_fact(inp: JudgeInput, fact: str, require_entity: bool) -> bool:
    ent = set(content_tokens(inp.entity))
    for s in sentences(inp.answer):
        if require_entity and not ent <= set(tokenize(s)):
            continue
        if contains_phrase(s, fact):
            return True
    return False


class RuleJudge:
    """Checks the gold facts and the component name. `noise` flips verdicts (seeded) to simulate a bad judge."""

    def __init__(self, noise: float = 0.0, seed: int = 0):
        self.noise, self.seed = noise, seed

    def __call__(self, inp: JudgeInput) -> bool:
        if not inp.answerable:
            verdict = inp.abstained
        else:
            verdict = (not inp.abstained) and all(_states_fact(inp, f, True) for f in inp.gold_facts)
        if self.noise and random.Random(f"{self.seed}:{inp.question}:{inp.answer}").random() < self.noise:
            return not verdict
        return verdict


class LenientJudge:
    """A plausible-looking judge with blind spots: it never checks the component, and never checks
    that an unanswerable question was refused."""

    def __call__(self, inp: JudgeInput) -> bool:
        if not inp.answerable:
            return True
        return (not inp.abstained) and all(_states_fact(inp, f, False) for f in inp.gold_facts)


class AlwaysPassJudge:
    def __call__(self, inp: JudgeInput) -> bool:
        return True


_VERDICT = re.compile(r"^\s*VERDICT:\s*(PASS|FAIL)\s*$", re.M)

PROMPT = """You are grading an answer from a documentation assistant.

Question: {question}

Documentation excerpt:
<<<
{context}
>>>

Answer to grade:
<<<
{answer}
>>>

PASS only if the answer states the requested value for the component named in the question and
claims nothing the excerpt does not support. If the excerpt does not contain the answer, the answer
must say it could not be found. Reply with exactly one line: VERDICT: PASS or VERDICT: FAIL"""


class TransportJudge:
    """LLM judge over any `prompt -> text` callable. UNTESTED against a live model."""

    def __init__(self, transport: Callable[[str], str]):
        self.transport = transport

    def prompt(self, inp: JudgeInput) -> str:
        return PROMPT.format(question=inp.question, context=inp.context, answer=inp.answer)

    def __call__(self, inp: JudgeInput) -> bool:
        reply = self.transport(self.prompt(inp))
        if not isinstance(reply, str):
            raise JudgeError("transport did not return text")
        found = {m.upper() for m in _VERDICT.findall(reply)}
        if not found:
            raise JudgeError("no 'VERDICT: PASS|FAIL' line in the reply")
        if len(found) > 1:
            raise JudgeError("reply contains conflicting verdicts")
        return found.pop() == "PASS"


# ---------------------------------------------------------------- labelled set, true by construction

def _gold_sentence(gs: GoldenSet, item, ref) -> str:
    ent = set(content_tokens(item.entity))
    p = gs.passage_map()[ref.passage]
    for s in sentences(p.text):
        if ent <= set(tokenize(s)) and contains_phrase(s, ref.fact):
            return s
    raise ValueError(f"no sentence with {ref.fact!r} in {ref.passage}")


def _corrupt_fact(fact: str) -> str:
    if re.search(r"\d+", fact):
        return re.sub(r"\d+", lambda m: str(int(m.group()) + 7), fact, count=1)
    if fact in REGIONS:
        return next(r for r in REGIONS if r != fact)
    return next(f"{t} team" for t in TEAMS if f"{t} team" != fact)


def _attr_phrase(text: str) -> str | None:
    low = text.lower()
    return next((a["phrase"] for a in ATTRS.values() if a["phrase"] in low), None)


def build_judge_labels(gs: GoldenSet, split: str = "dev", seed: int = 3) -> list[LabelledExample]:
    rng = random.Random(seed)
    entities_by_area: dict[str, list[str]] = {}
    for i in gs.items:
        entities_by_area.setdefault(i.area, [])
        if i.entity not in entities_by_area[i.area]:
            entities_by_area[i.area].append(i.entity)
    out: list[LabelledExample] = []
    items = gs.split(split)
    for idx, item in enumerate(items):
        pmap = gs.passage_map()
        if item.answerable:
            ctx = " ".join(pmap[g.passage].text for g in item.gold)
            sents = [_gold_sentence(gs, item, g) for g in item.gold]
            facts = tuple(g.fact for g in item.gold)
            kinds = ["correct", "paraphrase", "wrong_number", "wrong_entity", "abstain_answerable"]
            if len(item.gold) > 1:
                kinds.append("partial")
        else:
            ctx = ""
            sents, facts = [], ()
            kinds = ["abstained_unanswerable", "answered_unanswerable"]
        picked = kinds if not item.answerable else sorted({kinds[idx % len(kinds)], kinds[(idx + 2) % len(kinds)]})
        for kind in picked:
            label, abstained = False, False
            if kind == "correct":
                answer, label = " ".join(sents), True
            elif kind == "paraphrase":
                answer, label = "Per the docs: " + " ".join(sents), True
                if len(item.gold) == 1:
                    answer = f"Short answer: {facts[0]} ({item.entity})."
            elif kind == "wrong_number":
                answer = " ".join(s.replace(f, _corrupt_fact(f)) for s, f in zip(sents, facts))
            elif kind == "wrong_entity":
                other = rng.choice([e for e in entities_by_area[item.area] if e != item.entity])
                answer = " ".join(s.replace(item.entity, other) for s in sents)
            elif kind == "abstain_answerable":
                answer, abstained = ABSTAIN_TEXT, True
            elif kind == "partial":
                answer = sents[0]
            elif kind == "abstained_unanswerable":
                answer, label, abstained = ABSTAIN_TEXT, True, True
            else:  # answered_unanswerable: a real sentence about the same attribute of another component
                phrase = _attr_phrase(item.question)
                pool = [s for p in gs.passages for s in sentences(p.text)
                        if phrase and phrase in s.lower() and item.entity not in s.lower()]
                if not pool:
                    continue
                answer = rng.choice(pool)
            out.append(LabelledExample(kind, label, JudgeInput(
                item.question, answer, ctx, item.entity, facts, item.answerable, abstained)))
    return out


# ---------------------------------------------------------------------------------- calibration

@dataclass
class Calibration:
    n: int
    unjudged: int
    accuracy: float
    kappa: float
    kappa_lo: float
    kappa_hi: float
    min_kappa: float
    by_kind: dict[str, tuple[int, int]] = field(default_factory=dict)   # kind -> (n, judge wrong)

    @property
    def trustworthy(self) -> bool:
        """The whole interval must clear the bar, not just the point estimate."""
        return self.kappa_lo >= self.min_kappa


def calibrate(judge: Judge, labelled: list[LabelledExample], min_kappa: float = 0.7, min_labels: int = 30,
              min_per_class: int = 10, max_unjudged: float = 0.1) -> Calibration:
    pos = sum(e.label for e in labelled)
    if len(labelled) < min_labels or pos < min_per_class or len(labelled) - pos < min_per_class:
        raise CouldNotCalibrate(
            f"need >= {min_labels} labels with >= {min_per_class} of each class; "
            f"have {len(labelled)} ({pos} acceptable)")
    truth, said, by_kind, unjudged = [], [], {}, 0
    for e in labelled:
        try:
            v = bool(judge(e.input))
        except JudgeError:
            unjudged += 1
            continue
        truth.append(e.label)
        said.append(v)
        n, wrong = by_kind.get(e.kind, (0, 0))
        by_kind[e.kind] = (n + 1, wrong + (v != e.label))
    if unjudged / len(labelled) > max_unjudged:
        raise CouldNotCalibrate(f"{unjudged} of {len(labelled)} answers could not be judged (> {max_unjudged:.0%})")
    acc = sum(t == s for t, s in zip(truth, said)) / len(truth)
    k = cohens_kappa(truth, said)
    lo, hi = kappa_interval(truth, said)
    return Calibration(len(truth), unjudged, acc, k, lo, hi, min_kappa, by_kind)


def require_calibrated(judge: Judge, labelled: list[LabelledExample], **kw) -> Calibration:
    cal = calibrate(judge, labelled, **kw)
    if not cal.trustworthy:
        raise CouldNotCalibrate(
            f"judge not trustworthy: kappa {cal.kappa:.2f} (95% interval {cal.kappa_lo:.2f} to {cal.kappa_hi:.2f}), "
            f"needs the whole interval >= {cal.min_kappa}")
    return cal
