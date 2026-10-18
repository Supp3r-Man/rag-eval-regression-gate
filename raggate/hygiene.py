"""Golden-set hygiene. A regression gate is only as honest as the set behind it.

Errors make the gate refuse to judge (exit 2), because a verdict from a broken set is worse than no
verdict. Warnings are reported and do not block.

  error  duplicate_qid            two items share an id, so pairing runs by id is ambiguous
  error  unknown_split            split is not dev / heldout
  error  empty_split              a split has no questions
  error  missing_entity           an item has no component name, so answers cannot be checked
  error  gold_passage_missing     a gold passage id is not in the corpus
  error  fact_not_in_passage      the gold fact string does not occur in its gold passage
  error  answer_in_question       the question already contains the gold fact
  error  duplicate_question       identical question text twice in one split
  error  leak_question            near-identical question text in dev and held-out
  error  leak_gold_fact           the same (passage, fact) is the gold of questions in both splits
  warn   near_duplicate_question  near-identical question text within one split
  warn   shared_gold_fact         two questions in one split share a gold (passage, fact)
  warn   small_slice              a slice has too few held-out questions to test
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations

from .golden import GoldenSet
from .text import contains_phrase, normalize

NEAR_DUP_JACCARD = 0.9
SPLITS = ("dev", "heldout")


@dataclass(frozen=True)
class Finding:
    severity: str          # error | warn
    code: str
    ref: str               # qid, passage id or slice name
    message: str


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a | b) else 1.0


def check_golden(gs: GoldenSet, min_slice_n: int = 8) -> list[Finding]:
    out: list[Finding] = []
    err = lambda code, ref, msg: out.append(Finding("error", code, ref, msg))  # noqa: E731
    warn = lambda code, ref, msg: out.append(Finding("warn", code, ref, msg))  # noqa: E731
    passages = gs.passage_map()

    for qid, n in Counter(i.qid for i in gs.items).items():
        if n > 1:
            err("duplicate_qid", qid, f"{n} items share this id")
    for i in gs.items:
        if i.split not in SPLITS:
            err("unknown_split", i.qid, f"split {i.split!r}")
        if not i.entity:
            err("missing_entity", i.qid, "no component name")
        for g in i.gold:
            p = passages.get(g.passage)
            if p is None:
                err("gold_passage_missing", i.qid, f"{g.passage} is not in the corpus")
            elif not contains_phrase(p.text, g.fact):
                err("fact_not_in_passage", i.qid, f"{g.fact!r} not found in {g.passage}")
            if contains_phrase(i.question, g.fact):
                err("answer_in_question", i.qid, f"question contains the answer {g.fact!r}")
    for s in SPLITS:
        if not gs.split(s):
            err("empty_split", s, "no questions")

    norm = {i.qid: set(normalize(i.question).split()) for i in gs.items}
    exact = {i.qid: normalize(i.question) for i in gs.items}
    for a, b in combinations(gs.items, 2):
        same_split = a.split == b.split
        if exact[a.qid] == exact[b.qid]:
            if same_split:
                err("duplicate_question", a.qid, f"same text as {b.qid}")
            else:
                err("leak_question", a.qid, f"same text as {b.qid} in the other split")
        elif _jaccard(norm[a.qid], norm[b.qid]) >= NEAR_DUP_JACCARD:
            if same_split:
                warn("near_duplicate_question", a.qid, f"near-identical to {b.qid}")
            else:
                err("leak_question", a.qid, f"near-identical to {b.qid} in the other split")

    by_fact: dict[tuple[str, str], list] = defaultdict(list)
    for i in gs.items:
        for g in i.gold:
            by_fact[(g.passage, g.fact)].append(i)
    for (pid, fact), items in by_fact.items():
        splits = {i.split for i in items}
        if len(splits) > 1:
            err("leak_gold_fact", pid, f"fact {fact!r} is gold in both splits ({', '.join(i.qid for i in items)})")
        elif len(items) > 1:
            warn("shared_gold_fact", pid, f"{', '.join(i.qid for i in items)} share fact {fact!r}")

    held = gs.split("heldout")
    counts: Counter = Counter(s for i in held for s in i.slices())
    for s, n in sorted(counts.items()):
        if n < min_slice_n:
            warn("small_slice", s, f"only {n} held-out questions")
    return out


def errors(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.severity == "error"]
