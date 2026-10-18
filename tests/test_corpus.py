import re
from collections import Counter

from raggate.corpus import AREAS, build_golden
from raggate.golden import from_json, to_json
from raggate.text import contains_phrase


def test_build_is_deterministic():
    a, b = build_golden(), build_golden()
    assert to_json(a) == to_json(b)
    assert to_json(build_golden(seed=8)) != to_json(a)


def test_json_round_trip(gs):
    assert to_json(from_json(to_json(gs))) == to_json(gs)


def test_sizes_are_what_the_readme_claims(gs):
    assert len(gs.items) >= 250
    assert len(gs.split("heldout")) >= 150 and len(gs.split("dev")) >= 100
    assert {p.area for p in gs.passages} == set(AREAS)


def test_every_gold_fact_is_in_its_gold_passage(gs):
    pm = gs.passage_map()
    for item in gs.items:
        for g in item.gold:
            assert contains_phrase(pm[g.passage].text, g.fact), (item.qid, g)


def test_question_types_and_unanswerable_have_no_gold(gs):
    assert Counter(i.qtype for i in gs.items).keys() == {"numeric", "factoid", "multi", "paraphrase", "unanswerable"}
    for i in gs.items:
        assert i.answerable == (i.qtype != "unanswerable")
        assert len(i.gold) == (2 if i.qtype == "multi" else 1 if i.answerable else 0)


def test_split_is_by_component_so_no_component_spans_both_splits(gs):
    seen: dict[str, set[str]] = {}
    for i in gs.items:
        seen.setdefault(i.entity, set()).add(i.split)
    assert all(len(s) == 1 for s in seen.values())


def test_each_gold_fact_belongs_to_one_question_only(gs):
    pairs = [(g.passage, g.fact) for i in gs.items for g in i.gold]
    assert len(pairs) == len(set(pairs))


def test_stale_release_note_distractors_exist_with_a_different_value(gs):
    logs = [p for p in gs.passages if p.pid.endswith("-log")]
    assert len(logs) >= 10
    pm = gs.passage_map()
    for p in logs:
        main = pm[p.pid[:-4]]
        assert "Before version 3" in p.text and main.pid != p.pid


def test_no_payment_card_ssn_or_contact_data_in_corpus_or_questions(gs):
    """Organisation rule: no PCI or HIPAA data. The corpus is made-up infrastructure documentation."""
    blob = " ".join(p.text for p in gs.passages) + " " + " ".join(i.question for i in gs.items)
    assert not re.search(r"\b(?:\d[ -]?){13,19}\b", blob)           # card-number-shaped runs
    assert not re.search(r"\b\d{3}-\d{2}-\d{4}\b", blob)             # SSN-shaped
    assert not re.search(r"[\w.]+@[\w.]+\.\w+", blob)                # email addresses
