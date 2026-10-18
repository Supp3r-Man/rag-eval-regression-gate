"""Golden-set data model: passages, questions, gold evidence, and JSON round-tripping."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass(frozen=True)
class Passage:
    pid: str
    area: str
    title: str
    text: str


@dataclass(frozen=True)
class GoldRef:
    """One piece of gold evidence: the passage that holds the fact, and the fact string itself."""
    passage: str
    fact: str


@dataclass(frozen=True)
class GoldItem:
    qid: str
    question: str
    qtype: str            # numeric | factoid | multi | paraphrase | unanswerable
    area: str
    split: str            # dev | heldout
    entity: str = ""      # the component the question is about; an answer must mention it
    gold: tuple[GoldRef, ...] = field(default_factory=tuple)

    @property
    def answerable(self) -> bool:
        return len(self.gold) > 0

    def slices(self) -> tuple[str, ...]:
        return (f"qtype:{self.qtype}", f"area:{self.area}")


@dataclass
class GoldenSet:
    passages: list[Passage]
    items: list[GoldItem]

    def passage_map(self) -> dict[str, Passage]:
        return {p.pid: p for p in self.passages}

    def split(self, name: str) -> list[GoldItem]:
        return [i for i in self.items if i.split == name]

    def by_qid(self) -> dict[str, GoldItem]:
        return {i.qid: i for i in self.items}


def to_json(gs: GoldenSet) -> dict:
    return {
        "passages": [asdict(p) for p in gs.passages],
        "items": [
            {**{k: v for k, v in asdict(i).items() if k != "gold"},
             "gold": [asdict(g) for g in i.gold]}
            for i in gs.items
        ],
    }


def from_json(d: dict) -> GoldenSet:
    passages = [Passage(**p) for p in d["passages"]]
    items = [
        GoldItem(**{**i, "gold": tuple(GoldRef(**g) for g in i.get("gold", []))})
        for i in d["items"]
    ]
    return GoldenSet(passages, items)


def save_golden(gs: GoldenSet, path: str | Path) -> None:
    Path(path).write_text(json.dumps(to_json(gs), indent=1, sort_keys=True), encoding="utf-8")


def load_golden(path: str | Path) -> GoldenSet:
    return from_json(json.loads(Path(path).read_text(encoding="utf-8")))
