"""Synthetic corpus and golden set for a fictional product ("Brightwave") documentation site.

Everything is generated from a seed, so the golden set is reproducible. Nothing here is real data:
the components, numbers and teams are made up. The generator deliberately builds the traps a real
RAG system meets: many components share words ("gateway", "queue", "worker"), every component has
the same attribute vocabulary ("default timeout"), some facts live in a separate addendum passage,
some questions are paraphrased so they share no attribute words with the answer sentence, and some
ask for an attribute the component simply does not have (the right answer is to abstain).
"""
from __future__ import annotations

import random

from .golden import GoldItem, GoldRef, GoldenSet, Passage

AREAS: dict[str, list[str]] = {
    "auth": [
        "token issuer", "session store", "key vault", "login gateway", "consent service",
        "identity sync worker", "device registry", "audit log writer", "password reset service",
        "directory connector", "signing service", "access review queue", "recovery code service",
        "role mapper",
    ],
    "webhooks": [
        "event delivery worker", "event replay queue", "webhook signing service", "endpoint registry",
        "delivery log writer", "retry scheduler", "subscription manager", "event export queue",
        "payload validator", "dead letter queue", "webhook gateway", "test event sender",
        "schema registry", "delivery metrics collector",
    ],
    "storage": [
        "object store", "blob gateway", "archive worker", "export worker", "snapshot scheduler",
        "replication service", "upload gateway", "index builder", "cleanup worker", "quota tracker",
        "backup service", "restore queue", "tiering worker", "checksum verifier",
    ],
    "edge": [
        "api gateway", "rate limiter", "quota service", "burst controller", "request router",
        "edge cache", "throttle worker", "usage meter", "key registry", "traffic shaper",
        "abuse detector", "limit sync worker", "ip allow list service", "cors policy service",
    ],
    "deploy": [
        "release scheduler", "build runner", "artifact registry", "rollout controller",
        "config service", "canary analyzer", "deploy gateway", "environment manager",
        "rollback worker", "promotion queue", "secrets syncer", "release log writer",
        "approval service", "drift detector",
    ],
}

REGIONS = ["eu-north", "us-east", "us-west", "ap-south", "eu-west", "ca-central"]
TEAMS = ["Platform", "Core", "Edge", "Insights", "Reliability"]

# name -> (noun phrase used in passages and questions, kind, value maker, paraphrased question)
# The paraphrase deliberately avoids the noun phrase: it is what a user types, not what the docs say.
ATTRS: dict[str, dict] = {
    "timeout": dict(phrase="default timeout", kind="numeric",
                    make=lambda r: f"{r.choice([5, 10, 15, 20, 30, 45, 60, 90, 120])} seconds",
                    para="What timeout does the {e} apply by default?"),
    "retry": dict(phrase="retry limit", kind="numeric",
                  make=lambda r: f"{r.randint(2, 9)} attempts",
                  para="How many retries does the {e} permit?"),
    "payload": dict(phrase="maximum payload size", kind="numeric",
                    make=lambda r: f"{r.choice([1, 2, 5, 10, 25, 50, 100])} MB",
                    para="How large can a payload for the {e} be?"),
    "retention": dict(phrase="retention period", kind="numeric",
                      make=lambda r: f"{r.choice([7, 14, 30, 60, 90, 180, 365])} days",
                      para="How long does the {e} hold on to old records?"),
    "rate": dict(phrase="default rate limit", kind="numeric",
                 make=lambda r: f"{r.choice([60, 120, 300, 600, 1200])} requests per minute",
                 para="How many calls will the {e} accept in a minute?"),
    "tokenlife": dict(phrase="token lifetime", kind="numeric",
                      make=lambda r: f"{r.choice([5, 10, 15, 30, 60, 120])} minutes",
                      para="How long do tokens from the {e} last?"),
    "region": dict(phrase="home region", kind="factoid",
                   make=lambda r: r.choice(REGIONS),
                   para="Where is the {e} hosted?"),
    "owner": dict(phrase="owner", kind="factoid",
                  make=lambda r: f"{r.choice(TEAMS)} team",
                  para="Which team owns the {e}?"),
}

_FILLER_ENTITY = [
    "Operators of the {e} should review its dashboards every week.",
    "Changes to the {e} are announced in the release notes.",
    "The {e} emits structured logs that include a request identifier.",
]
_FILLER_GENERIC = [
    "The timeout and retry settings can be tuned through the admin console.",
    "Limits such as payload size and rate limit are reviewed each quarter.",
    "See the runbook for escalation paths and the on-call rotation.",
]


def _fact_sentence(rng: random.Random, attr: str, entity: str, value: str) -> str:
    phrase = ATTRS[attr]["phrase"]
    value_text = f"the {value}" if attr == "owner" else value
    variant = rng.randrange(3)
    if variant == 0:
        return f"The {phrase} of the {entity} is {value_text}."
    if variant == 1:
        return f"For the {entity}, the {phrase} is {value_text}."
    return f"In the {entity}, the {phrase} is set to {value_text}."


def _title(entity: str) -> str:
    return entity.title()


def build_golden(seed: int = 7, dev_fraction: float = 0.4) -> GoldenSet:
    rng = random.Random(seed)
    passages: list[Passage] = []
    entities: list[tuple[str, str]] = []                  # (area, entity)
    facts: dict[tuple[str, str], dict[str, tuple[str, str]]] = {}    # -> attr -> (passage id, value)
    absent: dict[tuple[str, str], list[str]] = {}
    addendum_attrs: dict[tuple[str, str], list[str]] = {}
    main_attrs: dict[tuple[str, str], list[str]] = {}

    attr_names = list(ATTRS)
    for area, names in AREAS.items():
        passages.append(Passage(
            f"{area}-overview", area, f"Overview of {area} components",
            f"Overview of {area} components. Every component has a default timeout, a retry limit, "
            "a retention period and a rate limit. Values differ per component, so check the page "
            "of the component you use."))
        with_addendum = set(rng.sample(range(len(names)), len(names) // 2))
        for idx, entity in enumerate(names):
            key = (area, entity)
            entities.append(key)
            order = attr_names[:]
            rng.shuffle(order)
            main = order[:3]
            add = order[3:5] if idx in with_addendum else []
            main_attrs[key], addendum_attrs[key] = main, add
            absent[key] = [a for a in order if a not in main and a not in add]
            pid = f"{area}-{idx + 1:02d}"
            facts[key] = {}
            sentences = []
            for attr in main:
                value = ATTRS[attr]["make"](rng)
                facts[key][attr] = (pid, value)
                sentences.append(_fact_sentence(rng, attr, entity, value))
            sentences.append(rng.choice(_FILLER_ENTITY).format(e=entity))
            sentences.append(rng.choice(_FILLER_GENERIC))
            rng.shuffle(sentences)
            passages.append(Passage(pid, area, _title(entity), f"{_title(entity)}. " + " ".join(sentences)))
            if idx % 5 == 2:
                # A stale release-note passage: same component and attribute words, an OLD value.
                attr = rng.choice(main)
                old_value = ATTRS[attr]["make"](rng)
                while old_value == facts[key][attr][1]:
                    old_value = ATTRS[attr]["make"](rng)
                lead = "owned by" if attr == "owner" else "set to"
                passages.append(Passage(
                    f"{pid}-log", area, f"{_title(entity)} release notes",
                    f"Release notes for the {entity}. Before version 3, the {ATTRS[attr]['phrase']} "
                    f"of the {entity} was {lead} {old_value}."))
            if add:
                apid = f"{pid}-add"
                asents = []
                for attr in add:
                    value = ATTRS[attr]["make"](rng)
                    facts[key][attr] = (apid, value)
                    asents.append(_fact_sentence(rng, attr, entity, value))
                passages.append(Passage(
                    apid, area, f"{_title(entity)} addendum",
                    f"Addendum for the {entity}. " + " ".join(asents)
                    + " This addendum supersedes earlier guidance."))

    # Split by component, so no component's facts appear in both dev and held-out.
    order = entities[:]
    rng.shuffle(order)
    n_dev = round(len(order) * dev_fraction)
    split_of = {key: ("dev" if i < n_dev else "heldout") for i, key in enumerate(order)}

    raw: list[dict] = []
    for key in entities:
        area, entity = key
        used: set[str] = set()
        if addendum_attrs[key]:
            m = rng.choice(main_attrs[key])
            a = rng.choice(addendum_attrs[key])
            used |= {m, a}
            pm, vm = facts[key][m]
            pa, va = facts[key][a]
            raw.append(dict(
                question=f"What are the {ATTRS[m]['phrase']} and the {ATTRS[a]['phrase']} of the {entity}?",
                qtype="multi", key=key,
                gold=(GoldRef(pm, vm), GoldRef(pa, va))))
        for attr in main_attrs[key] + addendum_attrs[key]:
            if attr in used:
                continue
            pid, value = facts[key][attr]
            if rng.random() < 0.12:
                raw.append(dict(question=ATTRS[attr]["para"].format(e=entity), qtype="paraphrase",
                                key=key, gold=(GoldRef(pid, value),)))
            else:
                raw.append(dict(question=f"What is the {ATTRS[attr]['phrase']} of the {entity}?",
                                qtype=ATTRS[attr]["kind"], key=key, gold=(GoldRef(pid, value),)))
        if rng.random() < 0.65 and absent[key]:
            attr = rng.choice(absent[key])
            raw.append(dict(question=f"What is the {ATTRS[attr]['phrase']} of the {entity}?",
                            qtype="unanswerable", key=key, gold=()))

    rng.shuffle(raw)
    items = [
        GoldItem(qid=f"q{i + 1:04d}", question=r["question"], qtype=r["qtype"], area=r["key"][0],
                 split=split_of[r["key"]], entity=r["key"][1], gold=tuple(r["gold"]))
        for i, r in enumerate(raw)
    ]
    return GoldenSet(passages, items)
