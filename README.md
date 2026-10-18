# rag-eval-regression-gate

![ci](https://github.com/Supp3r-Man/rag-eval-regression-gate/actions/workflows/ci.yml/badge.svg)

A CI **regression gate for a RAG system**. Given a golden set and two runs (baseline and candidate), it scores retrieval and generation separately, says **which stage lost each failed question**, decides pass/fail with **paired-bootstrap confidence intervals, Holm correction across every metric and slice, and a stricter no-statistics rule for critical slices**, refuses to judge when the set or the runs are broken, and tells you **how big a drop it could actually have seen**.

> **What is verified, and what is not.** 128 tests pass locally on Python 3.13, stdlib only. The CI workflow (Linux, Python 3.11 and 3.13) also ran green after the first push. I also broke the code 14 ways on purpose (Holm, the critical rule, `min_drop`, attribution order, entity check, numeric-drift check, leakage check, crash exit code, errored-run check, bootstrap p, judge unjudged-limit, kappa, recall, context budget) and the tests failed each time; the first pass left one survivor (`min_drop` ignored), which is why `test_min_drop_separates_statistically_detectable_from_practically_relevant` exists. **The five RAG pipelines are toys**: a BM25 retriever and an extractive, lexical "generator" with one knob turned each, so the gate has systems with *known* behaviour to judge. The corpus is a made-up product-documentation site generated from a seed (no real or personal data). **No LLM was called.** The `TransportJudge` adapter is untested against a live model, and the judge labels are synthetic (true by construction), not human annotations. Nothing here says how a real RAG system scores.

## The idea: "pass rate fell 9 points" is not a diagnosis

```
golden set ──► run pipeline ──► score every question ──► attribute each failure to a stage ──► gate
 (questions,    (saved as         retrieval:  recall@k, MRR, nDCG       retrieval_miss          PASS / FAIL /
  gold passages  JSON, so the     generation: fact coverage,            ranking_miss            INCONCLUSIVE
  + gold facts)  scorer is the    faithfulness                          context_truncation      (exit 0 / 1 / 2)
                 only judge)      verdict:    pass / fail               generation_error
                                                                        unanswerable_mishandled
```

A question **passes** when every gold fact is stated in an answer sentence *that also names the right component* (so "30 seconds" about the wrong service does not count) and every answer sentence is supported by one sentence of the retrieved context. An unanswerable question passes only if the pipeline **abstains**.

## Failure attribution

The earliest stage that lost any piece of gold evidence is blamed, because a later stage cannot recover what an earlier one dropped:

| Class | Meaning |
|---|---|
| `retrieval_miss` | a gold passage is not in the retrieved top-10 at all |
| `ranking_miss` | every gold passage was retrieved, but one ranks below the context cut (> k) |
| `context_truncation` | gold passages made the cut, but the context budget chopped the fact out of the prompt |
| `generation_error` | the fact *was* in the prompt and the answer is still wrong, unsupported, or an abstention |
| `unanswerable_mishandled` | the corpus cannot answer it and the pipeline answered anyway |

## The gate

Two rules, on purpose (`raggate/gate.py` has the reasoning):

1. **Statistical rule** for the 6 metrics and every qtype/area slice. A test FAILS only if its one-sided paired-bootstrap p-value, **Holm-adjusted over the whole family**, is below 0.05 **and** the observed drop is at least `min_drop` (2 pts). Holm keeps the family-wise false-alarm rate at alpha across ~14 tests; `min_drop` stops a detectable-but-meaningless wiggle from blocking a release. Resampling is by question, with the *same* resamples for every test, so the baseline/candidate pairing is never broken.
2. **Critical-slice rule.** In a critical slice (default `qtype:unanswerable`, where a regression means hallucinating, and `area:auth`) a **net loss of one question** fails the gate with no test, because a small slice has no power and you would not ship that regression. The price is noise-sensitivity: see the false-alarm table below.

**INCONCLUSIVE (exit 2)** instead of a verdict when: the golden set has hygiene errors, a run has errored questions, ids do not match the split, there are fewer than 60 questions, or a critical slice has no questions (nothing guards it). A crash anywhere also exits 2: *exit 1 means "regression found" and nothing else*.

## Measured on the five toy pipelines (`python -m raggate demo`)

Held-out split, 174 questions (148 answerable, 26 unanswerable). All numbers are from real runs of this code.

| Pipeline | What was changed | Pass rate | Recall@k | Faithfulness | Gate vs baseline |
|---|---|---|---|---|---|
| baseline | BM25 top-4, 1400-char context, abstains below 0.9 coverage | 86.2% | 100.0% | 100.0% | n/a |
| regressed_retriever | ingestion bug: addendum passages never indexed | 64.4% | 77.7% | 100.0% | **FAIL**: 44 `retrieval_miss`; every `multi` question fails (0/22) |
| regressed_generator | abstain threshold 0.9 to 0.55, 30% of numbers silently doubled | 64.9% | 100.0% | 82.2% | **FAIL**: faithfulness -17.8 pts, 23 unanswerable questions now answered, critical slice FAIL |
| tight_context | context budget 1400 to 520 chars | 81.0% | 100.0% | 100.0% | **FAIL**: 10 `context_truncation`; `multi` slice 95.5% to 59.1% |
| noisy-s1 | ~12% of questions get their top hit demoted (seeded) | 77.6% | 90.5% | 100.0% | **FAIL**: 16 `ranking_miss`, pass rate -8.6 pts |

Things this table shows that an average would hide:

- **Each regression has its own fingerprint in the attribution.** The retriever bug is 44 `retrieval_miss` and zero generation change; the generator change is `unanswerable_mishandled` 1 to 24 with retrieval metrics *identical* to baseline (recall, MRR and nDCG rows show `+0.0` and "ok").
- **`tight_context` is a slice story.** Overall pass rate moves 5 pts but `multi` questions, which need two passages in the prompt, fall 36 pts. The aggregate dilutes it.
- **Gating is symmetric about trade-offs.** `regressed_generator` *improves* paraphrased questions from 0/17 to 11/17 (a lower abstain threshold answers things the baseline refuses) while hallucinating on unanswerable ones. Gating `regressed_generator -> baseline` is better overall (+21 pts) and still FAILs, because the paraphrase slice drops significantly. That behaviour is a test.
- **A/A:** the baseline against itself passes with zero diff. `noisy-s1` against `noisy-s2` (same pipeline, different noise seed) has 8 questions newly failing and 15 newly passing, and passes: the statistical rule is quiet under pure churn. The critical-slice rule is *not* automatically quiet under churn, see below.

### Baseline weaknesses, stated plainly

Baseline passes 0 of 17 held-out **paraphrased** questions (the lexical generator cannot match "How large can a payload for the X be?" to "maximum payload size", so it abstains), and 6 more failures are stale release-note passages ranking above the real one. 24 of its 174 questions fail. **A gate can only see a slice get worse, so a regression in the paraphrase slice would be invisible: it is already at 0%.** The baseline was tuned on the dev split only (`python -m raggate tune`: abstain threshold 0.9 is the smallest value reaching the best dev pass rate); held-out was not used to pick any setting.

## What drop can this gate see? (`python -m raggate power`)

A gate that never fires is blind, not safe. The simulation draws paired outcomes (a question gets worse with probability `churn + drop`, better with probability `churn`) and counts how often the statistical rule fails the gate. "family" uses alpha/15, the threshold the weakest test faces in the real gate. 200 trials per cell; at 200 trials a rate has about +/-3 pts of sampling noise.

n = 174 (this repo's held-out size), min_drop 2 pts:

| true drop | deterministic pipeline (churn 0%): single / family | LLM-like noise (churn 4% per direction): single / family |
|---|---|---|
| 0 pts (false alarm) | 0.0% / 0.0% | 4.0% / 0.0% |
| 2 pts | 41.5% / 7.0% | 21.5% / 1.0% |
| 4 pts | 93.0% / 61.0% | 47.0% / 10.5% |
| 6 pts | 99.5% / 92.5% | 69.0% / 20.5% |
| 10 pts | 100% / 100% | 97.0% / 65.0% |
| 15 pts | 100% / 100% | 99.0% / 87.5% |

Honest reading: with a deterministic pipeline this gate reliably catches a 6-pt drop at n = 174; with realistic run-to-run noise you need a **10 to 15 pt** drop for 80%+ power at the strict threshold, and a **2 to 4 pt** drop is mostly invisible. Across other sizes (churn 4%, family): the smallest simulated drop with >= 80% power is 15 pts at n = 200 and 10 pts at n = 400; at n = 50 and 100 no drop on the grid (up to 15 pts) reaches it. More questions or repeated runs per question are the only fixes. The "single" false-alarm rate reached 7.0% at n = 400 (14 of 200 trials), above the nominal 5%: percentile-bootstrap p-values are slightly liberal, which is one more reason `min_drop` and Holm are there.

**Critical-slice false alarms on an unchanged pipeline** (exact calculation, not simulation): with 0% churn the rule never fires; with a noisy pipeline it fires often.

| slice size | churn 2% | churn 5% |
|---|---|---|
| 10 questions | 15.5% | 27.5% |
| 20 questions | 24.1% | 35.0% |
| 40 questions | 32.5% | 39.8% |

So **"any net loss fails" is right for deterministic pipelines and wrong for noisy ones.** For a sampling pipeline, raise `critical_min_net_drop` or run each question several times.

## Judge calibration (`python -m raggate judge`)

If an LLM judge grades faithfulness, calibrate it first. `calibrate` compares a judge with labelled answers using **Cohen's kappa** with a bootstrap interval (the **whole interval** must clear 0.7, not the point estimate), reports unreadable replies as "unjudged" (more than 10% stops calibration, exit 2) and breaks mistakes down **by kind of answer**, which says what the judge is blind to. 222 labelled answers built by known operations (correct, paraphrase, wrong number, right value for the wrong component, abstain on answerable, partial multi-fact, abstain/answer on unanswerable).

| Judge | Accuracy | Kappa [95% interval] | Trustworthy | Wrong on |
|---|---|---|---|---|
| rule (checks facts and component) | 1.00 | 1.00 [1.00, 1.00] | yes | nothing (see caveat) |
| rule + 15% flipped verdicts | 0.86 | 0.71 [0.62, 0.81] | no | everything, at random |
| lenient (never checks the component or refusals) | 0.79 | 0.59 [0.50, 0.69] | no | all 33 wrong-component, all 14 answered-unanswerable |
| always-pass | 0.41 | 0.00 | no | every bad answer |

The lenient judge is the point: **79% accuracy sounds fine and it is blind to exactly the two failures that matter.** Caveat: the rule judge scores 1.00 partly because it uses the same logic the labels were built from, so that row proves the code path, not that a real judge would do well. `TransportJudge` takes any `prompt -> text` function, requires exactly one `VERDICT: PASS|FAIL` line, treats conflicting or missing verdicts as `JudgeError` (unjudged, never a guess), and puts the answer in delimiters. **It has never been run against a live model.**

## Golden-set hygiene (`python -m raggate hygiene`)

A gate is only as honest as its set. Errors make the gate refuse to judge:

| Severity | Code | Catches |
|---|---|---|
| error | `duplicate_qid`, `unknown_split`, `empty_split`, `missing_entity` | structural problems that break pairing or answer checking |
| error | `gold_passage_missing`, `fact_not_in_passage` | gold evidence that cannot exist (token-boundary match: "5 MB" is not found in "15 MB") |
| error | `answer_in_question` | the question already contains the answer |
| error | `duplicate_question`, `leak_question`, `leak_gold_fact` | same question twice, or the same question or the same gold fact in both dev and held-out |
| warn | `near_duplicate_question`, `shared_gold_fact`, `small_slice` | not blocking, but dilutes the set or leaves a slice untestable |

The generated set splits **by component**, so no component's facts appear in both dev and held-out. Hygiene for unanswerable questions is the weak spot: nothing checks that the corpus really lacks the answer (here that holds by construction).

## Run it

```
python -m raggate demo                                  # everything above, ~5 s
python -m raggate run baseline --out base.json          # run a pipeline on the held-out split
python -m raggate run regressed_generator --out cand.json
python -m raggate gate base.json cand.json              # exit 0 pass / 1 regression / 2 could not judge
python -m raggate gate base.json cand.json --json       # machine-readable report
python -m raggate hygiene                               # golden-set checks
python -m raggate judge                                 # judge calibration
python -m raggate power --n 174 --churn 0.04            # what drop can the gate see (about 1 minute)
python -m raggate tune                                  # dev-only sweep of the abstain threshold
python -m raggate export --out golden.json              # write the golden set; pass back with --golden
python -m pytest
```

Pipelines: `baseline`, `regressed_retriever`, `regressed_generator`, `tight_context`, `noisy-s<seed>`. Wiring a real system means producing the same run JSON (`qid`, ranked `retrieved` ids, the `context` as placed in the prompt, `answer`, `abstained`); scoring and gating are separate from running.

In CI: `python -m raggate gate base.json cand.json` fails the job on exit 1, and a job that "could not judge" (exit 2) fails too, so a broken harness cannot pass silently.

## Known limits

- **Toy pipelines.** Lexical and extractive; nothing is learned. The harness is the subject, not the pipelines. `regressed_retriever` is an ingestion bug rather than a ranking-quality loss because lexical retrieval over entity names was too easy to degrade any other way.
- **Retrieval is nearly perfect at baseline** (recall 100%, MRR 97.6%), so retrieval metrics have little headroom to move except under a bug. Real retrievers are harder.
- **Baseline passes 0/17 paraphrased questions**, so that slice cannot show a regression. See above.
- **Pass/fail scoring is lexical** (token containment, same-sentence entity check). It cannot judge a correct paraphrased answer from a real model; that is what the judge is for, and the judge is untested live.
- **Synthetic labels and corpus**: templated text, one fact per sentence, made-up values. Real documents are messier.
- **Power numbers assume** independent per-question flips and a fixed churn; real flips correlate by topic.
- **Single seed, single golden set** (seed 7). Held-out is 174 questions; slices are 17 to 77 questions, which is why only large slice drops are detectable.
- Attribution blames the earliest stage, so when retrieval and generation both fail a question, retrieval gets the count.

## Layout

```
raggate/
  corpus.py       seeded synthetic docs + golden set (traps: shared names, stale notes, addenda, paraphrases)
  retrieval.py    BM25
  pipelines.py    the five toy pipelines, run files (JSON)
  metrics.py      recall/MRR/nDCG, fact coverage, faithfulness, pass/fail
  attribution.py  which stage lost each failed question
  stats.py        paired bootstrap, Holm, Cohen's kappa
  hygiene.py      golden-set checks
  gate.py         the verdict
  judge.py        judges, labelled set, calibration, LLM adapter (untested live)
  power.py        what the gate can detect
  report.py cli.py
tests/            128 tests
```

MIT licensed.
