# Margin-derived answer confidence — NEGATIVE RESULT

Status: **refuted with controls.** The plumbing ships default-off
(`emit_margins=False`, `answer_tau=0.0`) and is not on the active path.

`runs/` is gitignored, so this file is the durable record. Reproduce with
`python3 tools/rules_selective_eval.py`; outputs land under
`runs/rules_selective/`.

## The hypothesis

Every relation extractor computes a continuous quantity and thresholds it to a
boolean. Keeping that margin should give each edge a real confidence, which
should propagate to an answer-level confidence, which should support a
threshold `tau` that trades coverage for precision — i.e. the system should be
able to say "I don't know" on exactly the questions it gets wrong.

Predicted claim: *"we answer N% at high precision and correctly refuse the
rest, and a VLM cannot tell you which of its answers to trust."*

## Verdict

The claim is **not supported on this question set**, and the reason is
structural rather than a matter of tuning. The headline numbers look like the
prediction; the controls show they are not evidence for it.

## What the numbers look like before controls

| | AURC | E-AURC | tie spread | spread/E-AURC | distinct |
|---|---|---|---|---|---|
| baseline (frozen, all-tied) | 0.740539 | 0.617521 | 0.617521 | **1.00** | 2 |
| min (candidate) | 0.2291 | 0.1061 | 0.0688 | 0.65 | 24 |
| mean | 0.2237 | 0.1007 | 0.0688 | 0.68 | 25 |
| product | 0.2285 | 0.1055 | 0.0688 | 0.65 | 25 |

`coverage@risk<=0.05` goes 0.0 -> 0.4821; `coverage@risk<=0.10` goes 0.0 ->
0.5357. The tau sweep reaches 48% coverage at 3.7% selective risk — i.e. 96%
precision — and the deployed gate provably reproduces the curve at every tau.

Taken alone this reads as a clean success. It is not.

## Why it is not a success

`eval/selective.py` defaults to a **pessimistic** tie policy, so the all-tied
baseline is charged worst-case and *any* tie-breaking score beats it — including
a random one. The evaluator therefore computes seeded random-permutation
controls over the identical items (4000 trials):

| control | AURC | p05 | p95 |
|---|---|---|---|
| random | 0.4044 +/- 0.0612 | 0.3051 | 0.5058 |
| **random_within_outcome** | **0.2136 +/- 0.0323** | **0.1686** | **0.2735** |
| random_within_relation | 0.2657 +/- 0.0299 | 0.2186 | 0.3157 |
| oracle (= AURC*) | 0.1230 | | |
| **candidate (min)** | **0.2291** | inside the within-outcome band | |

`random_within_outcome` knows only whether the executor said empty / bindings /
abstain, and is otherwise noise. The margin score lands **inside its p05-p95
band**, and is marginally worse than its mean. All of the apparent gain is the
outcome-type ordering; the margin content adds nothing measurable.

65% of the remaining E-AURC is still tie-break — 27 of 56 items tie at
confidence 1.0, one of them wrong. The score is not inert like the baseline,
but it is largely inert.

**Plumbing is sound.** Re-running with `--no-margins` (frozen
`emit_margins=False`) reproduces the cached baseline bit-exactly:
`0.7405393328208909` / `0.61752056388782` / `0.61752056388782`. The delta comes
from margins being on, not from anything the reasoner change did.

## Three structural reasons it fails

### 1. The dominant error is not what abstention catches

| outcome x correctness | n |
|---|---|
| true_empty / empty | 27 |
| **miss / bindings** | **14** |
| true_answer / bindings | 4 |
| miss / empty | 4 |
| miss / abstain | 4 |
| false_answer / bindings | 3 |

The dominant failure is a *bindings* answer that omitted required UIDs — 14 of
25 errors. Not a false answer (3), and not an empty-vs-miss confusion (4). An
abstention threshold is the wrong instrument for an incomplete answer.

### 2. The near-miss band does not exist where these questions live

> **FOLLOW-UP: this loophole has since been closed, and the verdict did not
> change.** `graph.relations.base.sample_rejections` now selects the retained
> sample by margin instead of iteration order when `emit_margins=True`. The
> near-misses genuinely enter the sample -- measured per family on the
> Phase-8 scenes, retained maximum margin moves:
>
> | family | n | iteration-order max | margin-aware max |
> |---|---|---|---|
> | on_surface | 322 | 0.0000 | **0.4679** |
> | on_entity_surface | 326 | 0.0003 | **0.4150** |
> | near_surface | 268 | 0.0492 | **0.4795** |
> | attached_to | 365 | 0.1567 | **0.4726** |
> | on_entity_surface | 870 | 0.0029 | 0.0563 |
> | directional | 1959 | median 0.0971 | median **0.4807** |
>
> The `0.1567` and `0.0029` ceilings quoted below were exactly the artifacts
> of iteration-order truncation. With them removed, the candidate AURC still
> sits inside the `random_within_outcome` band, tie spread is still ~66% of
> E-AURC, and the bindings AUROC moves only 0.118 -> 0.1618 -- still far below
> chance. **The failure is not a sampling artifact.** The paragraphs below
> record the original measurement that motivated the fix.


`graph/relations/**` caps `rejection_samples` at 64 per extractor **in
iteration order, not by margin** — 1,472 retained of 10,934 actual (13.5%).
Consequences:

- 24 of the 31 empties had **zero** scoped rejections: the query anchored on an
  entity class absent from the scene, so no candidate ever existed. All 24 are
  correct.
- The 7 empties that did have candidates saw max margins of
  `[0.0, 0.0, 0.0, 0.0, 0.0001, 0.0003, 0.1567]`. The ON_ENTITY_SURFACE
  family's entire 64-sample tops out at 0.0029.

The real separator among empties is **anchor resolution** — a structural fact
about the query, not a calibration signal.

### 3. `min` is a cardinality detector, and cardinality runs backwards here

Correct bindings cite 1, 30, 37, 40 edges; wrong ones mostly cite 1-3. On the
bindings subset:

| signal | AUROC |
|---|---|
| min | **0.118** (below chance) |
| mean | **0.279** (below chance) |
| n_cited_edges | **0.794** |

`min`'s bindings AURC (0.9532) is *outside p95* of random (0.8105 +/- 0.0786):
actively worse than no information. The best predictor in the system is raw
citation count, which nobody designed as a confidence signal.

## Relation-type confound check

None of the 56 questions is directional, so the saturated-directional confound
cannot fire. Relation is in fact anti-aligned with the score:

| relation | n | accuracy | median confidence |
|---|---|---|---|
| ATTACHED_TO | 4 | 0.000 | 0.847 |
| CONTACTS_SURFACE | 4 | 0.000 | 0.745 |
| SUPPORTS_FLOOR | 4 | 0.000 | 0.582 |
| NEAR_SURFACE | 4 | 0.750 | 0.548 |
| ON_ENTITY_SURFACE | 40 | 0.700 | 1.000 |

`NEAR_SURFACE` is the most accurate family and receives the *lowest*
confidences. Holding relation constant (ON_ENTITY_SURFACE, n=40): AURC 0.1586,
E-AURC 0.1046, **spread/E-AURC = 0.92** — 92% tie artifact, and inside both
random bands.

So the curve is not relation type in disguise. It is **outcome type** in
disguise, which is the same class of problem.

## Tau sweep (end-to-end through the Router)

| tau | gated | coverage | selective risk | gate == curve |
|---|---|---|---|---|
| 0.000 | 0 | 0.9286 | 0.4038 | True |
| 0.550 | 6 | 0.8214 | 0.3913 | True |
| 0.700 | 11 | 0.7321 | 0.3415 | True |
| 0.900 | 18 | 0.6071 | 0.2059 | True |
| 0.999 | 22 | 0.5357 | 0.1000 | True |
| 1.000 | 25 | 0.4821 | 0.0370 | True |

The last column is a real check: the shippable gate realises the operating
points `eval/selective.py` claims, at every tau.

## What survives

A smaller, measured claim: **the router's five-way outcome taxonomy
(bindings / empty / unknown / abstain / parser_failure) is itself a usable
abstention signal. Per-edge geometric margins are not.**

## What would be worth testing next

The two signals that do separate here are (i) whether the query's anchor
resolved at all, and (ii) cited-edge count. Neither is a margin. Testing (i)
properly requires lifting the 64-rejection sampling cap in
`graph/relations/**`, which is currently truncated by iteration order rather
than by margin — that cap should be made margin-aware before any further
calibration work is attempted on this set.

Both observations rest on 56 questions across four Replica scenes and should
not be built on before a second dataset exists.
