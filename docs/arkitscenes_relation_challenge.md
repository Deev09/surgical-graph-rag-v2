# ARKitScenes NEAR relation challenge — protocol

Status: **scaffolding built, nothing scored.** The question manifest is
`DRAFT_AWAITING_OWNER`. The four-layer scorer passes synthetic tests and
refuses to run against a draft key.

## The question

> Does persistent 3D structure uniquely answer spatial questions that bounded
> multi-view visual reasoning misses?

The previous six-question kill test could not answer this. Its only relational
item required `ON_ENTITY_SURFACE`, which the delivered graph does not emit, so
it compared direct RGB against an **absent capability** rather than against
graph reasoning. See `docs/arkitscenes_representation_kill_test.md`.

## Why NEAR

`NEAR` is the one relation the delivered representation actually contains: 151
edges in `41069025`, 100 in `41069042`. Testing it is the difference between
measuring a representation and measuring a hole.

What the delivered graph can and cannot express, measured from the artifacts:

| form | expressible? | how |
|---|---|---|
| binary "is A near B?" | yes | edge present ⇔ AABB surface gap < 1.0 m |
| "what is near X?" over a closed roster | yes | every edge incident to X |
| comparative "is A closer to B or C?" | partly | compare two stored `distance_m` |
| cross-view proximity | yes | no per-view membership exists at all |
| support / on-surface | **no** | no such edge type is emitted |
| direction — "left of", "across from" | **no** | `canonical_forward`/`canonical_right` are `null` |
| containment | **no** | no containment relation exists |
| metric magnitude — "how far apart" | **no** | absent edges store no distance |

Two mechanical facts matter and were verified rather than assumed:

**Edge absence is a measurement, not missing data.** `extract_sparse_v2` is an
unguarded double loop over all C(n,2) pairs with one scalar test — no k-NN cap,
no top-k, no beam. The arithmetic closes exactly: C(35,2)=595 = 151 edges + 444
recorded rejections; C(23,2)=253 = 100 + 153. A missing edge therefore means
"measured, and at least 1.0 m apart".

**Comparatives survive a missing edge.** The test is strict less-than, so a
stored edge has d < 1.0 and a pruned pair has d ≥ 1.0; when exactly one of the
two pairs is stored, the ordering follows from the threshold alone. This is not
a knife-edge: the largest stored distance is 0.9919 / 0.9972 and the smallest
pruned distance is 1.0137 / 1.0049. Only when *neither* pair is stored must the
graph abstain.

## Four layers

Reported separately, always. A ceiling number is never deployable performance.

| layer | identity from | relations from | deployable |
|---|---|---|---|
| `geometry_relation_ceiling` | human-verified UID mappings | delivered geometry, same `aabb_to_aabb_surface` function the extractor uses | **no** |
| `delivered_graph` | learned `display_label` | delivered NEAR edges | yes |
| `blinded_rgb_vlm` | the images | the images | yes |
| `evidence_aware_hybrid` | routed | routed | yes |

Only the identity source changes between layers 1 and 2. No new metric is
introduced anywhere; the ceiling calls the extractor's own distance function
and reads each scene's own recorded threshold.

### Predeclared attribution

| observation | reading |
|---|---|
| ceiling correct, delivered wrong | naming, instance delivery or relation extraction binds |
| ceiling correct, delivered unanswered | the fact is expressible but the delivered system cannot address it |
| ceiling wrong | the current geometry/representation cannot answer the relation |
| ceiling unanswerable (no resolvable UID) | **instance delivery** binds — distinct from a geometry failure |
| graph uniquely correct where RGB misses | first real evidence for a hybrid |
| RGB matches or beats graph everywhere | do not expand the graph architecture |

`answer_*` functions never see an expected value; `grade()` is the only function
that reads the key, and a test AST-checks that the deployable layers cannot bind
a key-shaped name.

## The question set

12 questions, 7 in `41069025` and 5 in `41069042`. Manifest:
`eval/questions/arkitscenes_relation_challenge_v1.json`.

| form | n | threshold-dependent? |
|---|---:|---|
| comparative_near | 5 | no — ordinal only |
| binary_near | 5 | yes |
| near_set | 2 | yes |

Six are flagged `cross_view`.

**Selection provenance.** Anchors come from direct inspection of the raw RGB
plus the owner's `human_confirmed_scene_facts`. No graph edge, edge distance,
learned label or `obj_N` was consulted while choosing them. Choosing pairs by
what the graph already links would build a key that can only confirm the graph
— the same trap the support review sheet was built to avoid.

### The NEAR convention, and its declared confound

> Two objects are NEAR when the nearest points of their surfaces are within
> about one metre — close enough to reach by leaning or taking a single step.

The delivered extractor's threshold is **also 1.0 m**, recorded in each slice
manifest as a "provisional Replica-era constant; not a transfer claim". One
metre was chosen because it is the natural human reading of "near" at room
scale, but the coincidence is real and is declared here: **a binary item the
delivered system gets right cannot distinguish "the representation is correct"
from "the threshold happens to match the convention".** That is precisely why
the five comparative items, which use no threshold, carry the discriminative
weight of this set.

Items whose true gap lands within 0.1 m of one metre are threshold-determined
rather than geometry-determined. Which items those are will be reported only
*after* the key is returned; checking first would let edge distances influence
question selection.

## Evidence-sufficiency subtest (secondary)

The owner records, per question and independently of the answer, whether the
evidence is visible in **0 / 1 / 2+** views. Whatever lands in 0 or 1 becomes a
natural thin-evidence slice. No one-view case is manufactured.

The primary experiment asks whether 3D structure adds value. This one asks
whether the two-view gate reduces confident errors enough to justify its
coverage cost — and the report prints both sides: wrong answers suppressed
*and* correct answers suppressed. The previous run could not test the gate at
all, because every visual answer happened to cite two views.

## Continuation bar

**At least two delivered-graph-unique correct answers, preferably one per
scene**, before proposing a larger hybrid. A ceiling win is not a graph win.
This is a screening gate on two scenes, not a publication claim.

## Pre-registered: the delivered arm will abstain on every question

Measured on 2026-08-17, **after** the questions were fixed and **before** any
key exists, so it could not influence question selection.

The delivered label stage resolves exactly **one** anchor to exactly one
instance across both scenes — `long narrow desk` → `obj_14`. Every other anchor
is ambiguous (several instances claim the label) or absent (no instance claims
it at all).

| scene | resolves to one | ambiguous | absent |
|---|---|---|---|
| 41069025 | long narrow desk | kitchen counter, both waste containers, both tables | sofa, striped rug, pedestal fan, convector heater |
| 41069042 | — | bed, cream curtain, chest of drawers | radiator, window, white desk, framed picture |

The admitted vocabulary is the cause: `41069025` admits 15 distinct labels and
`41069042` admits 11, and most anchors a human would ask about are simply not
among them. `sofa` is absent from the admitted labels entirely — the same
failure that made `q4_sofa_present` wrong in the previous run.

**Consequence, stated before the review rather than after it:** the
`delivered_graph` layer abstains on 12/12. Delivered-graph-unique wins are
therefore **zero by construction**, and the continuation bar is unreachable on
this set — structurally the same way the previous kill test's accuracy clause
was unreachable. A zero here is a fact about instance naming, and must not be
read as evidence about graphs.

What this set can still establish: a decisive attribution (if the ceiling
answers what the delivered graph cannot, naming is binding — measured, not
argued), whether the representation could express these relations at all given
correct identity, and whether the sufficiency gate pays for its coverage cost.

What it cannot establish: whether persistent 3D structure uniquely answers what
RGB misses *in a deployable path*. That needs a delivered arm that can answer.

The owner's options are recorded in `owner_decision_required` in the manifest.
The recommendation is to proceed for the attribution while refusing to read the
zero as a graph result — while noting that the honest reading of both
experiments so far points at perception and vocabulary coverage, not
representation, as the binding stage.

## Two things the owner must adjudicate

**1. The 41069042 sealed-review facts do not match 41069042.** The review of
2026-08-09 records "a couch/sofa facing the kitchen area" and "the supplied
views show the kitchen from the couch perspective". Direct inspection of the
raw frames found no couch, no sofa, no kitchen and no counter anywhere in that
capture — it is a compact hotel bedroom (bed, nightstand, wall lamp, window
with blind, cream curtain, radiator, desk, wall TV, chest of drawers). Those
two facts *are* an accurate description of `41069025`. The ARKit annotation for
`41069042` independently agrees with the bedroom reading: its boxes contain
bed, tables, cabinets and tv_monitor. The likely explanation is that the
2026-08-09 screenshots were filed under the wrong scene id. The `41069042`
questions here are authored from that scene's own imagery.

**2. `41069025` contains two physically separate waste containers.** A tall
stainless bin with a dark lid at the kitchen/entrance boundary (frame
`41069025_750.511`) and a small grey wastepaper basket under the long desk at
the opposite end (`41069025_655.567`, `41069025_658.316`). They never appear in
one frame.

This does **not** revise the frozen v3 key or its preserved 4/6 result, and no
such revision was made. It does bear on how that result is read: on
`q2_trash_can_cardinality` the key says 1 and all four arms answered 2,
including a blinded model with no access to the object map. It also gives the
counting-convention problem a concrete resolution to make — "trash can" needs
to state whether it means the kitchen bin only or any waste container — before
the convention is inherited by this or any later key. Both containers are named
distinctly in this question set for exactly that reason, and one question uses
the pair as the strongest cross-view item in either scene.

## Artifact integrity

Read-only audit, both scenes. Geometry partitions verified with the existing
`geometry_signature()` from the kill test, not a reimplementation.

| | 41069025 | 41069042 |
|---|---|---|
| entities (rgb_tight) | 35 | 23 |
| NEAR edges | 151 | 100 |
| other edge types | none | none |
| structural surfaces | 0 | 0 |
| geometry signature | `f717194a…9bfd` | `8c6d998d…99fe` |
| graph ↔ entity partition | exact match | exact match |
| raw RGB frames | 10090 | 5882 |
| stored distance range (m) | 0.0 – 0.9919 | 0.0 – 0.9972 |
| NEAR threshold | 1.0 m, `aabb_surface` | 1.0 m, `aabb_surface` |

`rgb_tight` entity manifest `a61f83f3…64e7` / `3d24f910…6153`; sealed graph
manifest `adb53b80…8865` / `b3a03d71…2c96`.

**Declared confounder — two label arms exist.** The graph is bound to the
*sealed* entity manifest, whose `display_label` values differ from the
`rgb_tight` manifest on 26/35 entities in `41069025` and on **23/23** in
`41069042` (admission 27 vs 17, and 21 vs 5). Geometry is byte-identical across
both; only labels diverge. The `delivered_graph` layer reads `rgb_tight`
labels, matching the previous kill test so the two experiments stay comparable,
and reuses the sealed graph's edges as geometry-only. The choice is recorded
here because it is a real fork, not a detail.

## Commands

```bash
.venv/bin/python3 tools/arkitscenes_relation_challenge_review.py sheet
```

```bash
.venv/bin/python3 tools/arkitscenes_relation_challenge_review.py packets
```

```bash
.venv/bin/python3 tools/arkitscenes_relation_challenge_review.py validate --returned RETURNED.json
```

The scorer's input map is written by the scorer, not the review kit — the
review kit never names or opens a graph artifact, and a test enforces that:

```bash
.venv/bin/python3 tools/arkitscenes_relation_challenge_score.py --questions eval/questions/arkitscenes_relation_challenge_v1.json --out runs/arkit_relation_challenge/report.json --emit-scene-inputs
```

Scoring is deliberately a separate step and refuses to run until both the
question manifest and the key say `OWNER_CONFIRMED` and the key pins the
manifest hash:

```bash
.venv/bin/python3 tools/arkitscenes_relation_challenge_score.py --questions eval/questions/arkitscenes_relation_challenge_v1.json --key eval/human_feedback/arkitscenes_relation_challenge_key_v1.json --scene-inputs runs/arkit_relation_challenge/scene_inputs.json --out runs/arkit_relation_challenge/report.json
```

## Not started, deliberately

Real scene scoring, key finalization, relation threshold tuning, graph
extraction changes, perception or label changes, new Mask3D/SAM runs, the
`47331972` download, ConceptGraphs, another dataset, router redesign, paper
claims.
