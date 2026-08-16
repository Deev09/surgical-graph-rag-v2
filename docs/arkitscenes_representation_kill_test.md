# ARKitScenes representation kill test (screening, partial)

## Question

On the same human-reviewed room questions, does a typed spatial graph add
answer value beyond an object/label map, and does a small evidence-aware router
beat either structured representation or direct visual QA alone?

This is deliberately a small kill test, not a new architecture. It changes no
perception stage, graph definition, threshold, human answer, or frozen result.

## Common question scope

Scene: `arkitscenes_41069025`.

Six human-keyed, room-observable questions are included: five inventory items
and one furniture-support item. Two key rows are excluded from the common
comparison:

- delivered-instance identity, because raw RGB has no access to `obj_N`;
- reported-counter extent, because raw RGB cannot observe a system's reported
  region.

Including either would favor the structured arms by definition rather than
measure room understanding.

## Inputs and integrity

- Human key SHA-256: `5f007f0d3cdb3f4d5fcf8a64b4f5620566ff53aa9b5e8df11916bd36e2adcac0`
- Answer-free direct-VLM packet SHA-256: `3d6edb30e12b67e0bc7fc68ceeec2731c07ce0424bae6d87baebd42d5cf7acd7`
- Common-question SHA-256: `cb2bdc43fc616d19ca1c3287ecf96d602667ab17b810551bfe6226e87a257f25`
- Entity/graph geometry signature: `f717194a42a7384aa7f2f9617a7c10e0708303b117a9f03f127328a191ff9bfd`

The graph was originally serialized beside an earlier label arm. The scorer
therefore does not trust its entity bundle name: it hashes every UID, AABB,
OBB, centroid, geometry handle, frame and representation identifier, and
requires an exact match to the current RGB-labelled entity partition before
reusing the geometry-only edges. The check passed.

The visual packet selects 18 frames in equal capture-time bins and chooses the
highest-information RGB image within each bin. This answer-free rule replaced
pose-only farthest-point selection after the latter visibly wasted several of
12 slots on blank walls. The final contact sheet includes the kitchen,
counters, trash can, sofa, cushions, striped rug, table, TV and curtains.

## Partial result

| arm | correct | wrong | unanswered | coverage | exact accuracy |
|---|---:|---:|---:|---:|---:|
| object/label map | 0 | 5 | 1 | 0.833 | 0.000 |
| current typed graph | 0 | 5 | 1 | 0.833 | 0.000 |
| direct visual VLM | pending | pending | pending | pending | pending |
| evidence-aware hybrid | pending | pending | pending | pending | pending |

The current graph adds no answer value on this scope. Its 151 edges are all
`NEAR`; the only relational key item asks for `ON_ENTITY_SURFACE`, so both
structured arms leave it unanswered. This is not evidence that graphs are
useless. It says the current graph does not yet contain the relation needed by
the available human question.

## What this run can and cannot establish

Determined before the blinded arm was scored, from the delivered artifacts
alone. It does not depend on the visual result.

The delivered graph contains 151 edges, all `NEAR`. The router sends a question
to the graph only when the question is `support_relation` **and** the graph
materializes an answer. Of the common six, only `q6_cushion_on_sofa` is
graph-eligible, and the graph answers it `null`. So all six hybrid rows fall
back to the visual arm plus the two-view abstention gate.

Two consequences follow:

1. The hybrid can only convert a visual answer into an abstention; it can never
   add a correct answer the visual arm did not already have. Therefore
   `hybrid_correct <= direct_correct`, so
   `gain = hybrid_accuracy - best_single_accuracy <= 0` and the accuracy clause
   (`gain >= 0.10`) is **unreachable in this configuration**, whatever the
   blinded model answers.
2. Only the safety clause can fire. A `proceed` here would therefore mean
   *evidence-sufficiency abstention reduced confident errors*, and would **not**
   mean the graph helped.

**This run cannot establish that explicit 3D structure improves spatial QA.**
The graph is not being tested against a question it can answer: its only
relational key item requires `ON_ENTITY_SURFACE`, which it does not emit. That
is a measured missing capability, not evidence against graphs.

Owner reading of the possible outcomes, recorded 2026-08-16:

| observed | reading |
|---|---|
| direct VLM wins, hybrid equal | use direct visual QA for the immediate demo |
| hybrid makes fewer mistakes by abstaining | evidence sufficiency is valuable; the graph still has not helped |
| any outcome | insufficient to claim graph benefit from this key |

### Named next task

A small, independently human-keyed **relation challenge** over relations the
graph actually contains, starting with `NEAR` — e.g. which objects are near the
TV, table, sofa or counter, asked across views. Compare direct RGB, object map
and graph on it. That is the first test that can actually answer the main
question: does persistent 3D structure uniquely answer spatial questions that
bounded multi-view visual reasoning misses?

The key must come from human inspection, not from the graph being scored. Do
not expand the architecture until that relation-heavy test shows at least one
reproducible graph advantage.

## Pending blinded visual run

The repository has no configured vision-model runtime or credentials. The
direct and hybrid rows are intentionally left pending rather than filled by a
model that has seen the answer key in its conversation context.

The generated handoff is under the gitignored run directory:

- `runs/arkit_representation_kill_test/41069025_common6/contact_sheet.jpg`
- `runs/arkit_representation_kill_test/41069025_common6/prompt.txt`
- `runs/arkit_representation_kill_test/41069025_common6/packet.json`

The direct response must pin the packet hash, use structured answer types, cite
valid frame IDs, and answer or explicitly return `unknown` for every question.
The hybrid only accepts a visual answer when it cites at least two supplied
views. Support routes to the graph only when the graph materializes an answer;
otherwise it falls back to visible evidence.

## Result viewer

`tools/arkitscenes_representation_kill_test_viewer.py` renders one offline HTML
page from a committed report:

```bash
.venv/bin/python3 tools/arkitscenes_representation_kill_test_viewer.py \
  --report runs/arkit_representation_kill_test/41069025_common6/report.json \
  --packet runs/arkit_representation_kill_test/41069025_common6/packet.json \
  --contact-sheet runs/arkit_representation_kill_test/41069025_common6/contact_sheet.jpg \
  --out runs/arkit_representation_kill_test/41069025_common6/result.html
```

The page contains no script and makes no external request; the contact sheet is
inlined as a data URI. Every tally, rate and decision value is printed verbatim
from `report.json`, so the page cannot disagree with the scorer. A partial
report renders the direct and hybrid columns as explicitly pending rather than
as scored zeros.

## Screening decision rule

Continue toward a larger hybrid only if it beats the best single arm by at
least 0.10 absolute exact accuracy, or reduces wrong answers by at least 30%
while retaining at least 60% coverage. Even a pass is only a reason to test
more questions and scenes; it is not a generalization claim.
