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

## Result

Scored once, 2026-08-16, against a blinded response from a fresh Claude vision
conversation (`anthropic / claude-opus-5`) that received only the contact sheet
and `prompt.txt`.

| arm | correct | wrong | unanswered | coverage | exact accuracy | false-confident rate |
|---|---:|---:|---:|---:|---:|---:|
| object/label map | 0 | 5 | 1 | 0.833 | 0.000 | 1.000 |
| current typed graph | 0 | 5 | 1 | 0.833 | 0.000 | 1.000 |
| direct visual VLM | 4 | 2 | 0 | 1.000 | 0.667 | 0.333 |
| evidence-aware hybrid | 4 | 2 | 0 | 1.000 | 0.667 | 0.333 |

Per question:

| question | human key | object map | typed graph | direct VLM | hybrid |
|---|---|---|---|---|---|
| `q1_rug_cardinality` | 1 | 0 ✗ | 0 ✗ | **1 ✓** | **1 ✓** |
| `q2_trash_can_cardinality` | 1 | 2 ✗ | 2 ✗ | 2 ✗ | 2 ✗ |
| `q3_counter_cardinality` | 1 | 2 ✗ | 2 ✗ | 2 ✗ | 2 ✗ |
| `q4_sofa_present` | true | false ✗ | false ✗ | **true ✓** | **true ✓** |
| `q5_cushion_cardinality` | 2 | 3 ✗ | 3 ✗ | **2 ✓** | **2 ✓** |
| `q6_cushion_on_sofa` | `cushion ON_ENTITY_SURFACE sofa` | unanswered | unanswered | **sofa ✓** | **sofa ✓** |

Four findings, ordered by how much they should change what gets built next.

**1. The graph contributed no uniquely correct answer.** Its 151 edges are all
`NEAR`; the only relational key item asks for `ON_ENTITY_SURFACE`, so both
structured arms left it unanswered. The typed-graph arm is identical to the
object-map arm on every question. This is not evidence that graphs are useless
— it says the delivered graph does not contain the relation the available human
question needs.

**2. Direct RGB answered the one relation question the graph could not.** The
support item (`q6`) is the graph's home turf, and RGB won it outright while
citing two views. On this key, structure was not merely unhelpful; it was
beaten on the question type it exists to serve.

**3. The hybrid is exactly the direct arm.** The router never reached the
graph, and the two-view abstention gate never fired — every visual answer cited
at least two valid frames. Gain over the best single arm is 0.0000 and
wrong-answer reduction is 0.0000. The gate cost nothing, and on this key it
bought nothing, because the visual arm never produced a thinly-evidenced answer
for it to catch.

**4. Two failures are common to all four arms.** Every arm overcounted trash
cans (2 vs 1) and kitchen counters (2 vs 1). No representation on offer repairs
these: they survive both the object map and independent visual reasoning. That
makes instance duplication the binding stage, not representation choice.

### Decision

`proceed = false`. Both clauses of the predeclared rule were missed at exactly
zero. This is the decision table's *"direct RGB wins and hybrid is equal"* row:
**use direct visual QA for the immediate demo, and make relation evidence the
next engineering task — not a bigger router.**

One scene, six questions, one blinded response. This is a screening result. It
does not generalize, and it is not a claim about embodied planning or room
understanding.

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

Both held. The measured gain was exactly `0.0000`, and the safety clause also
came in at `0.0000` because the abstention gate never fired. Recording this
prediction before the score is what makes the `proceed = false` above a test
outcome rather than a story fitted to the numbers afterwards.

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

## How the blinded visual run was obtained

The repository has no configured vision-model runtime or credentials, and the
direct arm must not be filled by a model that has seen the answer key in its
conversation context. The repo-aware session had already read the key path and
the partial report, so it was disqualified from answering.

The owner therefore ran it by hand on 2026-08-16: a fresh Claude vision
conversation received the contact sheet and `prompt.txt` and nothing else — no
repository context, no human key, no statement of the hypothesis. Its JSON was
returned unedited and committed verbatim as `direct_claude.json`. It was not
repaired after the key was visible.

The blinding is procedural, not enforced by the repository. It rests on the
owner's isolation of that conversation, which is why the provenance is written
down here rather than assumed.

The inputs are under the gitignored run directory:

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
