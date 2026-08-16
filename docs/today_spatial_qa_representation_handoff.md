# Today handoff: decide the spatial-QA representation

Status: **bounded one-day task**. Do not start a graph rewrite, new segmenter,
or new dataset run.

## The main research question

> **When does explicit, evidence-bearing 3D structure add measurable value
> over direct multi-view visual reasoning for spatial QA on real handheld room
> captures?**

This is the question behind the implementation. The goal is not to prove that
graphs are always better. The original product goal is a captured room that can
answer spatial questions faithfully, cite its evidence, preserve state across
views and say when it does not know. A graph, an object map and a VLM are
candidate representations for that goal, not goals by themselves.

## Hypothesis

1. **Visible semantics and inventory:** direct real-RGB reasoning should beat
   the current object map because the delivered labels are still unreliable.
2. **Persistent spatial facts:** a typed 3D graph should help when a question
   requires cross-view identity, metric geometry or a materialized relation
   such as support—not merely object appearance.
3. **Hybrid:** routing visible appearance questions to RGB and materialized
   relation questions to the graph, while abstaining when evidence is
   insufficient, should reduce confident errors without collapsing coverage.

The hypothesis is falsifiable. If the graph contributes no uniquely correct
answers and the router does not beat the best single arm, do not build a larger
hybrid. Improve the missing relation evidence first or use direct visual QA for
the demo.

## What is already implemented

Branch: `codex/hybrid-kill-test`

Implementation commit: `84cf630`

The existing tool compares the same common questions across:

1. RGB-labelled object map, without graph edges;
2. current typed graph;
3. blinded direct visual VLM;
4. a question-kind router with a two-view evidence requirement.

The six common questions are human-keyed and room-observable. Delivered UID
identity and reported-region extent are excluded because a visual model cannot
observe those system-internal values.

Current partial result on `41069025`:

| arm | correct | wrong | unanswered |
|---|---:|---:|---:|
| object map | 0 | 5 | 1 |
| typed graph | 0 | 5 | 1 |
| direct visual | pending | pending | pending |
| hybrid | pending | pending | pending |

The graph contains 151 `NEAR` edges and no `ON_ENTITY_SURFACE`, so it cannot
answer the one human support question. That is a measured missing capability,
not evidence against graphs in general.

## One-day scope

### Task 1 — blinded visual answer (15–30 minutes)

Use a **fresh Claude vision conversation with no repository history and no
human key**. Give it only:

- `runs/arkit_representation_kill_test/41069025_common6/contact_sheet.jpg`
- `runs/arkit_representation_kill_test/41069025_common6/prompt.txt`

Save its JSON verbatim as:

`runs/arkit_representation_kill_test/41069025_common6/direct_claude.json`

Do not repair an answer after seeing the key. A schema failure may be repaired
only by asking the same fresh context to reformat without changing answers.

### Task 2 — complete the four-arm score (30 minutes)

Run from the integration checkout after reviewing and fast-forwarding the
bounded branch:

```bash
.venv/bin/python3 tools/arkitscenes_representation_kill_test.py score \
  --key eval/human_feedback/arkitscenes_41069025_spatial_qa_key_v3_final.json \
  --entities runs/arkit_label_image_ab_41069025/rgb_tight/entities/manifest.json \
  --graph runs/arkit_vertical_slice/sealed_pair/41069025/graph/manifest.json \
  --packet runs/arkit_representation_kill_test/41069025_common6/packet.json \
  --direct-responses runs/arkit_representation_kill_test/41069025_common6/direct_claude.json \
  --out runs/arkit_representation_kill_test/41069025_common6/report.json
```

Record per-question answers, outcomes, cited frames, coverage, exact accuracy,
false-confident rate and the predeclared proceed/stop decision. Do not change
the questions, answers, router or gate after seeing the VLM result.

### Task 3 — make the decision inspectable (60–90 minutes)

Implement one static, self-contained HTML report generated from `report.json`.
It should show:

- the hypothesis and one-scene limitation;
- the 18-view contact sheet;
- each question with all available arm answers side by side;
- correct/wrong/unanswered state;
- visual frame citations and the graph relation type used;
- the final proceed/stop rule and its measured values.

It must calculate no new metric in JavaScript. Display only committed report
values. It must not claim generalization, embodied planning improvement or
human-level room understanding.

### Task 4 — choose, do not automatically expand (15 minutes)

Use this decision table:

| observation | decision |
|---|---|
| Hybrid gains at least 0.10 exact accuracy over the best single arm, or cuts wrong answers by at least 30% at coverage at least 60% | Build a second-scene, relation-heavy evaluation slice next; do not yet rewrite the system. |
| Direct RGB wins and hybrid is equal | Use direct visual QA for visible questions; the next engineering task is relation evidence, not a bigger router. |
| All arms fail the same inventory items | Perception/delivery remains binding; a graph cannot repair absent or wrong objects. |
| Graph uniquely answers a relation that RGB misses | This is the first concrete evidence supporting the hybrid representation; reproduce it on another reviewed scene. |
| Graph contributes no uniquely correct answer | Do not claim graph benefit from this key. Build an independent relation challenge set before more graph engineering. |

## Today’s definition of done

- A blinded, hash-pinned Claude response exists.
- All four arms are scored once under the same six questions.
- One self-contained result page makes the comparison inspectable.
- The result document states which stage is binding and names exactly one next
  engineering task.
- The branch is committed, tests run, and the working tree is clean.

This is enough for today. Do not download `47331972`, tune relation constants,
change the human key, run a new segmenter, or port ConceptGraphs.

## If the result says the current key is non-discriminative

The next task—not part of today's implementation—is an owner-reviewed relation
challenge slice. It should ask questions that can genuinely distinguish the
representations: support/contact, relative layout, global cross-view identity,
metric distance and partially occluded persistent state. It must be keyed from
human inspection, not derived from the same graph being scored. Only after that
slice exists is it meaningful to claim that graph structure improves spatial
QA or embodied planning.

## Paste into the repo-aware Claude task

```text
Canonical repo: ~/Desktop/surgical-graph-rag-v2
Integration branch: v2-calibration
Current integration HEAD before handoff: 813ded3
Candidate branch: codex/hybrid-kill-test
Your role: primary editor/integrator.

Main question: When does explicit, evidence-bearing 3D structure add measurable
value over direct multi-view visual reasoning for spatial QA on real handheld
room captures?

Read first: CLAUDE.md,
docs/arkitscenes_representation_kill_test.md, and
docs/today_spatial_qa_representation_handoff.md.

First inspect `git diff v2-calibration..codex/hybrid-kill-test`. If the bounded
scope is intact and both trees are clean, fast-forward the candidate branch
into v2-calibration. Then consume the blinded response I provide, run the
four-arm scorer exactly once, and implement only the static result viewer
described in the handoff. Update the results doc with measured values and the
predeclared decision. Commit and push a complete checkpoint.

Do not change perception, labels, graph extraction, human keys, router rules,
questions or gates. Do not start the relation challenge slice, ConceptGraphs,
another dataset, or another model experiment today.

Validation: run the new focused test, its two neighbouring scorer/crop tests,
`git diff --check`, and `tools/run_tests.py`. Report environmental/artifact
failures separately rather than calling them regressions.
```
