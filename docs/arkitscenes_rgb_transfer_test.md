# Direct multiview RGB — one untouched-scene transfer test

Status: **protocol frozen before the scene is downloaded or inspected.**
Nothing in this document was written with any knowledge of the test room.

## The question

> Does the fixed direct multiview RGB QA path transfer to one untouched real
> handheld room, with no prompt, view-selection or scoring changes?

This is a transfer test of an existing path. It is **not** a graph, grounding
or segmentation experiment, and none of those stages runs.

## The scene, and its predeclared fallback

Primary: **ARKitScenes `47331972`**. Not present locally; it must be
downloaded, and only the assets needed for RGB packet generation
(`lowres_wide/` frames, `lowres_wide.traj`, `lowres_wide_intrinsics/`).

Fallback, fixed here so it cannot become scene-shopping later: **if
`47331972` cannot be obtained**, use **`41069021`**, which is already present
under `~/Desktop/datasets/arkitscenes/Validation/` and has never been touched
by any experiment in this repo — no packet, no key, no entity manifest, no
graph, no mention in any report. The substitution, if it happens, is recorded
in the result with the reason. **No third option exists**: if neither scene is
usable the test does not run, and no scene is chosen after seeing data.

Either way the scene stays uninspected until this file is committed.

## Frame selection — reused verbatim, not re-tuned

`survey_frames` from `tools/arkitscenes_relation_challenge_review.py`: 18
equal temporal bins across the capture, and inside each bin the frame with the
highest answer-free information score (contrast + edge density, penalising
clipping). The score cannot know what any question is about. This is the same
rule the previous two packets used, unchanged.

## Question-generation procedure — fixed now, applied after the packet exists

The failure mode this guards against is authoring questions the model can
answer. The procedure removes the author's discretion at the point where that
bias would enter.

**Step 1 — anchor enumeration, three independent blind passes.** Three agents
each receive only the 18 selected frames and list human-nameable objects. They
do not see each other's lists, any question, any prior result, or anything
from this repo. An object becomes an **anchor** only if **at least two of the
three passes** list it. Single-pass sightings are discarded, including mine.

**Step 2 — anchor ordering, mechanical.** Anchors are ordered by the frame
index of their first appearance ascending, ties broken alphabetically. This
ordering, not judgement about answerability, determines which anchors are used.

**Step 3 — fixed template allocation.** Questions are allocated by template
and filled from the ordered anchor list:

| # | form | filled from |
|---|---|---|
| 3 | presence or cardinality | anchors ranked 1, 2, 3 |
| 3 | comparative distance | ordered triples over anchors ranked 1–6 |
| 2 | cross-view relation | the two lowest-combined-rank anchor pairs sharing **no** frame |

Eight questions. If fewer than two non-co-visible pairs exist naturally, the
shortfall is recorded and the set drops to 7 or 6 — no substitute pair is
invented, and the "if naturally present" condition is reported either way. If
fewer than six anchors survive step 1, the test does not run.

**Excluded by construction:** `obj_N` identities, annotation-derived wording,
graph-internal state, anything RGB cannot observe (direction/orientation,
support, containment, metric magnitude), and any counting question without a
stated convention.

## Conventions, stated before the scene is seen

**Counting.** An instance is a physically separate object of the named class.
Two objects of the same class in different parts of the room count as two even
if they never appear in one frame. This is stated because the previous key got
this wrong: `41069025` was keyed as having one trash can, and it has two
physically separate waste containers at opposite ends of the room, which all
four arms then "failed".

**NEAR.** Two objects are near when the nearest points of their surfaces are
within about one metre. Comparative questions are ordinal and use no threshold.

**Both conventions appear verbatim in the model prompt and on the owner sheet**,
so the two are answering the same question.

## Prompt and answer schema — composed from existing text, not rewritten

The prompt is assembled from the two prompts already in the repo, with no
rewording:

- presence/cardinality forms take the kill test's `[integer]` / `[boolean]`
  phrasing;
- comparative and binary-near forms take the relation challenge's phrasing,
  including the corrected `outcome` flag wording.

Schema `arkitscenes_rgb_transfer_responses_v1`: per item `id`, `outcome`
(`answer` | `unknown`), `answer`, `confidence` in [0,1], and
`evidence_frame_ids` drawn only from the 18 supplied ids. The response must pin
the packet hash.

## Human review workflow

One self-contained offline sheet, same generator family as the relation
challenge review kit. For each question the owner records: the answer;
ambiguous yes/no; evidence visibility as **0 / 1 / 2+ views**, judged
independently of the answer; and free-text notes. The owner also confirms or
corrects the counting convention for any class the questions count.

**No UID mapping panel** — there are no delivered entities for this scene and
none will be produced. Estimated owner time: **20–30 minutes** for 6–10
questions.

## Scoring

Exactly once, after the blinded response is hash-pinned.

- **exact accuracy** = correct / scored items, where scored excludes items the
  owner marked ambiguous.
- **answer coverage** = (correct + wrong) / scored items.
- Items the owner marked ambiguous are excluded from both, and reported.

## Predeclared gates — both required

| gate | threshold |
|---|---|
| exact accuracy | **≥ 0.60** |
| answer coverage | **≥ 0.80** |

Plus, as conditions on the run rather than numbers: no prompt, view-selection,
question, convention or scoring adjustment after the scene is inspected; and
**one run only**.

## Stopping rule

If either gate fails, or any condition above is violated: **the demo remains a
fixed evaluation replay and no transfer claim is made.** Do not retune and
re-run, do not try a third scene, and do not soften a gate.

## Interpretation limits

A pass supports exactly one sentence:

> Under a fixed procedure, direct multiview RGB transferred to one previously
> untouched ARKitScenes handheld room.

It does not support generalization, a claim about handheld capture in general,
a claim about room understanding, or any comparison to the graph path — which
is not run here and whose existing results stand unchanged.

One room, 6–10 questions, one blinded response, one human reviewer.

## Execution order

1. Commit this protocol. ← the scene is untouched at this point
2. Download only the RGB assets needed for packet generation.
3. Generate the deterministic answer-free packet and the review sheet.
4. **Stop for owner review.**
5. Fresh blinded vision context, packet and prompt only.
6. Hash-pin the response before opening the key.
7. Score once.
8. Generate the offline result page and update the narrative.

## Not run

Mask3D, SAM, grounding, graph extraction, the live API. No existing result,
key, packet, report or demo artifact is modified.
