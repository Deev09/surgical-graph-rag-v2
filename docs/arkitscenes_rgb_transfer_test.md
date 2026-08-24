# Direct multiview RGB — one untouched-scene transfer test

Status: **complete — run 2 scored 2026-08-23. Both gates failed; no transfer claim.**
Run 1 void; protocol amended and re-frozen before run 2 regenerated.
The original protocol below was written with no knowledge of the test room.
See Amendment 1 at the end for what changed and why.

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

---

# Amendment 1 — run voided and regenerated, 2026-08-23

**Run 1 is void. Nothing from it is scored, and no transfer claim rests on it.**

## Why, and why this is not tuning

The blinded model never ran. No response was requested, no answer existed, and
nothing was scored. The no-adjustment rule exists to prevent selecting on
outcomes; there was no outcome to select on. What is being fixed is a
question-generation defect that was flagged **before** the owner answered.

The run-1 key is preserved **in commit `6ee7715`**. It is the evidence the
defect was real and it stays in the record. **Corrected 2026-08-24:** that path,
`eval/human_feedback/arkitscenes_rgb_transfer_key_47331972.json`, now holds the
**run-2** key (10 items, none ambiguous, questions hash `520074c2…`), which
overwrote the run-1 key at HEAD. To read the run-1 key, check out `6ee7715`.

## What the defect was

The owner marked 4 of 8 items ambiguous, leaving **n = 4**: one cardinality,
two presence, one comparative, and **zero cross-view**. Two failures follow:

1. **The coverage gate became an anti-abstention gate.** At n = 4, coverage
   ≥ 0.80 requires 4 of 4 answered, because 3/4 = 0.75 fails. A single honest
   `unknown` would fail the run. That gate was written to catch a model
   abstaining its way out of being wrong; at this sample size it instead
   punishes the exact behaviour the product path is built on.
2. **The surviving mix could not support the claim.** Half the scored set was
   presence — the easiest category and the least spatial — with no cross-view
   item at all. A clean 4/4 would have licensed "RGB can tell whether a sofa
   exists in an unseen room", not the intended sentence.

Root cause of the exclusions: six of eight questions said "in this room", and
the capture is a **9.99 × 13.53 m single-storey multi-space floor** with a
staircase and adjoining rooms, not one room. The phrase does not denote, so
several items had no determinate answer.

## The three fixes, fixed before regenerating

1. **Scope wording.** Cardinality and presence questions now say "in the
   captured space", defined in the conventions as everything visible anywhere
   in the supplied views. Comparative and near questions need no scope and are
   unchanged.
2. **Anchor ordering.** Was first-appearance ascending, which put a small
   wall print — 2 of 3 passes, seen in 3 frames — at rank 0, where it carried
   three of eight questions. Now ordered by **passes agreeing descending, then
   frames-seen descending, then first appearance ascending**. Still entirely
   mechanical, still nothing to do with whether any system can answer; it
   simply stops a briefly-glimpsed object dominating the set.
3. **Ten questions, not eight**, for headroom against exclusions: 3
   presence/cardinality, 4 comparative, 3 cross-view. Cross-view gains a slot
   because it lost both last time and it is the category the work turns on.

Everything else is unchanged: the 18-view selection, the prompt composition,
the answer schema, the conventions themselves, the gates, the stopping rule,
one run only, and the interpretation limits.

## The anchor passes are reused, not re-run

The three blind enumeration passes are reused verbatim. They enumerated
objects from the 18 frames with no access to any question, any wording, or any
ordering rule, so none of the three fixes touches what they saw or reported.
Re-running them would spend thirteen minutes to obtain the same evidence and
would introduce sampling noise between run 1 and run 2 for no gain.

## Gate arithmetic at the new size

At n = 10, accuracy ≥ 0.60 needs 6 correct and coverage ≥ 0.80 needs 8
answered, so up to two honest abstentions are affordable. If exclusions again
drive the scored set below **six** items, the run is void and reported as
such rather than scored — recorded here so it is a rule and not a judgement
made afterwards.

# Amendment 2 — relational slots restricted to unique-referent anchors

Applied immediately after amendment 1 regenerated, before any blinded response
existed, and before the owner was asked to review anything.

Amendment 1's ordering fix promoted the best-attested anchors — but
best-attested turns out to correlate with *numerous*. Of the top twelve
anchors, exactly one had a unique referent. Six of the ten regenerated
questions read "the cushion", "the framed picture", "the wall power socket",
where the three passes counted four, about nine, and three.

A singular definite description with no unique referent is ill-posed in the
same way "in this room" was ill-posed on a multi-room floor. It is detectable
without any model output, and it would have collapsed the scored set a second
time.

**The rule:** comparative and cross-view slots draw only from anchors where all
three passes agreed there is **exactly one**. 27 of the 48 admitted anchors
qualify, so the constraint costs nothing. Presence and cardinality slots are
untouched — "how many framed pictures are in the captured space" is a good
question precisely *because* there are several.

Nothing about answerability enters: the filter is a count agreed by three
observers who never saw a question.

## A limit on amending

This is the second structural amendment. Repeated amendment is itself a risk —
at some point "fix the generator" becomes "iterate until the questions look
right", which is the behaviour the protocol exists to prevent.

So: **no further structural amendment to this generator on this scene.** If
run 2's returned key again drives the scored set below six, that is reported as
a finding about the fixed-procedure generator — that it does not produce a
usable question set on this capture — and the transfer test is abandoned rather
than amended a third time. The demo stays a recorded replay and no transfer
claim is made.

---

# Result — run 2, scored once, 2026-08-23

Blinded response hash-pinned at `e193e6f` before scoring. Key at `45f8ec9`.
Packet `33355c8a…`, questions `520074c2…`. Scored exactly once.

## Both gates fail. No transfer claim is made.

| gate | required | measured | |
|---|---:|---:|---|
| exact accuracy | ≥ 0.60 | **0.50** | fail |
| answer coverage | ≥ 0.80 | **0.50** | fail |
| scored items | ≥ 6 | 10 | pass |

**5 correct, 0 wrong, 5 unanswered.** Per the stopping rule: the demo remains a
fixed evaluation replay, nothing is retuned, no third scene is tried, and no
gate is softened.

## What the failure actually is

It is not an accuracy failure. **Accuracy when answered is 1.000** — every one
of the five answers was right, and the model never once answered wrongly. The
false-confident rate is zero.

The failure is entirely coverage. The model abstained on half the set, and
`exact_accuracy` counts an abstention as a non-correct item, so a perfectly
calibrated abstainer fails the accuracy gate too. Both gates failed on the same
underlying behaviour.

| form | n | correct | unanswered |
|---|---:|---:|---:|
| presence | 3 | 3 | 0 |
| comparative | 4 | 2 | 2 |
| cross-view | 3 | **0** | **3** |

The gradient is clean and it is the interesting result: presence answered and
correct everywhere; comparative answered half; **cross-view answered not at
all**. Confidence tracks it honestly — 0.96 and 0.93 on presence, 0.36 and 0.48
on the two comparatives it did attempt.

Both zero-view items are cross-view, and it declined both. On the six-item
thin-evidence slice it answered only two, and got both right.

**Form naming, clarified 2026-08-24.** The three non-co-visible items are called
*cross-view* in this document; the scorer emits them under the form name
`binary_near` with `cross_view = true` on each row. Counts are identical (3 items,
0 correct, 3 unanswered); only the label differs. Source:
`eval/results/project_census_v1/arkit_rgb_transfer_47331972_score.json`.

## Reading this honestly

Direct multiview RGB **did not transfer** under this procedure. What transferred
is its *calibration*: on an unseen room it answered only what it could see and
was right every time it spoke. What did not transfer is its *reach* — the
spatial questions that motivate this work, the cross-view ones, it declined
wholesale.

That is a more useful negative than a noisy pass would have been. The previous
scenes' 7/10 came from a set where cross-view items were answerable from
co-visible evidence; here, with three genuinely non-co-visible pairs, the RGB
path produced nothing at all.

An honest abstainer is the right failure mode for a product — but a product
that abstains on half the questions, and on all of the ones that make the
product distinctive, is not yet a product.

## What this does not license

No claim that RGB cannot do this, from one room and ten questions. No comparison
to the graph path, which did not run here. No retune-and-retry: the protocol
forbids it and the result stands as measured.

The prior results are untouched. `41069025` and `41069042` still stand exactly
as reported, and the recorded-replay demo is unchanged and still valid for what
it claims to be.
