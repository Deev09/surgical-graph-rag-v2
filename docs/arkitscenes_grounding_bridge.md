# Oracle-free language-to-entity grounding bridge — protocol

Status: **rules declared before any prediction was made.** This document is
committed with the implementation and before the sidecar is opened against the
human mappings.

## Why grounding and not more relations

The relation challenge established that the delivered graph holds the spatial
content and the deployable path cannot reach it. The stored-edge replay matched
the recomputed geometry ceiling 12/12, so relation extraction is cleared on that
slice, while the delivered arm scored **0 correct of 12** because its learned
names could not resolve a single question anchor but one.

So one stage is replaced — how a natural-language anchor becomes a uid — and
nothing else:

    natural-language anchor
            |
    multiview RGB crops of every delivered entity + pinned OpenCLIP
            |
    UID candidate, or abstention
            |
    existing serialized NEAR edges, unchanged
            |
    answer

## Input facts, recorded before building

- **`embedding_ref` is `None` on every delivered entity**, in both scenes.
  There is no persisted image embedding to reuse; only the top-3
  `semantic_hypotheses` survive from the label stage. The bridge must therefore
  re-encode crops itself rather than read a stored vector.
- Delivered entities: 35 in `41069025`, 23 in `41069042`.
- The label stage that produced `display_label` used
  `image_source = arkitscenes_rgb_crop_pad0.15_mark0`, model
  `ViT-B-32-quickgelu` / `openai`, `min_top1_score = 0.28`.

## Frozen — nothing below is regenerated or retuned

Delivered instances, segmentation, graph nodes and edges, relation thresholds,
questions, human keys, and the blinded RGB responses. The bridge reads the
entity manifest for geometry handles only, and the mesh and capture frames for
crops. It writes one new prediction sidecar and nothing else.

## Reused, not rebuilt

- `extractors.arkitscenes_rgb_crops.RgbCropSource` with the exact `rgb_tight`
  configuration: `stride=6`, `n_views=3`, `context_pad=0.15`,
  `mark_target=False`. Same deterministic view selection the label stage used.
- `segmenter.clip_labeler` weights and prompts: `ViT-B-32-quickgelu` /
  `openai`, the three pinned prompt templates, CPU, eval mode, `no_grad`.
  Weights resolve to the locally cached snapshot
  `e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31`.
- No new model, no GPU run, no vocabulary training, no fine-tuning.

The one addition to the labeler is a public per-image embedding accessor.
`ClipLabeler.classify` mean-pools across an instance's views before scoring,
which destroys exactly the per-view detail the admission rule needs. The
addition is additive; `classify` is untouched and byte-identical.

## The rules, fixed before evaluation

**Phrase set.** For an anchor, the anchor name plus its declared synonyms from
`object_synonyms` in the frozen question manifest. No synonym is added,
removed or reworded for this experiment.

**Text encoding.** Each phrase through the three pinned prompt templates,
mean-pooled over templates, L2-normalised — the same construction
`ClipLabeler._text_embeddings` uses.

**Image encoding.** Each delivered entity's ranked RGB crops, encoded
individually and L2-normalised. Deliberately *not* mean-pooled: per-view
identity is the evidence the admission rule is written against.

**A "view slot"** is a rank index `k` over an entity's own ranked crops
(`k = 0, 1, 2`). Two entities' slot-`k` crops come from different capture
frames, chosen independently by the frozen view-selection rule. An entity with
fewer than `k+1` usable crops does not compete in slot `k`.

**Per-slot score.** `score_k(entity, anchor) = max over the anchor's phrases of
cosine(image_embedding(entity, slot k), text_embedding(phrase))`. Max over
phrases, because synonyms are alternative names for one object and any single
match is evidence for it; averaging would penalise an anchor for carrying a
rare synonym.

**Aggregate score.** `aggregate(entity, anchor) = mean over that entity's
available slots of score_k`. Mean, not max, so one lucky crop cannot carry an
entity.

**Ranking.** Entities ranked by `aggregate` descending, ties broken by uid
ascending, so the output is fully deterministic.

**Admission — no numeric threshold anywhere.** Let `top` be the highest
aggregate entity and `winner_k` the highest `score_k` entity in slot `k`.

> Admit `top` if and only if `top == winner_k` for **at least two distinct view
> slots**. Otherwise abstain.

No confidence cutoff is applied and none is swept. The only gate is
cross-view agreement, which is a property of the evidence rather than a tuned
constant.

**Absent objects are not special-cased.** The bridge has no way to know that
the striped rug and the white radiator were never delivered, and giving it one
would mean reading the human key. If the cross-view rule admits a uid for an
absent object, that is counted as a **precision error** and reported as one.
It is not suppressed after the fact.

## Hard separation

The bridge and its runner must not import or read human keys, uid mappings,
annotation boxes, oracle labels, or any evaluation module. Enforced by AST
guards over the module source and its import graph, not by convention.

The prediction sidecar is written and hashed **before** the human mappings are
opened. Evaluation is a separate tool that reads the finished sidecar.

## Predeclared success gates — all three required

| gate | threshold |
|---|---|
| anchor precision | **≥ 0.80** |
| anchor coverage | **≥ 0.60** over human-resolvable delivered anchors |
| deployable graph-unique QA wins vs the frozen blinded RGB arm | **≥ 2** |

Definitions, fixed here:

- An anchor is **human-resolvable** if the owner's returned mapping gave it a
  uid — not `none / missing`, not `ambiguous`.
- **Precision** = correct admissions / **all** admissions. An admission for an
  anchor with no human uid is automatically incorrect; the denominator is not
  quietly restricted to the resolvable ones.
- **Coverage** = admissions that are correct and human-resolvable / all
  human-resolvable anchors.
- A **graph-unique win** is an item the grounded delivered path answers
  correctly and the frozen blinded RGB arm does not. Ceiling wins never count.

## Predeclared stop

If any gate fails: **stop the graph-centered answer path and use direct
multiview RGB for the product.** Do not tune prompts, thresholds, synonyms or
view rules and re-run. This is written before the numbers exist for the same
reason the reachability note was written before the last score.

## What grounding cannot fix

The striped rug and the white radiator are absent from the delivered
partition. Grounding cannot invent an instance that was never delivered; those
items stay lost until segmentation or delivery changes. That is a separate
problem and is not what these gates measure.

---

# Result — scored 2026-08-19, after the prediction was sealed

The sidecar was committed and pushed at `951269a` **before** the human
mappings were opened. Prediction
`51b56774262146de905ae8987f3fe8d5ddf970db6ae79a189b3854211a2a0f26`, reproduced
identically across two independent runs.

## All three gates fail

| gate | required | measured | verdict |
|---|---|---:|---|
| anchor precision | ≥ 0.80 | **0.583** | fail |
| anchor coverage | ≥ 0.60 | **0.467** | fail |
| deployable graph-unique wins vs blinded RGB | ≥ 2 | **0** | fail |

17 anchors, 15 human-resolvable, 12 admitted, 7 correct.

Per scene the split is stark: `41069025` reaches precision 0.75 / coverage
0.667, while `41069042` reaches **0.25 / 0.167**. A bridge that works in one
room and not the other is not a bridge.

## The bridge did move the arm, and it was not close to enough

| layer | correct | wrong | unanswered | accuracy |
|---|---:|---:|---:|---:|
| `delivered_graph` (exact label match) | 0 | 0 | 10 | 0.000 |
| **`grounded_delivered_graph`** | **2** | **0** | 8 | **0.200** |
| `stored_graph_human_identity` (perfect identity) | 7 | 1 | 2 | 0.700 |
| `blinded_rgb_vlm` | 7 | 2 | 1 | 0.700 |

Grounding closed roughly a fifth of the gap between exact-label matching and
perfect identity. Every question it answered, it answered correctly — but that
is composition luck, not a safety property: a question abstains if **any** of
its anchors abstains, so the five mis-grounded anchors mostly landed in
questions that were already abstaining for another anchor. On a question whose
anchors all resolved wrongly, this arm would answer confidently and wrongly.

Zero grounded graph-unique wins. Every item the grounded arm answered, blinded
RGB also answered correctly.

## Four failures worth naming

**Two anchors collided on one uid, twice.** `coffee table` and `round dining
table` both grounded to `obj_5`; `white radiator` and `cream curtain` both
grounded to `obj_2` in `41069042`. Nothing in the rule enforces that distinct
anchors take distinct entities, and both scene-25 tables share the delivered
label vocabulary's single `table`.

**It confidently grounded an object that does not exist.** `white radiator` is
`none / missing` in the owner's mapping, and the bridge admitted `obj_2` for it
with **three** agreeing view slots — maximum cross-view agreement for an object
that was never delivered. The protocol predeclared that this counts as a
precision error rather than being suppressed, and it did. Cross-view agreement
measures *consistency*, not *existence*.

**The abstention rule mostly fired on things that do exist.** Of five
abstentions, four (`long narrow desk`, `window`, `white desk`, `framed picture`)
are human-resolvable objects the bridge failed to reach; only `striped rug` is
a genuinely absent object it correctly declined. The rule is not selecting for
existence.

**`bed` grounded to `obj_18` rather than `obj_6`.** The single largest, most
distinctive object in `41069042` — 54,217 vertices — was mis-grounded with two
agreeing slots.

## Predeclared stop: fires

The protocol said, before any number existed:

> If any gate fails: stop the graph-centered answer path and use direct
> multiview RGB for the product. Do not tune prompts, thresholds, synonyms or
> view rules and re-run.

All three failed. **Stop.** Nothing was tuned and nothing was re-run after
seeing these numbers, and no follow-on experiment is proposed here.

Direct multiview RGB is the product path. It scored 0.700 on this key, the same
as the identity oracle, while needing no mapping, no bridge and no graph.

## What this closes, and what it does not

**Scope of the stop, stated precisely.** This closes **the pinned OpenCLIP
crop-based grounding bridge** — one encoder, one crop configuration, one
admission rule, on this key. It does **not** close grounding research in
general, and no result here licenses the claim that language-to-entity
grounding is impossible.

What the stop rule does is prevent trying endless variants against the same
seventeen anchors, where each variant would be selected on numbers it had
already seen. That is the failure the rule exists to prevent, and it is a
different thing from a claim about grounding as a field.

Closed: whether soft multimodal grounding over the *current* delivered entities,
with this encoder and this rule, can unlock the spatial information the graph
already holds. On two scenes, seventeen anchors and one pinned encoder — it
cannot, by a wide margin on all three gates.

Not closed, and deliberately not attempted: whether a different grounding
mechanism, a per-entity embedding persisted at delivery time, exclusivity
constraints across anchors, or better instance delivery would change this. Those
are hypotheses, not findings, and the stop rule exists precisely so they are not
tried on this key by whoever reads a 0.583 and feels close.

The rug and the radiator remain an instance-delivery problem. Grounding never
had a path to them.
