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
