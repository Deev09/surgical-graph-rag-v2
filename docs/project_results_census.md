# Spatial project — facts-only results census

Compiled 2026-08-23 at repository HEAD `c8e200d`; **reconciliation pass applied
2026-08-24 from `58af9cd`** (see [Reconciliation pass](#reconciliation-pass)). **No experiment was run, no
threshold, key, evaluation module or system behaviour was changed, and no result
was reinterpreted or retuned to produce this document.** It records what the
committed artifacts already state.

Companion registry: [`docs/project_results_registry.csv`](project_results_registry.csv)
— **271 rows**. Every number below cites its `result_id` and its source path.

## How to read this

**Scope is not decoration.** Six values are used and they are not interchangeable:

| scope | meaning |
|---|---|
| `deployable` | a path that could ship, evaluated end-to-end: no human key, no oracle in the answer path, and the metric's denominator is not oracle-selected |
| `oracle_free_component_eval` | the prediction path is oracle-free, but the **denominator is oracle-selected** — a component evaluation of one stage, **not** end-to-end deployable performance |
| `delivered` | the delivered pipeline's own output, scored against an oracle or key |
| `proposal_ceiling` | oracle-evaluated upper bound on proposals — **not** delivered instances, **not** a QA gain |
| `identity_oracle` | consumes human-supplied identity or mapping — a bound, **never** system performance |
| `definition-change` | the benchmark or key definition moved, not the model |
| `bug-diagnostic` | a correctness fix; never a model before/after |

**One row, one denominator.** No row in the registry mixes metrics with different
denominators, and no cell below combines them. Where a scene has two legitimate
denominators (entity-only versus including structural classes) they are separate rows.

**Source precedence.** `runs/` and `data/` are gitignored in this repository, so most
numeric run artifacts are **not tracked**. Where a value exists only under `runs/`, the
primary source is the tracked narrative document that reports it, and the gitignored
path is named in the row's notes. **28 of 271 rows cite an untracked primary source**;
those values cannot be reproduced from the repository alone.

**`—` means unverifiable, never inferred.** Every `—` is explained in
[Values that could not be verified](#values-that-could-not-be-verified).

---

## Table 1 — Entity delivery and proposal coverage, by scene

Replica, frozen Mask3D reference bundle `ms02`. Ground-truth entity counts are 47 /
45 / 53 as given. **The three counts that circulate under the word “matched” are
separated here and must not be conflated.**

| metric | office_0 | room_1 | room_2 | denominator | scope |
|---|---|---|---|---|---|
| GT entities | 47 | 45 | 53 | — | reference |
| predicted instances, **incl. structural-matching** | 23 `[A01]` | 26 `[A21]` | 23 `[A44]` | — | delivered |
| predicted **entities** in delivered bundle | 18 `[A02]` | 22 `[A22]` | 22 `[A45]` | — | delivered |
| any-overlap matches, **incl. structural** | 21/66 `[A03]` | 26/55 `[A23]` | 22/61 `[A46]` | oracle instances (66 / 55 / 61) | delivered |
| any-overlap matches, **entity-only** | 16 `[A04]` | 22 `[A24]` | 21 `[A47]` | GT entities | delivered |
| **matches @ IoU 0.50** | 12/47 (0.26) `[A05]` | 17/45 (0.38) `[A25]` | 18/53 (0.34) `[A48]` | GT entities | delivered |
| matches @ IoU 0.25 | 12/47 `[A06]` | 19/45 `[A26]` | 21/53 `[A49]` | GT entities | delivered |
| matches @ IoU 0.75 | 9/47 `[A07]` | 13/45 `[A27]` | 15/53 `[A50]` | GT entities | delivered |
| median matched IoU (per scene) | — | — | — | — | see note |
| failure: recovered | 12 `[A08]` | 17 `[A28]` | 18 `[A51]` | GT entities | delivered |
| failure: merged | 13 `[A09]` | 20 `[A29]` | 22 `[A52]` | GT entities | delivered |
| failure: lost_by_resolver | 0 `[A10]` | 0 `[A30]` | 1 `[A53]` | GT entities | delivered |
| failure: no_raw_proposal | 22 `[A11]` | 8 `[A31]` | 12 `[A54]` | GT entities | delivered |

Sources: `docs/c1_closeout.md` (a519a8f) for the IoU and failure-class columns;
`eval/predictions/phase8_c2/replica_*_c2_labels.json` (4f38766) for entity-only
any-overlap; `runs/phase8_c1/ms02/replica_*_exact_eval.json` (**gitignored**) for the
prediction counts, the 0.25 / 0.75 columns and the incl.-structural matches.

**Median matched IoU is `—` per scene.** The only committed statement is a *range
across four scenes*, 0.80–0.89 `[A69]` (`docs/c1_closeout.md`). No committed artifact
reports a per-scene median, so the per-scene cells are not filled and were not inferred.

### Proposal ceilings — oracle-evaluated, not delivered

Every row here is `proposal_ceiling`: an oracle-guided upper bound on what the
proposal set *could* support. **These are not delivered instances and not QA gains.**

| arm | office_0 | room_1 | room_2 | scope |
|---|---|---|---|---|
| Mask3D viable raw | 13/47 `[B04]` | 21/45 `[B07]` | 20/53 `[B01]` | proposal_ceiling |
| P1 alone | 12/47 `[B05]` | 13/45 `[B08]` | 25/53 `[B02]` | proposal_ceiling |
| **pooled (Mask3D + P1)** | 19/47 `[B06]` | 26/45 `[B09]` | 33/53 `[B03]` | proposal_ceiling |

All nine values verify exactly against `docs/c1_p1_multiview_proposals_protocol.md`
(f5737dd) and `docs/c1_composition_ceiling.md` (98b73eb).

**What P1 is, and its fixed settings.** 40 gravity-aligned multiview mesh renders →
SAM 2.1 Hiera-L masks → lift and fuse into 3D proposals → pool with Mask3D. Viability
is per-entity best-single-proposal at vertex IoU ≥ 0.50. Recorded in
`docs/c1_p1_multiview_proposals_protocol.md` (f5737dd). See
[Values that could not be verified](#values-that-could-not-be-verified) for what the
artifacts do **not** state about transfer.

**Do not place these beside Table 1's delivered rows as a before/after.** room_2 shows
why: the delivered pipeline recovers 18/53 (0.34) `[A48]` at IoU 0.50, while the Mask3D proposal
ceiling on the same scene is 20/53 `[B01]` and the pooled ceiling 33/53 `[B03]`. Those are
three different questions with the same denominator.

---

## Table 2 — Semantic-label experiments

ARKitScenes. **Matched-instance classification — not detection, and not
human-ground-truth spatial QA.** The denominator is instances already matched to an
annotation box by the evaluator, so every row here is
`oracle_free_component_eval`: the prediction path uses no oracle, but the
denominator does. **These rows are not end-to-end deployable performance and must
not be quoted as such.** Reclassified from `deployable`/`delivered` in the
reconciliation pass.

| scene | arm | top-1 | top-3 | admission precision | scope |
|---|---|---|---|---|---|
| 41069021 | splat | 0/7 `[C05]` | 0/7 `[C06]` | 0.00 `[C19]` | oracle_free_component_eval |
| 41069021 | **rgb_tight** | 5/7 `[C07]` | 7/7 `[C08]` | 0.71 `[C20]` | oracle_free_component_eval |
| 41069021 | rgb_context | 3/7 `[C17]` | 5/7 `[C18]` | 0.43 `[C21]` | oracle_free_component_eval |
| 41069025 | splat | 1/9 `[C09]` | 4/9 `[C10]` | 0.20 `[C22]` | oracle_free_component_eval |
| 41069025 | **rgb_tight** | 5/9 `[C11]` | 8/9 `[C12]` | 0.56 `[C23]` | oracle_free_component_eval |
| 41069042 | splat | 0/5 `[C13]` | 0/5 `[C14]` | 0.00 `[C24]` | oracle_free_component_eval |
| 41069042 | **rgb_tight** | 2/5 `[C15]` | 3/5 `[C16]` | 0.50 `[C25]` | oracle_free_component_eval |
| **pooled** | splat | 1/21 `[C01]` | 4/21 `[C03]` | — | oracle_free_component_eval |
| **pooled** | **rgb_tight** | 12/21 `[C02]` | 18/21 `[C04]` | — | oracle_free_component_eval |

Source: `docs/arkitscenes_rgb_label_results.md` (c700df3). Per-scene top-1 sums to the
pooled figure exactly (0+1+0 = 1/21; 5+5+2 = 12/21), as does top-3 (0+4+0 = 4/21;
7+8+3 = 18/21).

**Held fixed, stated in the source:** OpenCLIP weights, the class vocabulary, `top_k=3`,
`min_top1_score=0.28`, the evaluator, the delivered Mask3D partitions, and greedy
IoU ≥ 0.50 matching. **Only the label images changed.** Two of those constants are
nonetheless disputed between sources — see contradictions C-2 and C-4.

The source states its own limits: *“This fixes naming, not detection”* and
*“`min_top1_score=0.28` was chosen against splat scores … **not** calibrated for RGB.”*

---

## Table 3 — Relation and QA experiments

**Denominators differ between blocks and the blocks must not be merged.** The kill
test has 6 questions; the relation challenge has 12 authored, 10 scored; the transfer
test has 10 scored.

### 3a · Representation kill test — 41069025, 6 questions

| arm | correct | wrong | unanswered | exact accuracy | scope |
|---|---|---|---|---|---|
| object map | 0 / 6 `[F01]` | 5 / 6 `[F02]` | 1 / 6 `[F03]` | 0.000 `[F05]` | delivered |
| typed graph | 0 / 6 `[F07]` | 5 / 6 `[F08]` | 1 / 6 `[F09]` | 0.000 `[F11]` | delivered |
| blinded RGB | 4 / 6 `[F13]` | 2 / 6 `[F14]` | 0 / 6 `[F15]` | 0.667 `[F17]` | deployable |
| hybrid router | 4 / 6 `[F19]` | 2 / 6 `[F20]` | 0 / 6 `[F21]` | 0.667 `[F23]` | deployable |

Screening rule outcome: accuracy gain 0.0000 (required >= 0.10) — MISSED `[F25]`, wrong-answer reduction 0.0000 (required >= 0.30) — MISSED `[F26]`,
proceed false `[F27]`. Source: `docs/arkitscenes_representation_kill_test.md` (e14d8fe).

### 3b · NEAR relation challenge — 41069025 + 41069042, 10 scored of 12

| layer | identity from | correct | wrong | unans. | coverage | scope |
|---|---|---|---|---|---|---|
| geometry ceiling | human | 7 / 10 `[F28]` | 1 / 10 `[F29]` | 2 / 10 `[F30]` | 0.800 `[F31]` | **identity_oracle** |
| stored-edge replay | human | 7 / 10 `[F35]` | 1 / 10 `[F36]` | 2 / 10 `[F37]` | 0.80 `[F38]` | **identity_oracle** |
| delivered graph | learned labels | 0 / 10 `[F40]` | 0 / 10 `[F41]` | 10 / 10 `[F42]` | 0.00 `[F43]` | delivered |
| grounded graph | grounding bridge | 2 / 10 `[F45]` | 0 / 10 `[F46]` | 8 / 10 `[F47]` | 0.20 `[F48]` | delivered |
| blinded RGB | the images | 7 / 10 `[F50]` | 2 / 10 `[F51]` | 1 / 10 `[F52]` | 0.90 `[F53]` | deployable |
| hybrid | routed | 7 / 10 `[F55]` | 2 / 10 `[F56]` | 1 / 10 `[F57]` | 0.900 `[F58]` | deployable |

**The two 7/10 ceiling rows are bounds, not performance.** They consume
human-supplied object identity. The only deployable structured arms are the delivered
graph at 0 / 10 `[F40]` and the grounded graph at 2 / 10 `[F45]`.

Cross-layer: stored-edge replay agrees with the geometry ceiling 12 / 12 (agreement rate 1.0, 0 disagreements) `[F63]`;
7 / 12 (= 7 of the 10 scored items) `[F64]` sit in the bucket *ceiling correct, delivered unanswered*;
delivered-graph-unique correct answers 0 (required >= 2) — BAR NOT MET `[F60]`.
2 / 12 excluded -> 10 scored `[F65]` items were excluded as owner-ambiguous.
Source: `docs/arkitscenes_relation_challenge.md` (b33abed).

### 3c · Grounding bridge — three predeclared gates

| gate | required | measured | scope |
|---|---|---|---|
| anchor precision | ≥ 0.80 | 0.583 (required >= 0.80) — FAIL `[F67]` | deployable |
| anchor coverage | ≥ 0.60 | 0.467 (required >= 0.60) — FAIL `[F68]` | deployable |
| graph-unique wins vs RGB | ≥ 2 | 0 (required >= 2) — FAIL `[F69]` | deployable |

Admissions: 12 / 17 admitted; 15 / 17 human-resolvable; 7 admissions correct `[F70]`. Per scene, precision 0.75 `[F71]` / 0.25 `[F73]` and coverage
0.667 `[F72]` / 0.167 `[F74]`. 1 of 12 admissions `[F75]`.
Source: `docs/arkitscenes_grounding_bridge.md` (d6baa42);
prediction sidecar `eval/predictions/arkitscenes_anchor_grounding_v1.json` (tracked).

### 3d · Unseen-scene transfer — 47331972, 10 scored

| metric | value | scope |
|---|---|---|
| correct | 5 / 10 `[F76]` | deployable |
| wrong | 0 / 10 `[F77]` | deployable |
| unanswered | 5 / 10 `[F78]` | deployable |
| exact accuracy (gate ≥ 0.60) | 0.50 (required >= 0.60) — FAIL `[F82]` | deployable |
| answer coverage (gate ≥ 0.80) | 0.50 (required >= 0.80) — FAIL `[F83]` | deployable |
| accuracy when answered | 1.000 `[F85]` | deployable |
| false-confident rate | 0 (zero) `[F86]` | deployable |
| presence items | 3 / 3 correct, 0 unanswered `[F80]` | deployable |
| comparative items | 2 / 4 correct, 2 unanswered `[F81]` | deployable |
| **cross-view items** | 0 / 3 correct, 3 / 3 unanswered `[F79]` | deployable |
| transfer claim | NONE — both gates failed; demo remains a fixed evaluation replay `[F88]` | — |

Source: `docs/arkitscenes_rgb_transfer_test.md` (c8e200d); key and response tracked at
`eval/human_feedback/arkitscenes_rgb_transfer_{key,response}_47331972.json`.

### 3e · Support relations — two separate experiments, never merged

Different datasets, different relation definitions, different denominators.

| experiment | metric | value | scope |
|---|---|---|---|
| E1 Replica semantics-v2, room_2 | support citation hits, before | 5/20 `[E01]` | proposal_ceiling |
| E1 Replica semantics-v2, room_2 | support citation hits, after | 16/20 `[E02]` | **definition-change** |
| E1 Replica semantics-v2, room_2 | support citation precision | 0.94 `[E03]` | **definition-change** |
| E1 transfer, room_1 | scene aggregate micro-precision | 0.58 `[E10]` | **definition-change** |
| E1 transfer, office_0 | scene aggregate micro-precision | 0.36 `[E12]` | **definition-change** |
| E1 track decision | S2 proceed rule | STOP_TRACK `[E21]` | **definition-change** |
| E2 ARKit 41069025 | owner-confirmed positives recovered | 1/3 `[E27]` | delivered — **exploratory** |
| E2 ARKit 41069025 | precision on keyed pairs | 1.0 `[E28]` | delivered |
| E2 ARKit 41069025 | recall on keyed positives | 0.3333333333333333 `[E29]` | delivered |
| E2 ARKit 41069025 | candidates proposed / pairs evaluated | 1/714 `[E30]` | delivered |

**E2 is exploratory and must not become a headline.** Two of its three keyed positives,
`obj_9→obj_13` and `obj_23→obj_13`, appear only in the FINAL owner-corrected key and have
no independent returned-form record; the key attributes the omission to a review-form
defect. The entire recall denominator therefore rests on pairs captured once. Rows
`[E27]`, `[E28]`, `[E29]` carry this caveat in the registry.

E2 miss attribution: the threshold-reachable miss is `obj_23->obj_13` at footprint
overlap 0.4545 `[E32]` against a 0.50 gate; the segmentation-evidence miss is
`obj_1->obj_14` with 0 of 7 patches in the contact band; mode = "evidence_missing" `[E34]`.
Sources: `docs/semantics_v2_track_protocol.md` (ef04997);
`eval/human_feedback/arkitscenes_41069025_support_relation_key_v1.json` (tracked).

---

## Table 4 — Experiment deltas

One row per experiment that changed exactly one variable. **A ceiling never appears as
an “after” against a delivered “before”.**

| experiment | single changed variable | before | after | scenes | metric family | scope | what it establishes |
|---|---|---|---|---|---|---|---|
| RGB label input A/B | labeler **input images only** (splat → real capture RGB crops, `context_pad=0.15`) | 1/21 `[C01]` | 12/21 `[C02]` | 41069021, 41069025, 41069042 | matched-instance top-1 | delivered → deployable | Naming improves when the classifier sees photographs instead of point splats. Says nothing about detection. |
| RGB label input A/B | same | 4/21 `[C03]` | 18/21 `[C04]` | same three | matched-instance top-3 | delivered → deployable | Same effect at top-3. |
| Crop-context control | crop context only (`0.15` → `0.60`, target dimmed) | 5/7 `[C07]` | 3/7 `[C17]` | 41069021 | matched-instance top-1 | deployable | More context **hurts**; the gain is object texture, not room gist. |
| Graph-consistency QA | label stage only; same geometry, graph and questions | 13/14 `[C32]` | 3/14 `[C33]` | three ARKit scenes | abstentions / 14 questions | delivered → deployable | Naming makes the graph *askable*. Scored against the graph's own neighbours, so it cannot show the neighbours are spatially right. |
| P1 proposal pooling | added 40-render SAM 2.1 Hiera-L multiview proposals, pooled with Mask3D | 20/53 `[B01]` | 33/53 `[B03]` | room_2 | proposal viability @ IoU 0.50 | **proposal_ceiling** | Raises an oracle-evaluated *ceiling*. Not a delivered gain and not a QA gain. |
| P1 proposal pooling | same | 13/47 `[B04]` | 19/47 `[B06]` | office_0 | proposal viability @ IoU 0.50 | **proposal_ceiling** | As above. |
| P1 proposal pooling | same | 21/45 `[B07]` | 26/45 `[B09]` | room_1 | proposal viability @ IoU 0.50 | **proposal_ceiling** | As above. |
| Semantics-v2 support | relation definition and key | 5/20 `[E01]` | 16/20 `[E02]` | replica_room_2 | support citation hits / 20 | **definition-change** | The benchmark definition moved. Not a model improvement, and it did not transfer: 0.58 `[E10]` on room_1, 0.36 `[E12]` on office_0. Track STOP_TRACK `[E21]`. |
| Identity source, structured arm | **identity only** — learned labels → human-verified UIDs, same stored edges | 0 / 10 `[F40]` | 7 / 10 `[F35]` | 41069025 + 41069042 | relation QA correct / 10 | delivered → **identity_oracle** | The information is present in the graph. **The “after” is a bound, not a shippable result.** |
| Grounding bridge | identity only — exact label match → oracle-free multimodal grounding | 0 / 10 `[F40]` | 2 / 10 `[F45]` | 41069025 + 41069042 | relation QA correct / 10 | delivered | A deployable identity stage recovers 2 of the 7 the oracle reaches; all three gates still fail. |
| Scene transfer, RGB arm | **scene only** — same prompt, view rule and scoring | 7 / 10 `[F50]` | 5 / 10 `[F76]` | 41069025+41069042 → 47331972 | relation QA correct / 10 | deployable | Accuracy falls and coverage halves on an unseen room; cross-view goes to 0 / 3 correct, 3 / 3 unanswered `[F79]`. |
| Camera convention fix | axis convention `[1,-1,-1]` → plain OpenCV | see D-1/D-2 | see D-1/D-2 | 41069021 | depth reprojection error | **bug-diagnostic** | A correctness prerequisite. All crop/label measurements made under the flip are invalidated. **Not a model before/after.** |
| Frame-stride fix | stride applied after pose matching instead of before | 122 usable frames | 1,878 usable frames | 41069021 | usable posed frames | **bug-diagnostic** | A sampling bug, not a method change. |

Sources are the per-row `result_id` entries in the registry; the two bug-diagnostic rows
are `docs/arkitscenes_rgb_label_results.md` (c700df3) and
`extractors/arkitscenes_rgb_crops.py` (7e87718), which **disagree numerically** — see D-1 and D-2.

---

## Contradictions found

Reported, **not resolved**. Each is a disagreement between two artifacts, or inside one
artifact. None was settled by assumption, and no registry row silently picks a winner.

### Area A

**A-1 · office_0 / room_1 / room_2 exact-eval numbers: TWO reports exist per scene under runs/phase8_c1/ with nearly identical filenames, produced at DIFFERENT operating points. Anyone citing the wrong path gets different entity-match, predicted-instance and any-overlap-match counts. Not a disagreement abou**

- `runs/phase8_c1/ms02/replica_<scene>_exact_eval.json (bundle_dir = runs/phase8_c1/bundles_ms02/<scene>; the FROZEN MIN_SCORE=0.2 re-resolution; matches` → office_0: n_pred_instances 23, n_matched 21, entity matches@0.5 = 12/47 | room_1: 26, 26, 17/45 | room_2: 23, 22, 18/53
- `runs/phase8_c1/replica_<scene>_exact_eval.json (bundle_dir = notebooks/bundle_<scene>; the ORIGINAL Colab MIN_SCORE=0.4 run)` → office_0: n_pred_instances 20, n_matched 19, entity matches@0.5 = 12/47 | room_1: 22, 22, 15/45 | room_2: 21, 21, 17/53

**A-2 · The MIN_SCORE recorded for the frozen ms02 reference bundles**

- `docs/c1_closeout.md (tracked, commit a519a8f)` → 'Mask3D @ MIN_SCORE=0.2 / min_vertices=20' — the frozen operating point
- `runs/phase8_c1/ms02/replica_<scene>_exact_eval.json, field segmenter.config_params_json (untracked)` → top-level '"min_score": 0.4' — a grep for min_score in the frozen-0.2 report returns 0.4. A further stated fact, not offered as a resolution: runs/phase8_c1/bundles_ms02/<scene>/meta.json nests '"reresolved_locally": {"min_score": 0.2, "min_vertices": 20, "note": "operating-point re-resolution from 

**A-3 · How many office_0 predictions 'matched' oracle objects — three different counts circulate under the word 'matched', and the docs do not name the criterion at the point of use**

- `docs/c2_matched_labels_protocol.md (tracked, 4f38766) / eval/predictions/phase8_c2/replica_office_0_c2_labels.json (tracked, 4f38766), column headed '` → 16 — any-overlap greedy pairs restricted to entity classes, NO IoU threshold (criterion recovered from tools/c2_run.py, not stated in the doc). Denominator 47 entities. Room_1: 22; room_2: 21.
- `docs/c1_closeout.md (tracked, a519a8f) 'entity matches @IoU0.5' vs runs/phase8_c1/ms02/replica_office_0_exact_eval.json field n_matched` → 12/47 at IoU >= 0.50 (closeout), and separately 21/66 as n_matched = any-overlap greedy pairs INCLUDING structural classes over 66 oracle INSTANCES (exact_eval). Room_1: 17/45 and 26/55; room_2: 18/53 and 22/61.

**A-4 · room_2 Mask3D oracle-guided selection ceiling at IoU 0.50 — two ceiling numbers, measured differently, both stated**

- `docs/c1_composition_ceiling.md (tracked, 98b73eb), stage 0 'best single mask (selection)' row, Mask3D @0.2 column` → 0.38 (20/53) — per-entity best single mask over the saved raw masks
- `docs/c1_m2c_protocol.md (tracked, 98b73eb), 'Mask3D joint ceiling' row` → 19/53 — jointly compatible nomination materialized through the frozen resolver

### Area B

**B-1 · What 'Mask3D on Replica room_2' equals, when placed in a like-for-like column beside P1's 25/53 proposal ceiling. The two numbers are different quantities (viable-raw proposal ceiling vs delivered dense assignment), but a reader assembling a results table from these docs can take either as 'the Mask**

- `docs/c1_p1_multiview_proposals_protocol.md (line 38, commit f5737dd), agreeing with docs/c1_closeout.md line 71 and docs/c1_composition_ceiling.md lin` → 20/53 - 'Mask3D viable raw @0.5'
- `docs/arkitscenes_mask3d_contract.md (line 149, commit be668c3), in the table 'Normalising by each mechanism's own Replica rate under the same configur` → 18/53 = 34% - 'Mask3D ms02 | Replica room_2', set directly against 'P1 render-and-lift | Replica room_2 | 25/53 = 47%' in the row above it

**B-2 · Whether the pooled 33/53 on room_2 is attributed to P1 or to Mask3D. Arithmetically both statements are consistent with a per-entity max over the two banks, but they credit opposite arms and cannot both be used as a headline.**

- `docs/c1_p1_multiview_proposals_protocol.md line 234 and docs/results_narrative.md line 144 / docs/paper_draft.md line 136 (commits f5737dd, 99d3899)` → P1 raised entity viability from 20/53 to 33/53 (P1 contributes +13 newly viable entities over Mask3D)
- `docs/arkitscenes_mask_coverage_protocol.md line 340 (commit 49d51c0)` → Mask3D 'already contributed +8 entities on Replica (25/53 -> 33/53 pooled)' (Mask3D contributes +8 over P1 alone)

### Area C

**C-1 · Status of scenes 41069025 and 41069042: blinded-sealed vs human-inspected held-out transfer**

- `docs/arkitscenes_rgb_label_results.md — results table rows` → '41069025 (sealed)' and '41069042 (sealed)'; the qualitative audit cites eval/human_feedback/arkitscenes_sealed_visual_review_2026-08-09.json
- `docs/arkitscenes_rgb_label_results.md — Scope section, same file` → '41069025 and 41069042 are **human-inspected held-out transfer scenes**, not blinded sealed scenes — they were visually reviewed before this run.'

**C-2 · Exact OpenCLIP model tag used for the label stage**

- `docs/arkitscenes_rgb_label_results.md, 'What varied' section` → 'Same OpenCLIP ViT-B-32/openai weights'
- `segmenter/clip_labeler.py:17-18 (commit 5578fda), corroborated by docs/arkitscenes_grounding_bridge.md (commit d6baa42) and by the gitignored run mani` → MODEL_NAME = 'ViT-B-32-quickgelu'; PRETRAINED = 'openai' — grounding_bridge.md states 'ViT-B-32-quickgelu / openai' and every runs/arkit_label_image_ab*/*/entities/manifest.json records model='ViT-B-32-quickgelu'

**C-3 · Whether end-to-end QA has been run for these arms**

- `docs/arkitscenes_rgb_label_results.md — Limits section` → 'Matched-instance accuracy only. **End-to-end QA has not been run** — the RGB image source is not yet wired into tools/arkit_vertical_slice.py, which still labels from splats.'
- `docs/arkitscenes_rgb_label_results.md — second half of the same file, and tools/arkitscenes_e2e_qa_ab.py (commit c700df3)` → A full end-to-end graph-consistency QA table is reported for both arms across all three scenes ('Pooled abstentions 13/14 → 3/14'), produced by tools/arkitscenes_e2e_qa_ab.py

**C-4 · How the classification vocabulary is constructed**

- `docs/c2_matched_labels_protocol.md (commit 4f38766) — the predeclared labeler protocol this stage descends from` → '**Vocabulary**: the sorted set of object class_names of the SCENE (from info_semantic.json) ... a declared vocabulary leak'
- `docs/arkitscenes_rgb_label_results.md (c700df3) and extractors/learned_labels.py (bdfec68)` → 'same 41-class GLOBAL_INDOOR_VOCABULARY_V1' — a fixed, scene-independent 41-entry tuple hardcoded in extractors/learned_labels.py and verified to contain exactly 41 classes

### Area D

**D-1 · Direct (no-flip) median absolute mesh-vs-sensor-depth error for the corrected OpenCV camera convention on ARKitScenes 41069021**

- `docs/arkitscenes_rgb_label_results.md:104-107 (commit c700df3, 2026-08-10)` → 1.4-5.2 cm median error direct
- `extractors/arkitscenes_rgb_crops.py:49-54 (commit 7e87718, 2026-08-09) — restated as a range in docs/repair_arm_design_note.md:93 and segmenter/rgb_mu` → 4.1, 3.2 and 2.5 cm on frames 305.377 / 380.363 / 455.366 — i.e. range 2.5-4.1 cm

**D-2 · Erroneous flipped-axis ([1,-1,-1]) median absolute mesh-vs-sensor-depth error on ARKitScenes 41069021**

- `docs/arkitscenes_rgb_label_results.md:107 (commit c700df3)` → 23.8-68.3 cm flipped
- `extractors/arkitscenes_rgb_crops.py:52-53 (commit 7e87718) — restated as '37-98 cm' in docs/repair_arm_design_note.md:93 and segmenter/rgb_multiview_r` → 96.8, 98.2 and 36.9 cm on frames 305.377 / 380.363 / 455.366 — i.e. range 36.9-98.2 cm

**D-3 · Whether end-to-end QA had been run on the RGB label arm, stated twice inside one file**

- `docs/arkitscenes_rgb_label_results.md, 'Limits' section` → 'End-to-end QA has not been run - the RGB image source is not yet wired into tools/arkit_vertical_slice.py, which still labels from splats.'
- `docs/arkitscenes_rgb_label_results.md, '# End-to-end GRAPH-CONSISTENCY QA' section of the same file` → A full end-to-end table is reported (pooled abstentions 13/14 -> 3/14 across three scenes, 14 questions), produced by tools/arkitscenes_e2e_qa_ab.py

### Area E

**E-1 · E2 judgement for pair obj_11->obj_8 (ARKitScenes 41069025 support key)**

- `eval/human_feedback/arkitscenes_41069025_support_truth_returned.json (tracked, commit 4fc0827, status OWNER_REVIEWED)` → "supports"
- `eval/human_feedback/arkitscenes_41069025_support_relation_key_v1.json (tracked, commit 25126ec, status FINAL)` → "does_not_support" (source "owner_correction", superseded "supports", rationale "obj_8 is the ceiling-height slab and obj_11 is below it, so obj_11 cannot rest on it")

**E-2 · E2 judgement for pair obj_7->obj_20**

- `eval/human_feedback/arkitscenes_41069025_support_truth_returned.json (tracked, 4fc0827)` → "supports"
- `eval/human_feedback/arkitscenes_41069025_support_relation_key_v1.json (tracked, 25126ec)` → "does_not_support" (owner_correction, superseded "supports", rationale "vertically adjacent/overlapping, not a gravity-support pair")

**E-3 · E2 judgement for pair obj_32->obj_3**

- `eval/human_feedback/arkitscenes_41069025_support_truth_returned.json (tracked, 4fc0827)` → "supports"
- `eval/human_feedback/arkitscenes_41069025_support_relation_key_v1.json (tracked, 25126ec)` → "does_not_support" (owner_correction, superseded "supports", rationale "obj_32 is a tall vertical region, not resting on obj_3")

**E-4 · E2 presence/judgement of the two cushion pairs obj_9->obj_13 and obj_23->obj_13 - the pairs that carry the entire E2 recall denominator**

- `eval/human_feedback/arkitscenes_41069025_support_truth_returned.json (tracked, 4fc0827)` → ABSENT - neither pair appears in the returned truth (52 rows)
- `eval/human_feedback/arkitscenes_41069025_support_relation_key_v1.json (tracked, 25126ec)` → both "supports" (source "owner_correction"); the key's `coverage` block lists them as rows_omitted with reason "the sheet's Build JSON only emits rows with a checked radio... Omission here is a form defect, not a retraction". Corroborated by eval/human_feedback/arkitscenes_41069025_support_truth_cor

**E-5 · E2 total count of `supports` judgements for scene 41069025**

- `eval/human_feedback/arkitscenes_41069025_support_truth_returned.json (tracked, 4fc0827) - 52 rows` → 4 supports (obj_1->obj_14, obj_11->obj_8, obj_7->obj_20, obj_32->obj_3)
- `eval/human_feedback/arkitscenes_41069025_support_relation_key_v1.json (tracked, 25126ec) - 54 rows, judgement_counts` → 3 supports (obj_9->obj_13, obj_23->obj_13, obj_1->obj_14); judgement_counts {does_not_support 42, supports 3, unsure 9}. Only obj_1->obj_14 is a `supports` in BOTH files.

**E-6 · E1: what quantity the "0.58" transfer-precision figure on room_1 actually measures**

- `docs/semantics_v2_track_protocol.md (tracked, ef04997), echoed by docs/results_narrative.md and docs/paper_draft.md (tracked, 99d3899), which place "p` → 0.58 (room_1)
- `runs/semantics_v2/s2_report.json (GITIGNORED, untracked)` → room_1 A_v2 micro_precision 0.5806 (scene aggregate, all relations) vs room_1 A_v2 ON_ENTITY_SURFACE precision 0.5294 (9 hit / 17 cited). The quoted 0.58 is the SCENE AGGREGATE, not the support relation's own precision. On office_0 the two coincide (both 0.3636), which is why the pairing reads as co

**E-7 · E1: what the single figure "0.94" denotes for room_2 A-v2 - it is used for two different quantities in one sentence**

- `docs/semantics_v2_track_protocol.md (tracked, ef04997): "aggregate ... precision (0.94) PASS; support hits **16/20 @ P 0.94** PASS"` → 0.94 used for BOTH the scene aggregate micro-precision and the ON_ENTITY_SURFACE citation precision
- `runs/semantics_v2/s2_report.json (GITIGNORED, untracked)` → replica_room_2 A_v2 micro_precision = 0.9375; replica_room_2 A_v2 per_relation.ON_ENTITY_SURFACE.precision = 0.9412. Two distinct values that both round to 0.94.

**E-8 · E1: which scenes fail the declared all-scenes micro-precision floor of 0.80**

- `docs/semantics_v2_track_protocol.md S2 verdict prose (tracked, ef04997)` → "all-scenes precision floor FAIL (office_0 0.36, room_1 0.58)" - names two scenes
- `docs/semantics_v2_track_protocol.md S2 verdict TABLE (same file, same commit)` → room_0 A-v2 micro-P is 0.79, also below the 0.80 floor, but room_0 is not named in the prose. Reported as observed; not resolved here.

### Area F

**F-1 · Whether the blinded RGB and hybrid layers of the relation challenge were scored**

- `docs/arkitscenes_relation_challenge.md (commit b33abed), status line and Result table — 'Status: complete — all five layers scored 2026-08-17', with b` → scored, complete
- `docs/arkitscenes_relation_challenge.md (same file, same commit), section '### Still pending'` → 'The blinded RGB and hybrid layers' are still pending; 'the secondary subtest cannot run until that arm does'

**F-2 · Size of the relation-challenge thin-evidence slice (owner-recorded 0- or 1-view items among the 12 authored)**

- `docs/arkitscenes_relation_challenge.md (b33abed), section 'The sufficiency gate still could not be tested'` → 'Seven of twelve items landed in the thin-evidence slice (0 or 1 owner-recorded view)' — matches gitignored runs/arkit_relation_challenge/report.json thin_evidence_subtest/n_thin = 7
- `docs/arkitscenes_relation_challenge.md (same file, same commit), section '### Still pending', and the protocol section 'Six are flagged cross_view'` → 'Six of the twelve items carry an owner-recorded visibility of 0 or 1 view'

**F-3 · Which returned key lives at eval/human_feedback/arkitscenes_rgb_transfer_key_47331972.json**

- `docs/arkitscenes_rgb_transfer_test.md (c8e200d), Amendment 1 — 'The returned key is preserved at eval/human_feedback/arkitscenes_rgb_transfer_key_4733` → run-1 key: 8 items, 4 owner-ambiguous, n = 4 scored, questions hash 42a76fbc...
- `the tracked file itself at HEAD (last touched 45f8ec9, 'Record the run 2 key: 10 of 10 answered, none ambiguous')` → run-2 key: 10 items, 0 owner-ambiguous, n = 10 scored, questions hash 520074c2... — the run-1 key was overwritten at that path and survives only inside commit 6ee7715

**F-4 · Form label applied to the three non-co-visible items in the 47331972 transfer test (counts are identical: 3 items, 0 correct, 3 unanswered)**

- `docs/arkitscenes_rgb_transfer_test.md (c8e200d), per-form result table` → 'cross-view', n = 3
- `runs/arkit_rgb_transfer/47331972/score.json (GITIGNORED), by_form` → 'binary_near', n = 3 — the same three rows carry cross_view = true in score.json/rows, so the values agree but the form name printed in the tracked doc is not the name the scorer emits

**25 contradictions total.**

---

## Values that could not be verified

Each is written `—` wherever it would otherwise appear. **None was inferred.**

### Area A

- **Per-scene median matched IoU for office_0, room_1 and room_2 at the frozen Mask3D @0.2 operating point** — Not stated in any artifact, tracked or untracked. tools/c1_exact_eval.py (tracked, 97ff382) computes np.median over report['matches'] but only PRINTS it to stdout in main(); the value is never written into the JSON report, and no stdout log is committed. The only median figure anywhere is the four-scene range '0.80–0.89' in docs/c1_closeout.md. I did not compute per-scene medians from the raw match lists, per the no-
- **The definition behind the '0.80–0.89' median matched IoU range in docs/c1_closeout.md** — The doc does not say whether the median is taken over all any-overlap matches (which include structural classes) or over entity-only matches, does not attribute endpoints to scenes, and does not restate the operating point for that sentence.
- **Integer numerator for the aggregate uid micro-R vs the human key, for room_1 and room_2** — No artifact states the aggregate as a fraction — only the rounded ratios (0.1143, 0.2449). Per-relation n_hit integers exist only in the gitignored runs/mvp_v0/replica_<scene>_mvp.json; summing them would be reconstructing a number, so it is not reported as a numerator.
- **Aggregate denominator for room_1's uid micro-R as a single stated figure** — room_2's denominator (49) is explicitly stated in docs/c1_composition_ceiling.md, but no source states room_1's total. It is only decomposable into the four per-relation n_expected values (SUPPORTS_FLOOR 7, CONTACTS_SURFACE 6, ATTACHED_TO 8, ON_ENTITY_SURFACE 14) in eval/questions/phase8/replica_room_1_qa.json and runs/mvp_v0/replica_room_1_mvp.json.
- **office_0 uid micro-P vs the human key, as a number** — It is undefined, not zero: C1 cites zero members on office_0's only exhaustive relation family. docs/c1_p2_composer_protocol.md records it literally as 'null (0 cited)'; runs/mvp_v0/replica_office_0_mvp.json records micro_precision: null.
- **A tracked (committed) source for the predicted ENTITY count in the delivered C1 bundle (18 / 22 / 22 for office_0 / room_1 / room_2)** — The value appears only as variants.C1.n_entities in runs/mvp_v0/replica_<scene>_mvp.json, and its decomposition only in runs/phase8_c1/ms02/replica_<scene>_c1_run.json oracle_injection_summary. Both are under the gitignored runs/ tree; no doc in docs/ states these counts.
- **A tracked source for the any-overlap match count INCLUDING structural classes (21/66, 26/55, 22/61)** — Only runs/phase8_c1/ms02/replica_<scene>_exact_eval.json (gitignored) carries n_matched. Tracked sources define the criterion (tools/c1_exact_eval.py, docs/mesh_pipeline_contract.md) and explicitly warn against using it as detection recall, but never print the per-scene values.
- **office_0 QA metrics for SUPPORTS_FLOOR, CONTACTS_SURFACE and ATTACHED_TO against the human key** — office_0's human-verified key (eval/questions/phase8/replica_office_0_qa.json) contains no exhaustive rows for those relations — its only exhaustive family is ON_ENTITY_SURFACE (8 expected members). The metrics do not exist for that scene, rather than being missing.
- **Entity matches @ IoU 0.25 / 0.75 for any of the three scenes from a tracked source** — docs/c1_closeout.md publishes only the 0.50 column. The 0.25/0.75 counts exist only in the gitignored ms02 exact_eval reports.
- **Any deployable (no-oracle, no-human-key) QA number for these scenes/arm** — Every C1 QA row consumes oracle labels and oracle structural surfaces by construction — runs/phase8_c1/ms02/replica_<scene>_c1_run.json isolation_statement: 'labels and structural surfaces were INJECTED from oracle data for C1 isolation; only instance boundaries are learned'. docs/results_narrative.md lists 'a deployable raw-scene QA system' under 'Not demonstrated'. No row in this report is scope=deployable, and non

### Area B

- **A committed / git-tracked JSON report for ANY of the nine C1-P1 ceiling values.** — No tracked JSON in the repo contains them. `git ls-files eval/` returns no C1-P1 file, and `grep -rln 'c1p1|viable_pooled|viable_p1' eval/` returns nothing. The machine-generated reports runs/phase8_c1p1/replica_{room_2,office_0,room_1}_eval.json exist on disk and match every claimed value exactly, but `git check-ignore -v` confirms they are excluded by .gitignore line 12 ('runs/'). All nine values are therefore atte
- **P1-alone ceilings for office_0 (12/47) and room_1 (13/45) from the C1-P1 protocol itself.** — The Stage-2 verdict table (docs/c1_p1_multiview_proposals_protocol.md line 264-267) reports only pooled / baseline / delta / new-in-key / evidence / caps for the transfer scenes. P1-alone for those two scenes is stated only in docs/selector_v0_results.md (lines 77-78) and in the gitignored eval JSONs. The room_2 P1-alone 25 IS in the protocol (line 234).
- **Any delivered or deployable entity count produced from the P1 or pooled banks.** — None exists, by design. The protocol declares P1 'a proposal-ceiling experiment, not a deployable C1 replacement' (line 20-23) and the artifacts EVALUATION-ONLY (line 280). The successor experiment C1-P2 measured only an oracle-guided ceiling (31/53 materialized on room_2) and then STOPPED at Stage P2.0 - 'no composer is built' (docs/c1_p2_composer_protocol.md line 169), with Stages P2.1/P2.2 cancelled and budgets un
- **Independent verification that the SAM 2.1 checkpoint actually used matches the pinned sha256 2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318.** — The protocol states the weights are 'stored locally outside the repo' (line 337-339); the file is not in the working tree, so the hash cannot be recomputed here. The protocol asserts the notebook re-downloads and hard-fails on mismatch, and the Stage-1/2 verdicts state 'sha verified in-notebook', but that is the document's own claim, not an artifact I can check.
- **The number of Mask3D raw masks pooled in on office_0 and room_1 (room_2's 171 is stated).** — Only the pooled bank sizes (521, 569) and the P1 bank sizes (335, 406) are stated, in docs/selector_v0_results.md lines 84-86 and 77-78. Subtracting them would be my arithmetic, not a stated value, so I have not reported those counts as facts.
- **Whether the 40-view render / mask sidecar artifacts can be re-verified against their pinned hashes.** — docs/c1_artifact_manifest.json pins views_manifest / masks_sidecar / bank_npz sha256 per scene, but every referenced path is under runs/phase8_c1p1/ (gitignored), with authoritative copies stated to live on Google Drive. Verifying would require hashing untracked local files against the manifest, which is beyond a read-only check of tracked sources and was not requested.

### Area C

- **Numerators for the admission-precision column (top1-correct among admitted) in any tracked artifact** — docs/arkitscenes_rgb_label_results.md reports only the ratio (0.00 / 0.71 / 0.43 / 0.20 / 0.56 / 0.00 / 0.50) and the admitted count. The n_correct integers exist only in runs/arkit_label_image_ab*/*/label_eval.json, which is gitignored (.gitignore contains 'runs/'). They were not reconstructed by multiplication.
- **A pooled admitted count or pooled admission precision across the three scenes** — The pooled row of the results table leaves the 'admitted' and 'adm. precision' cells EMPTY. No source states a pooled value, and pooling them would require averaging, which is forbidden here.
- **A pooled 'matched instances out of annotated entities' figure (e.g. 21 of 44)** — Per-scene recovery is stated separately (7/18, 9/20, 5/6 in docs/arkit_vertical_slice_72h.md) and the pooled matched count 21 is stated in docs/arkitscenes_rgb_label_results.md, but no source states the pooled annotated total. Summing 18+20+6 would be reconstruction.
- **Count of vocabulary-ineligible instances among the 21 scored, and the vocab-eligible top-1/top-3 subtotals** — docs/arkitscenes_rgb_label_results.md raises the caveat ('classes absent from the 41-class vocabulary (notably curtain) cannot be scored correctly by any arm') but never quantifies it, and the tool computes elig_top1/elig_top3 without the doc reporting them. The gitignored runs/*/summary.json show vocab_eligible equal to matched in all three scenes (7, 9, 5), i.e. the caveat appears not to reduce the 21 denominator a
- **The weights sha256 actually used by this ARKit label A/B run, in a tracked artifact** — docs/arkitscenes_rgb_label_results.md does not record it. The full digest e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31 appears in tracked docs/arkitscenes_grounding_bridge.md but for the grounding-bridge reuse, and as the prefix 'e6d1bd7789aa4519…' in docs/c2_matched_labels_protocol.md for the Replica C2.0 runs. For this run it appears only in the gitignored entity manifests.
- **Human-ground-truth spatial QA for either arm** — Explicitly does not exist. tools/arkitscenes_e2e_qa_ab.py: 'A human-ground-truth answer key, built from annotations independently of the graph, does not exist yet and is what the repair arm will need.'
- **Any confidence interval, significance test, or variance estimate on the 1/21 → 12/21 change** — No source computes one. The only uncertainty statement in any source is qualitative and pre-declared in tools/arkitscenes_label_image_ab.py: 'n=7. This is screening evidence; 2/7 could be noise.'
- **rgb_context (the confound control) results on 41069025 and 41069042** — The arm was only run on the development scene. The doc's transfer reproduce line passes '--arms splat rgb_tight', and runs/arkit_label_image_ab_41069025/ and _41069042/ contain only splat and rgb_tight.
- **Deployable end-to-end performance of the rgb_tight labeler inside the product** — It is not wired in. docs/arkitscenes_rgb_label_results.md: 'the RGB image source is not yet wired into tools/arkit_vertical_slice.py, which still labels from splats.'

### Area D

- **Per-frame numbers and frame ids behind the 1.4-5.2 cm / 23.8-68.3 cm claim** — docs/arkitscenes_rgb_label_results.md:107 states only the two ranges. No tracked artifact breaks them down per frame, names the frames used, or names the scene. The only per-frame numbers anywhere in the repo are the 4.1/3.2/2.5 and 96.8/98.2/36.9 set in extractors/arkitscenes_rgb_crops.py, which is the contradicting measurement.
- **Denominator (pixel count) for any of the median depth-error figures** — extractors/arkitscenes_rgb_crops.py:53-54 says only 'across tens of thousands of depth pixels'. tools/arkitscenes_camera_alignment.py computes median_abs_error_m but no committed JSON report of its output exists; runs/ is gitignored and contains no camera-alignment or depth-alignment report directory.
- **Denominator for the 1878 usable frames (total raw 60 Hz lowres_wide frames in the dev stream)** — Neither docs/arkitscenes_rgb_label_results.md:108-110 nor extractors/arkitscenes_rgb_crops.py:210-214 states the raw frame count. A code comment at extractors/arkitscenes_rgb_crops.py:119 says '~1900 frames per instance', which is an approximation of the usable count, not the raw denominator.
- **The stride value in force when the buggy 122-frame count was produced** — Neither artifact states it. RgbCropSource's default is stride=6 and tools/arkitscenes_label_image_ab.py:79 uses stride=6, but no artifact says the 122 figure was measured at stride=6.
- **The scene the 122 -> 1878 figure was measured on** — docs/arkitscenes_rgb_label_results.md's bug list names no scene. extractors/arkitscenes_rgb_crops.py:58 says '1,878 exact-matched frames in this development stream' and tests/extractors/test_arkitscenes_rgb_crops.py pins DEV = .../Validation/41069021, but no artifact states the attribution explicitly, so it is not asserted as fact.
- **Per-fix attribution: WHICH earlier results were invalidated by the camera-convention fix versus by the stride fix** — docs/arkitscenes_rgb_label_results.md groups both under one heading, 'Two bugs that produced earlier invalid numbers', and never splits the invalidated set between them. extractors/arkitscenes_rgb_crops.py:64-66 attributes invalidation to the axis flip only ('all crop/label measurements made with the old axis flip are invalidated'), and segmenter/rgb_multiview_repair.py:52 says 'the bug that invalidated every earlier
- **An enumerated list of the specific earlier results invalidated by the camera-convention fix** — The invalidation is stated only as an unbounded class: 'all crop/label measurements made with the old axis flip' (extractors/arkitscenes_rgb_crops.py:64-65) and 'every earlier crop measurement' (segmenter/rgb_multiview_repair.py:52). No artifact lists the individual result ids, tables or documents covered.
- **WHICH invalidated results were rerun after each fix, and which were NOT rerun** — No artifact contains a rerun/not-rerun ledger for the ARKitScenes camera or stride fixes. docs/arkitscenes_rgb_label_results.md presents a post-fix A/B table but never states that it supersedes a specific earlier table, and never states that any invalidated result was left un-rerun. The closest in-artifact statements are: (a) 'Every splat row reproduces its committed baseline exactly, which is the evidence that the h
- **Whether any ARKitScenes result other than the label A/B (e.g. vertical slice, relation challenge, grounding bridge, repair arm) was rerun after the camera fix** — None of docs/arkit_vertical_slice_72h.md, docs/arkitscenes_relation_challenge.md, docs/arkitscenes_grounding_bridge.md or docs/repair_arm_design_note.md states that it was rerun because of the camera fix. docs/repair_arm_design_note.md:91-93 and segmenter/rgb_multiview_repair.py:46-52 only state that they IMPORT the corrected convention rather than restating it.
- **A committed JSON report for any of the camera/frame correction numbers** — runs/ and data/ are gitignored (.gitignore lines 12-13; confirmed with git check-ignore). The frame audit's raw output paths runs/frame_audit/frame_scale_audit.json, runs/frame_audit/tables.md and runs/frame_audit/<scene>_frame_estimate.json are gitignored, so docs/frame_and_scale_audit.md is the tracked primary source. No runs/ directory for the ARKitScenes camera-alignment depth measurement exists at all.
- **Any camera/frame correction content in docs/arkitscenes_mask3d_contract.md or docs/mesh_pipeline_contract.md** — Both were read in full. docs/arkitscenes_mask3d_contract.md (commit be668c3, 2026-08-08 — predates the camera fix) is a geometry-native path and states 'No rendered images anywhere in this path', so it neither reports nor is affected by the projection numbers; it records the mesh as 'gravity-aligned with up = +z (frame.kind="scene_canonical")' only as a fixed condition. docs/mesh_pipeline_contract.md (commit 5260bef,

### Area E

- **A tracked (committed) primary source for the E2 support-stage precision 1.0 and recall 0.3333333333333333** — Both values exist only in runs/arkit_support_calibration/arkitscenes_41069025_support_separability.json, which `git ls-files --error-unmatch` reports as untracked (runs/ is gitignored). `grep -rn "separability" docs/` and `grep -rn "evidence_probe" docs/` return NO matches, so there is no narrative doc in the repo that reports these numbers either. The instruction's fallback (cite the narrative doc that reports the r
- **A tracked primary source for the E2 segmentation-evidence verdict on obj_1->obj_14 (owner coverage 0.0494 vs owner_coverage_min 0.25; unassigned 0.4475; delivered_partition_unassigned_rate 0.5973)** — Only in runs/arkit_support_calibration/arkitscenes_41069025_obj_1_obj_14_evidence_probe.json (gitignored). The verdict LABEL "segmentation_evidence_missing" and its definition are tracked in tools/arkitscenes_support_evidence_probe.py (commit 58faaea), but the measured coverage numbers are not.
- **A tracked primary source for the semantics-v2 per-relation ON_ENTITY_SURFACE precision on the transfer scenes (room_1 9/17 = 0.5294; office_0 4/11 = 0.3636) and for the exact room_2 support precision 16/17 = 0.9412** — Only in runs/semantics_v2/s2_report.json (gitignored). The tracked docs report rounded scene-aggregate micro-P figures without numerators or denominators.
- **Any deployable (non-oracle) semantics-v2 support number** — The semantics-v2 S2 stage ran variant A only, and variant A "uses the dataset's oracle boxes and labels" (docs/paper_draft.md). The doc calls S2 "the new representation ceiling". Stage S3 (learned variants B / C1 / the pooled-bank ceiling under v2) was cancelled: "STOP_TRACK per the frozen rule... S3 is cancelled unspent." So 5/20 -> 16/20 has no deployable counterpart anywhere in the repo.
- **Which specific entities/uids make up room_2's 16 semantics-v2 support hits (or the 5 v1 hits)** — Neither docs/semantics_v2_track_protocol.md nor docs/results_narrative.md nor docs/paper_draft.md enumerates them, and the gitignored runs/semantics_v2/s2_report.json records only counts (n_cited / n_expected / n_hit), not uids. By contrast the ATTACHED_TO census does name uids (obj_42, obj_59).
- **An exhaustive scene-level support precision or recall for ARKitScenes 41069025** — The human key judges 54 of the scene's 1225 pairs (n_evaluated 714), and the 54 rows were selected by a system-biased rule (36 qualifying-patch rows + 29 stratified rejects + 2 owner-confirmed), not a random draw. The key's 3 positives are therefore a lower bound on the scene's true support relations, and E2's 1/3 cannot be extrapolated to the scene.
- **Whether moving the overlap gate from 0.50 to <= 0.4545 would actually recover obj_23->obj_13 without admitting false positives** — The separability report declines to answer and states so explicitly: "3 positives from one scene. A boundary that separates them is a hypothesis, not a calibration; a gate fitted to this would be fitted to two or three points." It records separable_in_2d=true with negatives_inside_that_box=[], but that is a two-point hypothesis, not a measured result. No threshold was changed (logic_changed=false, thresholds_changed=
- **Any combined / averaged E1+E2 support figure** — Refused by construction. E1 is Replica, oracle-boxed variant A, citation hits against QA answer keys under a CHANGED system definition (denominators 20, 14, 23). E2 is ARKitScenes 41069025, delivered segmentation, pair-level gravity-rest judgements against a human pair key (denominators 3, 1, 54, 714). Different datasets, different definitions, different denominators - they must never share a comparison_group or be p
- **A resolved reading of the returned-truth vs FINAL-key disagreements (which owner judgement is correct)** — Both artifacts are tracked and both are owner-attributed (returned = OWNER_REVIEWED; key = FINAL with corrections dated 2026-08-10). Per the task's rule, both are reported in contradictions and no winner is picked here.
- **A precision figure for E1's room_2 ATTACHED_TO under semantics-v2** — With 0 citations there is no denominator; the gitignored run records precision: null for replica_room_2 A_v2 ATTACHED_TO, and the tracked doc says only "0/14, zero citations".

### Area F

- **geometry_relation_ceiling coverage (0.800) and evidence_aware_hybrid coverage (0.900) for the relation challenge, from a TRACKED artifact** — No tracked doc states coverage for these two arms. The tracked relation-challenge doc gives only the tally (7/1/2 excl 2 and 7/2/1 excl 2) and accuracy 0.700; docs/direct_rgb_product_path.md publishes coverage for four arms only (RGB 0.90, stored graph 0.80, grounded 0.20, delivered 0.00) and omits these two. Both values were read only from GITIGNORED runs/arkit_relation_challenge/report.json and are recorded above w
- **Answered-item counts (the numerators of every coverage figure) as explicit integers** — No artifact — tracked or gitignored — stores 'answered' as its own field; only correct/wrong/unanswered tallies and the coverage ratio are stored. Deriving answered = correct + wrong would be reconstruction, so coverage numerators are recorded as '—' except where the numerator equals a directly stated count (grounded arm 2, transfer 5 correct + 0 wrong).
- **Explicit deployable / not-deployable labels for the four representation-kill-test arms** — docs/arkitscenes_representation_kill_test.md carries no deployable column and its gitignored report has no deployable or layer_kind field (unlike the relation-challenge report). The scopes assigned above are inferred ONLY from the documented facts that no kill-test arm consumes a human key or oracle identity and that the direct arm was blinded; the doc itself makes no such claim.
- **n_excluded_no_human_answer for the grounded_delivered_graph arm from a tracked artifact** — docs/arkitscenes_grounding_bridge.md states only correct/wrong/unanswered = 2/0/8 for that arm; the 'excl 2' figure for it appears only in GITIGNORED runs/arkit_relation_challenge/report.json. (The tracked relation-challenge doc does state excl 2 for the other five arms.)
- **The denominator convention behind 'false-confident rate' in both the kill test and the transfer test** — Neither tracked doc defines it, and neither the gitignored kill-test report nor the gitignored transfer score.json stores a numerator/denominator pair for it — only the scalar rate.
- **Any transfer-test comparison against the graph path on 47331972** — No graph, grounding or segmentation stage ran on 47331972 (stated in both docs/arkitscenes_rgb_transfer_test.md and the gitignored score.json limits). Only the direct multiview RGB arm exists for that scene, so no per-arm comparison is available and none may be constructed.

**52 unverifiable values total.**

---

## Reconciliation pass

Applied 2026-08-24 from `58af9cd`. **No experiment was run and no system behaviour
changed.** One deterministic diagnostic was re-measured with an existing tracked tool
— see D below — and every other change is a document correction or a scope label.

### Reclassification of all 25 reported items

| id | class | disposition |
|---|---|---|
| A-1 | superseded artifact | original 0.4 Colab reports superseded by frozen `ms02`; precedence below |
| A-2 | different operating point | inference threshold vs local re-resolution; relationship below |
| A-3 | different metric | three distinct match criteria, already separate registry rows |
| A-4 | different metric | best-single-mask vs jointly compatible; relationship below |
| B-1 | different metric | viable-raw proposal ceiling vs delivered dense assignment |
| B-2 | different metric | two decompositions of one pooled max; relationship below |
| C-1 | stale prose | **resolved** — table said *sealed*, Scope said *held-out*; now *held-out* |
| C-2 | stale prose | **resolved** — model tag corrected to `ViT-B-32-quickgelu` |
| C-3 | stale prose | **resolved** — the flat *E2E has not been run* claim withdrawn |
| C-4 | different metric | two different pipelines: Replica C2 scene vocabulary vs ARKit 41-class global |
| D-1 | missing tracked evidence | **resolved** — unprovenanced range withdrawn, authoritative report created |
| D-2 | missing tracked evidence | **resolved** — same |
| D-3 | stale prose | **resolved** — duplicate of C-3 |
| E-1 – E-5 | superseded artifact | returned review form superseded by the FINAL owner-corrected key; precedence below |
| E-6 | stale prose | **resolved** — `0.58` relabelled as scene aggregate, not support precision |
| E-7 | stale prose | **resolved** — the two quantities that both round to `0.94` separated |
| E-8 | stale prose | **resolved** — room_0 at 0.79 added to the floor-failure list |
| F-1 | stale prose | **resolved** — the *Still pending* section withdrawn |
| F-2 | stale prose | **resolved** — thin-evidence slice is **seven**, matching the report |
| F-3 | stale prose | **resolved** — the run-1 key is in commit `6ee7715`, not at that path |
| F-4 | stale prose | **resolved** — doc's *cross-view* vs scorer's `binary_near`; counts identical |

### The five pairs that are not contradictions — relationship and precedence

**A-2 · inference threshold 0.4 vs local re-resolution 0.2.** Two different stages of one
pipeline. Mask3D *inference* ran at `min_score = 0.4`, which is what
`segmenter.config_params_json` records; the delivered bundle was then *locally
re-resolved* at `0.2`, which is what `docs/c1_closeout.md` records and what
`meta.json.reresolved_locally` confirms. **Precedence:** for any delivered-instance
claim, `0.2` is the operating point. Both numbers are correct about different stages.

**A-4 · best-single-mask 20/53 vs jointly compatible 19/53.** Two different selection
ceilings on the same proposals. `20/53` allows the best single mask per entity
independently; `19/53` requires a jointly compatible nomination materialised through the
frozen resolver. **Precedence:** quote `20/53` for what the proposals contain, `19/53`
for what a resolver can simultaneously deliver. Never as a before/after.

**B-2 · P1's +13 vs Mask3D's +8.** Both describe the same pooled `33/53` as a per-entity
max over two banks. From Mask3D's `20/53`, P1 adds 13. From P1's `25/53`, Mask3D adds 8.
**Precedence:** the pooled figure is the only headline; neither increment may be quoted
as one arm's contribution without naming the baseline it is measured from.

**A-1 · original 0.4 reports vs frozen `ms02` reports.** Two report sets exist per scene
under `runs/phase8_c1/`. **Precedence: the `ms02` set supersedes the original.** It is
the frozen reference bundle every downstream artifact is pinned to, and it is the set
copied into the evidence pack. The original set is historical and must not be cited.

**E-1 – E-5 · returned review forms vs the FINAL owner-corrected key.** The returned form
is the raw review capture; the key is the owner's corrected, finalised judgement, with
each change carrying `source: owner_correction`, the superseded value, and a rationale.
**Precedence: the FINAL key governs all E2 numbers.** One residual evidentiary caveat,
recorded rather than resolved: the two cushion pairs carrying the entire E2 recall
denominator are absent from the returned form and present only in the key, which
attributes the omission to a form defect (*“the sheet's Build JSON only emits rows with
a checked radio”*). The 1/3 headline therefore rests on pairs with no independent
second capture.

### D · one authoritative camera-convention measurement

The two circulating ranges were **not** adjudicated by choosing between them, because
neither source recorded frames, masks and aggregation for both. Instead one authoritative
measurement was produced with the already-tracked tool
`tools/arkitscenes_camera_alignment.depth_alignment_metrics` — the same call the
dataset-guarded regression executes on every suite run — and recorded at
`eval/results/project_census_v1/camera_convention_depth_diagnostic.json`.

| | direct (OpenCV) | legacy `[1,-1,-1]` flip |
|---|---|---|
| authoritative measurement, 3 named frames | **2.5 – 4.1 cm** | **36.9 – 98.2 cm** |
| `extractors/arkitscenes_rgb_crops.py` docstring | 2.5 – 4.1 cm | 36.9 – 98.2 cm |
| `docs/arkitscenes_rgb_label_results.md`, before this pass | 1.4 – 5.2 cm | 23.8 – 68.3 cm |

Frames 305.377 / 380.363 / 455.366 on `41069021`, exact pose match within 0.001 s,
median absolute error over common pixels, no instance mask. The docstring values
reproduce exactly. **The document's ranges do not**, and that document recorded no frame
set, scene or aggregation, so they are classified as *missing tracked evidence* and
withdrawn rather than reconciled — they may have measured a different set, and nothing
tracked establishes what. The regression pins only *bounds* (direct ≤ 0.06 m, legacy ≥ 4×
direct), which both ranges satisfy, so the test never adjudicated them.

### Evidence pack

`eval/results/project_census_v1/` — **10** numeric JSON reports, each with its original
path, original sha256 and producing commit in `MANIFEST.json`. **No raw masks, meshes,
images, point clouds or dataset files, and not the `runs/` tree.** Large arrays are elided
and geometry keys dropped; the manifest records the sanitisation applied. Most originals
live under the gitignored `runs/` tree, which is why they are copied here at all.

The two ARKit label A/B summaries backing the chosen headline were added in the
documentation pass, and `MANIFEST.json` records two **known gaps** alongside them:

- **`41069021` has no summary in this working copy.** Its contribution to the pooled
  headline (`0/7 → 5/7` top-1, `0/7 → 7/7` top-3) rests on the tracked narrative alone.
  So `1/21 → 12/21` is reproducible **in part, not in whole**, from this pack: two of the
  three pooled scenes have a machine report here.
- **The `rgb_context` control has no machine report anywhere in this working copy.** The
  two summaries present carry only `splat` and `rgb_tight`. The control that *supports*
  the texture-not-room-gist interpretation `[C17]`, `[C18]`, `[C21]` therefore rests on the
  tracked narrative alone.

The camera diagnostic now records sha256 for its mesh, all three depth frames, the pose
trajectory and all three intrinsics files, and separates
`measurement_repository_state` (`58af9cd`, where the numbers were computed) from
`artifact_commit` (`fdb63a5`, where the file first landed) — a file cannot record the
commit that contains it.

### Follow-up documentation pass

Three audit defects found after the reconciliation commit and fixed here, all
documentation-only:

1. The strongest-result table still called `[C01→C02]` `deployable` and described it as
   *“simultaneously deployable in the answer path”*, contradicting its own corrected
   scope column. Both are corrected and the claim is withdrawn.
2. The `[C02]` registry note still carried stale prose beginning *“Scope='deployable'
   because…”*. Removed; no registry note now contradicts its scope column.
3. The evidence pack omitted the machine reports behind the chosen headline. Two of the
   three were added; the third does not exist in this working copy and is recorded as a
   known gap rather than left implicit.

### Genuine unresolved contradictions remaining: **0**

Of 25 reported items: 11 resolved as stale prose or missing evidence and corrected in
the source documents; 14 reclassified as different metrics, different operating points,
or superseded artifacts, each with its relationship and precedence written down.

This is not a claim that the evidence base is complete. **52 unverifiable values remain
unchanged** and are listed above; 28 registry rows still cite untracked primary sources,
now reduced in practice for the 7 reports copied into the evidence pack; and the E-4/E-5
provenance caveat above is recorded, not closed.

---

## Consistency checks run

| check | result |
|---|---|
| every row carries a scope from the six-value enum | pass — enforced at build time, 271/271 |
| numerator ÷ denominator agrees with the stated value | pass — 0 mismatches among rows with integer numerator and denominator |
| no row combines two denominators | pass by construction — one row, one denominator |
| the nine B ceiling values match the figures given in the brief | pass — `[B01]`–`[B09]` verify exactly |
| the four C pooled values match the brief | pass — `[C01]`–`[C04]` verify exactly; per-scene rows sum to pooled |
| the 47331972 transfer values match the brief | pass — `[F76]`–`[F79]`, `[F82]`, `[F83]` verify exactly |
| no `deployable` row is an oracle or ceiling arm | 8 candidates flagged by pattern, 7 were false positives on the string *oracle-free*; 1 genuine, see below |
| ceiling and oracle rows never labelled deployable | pass — 26 `proposal_ceiling` and 45 `identity_oracle` rows, none marked deployable |

**The one genuine scope flag: `[F62]`.** It is labelled `deployable` and reports
*RGB-unique wins over structure*, but the comparison is against the two
`identity_oracle` layers. The measured arm is deployable; the baseline it beats is not.
Its counterpart `[F61]` is correctly labelled `identity_oracle`. Both rows are marked
DIAGNOSTIC ONLY in the source and neither counts toward any gate. **Flagged, not
silently relabelled.**

---

## Strongest-result candidates

Several results compete; they are listed so the choice is visible rather than implied.

| candidate | value | scope | why it might win | why it might not |
|---|---|---|---|---|
| Labeler input A/B `[C01→C02]` | 1/21 → 12/21 top-1 | `oracle_free_component_eval` | one changed variable, three scenes, no tuning between them, ~12× effect | **not end-to-end**: the denominator is oracle-matched instances, so it says nothing about detection |
| Labeler input A/B `[C03→C04]` | 4/21 → 18/21 top-3 | `oracle_free_component_eval` | same design, larger absolute effect | **not end-to-end**, same denominator caveat; top-3 is a weaker criterion than top-1 |
| Graph-consistency QA `[C32→C33]` | 13/14 → 3/14 abstentions | deployable | end-to-end, makes the graph askable | scored against the graph's own neighbours, so it cannot show they are spatially correct |
| Pooled proposal ceiling `[B01→B03]` | 20/53 → 33/53 | **proposal_ceiling** | largest headline jump in the project | an oracle-evaluated bound, ineligible as a deployable result |
| Stored-edge replay agreement `[F63]` | 12/12 | **identity_oracle** | cleanly isolates relation extraction | a bound, and a null result rather than a gain |
| Unseen-scene calibration `[F85]`, `[F86]` | 1.000 when answered, 0 false-confident | `deployable` (end-to-end) | perfect precision on a never-seen room | only 5 answers, and both gates failed `[F82]`, `[F83]` |

**Chosen: `[C01→C02]`.** It isolates a single changed variable, spans three scenes with
no tuning between them, and a context control in the opposite direction `[C07→C17]`
supports the interpretation that the gain comes from object texture rather than room gist.

**It is not a deployable end-to-end result and must never be quoted as one.** Its scope is
`oracle_free_component_eval`: the prediction path uses no oracle, but the denominator is
instances the evaluator had already matched to an annotation box, so the number is
conditional on detection having succeeded and says nothing about detection itself. An
earlier revision of this section called it *“simultaneously deployable in the answer
path”*, which contradicted its own corrected scope column; that claim is withdrawn.

The strongest **end-to-end deployable** evidence in the project is a different and much
weaker set: the unseen-scene transfer result `[F76]`, `[F82]`, `[F83]`, which failed both
of its predeclared gates. The ceiling result `[B01→B03]` is numerically larger than either
but is an oracle-evaluated bound and ineligible for any headline.

---

## Five sentences

Revised in the reconciliation pass; the context control is described as *supporting* the
interpretation rather than confirming it.

**Problem:** Answering a spatial question about a real handheld room capture requires
resolving a natural-language object reference to a delivered 3D instance and then reading
a relation off that instance, and this project measured those stages separately rather
than collapsing them into one score `[A05]`, `[F40]`, `[F67]`.

**Existing limitation:** On the two ARKitScenes rooms the delivered structured arm
answered 0 of 10 relation questions `[F40]` and an oracle-free grounding bridge raised
that only to 2 of 10 `[F45]`, while the same stored relations under human-supplied
identity answered 7 of 10 `[F35]`.

**Method:** Each experiment changed exactly one variable — labeler input images
`[C01→C02]`, proposal source `[B01→B03]`, relation definition `[E01→E02]`, identity source
`[F40→F35]` and `[F40→F45]`, or scene `[F50→F76]` — and each arm is scoped so that an
identity oracle `[F28]`, `[F35]`, a proposal ceiling `[B01]`, and an oracle-free component
evaluation `[C02]` are never quoted as end-to-end deployable performance.

**Strongest result:** Replacing only the labeler's input images, with OpenCLIP weights,
vocabulary, admission threshold, evaluator, delivered partitions and IoU matching all held
fixed, raised pooled matched-instance top-1 from 1/21 to 12/21 and top-3 from 4/21 to
18/21 across three scenes with no tuning between them `[C01]`, `[C02]`, `[C03]`, `[C04]`,
and a context control in the opposite direction supports the interpretation that the gain
comes from object texture rather than room gist `[C07]`, `[C17]` — noting this is an
oracle-free component evaluation on an oracle-selected denominator, not end-to-end
performance.

**Meaning:** Correct relational information is present in the representation and
unreachable through the deployed identity stage `[F35]`, `[F40]`, `[F63]`, `[F64]`, and
grounding did not close that gap `[F45]`, `[F67]`, `[F68]`, `[F69]`; on a previously
unseen room the direct-RGB path answered 5 of 10 with zero wrong answers and 0 of 3
cross-view items `[F76]`, `[F77]`, `[F79]`, so neither path yet answers non-co-visible
spatial questions deployably.

## On the proposed interpretation

The interpretation offered for testing was:

> correct spatial information can exist in the 3D representation while remaining
> unreachable because object delivery and identity grounding fail; direct RGB improves
> visible-object naming but does not yet solve non-co-visible cross-view QA.

**Adopted in part.** Clause by clause against the registry:

| clause | verdict | evidence |
|---|---|---|
| information exists in the representation but is unreachable | **supported** | `[F35]` 7/10 under human identity vs `[F40]` 0/10 delivered; `[F63]` 12/12 replay agreement; `[F64]` 7/12 ceiling-correct-delivered-unanswered |
| …because **identity grounding** fails | **supported** | `[F67]`, `[F68]`, `[F69]` all three gates fail; `[F45]` grounded arm reaches 2/10 |
| …because **object delivery** fails | **partly — wrong dataset** | supported on Replica `[A05]`, `[A25]`, `[A48]`, `[A12]`, `[A32]`, `[A55]`; **no registry row records absent instances on the ARKit scenes where the relation QA was actually run** |
| direct RGB improves visible-object naming | **supported** | `[C01]`→`[C02]`, `[C03]`→`[C04]`, with control `[C07]`→`[C17]` |
| …but does not solve non-co-visible cross-view QA | **supported on one scene only** | `[F79]` 0/3 cross-view on 47331972; **no registry row breaks out cross-view performance within the relation challenge**, so the claim is not established on 41069025/41069042 |

The two gaps are recorded as unverifiable rather than filled by inference. Closing them
would need a delivered-instance audit against a human inventory on the ARKit scenes, and
a per-form breakdown of the relation-challenge RGB arm — neither exists in a committed
artifact today.

---

## Standing cautions this census does not resolve

- **Two exact-eval report sets exist** under `runs/phase8_c1/` with different numbers for
  the same scenes (A-1). Table 1 uses the `ms02` set throughout; the other set is not
  merged in, and which is authoritative is not settled here.
- **The frozen bundles' `min_score` is recorded as 0.2 in prose and 0.4 in the JSON**
  (A-2). Every Table 1 row inherits that ambiguity.
- **The ARKit support key does not match the returned owner truth** on five pairs (E-1
  to E-5), including the two cushion pairs that carry the headline 1/3.
- **`docs/arkitscenes_relation_challenge.md` contradicts itself** on whether the blinded
  RGB and hybrid layers were scored, and on whether the thin-evidence slice is six or
  seven items (F-1, F-2).
- **`docs/arkitscenes_rgb_transfer_test.md` Amendment 1 describes the run-1 key** at the
  path that now holds the run-2 key (F-3).

No experiment was run and no artifact outside these two files was modified.

