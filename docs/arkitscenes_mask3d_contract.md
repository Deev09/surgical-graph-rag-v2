# ARKitScenes × Mask3D — execution contract

Status: **EXECUTED 2026-08-08. STRONG PASS — 7/18 at IoU 0.50**, against a
6/18 strong gate and a 1/18 baseline, from 37 masks rather than 1,733
proposals. Verdict at the bottom; the contract above it is unedited.

Deliberately a contract, not a protocol: one dev scene, one configuration,
no sweep, fixed gates.

Follows `docs/arkitscenes_mask_coverage_protocol.md`, whose verdict closed
the render-and-lift line after four measured failures totalling +1 entity.

## The claim, stated at its actual size

This tests whether **geometry-native proposals transfer better than
render-and-lift on one real scan**, evaluated against vertex sets derived
from ARKitScenes annotation boxes.

It does not establish generalization. It is one scene, one oracle
construction, one checkpoint. A pass licenses the two sealed transfer scenes
and the pooled selection/NMS work — nothing further.

## Fixed conditions

| | |
|---|---|
| scene | `41069021` (dev) only |
| geometry | canonical mesh directly — `41069021_3dod_mesh_canonical.ply`, sha256 `ec219f56c1f9d79a1…`, 1,008,964 vertices. **No rendered images anywhere in this path.** |
| model | frozen OpenMask3D class-agnostic mask module, commit `3bc3fc52693b`, checkpoint sha256 `da4b68cb52c7f204…` |
| upstream params | `num_queries=150`, `use_dbscan=true`, `dbscan_eps=0.95` — untouched |
| resolve config | `MIN_SCORE=0.2`, `min_vertices=20` |
| sweep | none |
| sealed | `41069025`, `41069042` stay sealed unless a pass |

Vertex indexing is shared by construction: the canonical ply, the adapter's
`load_canonical_geometry`, the P1 bank (`n_vertices=1008964`) and the box
oracle all index the same 1,008,964 vertices, which is what makes the pooled
bank a legitimate union rather than two incompatible index spaces.

The mesh is gravity-aligned with up = +z (`frame.kind="scene_canonical"`,
z-extent 3.17 m = room height). ScanNet is z-up too, so this removes an
orientation confound rather than introducing one.

### `MIN_SCORE=0.2` resolved against the +8 reference — read this before comparing

The contract names Replica's `MIN_SCORE=0.2`. The Replica **+8 pooled**
figure did not use it: `p1_selector_eval.load_mask3d()` reads
`raw_masks.npz` and ignores scores entirely. These are different banks, and
on room_2 the difference is 171 masks versus 27.

Measured before running anything, so the gate is interpretable:

| Replica room_2 bank | K | ceiling @0.10 / 0.25 / 0.50 |
|---|---|---|
| Mask3D raw (source of the +8) | 171 | 39 / 31 / **20** |
| Mask3D `ms02` (this contract) | 27 | 28 / 26 / **18** |
| P1 render-and-lift | 534 | 40 / 33 / **25** |
| pooled P1 + raw | 705 | 46 / 42 / **33** |
| pooled P1 + `ms02` | 561 | 44 / 40 / **32** |

`MIN_SCORE=0.2` costs 2 entities while discarding 144 masks, so the
contract's configuration is a near-equivalent and much higher-precision
bank. It is **gated**; the raw bank is **reported alongside, not gated**, so
the +8 lineage stays comparable. Same reported-not-gated pattern as M1's
Replica anchor.

## Why 4/18 and 6/18 are the right numbers

Mask3D alone recovers **18/53 = 34%** of Replica entities at IoU 0.50 under
this exact configuration. Applied to ARKitScenes' 18 entities:

| gate | entities | = share | reading |
|---|---|---|---|
| stop | ≤3/18 | ≤17% | below half the Replica rate |
| **minimum pass** | **≥4/18** | 22% | ~⅔ of the Replica rate |
| **strong pass** | **≥6/18** | 33% | ~the full Replica rate |

For contrast, render-and-lift transferred at **1/18 = 5.6%** against its own
Replica rate of 25/53 = 47% — roughly an eighth. That is the failure this
experiment is trying to beat, and 2/18 under M1 is why clearing 1/18 was
explicitly rejected as a success criterion.

## Reported metrics

For each of the three banks — P1, Mask3D, pooled:

* ceiling at IoU **0.10 / 0.25 / 0.50**
* AR@**25 / 50 / 100 / 200**
* proposal count
* giant-mask rate (share of proposals > 15% of mesh vertices)
* zero-overlap rate (share with max IoU < 0.10 against any entity)

## Decision

| outcome | consequence |
|---|---|
| Mask3D ≥6/18 @0.50 | strong pass — unlock sealed scenes and pooled selection/NMS |
| Mask3D ≥4/18 @0.50 | pass — same unlock, weaker claim |
| Mask3D ≤3/18 @0.50 | **STOP.** Do not tune Mask3D on the dev scene. Geometry-native proposals do not transfer here either, and the finding is that the difficulty is the scan, not the proposal modality. |

Giant-mask rate and zero-overlap rate are **diagnostic, not gates** — they
explain a result, they do not license one. A pass earned entirely by
oversized blobs is reported as such.

## Out of scope

Parameter tuning of any kind, the two sealed scenes, additional datasets,
NMS or selection over the pooled bank, and any change to the P1 artifacts —
which stay byte-identical throughout.

---

# VERDICT — strong pass, 2026-08-08

One run, A100, 64.5 s. Provenance verified against the contract before
evaluation: mesh sha256 `ec219f56c1f9d79a1…`, 1,008,964 vertices, checkpoint
sha256 `da4b68cb52c7f204…`, `num_queries=150`, `use_dbscan=true`,
`dbscan_eps=0.95`. 182 raw masks → **37** at `MIN_SCORE=0.2` /
`min_vertices=20`.

| bank | K | @0.10 | @0.25 | **@0.50** | AR@25 | AR@100 | giant | zero-overlap |
|---|---|---|---|---|---|---|---|---|
| P1 render-and-lift | 1733 | 12 | 7 | **1** | 1 | 1 | 0.0% | 98.7% |
| **Mask3D `ms02`** (gated) | **37** | 11 | 11 | **7** | 4 | 7 | 0.0% | 62.2% |
| Mask3D raw (reported) | 182 | 16 | 14 | **7** | 3 | 4 | 0.0% | 64.8% |
| pooled P1 + `ms02` | 1770 | 14 | 13 | **7** | 1 | 3 | 0.0% | 98.0% |
| pooled P1 + raw | 1915 | 18 | 15 | **7** | 1 | 2 | 0.0% | 95.5% |

**Gate: 7/18 ≥ 6/18 — strong pass.** Sealed scenes and the pooled
selection/NMS work are unlocked.

The diagnostics say the pass is real rather than an artifact of bank
inflation: **giant-mask rate is 0.0%** in every bank, so nothing was earned
by oversized blobs, and zero-overlap fell from 98.7% to 62.2% — the bank
got 47× smaller and dramatically more precise at the same time.

## The result the M1 verdict was set up to interpret

M1 closed by naming the two readings its STOP outcome would have supported:
that "the difficulty is the scan, not the proposal modality." It did not
stop. It passed at the top band, so **the opposite holds — the modality was
the problem.**

Normalising by each mechanism's own Replica rate under the same
configuration:

| mechanism | Replica room_2 | ARKitScenes 41069021 | transfer |
|---|---|---|---|
| P1 render-and-lift | 25/53 = 47% | 1/18 = 5.6% | **12%** of its own rate |
| Mask3D `ms02` | 18/53 = 34% | 7/18 = 39% | **115%** of its own rate |

Mask3D does not merely survive the move to a real scan; it performs
marginally *better* here than on Replica. Render-and-lift retains an eighth
of its ability. That retroactively explains all four refuted protocols at
once: F1, R1 arm A, R1 arm C and M1 were each tuning inputs to a mechanism
that was itself the wrong mechanism for this data. Closing every render
statistic moved the ceiling by one entity because no render statistic was
ever the binding constraint.

Note the claim this does **not** license. One scene, one oracle
construction (annotation-box-derived vertex sets), one checkpoint. The
sealed scenes are the transfer test, and they remain sealed until run.

## Pooling is not additive at IoU 0.50, and that is measured

| IoU | P1 | Mask3D | both | P1-only | Mask3D-only | union |
|---|---|---|---|---|---|---|
| 0.50 | 1 | 7 | 1 | **0** | 6 | 7 |
| 0.25 | 7 | 11 | 5 | 2 | 6 | 13 |
| 0.10 | 12 | 11 | 9 | 3 | 2 | 14 |

At 0.50 P1's single entity is a strict subset of Mask3D's seven — **P1
contributes nothing the geometry-native bank does not already have.** The
two are complementary only at loose overlap (union 13 at 0.25, 14 at 0.10),
which is a statement about coarse localisation, not about usable instances.

## A regression the pooled banks expose in the selector

Mask3D alone reaches AR@25 = 4. Pooled with P1 it drops to **AR@25 = 1**,
and AR@100 falls from 7 to 3. The ceiling is unchanged at 7, so the entities
are present in the pooled bank and the ranker is burying them: the
oracle-free selector prefers 1,733 render-and-lift proposals over 37 sharply
better ones.

This is now the binding constraint on the pooled path and it is a
**selector** problem, not a proposal problem. Fixing it is the first item
the unlock makes available.

## What is still missed, and it is one class

Recovered @0.50: cabinet, chair, sofa, table ×3, tv_monitor.
Missed (11): **cabinet ×8**, chair, oven, table.

Eight of eleven misses are cabinets, against one cabinet recovered. A single
class dominates the residual, which is a far more tractable target than the
diffuse failure the render path produced. Whether that is Mask3D merging a
run of built-in units, or the annotation boxes splitting them differently
than the geometry does, is not determined by this run and should not be
guessed at — it is the obvious first question for the sealed scenes.

## Artifacts

* bundle — `runs/arkitscenes_mask3d/bundle_arkitscenes_41069021/`
  (`meta.json`, `raw_masks.npz`, `vertex_instance_ids.npy`,
  `instance_table.json`)
* report — `runs/arkitscenes_mask3d/arkitscenes_41069021_mask3d_eval.json`
* evaluator — `tools/arkitscenes_mask3d_eval.py`
