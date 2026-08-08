# ARKitScenes × Mask3D — execution contract

Status: **approved, awaiting the GPU run.** Deliberately a contract, not a
protocol: one dev scene, one configuration, no sweep, fixed gates.

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
