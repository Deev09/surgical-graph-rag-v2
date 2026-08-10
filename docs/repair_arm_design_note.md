# Multi-view 2D-mask → 3D instance repair arm — design note

Status: constants below were declared at Checkpoint A, **before** the repair
arm was implemented or run. Results are appended at Checkpoint B.

## What this arm is, and what it is not

It is a **detection repair** experiment: given the frozen Mask3D proposal
bank, can multi-view 2D masks from the real posed RGB capture add proposals
for objects Mask3D **missed**, and split objects Mask3D **overmerged**?

It is not about labels. `docs/arkitscenes_rgb_label_results.md` measured that
corrected `rgb_tight` crops improve matched-instance labels and
graph-consistency QA. That result stands and is unrelated: a better label on a
wrong region is still a wrong region. This arm changes which regions exist.

Explicitly out of scope, and enforced by the code structure rather than by
intent: labelling, relation extraction, graph QA, and any Open3DIS /
ConceptGraphs-style framework. The arm emits vertex sets and nothing else.

**Detection metrics here cannot support an end-to-end QA claim.** Every report
written by `eval/detection_repair.py` carries that sentence as a field.

## Why the evaluator was built first

The failure mode this ordering prevents is the one that has already cost this
project four refuted knobs: building a mechanism, then discovering the
measurement that would have refuted it. Two ordering locks, both mechanical:

1. `eval/detection_repair.py` never imports an oracle. Entities are injected
   by the CLI. A repo-level AST sweep pins this.
2. Proposals are finalized and sha256-stamped by `ProposalArtifact.finalize`
   before annotations are opened; `score_bank` re-verifies the digest and
   refuses a set that changed. `tools/arkitscenes_repair_eval.py` is
   AST-tested to import `load_oracle_entities` only after `build_proposals`
   has returned.

## Baseline

`mask3d_ms02` from `docs/arkitscenes_mask3d_contract.md`: the frozen GPU
bundle's raw masks at `min_score=0.2, min_vertices=20`, each carrying its own
Mask3D score. Reproduced independently through the new evaluator:

| scene | proposals | entities | recovered @0.25 | recovered @0.50 | giant | zero-overlap |
|---|---|---|---|---|---|---|
| 41069021 | 37 | 18 | 11 | 7 | 0.0% | 62.2% |
| 41069025 | 42 | 20 | 11 | 9 | 0.0% | 61.9% |

These match the recorded contract and Checkpoint 2 numbers exactly, and
`tests/tools/test_arkitscenes_repair_eval.py` pins them.

The delivered *dense partition* is a different artifact (winner-takes-all, and
the shipped one used `min_score=0.4`). Repair proposals pool with a proposal
bank, so the baseline has to be a proposal bank.

## Metric definitions

Fixed in `eval/detection_repair.py` so two banks cannot be scored two ways.

- **recovered entity** — annotated entity whose best vertex-IoU against *any*
  proposal is ≥ t. Ranking-free, so it is a *ceiling*: the bank contains the
  object; it does not mean a resolver would deliver it.
- **unique recovered** — recovered by pooled, not by baseline.
- **preserved** — recovered by baseline and still recovered by pooled. Pooling
  is additive and asserted to be, so a loss here is a pooling bug.
- **zero-overlap** — proposals with best IoU < 0.10 against every entity.
  **Not precision.** ARKitScenes boxes are not an exhaustive inventory, so a
  genuinely new real object with no box is counted as junk. This penalises the
  arm for succeeding at part of what it is for; it is reported anyway because
  the alternative is an unbounded proposal count.
- **giant mask** — proposal covering > 15% of the mesh.
- **top-100** — under a declared annotation-free ranking. Primary:
  `confidence` (each source's own score; Mask3D scores and repair consensus
  are *not* mutually calibrated, which is exactly the naive pooling a real
  system would do). Reported alongside: `size`.

## Mechanism (pre-registered)

Five stages. Everything downstream of stage 3 is existing frozen code.

1. **Frame selection** — pose-matched capture frames, subsampled, filtered to
   visibility-valid (enough of the frame is explained by the mesh), then
   greedily chosen for angular diversity of view direction.
2. **2D masks** — Felzenszwalb graph-based segmentation on the real RGB frame.
3. **Lifting** — front-surface visibility buffer built with the
   corrected OpenCV camera convention established in
   `extractors/arkitscenes_rgb_crops.py` (validated against sensor depth:
   2.5–4.1 cm median error, versus 37–98 cm for the invalidated axis flip).
   Each 2D region becomes the visible vertex set beneath it.
4. **Consensus fusion** — `segmenter/proposal_fusion.py`, unmodified: mesh-edge
   confidence = share of evidence-bearing co-visible views in which both
   endpoints share a region, then connected components at the frozen cuts
   {0.25, 0.50, 0.75} with the frozen dedupe rule. A component is a set of
   vertices that *multiple independent viewpoints agree* belong together.
5. **Repair classification** — each component is compared to the frozen Mask3D
   bank and emitted **beside** it, never replacing it:
   - `additional` — best containment in any Mask3D proposal < 0.50, i.e. the
     region is largely unexplained by the baseline.
   - `split` — contained ≥ 0.70 in one Mask3D proposal but ≤ 0.70 of its size,
     i.e. a piece of an overmerge. Emitted clipped to the parent.
   - anything else is dropped.

### Declared constants — set once, not swept

| constant | value | why |
|---|---|---|
| `FRAME_STRIDE` | 6 | 60 Hz capture; consecutive frames are the same viewpoint |
| `N_FRAMES` | ~~24~~ → 128 | corrected pre-evaluation; see below |
| `SURFACE_DEPTH_TOLERANCE_M` | 0.05 | added pre-evaluation; see below |
| `MIN_MESH_COVERAGE` | 0.60 | rejects frames aimed at unreconstructed space |
| `FELZ_SIGMA` / `FELZ_K` / `FELZ_MIN_SIZE` | 0.8 / 40 / 60 px | see the calibration note below |
| `MIN_CONSENSUS_VIEWS` | 3 | "multi-view" is the mechanism; 2 is a coincidence |
| `ADDITIONAL_MAX_CONTAINMENT` | 0.50 | majority-unexplained ⇒ candidate missing object |
| `SPLIT_MIN_CONTAINMENT` | 0.70 | mostly inside one parent ⇒ candidate piece |
| `SPLIT_MAX_SIZE_RATIO` | 0.70 | a "piece" must be materially smaller than its parent |
| `DUPLICATE_IOU` | 0.90 | a near-copy of a Mask3D proposal is not a repair |
| `MIN_PROPOSAL_VERTICES` | 200 | junk floor above the fusion floor of 20 |
| `MAX_PROPOSAL_FRAC` | 0.15 | equals the giant-mask threshold |
| `MAX_REPAIR_PROPOSALS` | 60 | keeps the pooled bank near 100 |

### Three changes made after Checkpoint A, all before any annotation was opened

Recorded rather than quietly folded in. None was informed by a measured
result: the arm emitted **zero** proposals until the first two were fixed, and
no annotation had been read.

**1. Lifting was wrong (bug).** The first implementation kept the single
nearest vertex per pixel. This mesh has 1,008,964 vertices and a frame has
49,152 pixels, so that marks at most 5% of the mesh visible and reports a
vertex as hidden because a neighbour 2 mm away won the pixel. Measured: 12k–43k
visible vertices per view, no mesh edge reached three co-visible views, zero
proposals emitted. Replaced with front-surface visibility — every vertex whose
depth is within `SURFACE_DEPTH_TOLERANCE_M = 0.05` of the nearest at its pixel.
That tolerance is tight (the buffer rasterises at full mesh density) and sits
well above the 2.5–4.1 cm mesh-versus-sensor-depth error already measured in
`extractors/arkitscenes_rgb_crops.py`.

**2. `N_FRAMES` 24 → 128 (under-provisioned by an order of magnitude).**
Measured share of mesh vertices seen in ≥ 3 selected views:

| N frames | 24 | 48 | 72 | 96 | 128 | 160 | 200 |
|---|---|---|---|---|---|---|---|
| ≥3 views | 5.6% | 37.4% | 54.4% | 65.7% | 71.7% | 75.9% | 78.9% |

At 24, only 5.6% of the mesh could even be evaluated by a 3-view consensus.
Set to the smallest N reaching 70% — the criterion was fixed before the curve
was read. This is how much evidence the mechanism gets, not a decision
threshold.

**3. Felzenszwalb scale.** The note originally declared the paper's
`k=300 / min_size=120`. On these 256×192 frames that yields a median of 18
regions per frame with one region covering more than half the frame — walls
and floors, not object-scale masks. The criterion for the correction was
stated first and is annotation-free: *median regions per frame in 40–120, and
the largest region below a third of the frame*, measured on five evenly spaced
development frames. `k=40 / min_size=60` gives median 107 and 31%; `k=100`
gives median 80 and 50%. This is calibration of the mask source to the scale
it was supposed to operate at, not tuning against a measured result — no
annotation-based number for this arm existed when it was made. It was not
touched again.

Two constants deserve explicit caveats rather than credit:

- `MAX_PROPOSAL_FRAC` equals the giant-mask threshold, so **the "giant-mask
  rate stays zero" gate is satisfied by construction, not by evidence.** It is
  a guardrail, and it should be read as one.
- `MAX_REPAIR_PROPOSALS = 60` keeps baseline + repair ≈ 97 proposals, so
  "top-100" is very nearly the whole bank. That makes the junk gate a
  whole-bank statement instead of a ranking artifact — but it also means the
  arm is capped before it is measured, and a cap is a form of tuning even when
  it is declared in advance.

## Development plan and gates

Develop on `41069021`. Verify the human-audited cases on `41069025`
(`eval/human_feedback/arkitscenes_sealed_visual_review_2026-08-09.json`):
the overmerged whole-plane `obj_8` and the sofa cushions `obj_9` / `obj_23`.

Gates, all five required before the unseen scene is touched:

1. pooled adds ≥ 2 unique annotation matches at IoU 0.50
2. covers ≥ 1 audited missing/overmerged case
3. preserves all baseline matches
4. pooled giant-mask rate is 0
5. top-100 zero-overlap worsens by ≤ 10 pp (ranking: `confidence`)

`47331972` (seed 20260810) stays undownloaded and uninspected until all five
pass and the implementation is locked. If they fail, the failure mechanism is
reported and the arm stops — no threshold sweep.

### One case cannot be scored by annotation IoU


The audited **missing curtains** are recorded in `41069042`, not `41069025`,
and the human review states that scene's six boxes contain "no sofa/couch,
curtain, rug, or trash-can class". `curtain` has no annotation box in any of
the three scenes. A curtain recovery therefore **cannot** appear as a unique
annotation match, and gate 1 cannot be satisfied by it. It is checkable only
by inspection of the emitted vertex set. Gate 2 is supplied by the operator
from the human-feedback record for exactly this reason: it is not derivable
from IoU.

---

# Checkpoint B — development result: FAILED, arm not locked

The arm is implemented, runs end to end, and **does not clear the gates.**
`47331972` was not downloaded, inspected or touched.

## Result

Pooled = frozen `mask3d_ms02` + repair, scored by `eval/detection_repair.py`
against annotation boxes opened only after the proposal artifact was hashed.

| scene | bank | proposals | @0.25 | @0.50 | giant | zero-overlap |
|---|---|---|---|---|---|---|
| 41069021 | baseline | 37 | 11 | 7 | 0.0% | 62.2% |
| 41069021 | repair only | 52 | 1 | **0** | 0.0% | 88.5% |
| 41069021 | pooled | 89 | 11 | 7 | 0.0% | 77.5% |
| 41069025 | baseline | 42 | 11 | 9 | 0.0% | 61.9% |
| 41069025 | repair only | 35 | 1 | **0** | 0.0% | 97.1% |
| 41069025 | pooled | 77 | 11 | 9 | 0.0% | 77.9% |

| gate | 41069021 | 41069025 |
|---|---|---|
| ≥ 2 unique matches @0.50 | **FAIL** (0) | **FAIL** (0) |
| ≥ 1 audited case | **FAIL** | **FAIL** |
| preserves baseline matches | PASS (0 lost) | PASS (0 lost) |
| giant-mask rate 0 | PASS (by construction) | PASS (by construction) |
| top-100 zero-overlap ≤ +10 pp | **FAIL** (+15.4 pp) | **FAIL** (+16.0 pp) |

Three of five fail on both scenes, in the same way. The two that pass carry no
weight: giant-mask rate is structurally guaranteed, and preservation is
guaranteed by additive pooling, which the evaluator asserts.

## Failure mechanism

Diagnosed on 41069021 after the gates were scored. **The failure is in stage 4
— consensus fusion — and specifically in its interaction with the mask
source. It is not in the camera convention, the lifting, or frame selection,
all of which measure as working.**

What works:

- Frame selection reaches 128 visibility-valid, angularly diverse frames; 71.7%
  of mesh vertices are seen in ≥ 3 of them.
- Lifting is healthy: 81.9% of the 2.9M mesh edges carry view evidence, at a
  median of 6 co-visible views (p90 = 11).
- The 2D masks are object-scale in 2D: median 103 regions per frame, 94 of
  which lift to ≥ 20 vertices.

What fails — the consensus graph never separates the scene into objects:

| cut | components | largest component | ≥ 200 vertices |
|---|---|---|---|
| 0.25 | 12,638 | 739,103 (**73.3%** of mesh) | 56 |
| 0.50 | 12,723 | 738,565 (**73.2%** of mesh) | 56 |
| 0.75 | 17,054 | 690,011 (**68.4%** of mesh) | 86 |

At every cut the graph is **one room-sized blob plus dust**. The blob is
discarded by the fusion module's 40%-of-mesh cap, and what survives to be
emitted is the dust around its edges: the 52 emitted proposals on 41069021
have median 348 vertices, median planarity 0.006–0.012, and median extent
0.29 m × 0.16 m — flat ~30 cm surface patches, against entities of
2,385–40,280 vertices. 12 of 18 annotated entities have **0.0%** of their
vertices touched by any repair proposal.

The cause is visible in the edge-confidence distribution: on evidence-bearing
edges the median confidence is 1.00, and 98.5% / 96.5% / 85.7% of them survive
the 0.25 / 0.50 / 0.75 cuts. Edge confidence is *the fraction of co-visible
views in which both endpoints land in the same 2D region*. For an edge to be
cut, its endpoints must fall in different regions in more than a quarter of
those views. Colour-region boundaries at 256×192 rarely coincide with the 3D
object boundary and move with viewpoint and exposure, so an edge crossing a
real object boundary still scores near 1.0 — the signal "this edge crosses an
object boundary" is largely absent from region-agreement, whatever the cut.

That is a structural mismatch, not a threshold miss. Raising the cut from 0.25
to 0.75 moves the largest component by 5 percentage points while multiplying
fragments; there is no middle where components are object-sized. This fusion
rule was designed for SAM masks on rendered views, where region boundaries are
object-shaped and stable across viewpoints. It does not transfer to
boundary-unstable colour regions.

## The audited cases

Untouched, on both counts:

- **Overmerged kitchen counter plane (41069025).** The audited `obj_8` is
  identifiable in the baseline bank as proposal 8, 132,173 vertices, extent
  7.38 × 5.95 × 0.20 m — matching the human review's "7.38 x 5.95 x 0.20 m
  whole-plane segment" exactly. The arm emitted **one** split piece for it,
  covering **0.2%** of it. The next two largest baseline proposals received
  none.
- **Sofa cushions (41069025).** No repair proposal reaches IoU 0.50 with any
  annotated entity, and repair-only zero-overlap is 97.1%.
- **Missing curtains.** Not evaluable here in any case — recorded in 41069042,
  and `curtain` has no annotation box in any of the three scenes.

## What this does and does not refute

It refutes **this mask source through this fusion rule**. It does not refute
multi-view 2D→3D repair as a mechanism, because Felzenszwalb colour regions
are a deliberately weak stand-in for a learned segmenter (this environment has
numpy and PIL only), and the diagnosis lands squarely on boundary instability
— the property a promptable segmenter is specifically better at.

The measurements that do transfer, independent of the mask source:

- the corrected OpenCV projection lifts real posed RGB onto this mesh well
  enough to give 81.9% of mesh edges multi-view evidence at a median of 6
  views;
- one-vertex-per-pixel id buffers are unusable at this mesh/image scale
  (1,008,964 vertices against 49,152 pixels) — surface-tolerance visibility is
  required;
- 24 views is far too few: ≥3-view co-visibility rises 5.6% → 71.7% between 24
  and 128 frames.

**No end-to-end or QA claim follows from any of this.** These are detection
metrics on proposal banks.

## Stop condition

Per the brief, the arm stops here rather than being tuned. Threshold changes
that would obviously move the numbers — lowering `MIN_PROPOSAL_VERTICES`,
raising `MAX_REPAIR_PROPOSALS`, relaxing `MIN_CONSENSUS_VIEWS` — were **not**
tried, because the diagnosis says the components are the wrong shape, not the
wrong size, and none of them addresses a graph whose largest component is 73%
of the mesh.

The next experiment, if the owner wants one, is a **mask-source swap behind
the same evaluator**: `segment_frame` is the only seam, the evaluator and its
gates are already fixed and committed, and the failure above gives a specific
prediction to test — that a boundary-stable segmenter moves the largest
component well below 73% of the mesh. That prediction is checkable *before*
any annotation is opened, from the component table alone.

`47331972` remains undownloaded and uninspected.

---

# Checkpoint C — SAM 2.1 repair arm, local half complete, awaiting GPU

The Felzenszwalb + mesh-edge-consensus arm above is a closed negative and is
not tuned further. This is a different mechanism sharing the same evaluator.

`eval/detection_repair.py` is **byte-identical to Checkpoint A** (zero diff).
Its gates, IoU grid, zero-overlap definition, giant-mask threshold and
baseline reproduction are unchanged.

## The architectural correction

SAM masks are overlapping, non-exhaustive object hypotheses. The previous arm
treated 2D masks as an exhaustive per-pixel partition and paid for it: every
pixel belonged to exactly one region, so "do these adjacent vertices share a
region" was ~1.0 even across real object boundaries, and the consensus graph
was one blob covering 68–73% of the mesh. Three properties are now structural
and pinned by test:

- **no partition is ever formed.** Every mask is lifted independently; a pixel
  in no mask contributes nothing, a pixel in three masks feeds three
  hypotheses.
- **background is never a region.** No complement, no "everything else".
- **association is between masks, in 3D** — not between vertices.

Nested masks are kept, not resolved: SAM emits a sofa and its cushions and
both are legitimate entries in a proposal bank.

## Pipeline

1. `tools/arkitscenes_repair_frames.py` — select 32 visibility-valid frames,
   copy the exact PNGs, hash-pin each one and the ordered selection, pack a
   portable tar for the GPU stage. Poses and intrinsics stay local; a
   post-write check asserts nothing else reaches the tar.
2. `notebooks/repair_sam2_colab.ipynb` — SAM 2.1 Hiera-L at the existing pin
   (sam2 @ `2b90b9f5…`, checkpoint sha `2647878d…`, AMG parameters and seeds
   from `docs/c1_p1_multiview_proposals_protocol.md`). Emits a sidecar
   carrying the selection hash.
3–5. `segmenter/sam_multiview_repair.py` — independent lifting, 3D
   association, cluster fusion.
6–7. `tools/arkitscenes_repair_propose_sam.py` — emit beside Mask3D, finalize
   and hash the bank before evaluation.

The propose CLI refuses a sidecar whose selection hash, sam2 commit or
checkpoint sha does not match, and refuses a frame list whose source PNGs have
moved. A sidecar from a different selection would lift masks onto the wrong
geometry with no other symptom.

## Frame selection was rebuilt, and it matters more than anything else here

The previous arm maximised angular spread by farthest-point sampling. That
optimises *against* association: masks can only cluster if the same object
appears in two or more frames. Selection is now greedy on coverage
**multiplicity** — maximising the number of vertices seen at least three times
— subject to every pair of chosen view directions differing by at least 8°.
Diversity is a constraint; overlap is the objective.

Measured on 41069021 at the brief's 32-frame ceiling, over the 37 frozen
Mask3D objects:

| selection | ≥2 views @50% | ≥3 views @50% | mesh seen |
|---|---|---|---|
| farthest-point on direction | 18/37 | 7/37 | 70.0% |
| multiplicity-greedy, 8° floor | **24/37** | **17/37** | **75.0%** |

Annotation-free, measured before any SAM inference.

## Loopback self-check — a necessary condition, verified before spending a GPU

`tools/arkitscenes_repair_loopback.py` runs stages 3–5 on *synthetic perfect
masks* rendered from known 3D instances (frozen Mask3D proposals, chosen
without annotations). If the geometry cannot recover an object it is handed a
perfect mask of, that failure is free to find now instead of being mistaken
for "SAM missed it" afterwards.

Result on 41069021: **8/12 probes recovered at IoU 0.50**, median IoU 0.60,
all 12 visible in ≥2 selected frames.

It also found a real defect. `MIN_CLUSTER_ANGULAR_SPREAD_DEG` was an arbitrary
20°; three probes associated cleanly at 3D IoU 0.60–0.63 and were discarded
because their only two supporting views were 10.4° apart. Selection already
guarantees ≥8° between every chosen pair, so a larger per-cluster threshold
re-litigates a decision already made against a frame set built not to satisfy
it. It is now equal to the selection floor. Changed on synthetic evidence,
before any SAM inference, with no annotation open.

**The four remaining misses are a measured prediction, not noise.** Each fuses
to ~0.45× its probe's vertex count (probes of 149k, 36k, 36k, 29k vertices →
IoU 0.435, 0.427, 0.427, 0.455), while every recovered probe fuses to
0.60–0.92×. The failure is under-*coverage*, not mis-segmentation: at 32
frames the selected views collectively see under half of the largest objects.
So before SAM runs, the expected behaviour is:

- objects supported by 4+ views recover well (IoU 0.53–0.88 with perfect masks);
- large objects supported by 2–3 partial views land near IoU 0.43–0.46 and
  **miss the 0.50 gate**;
- this ceiling is a property of the brief's 16–32 frame budget, not of the
  fusion rule, and no threshold in this module moves it.

## Declared constants

| constant | value |
|---|---|
| `N_FRAMES` | 32 (top of the brief's 16–32 band) |
| `MIN_ANGULAR_SEPARATION_DEG` | 8.0 |
| `COVERAGE_MULTIPLICITY_CAP` | 3 |
| `MIN_MASK_VERTICES` | 200 |
| `MAX_MASK_FRAC` | 0.15 (= giant-mask threshold) |
| `ASSOC_ANCHOR_STRIDE` | 16 |
| `ASSOC_IOU` | 0.30 |
| `ASSOC_CONTAINMENT` / min size ratio | 0.80 / 0.50 |
| `MIN_SUPPORT_VIEWS` | 2 |
| `MIN_CLUSTER_ANGULAR_SPREAD_DEG` | = selection floor (8.0) |
| `VERTEX_VOTE_FRACTION` | 0.50 |
| `MIN_MEMBER_IOU` | 0.20 |

Classification thresholds (`ADDITIONAL_MAX_CONTAINMENT`,
`SPLIT_MIN_CONTAINMENT`, `SPLIT_MAX_SIZE_RATIO`, `DUPLICATE_IOU`,
`MIN_PROPOSAL_VERTICES`, `MAX_PROPOSAL_FRAC`, `MAX_REPAIR_PROPOSALS`) are
**imported** from the previous arm, not restated, so the two differ only in how
a candidate region was produced.

`MIN_SUPPORT_VIEWS = 2` rather than 3 is a deliberate, recorded compromise: at
32 frames only 17 of 37 objects are seen three times, so requiring three would
discard half the recoverable objects before SAM runs.

The giant-mask gate remains satisfied **by construction** (`MAX_MASK_FRAC` =
`MAX_PROPOSAL_FRAC` = the evaluator's threshold) and still carries no
evidential weight.

## Status: blocked on the GPU stage

Everything except SAM inference is implemented, tested (100/100 test files)
and exercised on real geometry. The frame bundle for 41069021 is built and
pinned:

- 32 frames selected from 273 visibility-valid candidates
- 75.0% of the mesh seen in ≥1 view, 57.2% in ≥2, 40.0% in ≥3
- `selection_sha256` `53982b5bf1166163…`, 2.2 MB upload tar

The pinned configuration is CUDA bf16 on an A100. This machine is an Apple
M4 Pro with no CUDA and no torch installed, so inference cannot run here.
41069025 is not prepared and 47331972 is untouched.

### Post-GPU path verified before the GPU session

The whole downstream half was exercised on a **synthetic sidecar** built in the
exact wire format (packbits masks, scores, shapes, pinned env) over the real
development scene, run through the real CLIs into a scratch directory:

```
propose_sam  45 lifted masks -> 13 clusters -> 10 supported -> 5 emitted
repair_eval  pooled 42 proposals, gates evaluated, report written
```

So sidecar parsing, the pin/selection joins, lifting, association, fusion,
finalization, additive pooling and gate evaluation all run end to end. **The
numbers from that run are meaningless** — the synthetic masks were rendered
*from* the Mask3D proposals, so the arm can only rediscover the baseline — and
they were written to a scratch path, not to `runs/`. It is a plumbing check: a
crash discovered after a GPU session would waste the session.

`tests/tools/test_arkitscenes_repair_propose_sam.py` pins the cheap half of
that (selection-hash mismatch, off-pin model, frame-count and shape mismatch,
empty-mask frames) so it cannot rot between sessions.

## Handoff

Run `notebooks/repair_sam2_colab.ipynb` with
`runs/arkitscenes_repair/arkitscenes_41069021/repair_frames_arkitscenes_41069021.tar.gz`,
then drop the sidecar back and run the two commands in the notebook's closing
cell. Gate failure on 41069021 stops the arm; 41069025 is prepared only if it
passes, and 47331972 stays untouched either way.
