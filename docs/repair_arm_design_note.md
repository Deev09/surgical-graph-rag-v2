# Multi-view 2D-mask → 3D instance repair arm — design note

Status: **TRACK CLOSED PERMANENTLY at Checkpoint I.** Four mechanisms plus one
correctness replay were measured against one unchanged evaluator; none added a
single annotation match at IoU 0.50. The closure and what survives it are at
the end of this file.

Constants below were declared at Checkpoint A, **before** the repair arm was
implemented or run. Results are appended per checkpoint.

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

---

# Checkpoint D — SAM 2.1 arm, 41069021 result: FAILED. Stop; 41069025 not prepared.

GPU run completed on the pin: A100-SXM4-80GB, torch 2.11.0+cu128, bf16, 27.4 s,
selection `53982b5b…`, checkpoint `2647878d…`, sam2 `2b90b9f5…`. 735 masks over
32 frames, no empty frames. All three joins verified by the propose CLI before
lifting.

Local: 570 lifted masks → 318 raw clusters → 94 supported → 60 emitted
(42 additional, 18 split), median 3,163 vertices.

## Result against the brief's four proceed conditions

| condition | value | |
|---|---|---|
| ≥ 2 unique entities at IoU 0.50 | **0** | **FAIL** |
| existing Mask3D matches preserved | 0 lost | PASS |
| giant-mask rate zero | 0.0% | PASS (by construction) |
| top-100 zero-overlap ≤ +10 pp | **+12.1 pp** | **FAIL** |

| bank | proposals | @0.25 | @0.50 | zero-overlap |
|---|---|---|---|---|
| mask3d_ms02 | 37 | 11 | 7 | 62.2% |
| repair only | 60 | 3 | 0 | 81.7% |
| pooled | 97 | **12** | 7 | 74.2% |

The pooled bank gains **one** unique entity at IoU 0.25 (a cabinet, best IoU
0.010 → 0.300) and none at 0.50.

The evaluator also reports a fifth gate, `covers_an_audited_case`. It is a
Checkpoint-A gate, is not among this brief's four proceed conditions, and is
**not applicable to 41069021**: the human-feedback record contains audited
cases only for 41069025 and 41069042. The evaluator was not modified.

## Failure mechanism: high precision, low recall — SAM returns parts

For every entity where a repair proposal overlaps at all, the proposal sits
almost entirely *inside* the annotated object and covers a small fraction of
it:

| entity | entity verts | proposal verts | IoU | precision | recall |
|---|---|---|---|---|---|
| sofa | 36,258 | 5,529 | 0.152 | **0.997** | 0.152 |
| table | 10,431 | 2,439 | 0.234 | **1.000** | 0.234 |
| tv_monitor | 4,806 | 1,506 | 0.311 | **0.993** | 0.311 |
| table | 17,453 | 3,276 | 0.176 | 0.947 | 0.178 |
| cabinet | 11,226 | 1,924 | 0.144 | 0.862 | 0.148 |
| oven | 13,417 | 1,234 | 0.074 | 0.823 | 0.076 |
| cabinet | 26,384 | 2,854 | 0.091 | 0.855 | 0.092 |

Median emitted proposal is 3,163 vertices against a median entity of 10,231.
The proposals are **correct object parts, not wrong regions** — a cabinet door,
a table top, a sofa cushion. IoU 0.50 requires recall ≥ 0.50 even at perfect
precision; observed recall is 0.08–0.31, so the gate is unreachable by
arithmetic, not by a threshold being slightly off.

Two compounding causes, neither fixable by a constant in this module:

1. **The pinned AMG configuration returns part-level masks at this
   resolution.** `points_per_side=32` on a 256×192 frame over-segments objects
   into parts. The pin is the frozen C1-P1 configuration and was not varied.
2. **Annotation targets are OBB vertex sets** — everything inside an oriented
   box — so an object part can never reach 0.50 against one even when it is a
   perfectly clean part.

The 60-proposal cap was binding (78 candidates, 18 dropped) but is **not** what
failed the gate: raising it adds more parts, and parts do not raise recall.

This is a different failure from Checkpoint B. That arm produced flat ~30 cm
patches unrelated to objects, leaving 12 of 18 entities untouched. This arm
touches 12 of 18 entities with precision 0.82–1.00 and fails on extent.

## What the loopback did and did not predict

The loopback predicted large objects landing near IoU 0.43–0.46. The real run
came in far lower (0.07–0.31). The gap is the loopback's own design: it
rendered **whole-object** synthetic masks, so it could validate lifting,
association and fusion, but structurally could not simulate part-level
fragmentation — the thing that actually decided the outcome. It was labelled a
necessary-condition check and it was one; it is now also a recorded example of
what such a check cannot see. A future version should render *part-level*
probes.

## Stop

Per the brief, 41069021 fails and **41069025 is not prepared**: no frame
bundle, no GPU run. `47331972` remains undownloaded and uninspected.

The audited counter/cushion cases are recorded for 41069025 and therefore
**were not evaluated** — reporting on them requires the run the gate forbids.

No constant was tuned against this result.

---

# Checkpoint E — oracle-guided composition ceiling: part assembly cannot reach 0.50

Zero GPU. Same pinned sidecar, no SAM parameter changed. **Diagnostic only** —
parts are chosen with the annotation in hand, so every number is an upper bound
no oracle-free assembler can exceed. The report is stamped
`oracle_guided: true`, `deployable: false`, and a test AST-checks that the tool
cannot mint a `ProposalArtifact` or reach the gate sheet.

Development stop held: 41069025 not run, 47331972 untouched.

## Ceiling

Greedy union maximising IoU per entity, over three banks of increasing
permissiveness. 11 of 18 entities are missed by Mask3D at IoU 0.50.

| bank | parts | reach 0.50 | **of the 11 missed** | median IoU | median recall |
|---|---|---|---|---|---|
| emitted (60) | 1 → 16 | 0 → 0 | **0** | 0.142 → 0.242 | 0.152 → 0.275 |
| supported (94) | 1 → 16 | 1 → 1 | **0** | 0.202 → 0.309 | 0.253 → 0.374 |
| pre_support (318) | 1 | 1 | 0 | 0.256 | 0.342 |
| pre_support (318) | 2 | 1 | 0 | 0.297 | 0.440 |
| pre_support (318) | 4 | 1 | 0 | 0.346 | 0.487 |
| pre_support (318) | 8 | 2 | **0** | 0.401 | 0.496 |
| pre_support (318) | 16 | 2 | **0** | 0.409 | 0.496 |

**Zero previously missed entities become reachable at IoU 0.50 under any bank
at any budget.** Best value seen anywhere is 0.777, on the `tv_monitor` that
Mask3D already delivers at 0.945. Everything saturates past 8 parts.

Greedy was checked against the exhaustive best pair on every entity: genuine
shortfalls are 0 / 0 / 1 across the three banks, worst +0.0055. (The first run
of the tool reported 9 shortfalls on `pre_support`; that was a rounding
artifact in the comparison — unrounded exhaustive against 4-decimal greedy —
and is fixed. It did not affect any ceiling value.) So the ceiling is tight,
not a loose greedy lower bound.

**Per the decision rule, the next mechanism is NOT oracle-free part assembly.**
The parts to assemble are not there.

## Why: it is a precision wall, not a coverage wall

Two decompositions, both on the 32 selected frames.

Coverage is *not* the binding constraint:

| quantity | median over 18 entities |
|---|---|
| entity fraction visible in ≥1 selected frame | **0.795** |
| of that visible part, fraction inside ≥1 SAM mask | **0.843** |

So the raw material for ~0.67 recall exists. But taking it destroys precision.
Compare the IoU-optimal union against the recall-maximal union (every
`pre_support` part touching the entity):

| union | median IoU | median precision | median recall |
|---|---|---|---|
| greedy, ≤16 parts, IoU-optimal | 0.409 | 0.615 | 0.496 |
| all touching parts, recall-maximal | **0.073** | **0.078** | 0.611 |

Buying the last 0.115 of recall costs 0.54 of precision. The parts holding the
rest of each object are ~13× larger than their intersection with it: they are
surface regions spanning several objects at once (a counter run continuing into
cabinet and wall). The bank contains, for each object, a few clean parts and
then nothing usable.

## What this evidence does and does not support

It does **not** support raising `points_per_side` or upsampling frames. Both
make masks *smaller and more numerous*. The measured gap is not "masks too
coarse to be precise" — precision at the IoU-optimal union is already 0.615 and
the uncovered-visible surface is only 15.7%. The gap is "no available mask
covers the rest of the object without also covering its neighbours". Finer
seeding addresses the 15.7%, not the precision collapse, and would enlarge the
318-cluster bank that already fails. **I am not claiming finer masks would
fail — I am declining to claim they would help, because nothing measured here
says so.**

The one change with direct supporting evidence is the **view budget**: 20.5% of
a median entity is never seen by any of the 32 frames, and that is a hard cap
on recall independent of the mask source. It is also not sufficient on its own
— lifting visibility from 0.795 to 1.0 does not fix a precision collapse from
0.615 to 0.078.

If the owner wants one more measurement before changing anything, the cheapest
discriminating test is: re-run the ceiling with the SAME sidecar but parts
**intersected with mesh-geometry connected components**, i.e. cut the
over-extended masks at geometric discontinuities. That is zero-GPU, uses no new
mask source, and directly tests whether the precision collapse is caused by
masks bleeding across geometric boundaries — the one hypothesis consistent with
all the numbers above. If cut parts assemble to 0.50, the fix is geometric
boundary snapping, not a new pin.

---

# Checkpoint F — topology-only boundary cut: 0 genuine recoveries. Stop.

Zero GPU, same pinned sidecar, no SAM parameter changed. Development stop held:
41069025 not run, 47331972 untouched. `eval/detection_repair.py` byte-identical
to Checkpoint A.

## What was run

For each of the 570 lifted SAM masks: take its vertices, induce the
mesh-adjacency subgraph on exactly those vertices, split into connected
components. **No distance, normal, depth, curvature or learned threshold** —
`connected_components` does not even take coordinates, and a test asserts its
signature. The only threshold is the existing `MIN_MASK_VERTICES = 200`. Every
qualifying component is emitted; none is selected against annotations. The bank
is finalized and sha256-stamped (`85cd974dd86112fb…`) before the ceiling opens
a single box.

## Components

| quantity | value |
|---|---|
| components before min-size | **51,446** |
| components emitted | **868** (835 unique vertex sets) |
| masks that split | 556/570 (97.5%), median 38, mean 90.3, max 1,037 |
| mass removed by min-size | **14.4%** (305,346 of 2.1M mask-vertices) |
| component size | min 200, p10 251, median 736, p90 4,929, max 33,824 |

The size floor discards 98% of components but only 14.4% of mass: almost all
components are sub-200-vertex speckle at occlusion edges.

## Ceiling, same ≤1/2/4/8/16 budgets

| bank | props | zero-ovl | ≤1 | ≤2 | ≤4 | ≤8 | ≤16 | **missed reached** | **1-part** | **assembled** |
|---|---|---|---|---|---|---|---|---|---|---|
| emitted | 60 | 81.7% | 0 | 0 | 0 | 0 | 0 | **0** | 0 | 0 |
| supported | 94 | 81.9% | 1 | 1 | 1 | 1 | 1 | **0** | 0 | 0 |
| pre_support | 318 | 88.4% | 1 | 1 | 1 | 2 | 2 | **0** | 0 | 0 |
| topology_cut | **868** | **94.3%** | 1 | 1 | 2 | 2 | 4 | **1** | **0** | **1** |

Median precision / recall at ≤16 parts: `pre_support` 0.615 / 0.496;
`topology_cut` 0.762 / 0.441.

At **1 part**, topology cutting raises median precision from 0.556 to **0.950**
— the cleanest parts this arm has produced. Recall at 1 part falls to 0.195.

## Answers to the five questions asked

1. **Components and sizes** — above. 51,446 raw, 868 emitted, median 736.
2. **Newly reachable missed entities at IoU 0.50** — **one**, versus zero for
   every other bank.
3. **Precision/recall** — precision up substantially (0.556 → 0.950 at 1 part;
   0.615 → 0.762 at ≤16), recall down (0.496 → 0.441 at ≤16).
4. **Proposal and zero-overlap growth** — 318 → 868 proposals (+173%),
   zero-overlap 88.4% → 94.3% (+5.9 pp).
5. **Source of the recovery — actual disconnected surfaces, or oracle assembly?**
   **Oracle assembly.** Single-part recoveries of missed entities are **zero**
   in every bank at every budget. The one topology_cut recovery needed 16
   oracle-selected fragments. No annotation-free assembler reproduces that.

## Why topology cut in the wrong places

The hypothesis was that over-extended masks bleed across boundaries that are
*disconnected* on the mesh. Measuring the annotated entities directly refutes
it — and refutes a "shredded mesh" reading too:

| | median over 18 entities |
|---|---|
| largest mesh-connected component's share of an entity | **0.882** |
| entities whose largest component holds ≥90% of them | 8/18 |

Entities are largely **connected**: a single component could reach IoU 0.88 for
a median entity, and `tv_monitor` is exactly one component. So the mesh is not
shredded and connectivity is not the missing signal.

What that means is that connectivity carries almost no *object-boundary*
information in this scene. Objects rest on floors and against walls, and those
contacts are mesh-connected, so a mask spanning cabinet-and-wall cannot be cut
by adjacency. Topology instead cuts at **reconstruction holes** — occlusion
gaps and scan dropouts — producing 51,446 fragments that are clean (precision
0.95) but small (median 736) and located wherever the scanner failed, not
wherever objects end.

## Stop

The literal count is **1 newly reachable missed entity, not 0**, so the stated
stop condition is not met word-for-word. Stopping anyway, and flagging the
judgement rather than acting on it:

- genuine, single-part recoveries are **0**;
- the one recovery is 16-fragment oracle assembly, unreachable without the
  annotation;
- 1 is below the standing bar of 2;
- the diagnosis says the cue itself is wrong, so tuning around it is unlikely
  to convert.

**No normal or depth threshold tuning has been started, and none will be
without another decision.** For that decision, the relevant evidence is
double-edged: object boundaries that adjacency cannot cut — a cabinet meeting a
wall — usually *are* normal discontinuities, so a normal-based cut is the one
mechanism this measurement does not rule out. It equally does not support it:
nothing here measures whether normal discontinuities coincide with the specific
boundaries these masks straddle, and the same cut would also fire on every
interior crease of a sofa. That is a measurement, not a plan.

---

# Checkpoint G — detector-guided SAM arm, built and awaiting one GPU run

Checkpoints B, E and F are closed negatives and are not revisited. No normal,
depth or connectivity threshold experiment was run.

`eval/detection_repair.py` is **byte-identical to Checkpoint A** (zero diff).
41069025 not prepared; 47331972 untouched.

## The single change under test

Every prior arm segmented *without being told what to look for*. The
class-agnostic SAM arm seeded a 32×32 point grid and produced object PARTS —
precision 0.82–1.00, recall 0.08–0.31 — and no union of those parts reached
IoU 0.50 for a missed entity, before or after topology splitting.

Nothing in that says SAM cannot outline a whole object. It says nothing told it
which object to outline. So the variable here is the **prompt**: Grounding DINO
boxes over the fixed 41-class vocabulary, each box prompting the *same* SAM 2.1
checkpoint. Same 32 frames, same lifting, same geometric association
thresholds, same evaluator, same gates.

## Pins, fixed before evaluation

| | |
|---|---|
| detector | `IDEA-Research/grounding-dino-base` |
| thresholds | `box_threshold=0.35`, `text_threshold=0.25` — published defaults, **no sweep** |
| vocabulary | `GLOBAL_INDOOR_VOCABULARY_V1`, 41 classes, imported not copied |
| segmenter | SAM 2.1 Hiera-L, sha `2647878d…`, box prompts, `multimask_output=False` |
| frames | the identical 32-frame bundle, selection `53982b5b…` |

Recorded in `docs/grounding_dino_pin.json`. The detector **revision and weight
sha are null until the first run**: the notebook resolves and prints them, and
they go into the pin file by the freeze commit before any result is claimed —
the same procedure `docs/c1_p1_multiview_proposals_protocol.md` used for the
SAM weights. Nothing was invented ahead of the download, and the CLI reports
the run as `UNPINNED` while those fields are null.

The propose CLI refuses a sidecar on any of five joins: frame selection hash,
SAM commit, SAM checkpoint sha, detector identity/revision/sha once frozen,
either threshold differing from the pin, or a vocabulary that does not hash to
the repo's list. Refusing a threshold mismatch is what "no sweep" means
mechanically rather than as a promise.

## What is structural

- **No background or complement proposal.** Only detected boxes produce masks.
  A test AST-scans the module for mask inversion, `logical_not`, `invert` and
  `setdiff1d` and fails on any of them.
- **Association requires compatible labels AND 3D overlap.** Two masks merge
  only if the detector named the same class and they occupy the same surface.
  The geometric half is *imported* from the class-agnostic arm, so the two
  differ only in the prompt and this rule.
- **Provenance survives to the proposal.** Detector label, score, box and frame
  ride from the sidecar into each emitted proposal's notes.
- Phrases that do not resolve to exactly one vocabulary entry are **dropped,
  not guessed**, and counted in the diagnostics.

## Two defects the pre-GPU smoke test caught

The whole post-GPU path was exercised on a synthetic sidecar in the real wire
format, into a scratch directory.

1. **Split proposals lost their label.** Labels were re-attached by matching a
   proposal's first vertex to its cluster's, but a `split` piece is clipped to
   its Mask3D parent, so that match fails whenever clipping removes the head.
   Two of five proposals came out labelled `?`. Fixed by threading the source
   cluster index through `classify_clusters` diagnostics — an additive change
   that leaves the function's signature and behaviour alone.
2. Re-running the class-agnostic arm afterwards reproduces proposal sha
   `39bcd061c21ae464…` **exactly**, confirming the Checkpoint D result is
   untouched by that patch.

Both are now regression-tested, including a fixture that genuinely exercises
head-clipping — the first attempt clipped only the tail and would have passed
vacuously.

## Status

Everything except detector+SAM inference is implemented and tested
(104/104 test files). The run is one Colab session on the existing bundle.

**If this arm also produces zero unique IoU-0.50 recoveries, the repair track
closes** — no further mask, geometry or threshold variant.

---

# Checkpoint H — detector-guided SAM result, and CLOSURE of the repair track

Run on the pin: A100-SXM4-40GB, 8.5 s, 126 boxes over the same 32 frames,
selection `53982b5b…`. Detector pin **frozen before scoring** —
`IDEA-Research/grounding-dino-base` revision `12bdfa31…`, weights
`5548f844…` — at the published thresholds 0.35 / 0.25, vocabulary hash
verified against the repo's 41 classes. 113 masks lifted → 66 label-consistent
clusters → 23 supported → **13 emitted**.

## Result

| bank | proposals | @0.25 | @0.50 | giant | zero-overlap |
|---|---|---|---|---|---|
| mask3d_ms02 | 37 | 11 | 7 | 0.0% | 62.2% |
| repair only | 13 | 4 | **0** | 0.0% | 46.2% |
| pooled | 50 | **12** | 7 | 0.0% | **58.0%** |

| condition | value | |
|---|---|---|
| ≥ 2 unique entities at IoU 0.50 | **0** | **FAIL** |
| Mask3D matches preserved | 0 lost | PASS |
| giant-mask rate zero | 0.0% | PASS (by construction) |
| top-100 zero-overlap ≤ +10 pp | **−4.2 pp** | PASS |

**This is the best repair bank the track produced** — 13 surgical proposals,
sensible labels, one new entity at IoU 0.25 (an `oven` Mask3D had at 0.012), and
the only arm whose pooled *proportion* of zero-overlap proposals fell. It still
recovers **zero** unique entities at IoU 0.50.

Stated precisely, because the rate alone flatters it: **proportional
contamination fell from 62.2% to 58.0%, while the absolute number of
zero-overlap proposals rose from 23 to 29.** The pool is a smaller share junk
and a larger amount of junk. Both are true and the second is what a downstream
consumer pays.

## Same wall as every other arm: recall, not precision

Best repair proposal per entity it touches:

| entity | baseline | repair IoU | precision | recall | label |
|---|---|---|---|---|---|
| table (17,453) | 0.531 | 0.380 | **0.962** | 0.385 | desk |
| oven (13,417) | 0.012 | 0.312 | 0.744 | 0.349 | oven |
| table (10,431) | 0.697 | 0.273 | 0.650 | 0.321 | table |
| table (2,385) | 0.616 | 0.217 | 0.250 | **0.622** | table |
| cabinet (11,226) | 0.086 | 0.185 | 0.235 | 0.464 | counter |

Precision reaches 0.96; recall never exceeds 0.62 and is usually 0.02–0.48. IoU
0.50 needs recall ≥ 0.50 at perfect precision. One over-extended `counter`
detection (22,113 vertices) straddles five separate cabinet entities — the
counter/cabinet-run ambiguity the human review already flagged on 41069025.

13 of 127 detections were dropped for phrases that resolve to no single class
(`armchair chair` ×10, `blinds window`, `stool table`, one empty). A span-
parsing improvement could recover about ten detections. It was **not** made:
the track closes rather than iterating.

## Closure

Per the brief, zero unique IoU-0.50 recoveries from detector-guided SAM closes
the repair track. Four mechanisms, one unchanged evaluator, one scene:

| arm | proposals | unique @0.50 | failure |
|---|---|---|---|
| B — Felzenszwalb + mesh-edge consensus | 52 | 0 | consensus graph is one 73%-of-mesh blob; emitted flat 30 cm patches |
| D — SAM 2.1 automatic masks | 60 | 0 | masks are object parts: precision 0.82–1.00, recall 0.08–0.31 |
| F — topology-only split of those masks | 868 | 0 | cuts at reconstruction holes, not object boundaries; 1 oracle-assembled hit, 0 genuine |
| H — Grounding DINO boxes → SAM 2.1 | 13 | **0** | precision up to 0.96, recall ≤ 0.62 |

And the oracle-guided ceiling (E) showed the parts are not merely
unassembled — with the annotation in hand, no union of ≤16 parts from any bank
reached IoU 0.50 for any missed entity.

**What is established.** Multi-view 2D→3D lifting on this capture works: the
corrected OpenCV convention, front-surface visibility and multiplicity-greedy
frame selection give 79.5% entity visibility and 84.3% mask coverage of what is
visible, and the loopback recovers 8/12 probes from perfect masks. The
machinery is sound. What no 2D mask source tested here supplies is an extent
that matches an ARKitScenes oriented-box entity: every arm lands high-precision
and low-recall against that target.

**What is not established, and should not be inferred.** That 2D-guided repair
is impossible; that a different detector, prompt granularity or frame budget
would fail; or anything at all about labelling, relations or QA. These are
detection metrics on proposal banks for one scene.

**Scope actually consumed.** 41069021 only. `41069025` was never prepared —
no frame bundle, no GPU run — so the audited counter/cushion cases were never
evaluated by any arm. `47331972` was never downloaded or inspected.

**What survives for reuse.** `eval/detection_repair.py`, byte-identical since
Checkpoint A: an annotation-free, hash-ordered, additively-pooled detection
evaluator with a reproduced Mask3D baseline (7/18 and 9/20 at IoU 0.50) pinned
by test. Any future proposal source can be scored against it without
re-litigating the measurement.

---

# Checkpoint I — candidate-label-set replay, and PERMANENT closure

No GPU. Identical detector sidecar, byte for byte. Detector and SAM pins,
thresholds, frames, vocabulary, fusion constants, evaluator and gates all
unchanged.

## The correctness defect

Checkpoint H discarded 13 of 127 detections whose phrase did not resolve to
exactly one class — 10 of them `armchair chair`, plus `blinds window`,
`stool table` and one empty string. That was wrong. Grounding DINO returning a
span across two adjacent prompt entries is the model saying "armchair or
chair": real evidence, not failed inference.

Each detection now carries the **set** of classes its phrase could name, and
association requires those sets to **intersect** rather than match.
`{armchair, chair}` joins `{chair}`; `{sofa}` and `{cushion}` still never join;
an empty set intersects nothing and cannot associate. Genuinely unknown
phrases (`teapot`, `""`) still resolve to the empty set and are still dropped —
ambiguity between known classes is evidence, an unknown word is not.

Cluster reporting uses the intersection of member sets when non-empty
(`armchair|chair`), else the most-admitted class, since single-linkage can
chain `{a,b}`–`{b,c}`–`{c,d}` into an empty intersection.

## Replay result

| | strict (H) | candidate sets (I) |
|---|---|---|
| detections used | 114 | **127** |
| lifted masks | 113 | **125** |
| clusters supported | 23 | 27 |
| proposals emitted | 13 | **15** |
| repair-only zero-overlap | 46.2% | **40.0%** |
| pooled zero-overlap, proportion | 58.0% | **55.8%** (−6.4 pp vs baseline) |
| pooled zero-overlap, absolute | 29 | **29** (baseline 23) |
| unique @ IoU 0.25 | +1 (oven) | +1 (oven) |
| **unique @ IoU 0.50** | **0** | **0** |

Recovering the discarded evidence did exactly what it should — two more
proposals, two `armchair|chair` clusters, and the lowest zero-overlap
*proportion* of any arm. It moved nothing at IoU 0.50.

The proportion is not the whole picture and should not be quoted alone:
**proportional contamination fell from 62.2% (baseline) to 55.8%, while the
absolute count of zero-overlap proposals rose from 23 to 29.** Adding 15
proposals of which 6 overlap no annotated entity lowers the ratio and raises
the count. A consumer filtering the bank sees the ratio; a consumer processing
every proposal sees the count.

Gate sheet: preserved matches PASS (0 lost), giant-mask rate PASS (0, by
construction), top-100 zero-overlap PASS (−6.4 pp proportional; +6 absolute, 23 → 29), unique @0.50
**FAIL** (0).

Both runs are kept side by side under
`runs/arkitscenes_repair/arkitscenes_41069021/`: the strict artifacts are
suffixed `.strict`, the replay overwrites the plain names.

## Permanent closure

Per the brief, unique recovery at IoU 0.50 remained zero, so the repair track
closes permanently.

| arm | proposals | unique @0.50 |
|---|---|---|
| B — Felzenszwalb + mesh-edge consensus | 52 | 0 |
| D — SAM 2.1 automatic masks | 60 | 0 |
| F — topology-only split | 868 | 0 |
| H — Grounding DINO → SAM 2.1 | 13 | 0 |
| I — same, candidate-label sets | 15 | **0** |
| E — oracle-guided ceiling over all part banks | ≤16 parts | 0 |

The last row is the one that makes this a closure rather than a pause: even
choosing parts *with the annotation in hand*, no union of ≤16 from any bank
reached IoU 0.50 for a single entity Mask3D missed.

**One post-result change is on the record.** The candidate-set rule was written
after the strict result had been scored. It is a correctness fix authorised as
such, it can only *add* evidence — 13 detections that were being thrown away —
and it moved the gated metric not at all. It is not tuning toward a number, but
it is not blind either, and it should be read that way.

**What no longer needs re-litigating.** Multi-view lifting on this capture is
sound: 79.5% entity visibility, 84.3% mask coverage of what is visible, 8/12
loopback probes recovered from perfect masks. Every arm fails the same way —
high precision, low recall against ARKitScenes oriented-box extents.

**Never touched.** `41069025` has no frame bundle and no run, so the audited
counter/cushion cases were never evaluated by any arm; that gap is open, not
answered. `47331972` was never downloaded or inspected.

**Survives.** `eval/detection_repair.py`, byte-identical across all nine
checkpoints, with its reproduced Mask3D baseline pinned by test.

---

# Checkpoint J — human spatial-QA key for 41069025, and one score against it

Repair track stays closed. No repair run on 41069025; `47331972` untouched. No
perception change: the scorer loads finalized Lane A artifacts and reads them.

## The key

`eval/human_feedback/arkitscenes_41069025_spatial_qa_key_v1.json`, status
**DRAFT_PENDING_OWNER_CONFIRMATION**. Seven items, each restating a fact from
`arkitscenes_sealed_visual_review_2026-08-09.json`. Nothing was taken from
annotation boxes, Mask3D output, learned labels or any run artifact, and the
key was written and hashed before a single system answer was read.

It is **independent by construction**: every question is phrased about the
room, never about a delivered instance id, so it survives a perception change
and cannot become the system grading itself. A test enforces that no
`obj_N` appears in any question.

Four **unresolved UID mappings** are flagged rather than guessed — the single
true rug, the single true trash can, the main kitchen counter, and cushion
cardinality. The review confirms each exists (or diagnoses the impostor) but
never says which delivered instance it is, so no per-instance item is scored.
Those are the only things needing owner confirmation.

## Score: Mask3D delivered + rgb_tight, one run

**Superseded by the key v3 score below. Corrected 2026-08-10 after an external
review noticed this section still narrated the v1 DRAFT key while the committed
artifact had moved on — the stale text flattered the result by roughly 2x, in a
project whose whole claim is evidentiary hygiene. Recorded rather than
overwritten.**

The v1 DRAFT run (7 items) read **2 correct, 4 wrong, 1 unanswered**, 0.333
excluding unanswered. Two of those three non-failures did not survive
confirmation: `q5` asked only whether cushions were PRESENT, and it passed
solely because the sofa was mislabelled a cushion.

### Current, against key v3 FINAL — the number of record

`runs/arkit_spatial_qa/arkitscenes_41069025_human_spatial_qa.json`, 8 items,
after owner UID confirmation.

35 delivered instances, 27 with admitted labels, 8 anonymous. Graph: 151 edges,
all `NEAR`.

| item | expected | system | |
|---|---|---|---|
| q1 rug cardinality | 1 | 0 | WRONG |
| q2 trash-can cardinality | 1 | 2 (`obj_1`, `obj_12`) | WRONG |
| q3 counter cardinality | 1 | 2 (`obj_16`, `obj_27`) | WRONG |
| q4 sofa present | true | false | WRONG |
| q5 cushion cardinality | 2 | 3 (`obj_9`, `obj_13`, `obj_23`) | WRONG |
| q6 cushions rest on sofa | `cushion ON_ENTITY_SURFACE sofa` | — | **unanswered** |
| q7 counter is object-scale | true | true | ok |
| q8 cushion identity | {obj_9, obj_23} | {obj_9, obj_13, obj_23} | WRONG |

**1 correct, 6 wrong, 1 unanswered.** 0.143 excluding unanswered; 0.125
counting it as failure.

`q6` is `unanswered`, not `wrong`: the delivered graph emits no
`ON_ENTITY_SURFACE` edge of any type, because Checkpoint 2 recorded the
geometry-only floor census as `unknown` here and no support edge is promoted.
A system that cannot express an answer has not given a wrong one, and
collapsing the two would hide a missing capability behind a mistake.

## A confound the scorer flags automatically

`q4` fails and `q5` passes **for the same underlying error**. The review
diagnoses `obj_13` as the sofa; rgb_tight labels it `cushion`. So the system is
credited with finding cushions partly because it mislabelled the sofa as one,
and penalised for missing the sofa for the identical reason. The key declares
this as `sofa_cushion_coverage`-style coupling and the scorer prints it
whenever the signature (q4 wrong ∧ q5 correct) appears. **q5 should not be read
as evidence the cushions were found.** On a stricter reading the honest score
is 1/6 answerable.

## What this does and does not say

The two failures that are *not* label errors are informative: `q3` and `q7`
together show the system reports two counters, neither room-spanning, against
a scene with one — and the earlier detection work showed the room-spanning
plane (`obj_8`, 7.38 × 5.95 × 0.20 m) exists but is not labelled `counter`
under rgb_tight, which is why `q7` passes. `q7` passing is therefore weak
evidence, not a capability.

Seven items, one partial review, one scene. **This is not an end-to-end QA
benchmark result**, and the repair track's closure is unaffected by it.
