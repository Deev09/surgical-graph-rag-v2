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
3. **Lifting** — per-pixel nearest-visible-vertex buffer built with the
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
| `N_FRAMES` | 24 | enough co-visibility for a 3-view consensus; fusion cost is ~0.1 s/view |
| `MIN_MESH_COVERAGE` | 0.60 | rejects frames aimed at unreconstructed space |
| `FELZ_SIGMA` / `FELZ_K` / `FELZ_MIN_SIZE` | 0.8 / 300 / 120 px | standard Felzenszwalb defaults at this resolution |
| `MIN_CONSENSUS_VIEWS` | 3 | "multi-view" is the mechanism; 2 is a coincidence |
| `ADDITIONAL_MAX_CONTAINMENT` | 0.50 | majority-unexplained ⇒ candidate missing object |
| `SPLIT_MIN_CONTAINMENT` | 0.70 | mostly inside one parent ⇒ candidate piece |
| `SPLIT_MAX_SIZE_RATIO` | 0.70 | a "piece" must be materially smaller than its parent |
| `DUPLICATE_IOU` | 0.90 | a near-copy of a Mask3D proposal is not a repair |
| `MIN_PROPOSAL_VERTICES` | 200 | junk floor above the fusion floor of 20 |
| `MAX_PROPOSAL_FRAC` | 0.15 | equals the giant-mask threshold |
| `MAX_REPAIR_PROPOSALS` | 60 | keeps the pooled bank near 100 |

Two of these deserve explicit caveats rather than credit:

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
