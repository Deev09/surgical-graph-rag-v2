# ARKitScenes R1 — render splat density (draft protocol)

Status: **draft, unexecuted.** No code has been written. Baselines quoted
here were measured before it was drafted (`runs/arkitscenes_p1/`, commits
`7af4164` and `0c6e83e`).

Follows `docs/arkitscenes_fusion_evidence_protocol.md`, whose verdict
exonerated fusion as the binding constraint and directed the investigation
upstream. Related: `docs/c1_p1_multiview_proposals_protocol.md` (the frozen
render contract this modifies).

---

## Decision this experiment answers

ARKitScenes renders fill **53.7%** of pixels against Replica's **72.8%**,
and SAM masks cover **28.9%** of occupied pixels against **46.5%**. The
resulting bank has a ceiling of 1/18. Is that sparsity the binding
constraint, and does a denser splat fix it?

A pass adopts a denser render for real-capture geometry. A fail closes the
splat-density line and leaves the C1-P1 mechanism unsupported on real scans
— at which point the honest conclusion is that render-and-lift does not
transfer off synthetic meshes, and the next candidate is a different
proposal source entirely.

## Two hazards this design exists to avoid

**H-A: circular gating.** Splat size trivially raises pixel fill. Gating on
fill would report the knob's own setting back as a result. Fill is
**reported and never gated**. Every gate is on entity ceiling or proposal
geometry.

**H-B: two variables in one change.** A larger splat changes both what SAM
*sees* (denser RGB → different masks) and what a mask *lifts* (each vertex
paints more pixels → masks lift more vertices). A naive single-arm run
cannot tell which produced any gain.

The design separates them with an arm that costs no GPU:

| arm | RGB splat | id-buffer splat | SAM masks | isolates |
|---|---|---|---|---|
| **baseline** | 3×3 | 3×3 | existing | — |
| **C** | 3×3 | **5×5** | **existing, byte-identical** | lifting dilation alone |
| **A** | **5×5** | **5×5** | new run | lifting + perception |

Arm C reuses the mask sidecar already on disk **unchanged**, so SAM's output
is literally identical and only lifting differs. It is CPU-only. Run C
first: if C alone reaches A's gates, the GPU run is unnecessary and the
"SAM sees better" story is unsupported.

## Hypothesis

**H:** occupied-pixel mask coverage of 28.9% is too low for multiview
co-membership to bind objects together, and it is low because the render is
sparse rather than because SAM is weak. A 5×5 splat raises fill toward
Replica's regime, raising mask coverage and lifting the entity ceiling.

**Predicted direction, stated before the run.** Pixel fill rises; occupied
mask coverage rises; ceiling rises; median proposal size rises.

**Predicted failure mode, also stated before the run.** Denser lifting
raises co-membership confidence on every edge, exactly as the refuted
`masked` denominator did. At frozen cut thresholds that risks the same
over-merge into blobs — largest proposal reached **265,346 vertices (26% of
the mesh)** in F1. R3b exists to catch that, and a blob outcome is a FAIL,
not a partial win.

## Baseline (measured, dev scene 41069021)

| quantity | value |
|---|---|
| pixel fill / view | 53.7% (Replica room_2: 72.8%) |
| occupied px inside any mask | 28.9% (46.5%) |
| 2D masks / lifted | 457 / 449 |
| proposals | 1733 |
| median proposal size | 37 vertices |
| largest proposal | 142,278 (14.1% of mesh) |
| ceiling @IoU 0.50 / 0.25 / 0.10 | **1/18** · 7/18 · 12/18 |

## Frozen anchors

* SAM 2.1 pin, checkpoint SHA, and every automatic-mask-generator parameter.
  **Not touched.** `points_per_side` stays 32.
* Fusion: `evidence_denominator="covisible"`, cut thresholds 0.25/0.50/0.75.
  **Not touched, not swept.** The `masked` mode was refuted in F1 and is not
  combined with this change — one variable.
* Camera set, view count (40), canvas size (1024), VFOV, pitch, eye height,
  origin fraction. Only the splat kernel changes.
* Every Replica artifact. The Replica path keeps the 3×3 splat regardless of
  outcome; changing it would move frozen results and needs its own protocol.

## The change

`segmenter/view_render.py` gains a splat parameter defaulting to today's
kernel, plus the ability to render the id buffer at a different kernel from
the RGB — the mechanism arm C needs:

    SPLAT_OFFSETS_3X3   # current, default
    SPLAT_OFFSETS_5X5   # the single tested alternative

One alternative value. **No sweep.** 5×5 is chosen because it is the next
symmetric kernel, not because it was tuned.

## Isolation boundary

* Annotations remain readable only by `tools/arkitscenes_eval.py` and
  `tools/arkitscenes_selector_eval.py`, below their ORACLE BOUNDARY comments.
* **41069025 and 41069042 stay sealed.** They have never been fused,
  evaluated, or inspected under any condition and must not be touched in
  Stage 0 or Stage 1.
* Gates are fixed in this document before execution and are not adjusted
  after seeing results.

## Stage 0 — validity (no new inference)

| gate | predeclared criterion |
|---|---|
| W1 | at the default 3×3 kernel, re-rendered id buffers and RGB PNGs are **byte-identical** to the committed `views_arkitscenes_41069021/` artifacts (per-file sha256 from the existing manifest) |
| W2 | `tools/run_tests.py` green; scorecard 4 / 27 / 22 / 3; all six Replica bundle hashes unmoved |
| W3 | arm C consumes `c1p1_masks_arkitscenes_41069021.npz` with its sha256 unchanged — SAM output is provably reused, not regenerated |
| W4 | synthetic fixture: a single vertex rendered at 5×5 paints exactly 25 id-buffer pixels and 3×3 paints exactly 9, with correct clipping at canvas edges |

Any W failure: **STOP.** No arm runs.

## Stage 1a — arm C (CPU only, dev scene)

Re-render id buffers at 5×5, re-lift the **existing** masks, re-fuse, evaluate.

Reported, not gated: pixel fill, occupied-mask coverage, lifted-mask sizes.

**Decision rule, not an adoption gate:**

| arm C outcome | consequence |
|---|---|
| C alone meets every R gate below | Adopt C. **Do not run arm A** — the GPU spend would buy nothing and the perception story is unsupported. |
| C moves ceiling @0.50 by ≥2 entities but misses gates | Arm A's result is confounded by lifting dilation. Run A, and report A **relative to C**, never relative to baseline. |
| C moves ceiling @0.50 by <2 entities | Lifting dilation is not the mechanism. Any arm-A gain is attributable to perception. |

## Stage 1b — arm A (one GPU run, dev scene)

Render RGB and ids at 5×5, one SAM run, fuse, evaluate.

| gate | predeclared criterion |
|---|---|
| R1 | ceiling @IoU 0.50 ≥ **6/18** (baseline 1/18) |
| R2 | ceiling @IoU 0.25 ≥ **12/18** (baseline 7/18) |
| R3a | median proposal size ≥ **300** vertices (baseline 37) |
| R3b | **no proposal exceeds 15% of mesh vertices** (baseline 14.1%; F1's failure hit 26%) |
| R4 | ≤ 2,000 proposals and ≤ 2 GiB serialized |
| R5 | selector v1 recovers ≥ **0.80** of the new ceiling at k=100 |
| R6 | arm A exceeds arm C on ceiling @IoU 0.50 by ≥ **2** entities |

All seven must pass. Two clauses carry the lessons of F1:

* **R3a and R3b are two-sided.** Too small is fragments; too large is blobs.
  F1 was rejected for producing a quarter-of-the-mesh proposal, and this
  protocol will reject the same outcome rather than reading it as
  consolidation.
* **R6 is the anti-confound gate.** If arm A does not beat the free CPU arm,
  the denser *render* bought nothing beyond denser *lifting*, and adopting a
  GPU-dependent change on that evidence would be unjustified.

## Stage 2 — sealed transfer

Only if every Stage-1b gate passes. Run the identical committed configuration
once on 41069025 and once on 41069042, with no intervening change. Evaluate
both only after both banks are final.

| gate | predeclared criterion |
|---|---|
| S1 | ceiling @IoU 0.50 improves on **both** scenes versus their own 3×3 baseline, computed in the same run |
| S2 | both satisfy R3a and R3b |
| S3 | both stay within R4 caps |

Genuinely sealed: neither scene has been fused, evaluated, or inspected under
any condition.

## Budget, stopping rule, and decision

* Arm C: **no GPU.** CPU re-render (~10 s) plus fusion (~25 s).
* Arm A: **one** SAM run on the dev scene; two more only if Stage 1b passes.
* One splat value. No sweeps, no per-scene tuning, no rescue run.
* A run invalidated by a crash is rerun from scratch and reported.

| outcome | decision |
|---|---|
| W fails | STOP. Implementation defect. Nothing claimed. |
| C meets all R gates | Adopt arm C. No GPU spend. Record that perception was never the constraint. |
| A fails, R3b the cause | Over-merge, same failure as F1 by a different route. Negative result: the cut thresholds do not survive a changed confidence distribution, and **that** — not splat density — is the next question. |
| A fails otherwise | Negative result. Splat density is not the binding constraint. Close the render line; the honest reading is that render-and-lift does not transfer off synthetic meshes, and the next candidate is a different proposal source. |
| A passes, Stage 2 fails | Negative transfer. Not adopted. The dev gain is a 41069021 artifact. |
| Both pass | Adopt 5×5 for ARKitScenes. Replica stays 3×3 unless a separate protocol re-runs its gates. |

## Explicitly out of scope

* SAM parameters, checkpoint, or prompt configuration.
* Fusion changes of any kind, including combining this with the refuted
  `masked` denominator.
* Cut-threshold sweeps — even though the F1 verdict named cut calibration as
  an open question. That is a separate experiment.
* Canvas size, camera set, VFOV, view count.
* Any relation-threshold, reasoner, or QA change.

## Required artifacts and reporting

* the committed render diff and its W4 synthetic test;
* per-arm `bank_*.npz` sha256 and `*_bank.json`;
* pixel fill and occupied-mask coverage per arm, **reported as context, not
  as evidence of success**;
* the gate tables above with measured values filled in, pass or fail;
* the verdict committed to this file, negative results included.

## Sign-off

Drafted 2026-08-03. Arm structure and gate values approved by the owner
before any code was written. Stage 0 and Stage 1a executed 2026-08-03;
Stage 1b awaiting its single GPU run.

---

# 2026-08-03 interim — STAGE 0 PASSES · ARM C IS NULL · ARM A PENDING

## Stage 0 — all validity gates pass

| gate | criterion | measured |
|---|---|---|
| W1 | default 3×3 render byte-identical | 40/40 RGB sha match, 40/40 id sha match, `ids.npz` sha identical |
| W2 | Replica untouched | 84/84; 4 / 27 / 22 / 3; six bundle hashes unmoved |
| W3 | arm C reuses SAM output unchanged | 0/40 RGB mismatches, 40/40 id buffers dilated, sidecar sha `03fa67f9…` reused |
| W4 | splat kernels paint exactly k pixels | 3×3 → 9, 5×5 → 25; edge clipping clamps, never wraps; 6/6 |

The default path is bit-for-bit what it was before the kernel became a
parameter — asserted directly, not inferred from code shape.

## Stage 1a — arm C (CPU only): **NULL RESULT**

| | baseline 3×3/3×3 | arm C 3×3/5×5 |
|---|---|---|
| id-buffer fill | 53.7% | **74.6%** |
| proposals | 1733 | 1428 |
| median proposal size | 37 | 38 |
| largest proposal | 142,278 (14.1%) | 147,560 (14.6%) |
| ceiling @IoU 0.50 | **1** | **1** |
| ceiling @IoU 0.25 | 7 | 5 |
| ceiling @IoU 0.10 | 12 | 11 |

**Δ ceiling @0.50 = +0.** The decision rule's third row applies: *"C moves
ceiling @0.50 by <2 entities → lifting dilation is not the mechanism. Any
arm-A gain is attributable to perception."*

This is the arm's whole purpose and it did its job. Lifting dilation alone
raised id-buffer fill by 21 points and changed nothing that matters — the
ceiling is flat at 0.50 and slightly *worse* at looser thresholds. Arm A is
therefore unconfounded: whatever it produces is attributable to what SAM
sees, not to how masks lift.

**Note for the record, not a gate.** Larger splats occlude each other, so
distinct visible vertices per view *fell* — median 114,255 → 80,997 — even
as pixel fill rose. Fill and evidence are not the same quantity, which is
precisely why H-A forbids gating on fill.

## Stage 1b — ARM A FAILS. SPLAT DENSITY IS NOT THE BINDING CONSTRAINT.

| gate | criterion | measured | verdict |
|---|---|---|---|
| R1 | ceiling @IoU 0.50 ≥ 6/18 | **1** (baseline 1) | **FAIL** |
| R2 | ceiling @IoU 0.25 ≥ 12/18 | **8** (baseline 7) | **FAIL** |
| R3a | median proposal size ≥ 300 | **38** (baseline 37) | **FAIL** |
| R3b | largest ≤ 15% of mesh | 11.2% (baseline 14.1%) | pass |
| R4 | ≤ 2000 proposals | 1455 | pass |
| R5 | selector recovery ≥ 0.80 @k=100 | 1.00 | pass |
| R6 | arm A beats arm C by ≥2 @0.50 | **0** | **FAIL** |

R3b passed, so the decision table's *"A fails otherwise"* row applies:
**close the render line.**

### The change worked. The outcome did not.

Everything the hypothesis predicted upstream happened, and none of it
reached the ceiling:

| | fill | masks/view | occupied px in a mask | ceiling @0.50 |
|---|---|---|---|---|
| ARKitScenes baseline | 53.7% | 11.4 | 28.9% | **1/18** |
| **ARKitScenes arm A** | **74.6%** | **16.4** | **30.2%** | **1/18** |
| Replica room_2 | 72.8% | 15.5 | **46.5%** | **25/53** |

Arm A **matches or exceeds Replica on both render fill and mask count**.
SAM responded exactly as hoped — 43% more masks, 449 → 601 lifted. The
entity ceiling moved by zero, and median proposal size by one vertex.

### What the run newly identified

**Occupied-pixel mask coverage is the quantity that separates the datasets,
and it does not respond to render density.** It moved 28.9% → 30.2% while
fill moved 21 points and mask count moved 43%. Replica sits at 46.5%.

So ARKitScenes gets *more* masks that cover *no more surface*: SAM's masks
on real-scan renders are more numerous and more localised, and they do not
tile the visible surface the way they do on synthetic renders. That is a
segmenter-behaviour property, not a rasterisation one, and this protocol
freezes every SAM parameter — deliberately. Naming the next question is not
authorising it.

### Verdict

The measured chain from `7af4164` is now eliminated end to end:

    render density   ← tested here (arm A), null
    lifting          ← tested here (arm C), null
    fusion evidence  ← tested in F1, refuted
    mask coverage    ← the residual, newly isolated, untested

Per the decision table, the honest reading stands: **render-and-lift does not
transfer off synthetic meshes on the frozen C1-P1 configuration.** C1-P1's
result is a Replica result until something outside this configuration changes
it. That is a limitation of the mechanism, established by measurement across
three predeclared experiments, not an implementation defect.

Stage 2 was **not run.** 41069025 and 41069042 have never been fused,
evaluated, or inspected under any condition and remain sealed.

## Stage 1b — arm A inputs (rendered before the run above)

`views_arkitscenes_41069021.rgb5x5_id5x5/` rendered at 5×5 for both RGB and
ids; id-buffer fill 74.6%, against Replica room_2's 72.8%. Upload tar is
16.1 MB, 40 PNGs, `ids.npz` withheld as always.

**Defect found and fixed during this stage.** `tar_for_upload` used
`Path.with_suffix(".tar.gz")`, which parses `.rgb5x5_id5x5` as the suffix
and silently wrote every arm's upload to the **baseline filename**. Arm A's
tar overwrote the baseline's before this was caught. Both were rebuilt from
their own view directories and verified to carry distinct roots. Committed
with a comment naming the trap, because the failure is silent and would have
sent the wrong images to the GPU.

No gate is claimed for arm A. R1–R6 remain as written.
