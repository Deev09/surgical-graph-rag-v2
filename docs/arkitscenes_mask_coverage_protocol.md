# ARKitScenes M1 — mask coverage via the stability gate

Status: **EXECUTED 2026-08-08. REFUTED.** Coverage was closed — overshot,
in fact, past Replica's own baseline — and the ceiling did not move. The
mandatory anchor arm additionally showed the change **destroys** Replica.
Verdict and measured gate table at the bottom; the protocol as written above
it is unedited.

Baselines quoted below were measured before it was drafted
(`runs/arkitscenes_p1/`, commit `e348ada`).

Follows `docs/arkitscenes_render_density_protocol.md`, whose verdict closed
the render line and isolated occupied-pixel mask coverage as the residual.
Related: `docs/arkitscenes_fusion_evidence_protocol.md` (F1, refuted),
`docs/c1_p1_multiview_proposals_protocol.md` (the pins this breaks).

---

## Decision this experiment answers

R1 brought the ARKitScenes render **above** Replica on fill (74.6% vs 72.8%)
and mask count (16.4 vs 15.5 per view), and the entity ceiling did not move
from 1/18. The one input quantity that stayed stuck is **occupied-pixel mask
coverage: 30.2% against Replica's 46.5%.**

Is that gap the binding constraint — and does closing it lift the ceiling?

A pass means the frozen SAM parameterisation, not the mechanism, was what
failed to transfer. A fail exhausts the measured chain and settles the
question C1-P1 has been circling since the first ARKitScenes run.

## What the data already says, before any run

Measured across all 40 views of the dev scene:

| | masks | median area px | p90 area | total mask area / canvas | occ. coverage |
|---|---|---|---|---|---|
| ARK baseline (3×3) | 457 | 6,181 | 58,279 | 0.34 | 28.9% |
| ARK arm A (5×5) | 655 | 5,612 | 49,931 | 0.35 | 30.2% |
| Replica room_2 | 619 | **13,804** | **97,184** | **0.56** | **46.5%** |

**SAM produces smaller masks here, not fewer.** Replica's are 2.2× larger by
median from a similar count. Arm A got 43% *more* masks that were *smaller*,
leaving total mask area per canvas flat (0.34 → 0.35).

This is direct evidence against `points_per_side` as the lever: arm A already
ran that experiment by proxy — 39% more surface for seeds to land on produced
43% more masks and **+1.3 points of coverage**. More seeds yield more small,
overlapping masks, not coverage of new ground. `crop_n_layers` would push the
same direction and is excluded for the same reason.

## Hypothesis

**H:** on stippled real-scan renders, the large "whole-object" candidate that
`multimask_output=True` proposes has noisier boundaries than its part-level
siblings, scores lower on stability, and is cut by
`stability_score_thresh=0.95`. The surviving mask is the smaller part. Lower
the gate and the larger masks survive, coverage rises toward Replica's
regime, and co-membership finally binds objects together.

**Single variable:** `stability_score_thresh` **0.95 → 0.85**. One value.
Nothing else moves — `pred_iou_thresh` stays 0.80, `points_per_side` stays
32, checkpoint and commit stay pinned.

### The mechanism is inferred, and here is the weakness

Surviving stability scores pile up immediately above the cut on **both**
datasets — p05 is 0.9508 (ARK) and 0.9512 (Replica) against a 0.95 threshold.
So the gate is demonstrably binding, but **it is binding equally on both**,
and the rejected mass is not observable from any committed artifact. This
protocol therefore cannot claim in advance that the gate explains the
*difference* between the datasets — only that it is a live constraint on
ARKitScenes. M6 and the comparability arm below exist because of that gap in
the argument, not despite it.

## Three hazards this design exists to avoid

**H-A: circular gating.** Lowering a rejection threshold trivially raises
mask count and coverage. Coverage is therefore a **direction check, never a
success criterion** — the same role F3 and R3a played. Success is entity
ceiling and fixed-budget AR@k.

**H-B: bank inflation.** The ceiling is a max over the whole bank, so
admitting more masks can raise it mechanically without the bank becoming more
useful. Countered by capping the bank **and** gating on AR@k at a fixed
k=100, not on ceiling alone. This is a gate the two previous protocols did
not need and would have benefited from.

**H-C: this breaks a frozen pin — the largest cost so far.** F1 and R1 both
left the SAM configuration untouched, so ARKitScenes and Replica banks stayed
products of one parameterisation and remained comparable. Changing
`stability_score_thresh` ends that. **Mitigation: the same configuration is
run on Replica room_2 in the same experiment**, so a like-for-like comparison
survives. That anchor arm is mandatory, not optional.

## Frozen anchors

* Everything in the SAM config except `stability_score_thresh`: checkpoint
  sha, sam2 commit, `points_per_side=32`, `pred_iou_thresh=0.80`,
  `box_nms_thresh=0.7`, `crop_n_layers=0`, `min_mask_region_area=0`,
  `multimask_output=True`, seeds 0.
* The **3×3 render** — R1 failed, so 5×5 was not adopted and this builds on
  the adopted configuration, not the rejected one.
* Fusion: `evidence_denominator="covisible"`, cuts 0.25/0.50/0.75. F1
  refuted the alternative; it is not combined here.
* Every Replica *baseline* artifact. The anchor arm writes to suffixed paths
  and must leave `bank_replica_room_2.npz` and its sha untouched.

## Baseline (dev scene 41069021, adopted 3×3 config)

| quantity | value |
|---|---|
| occupied px in a mask | 28.9% (Replica room_2: 46.5%) |
| median mask area | 6,181 px (13,804) |
| proposals | 1733 |
| median proposal size | 37 vertices |
| largest proposal | 14.1% of mesh |
| ceiling @IoU 0.50 / 0.25 / 0.10 | **1/18** · 7/18 · 12/18 |
| AR@k=100 @IoU 0.50 | 1 |

## Isolation boundary

* Annotations stay behind the ORACLE BOUNDARY in `tools/arkitscenes_eval.py`
  and `tools/arkitscenes_selector_eval.py` only.
* **41069025 and 41069042 stay sealed.** Never fused, evaluated, or
  inspected under any condition, through three protocols. That holds here.
* Gates are fixed below before execution and are not adjusted afterwards.

## Stage 0 — validity (no new inference)

> **V1 was amended before execution, by owner decision on 2026-08-05.** As
> drafted it required a GPU re-run at 0.95 reproducing the committed sidecar
> byte-for-byte. That is **not executable as specified**: SAM runs
> off-machine and Colab does not guarantee the same GPU model, so V1 could
> fail for reasons unrelated to the change and halt a valid experiment under
> the STOP rule. It also implied a third GPU run against a stated two-run
> budget. The substitute below is structural and cannot be confounded by
> hardware. Recorded here rather than silently swapped; no other gate moved,
> and nothing had been run when the amendment was made.

| gate | predeclared criterion |
|---|---|
| V1 | **(amended, structural)** the `SAM2AutomaticMaskGenerator(...)` call is diffed against the frozen notebook at `f373791` and **exactly one keyword argument differs** — `stability_score_thresh`, `0.95` → the declared variable. The argument *set* must be unchanged, so nothing may be added or removed either. |
| V2 | `tools/run_tests.py` green; scorecard 4 / 27 / 22 / 3; all six Replica bundle hashes unmoved |
| V3 | fuse/eval accept a mask sidecar produced under a non-default threshold and tag every artifact with it, so two parameterisations cannot silently share a filename — the failure mode caught twice already in R1 |

Any V failure: **STOP.**

## Stage 1 — dev scene 41069021 at `stability_score_thresh=0.85`

One SAM run. One value. No sweeps.

| gate | predeclared criterion |
|---|---|
| M1 | ceiling @IoU 0.50 ≥ **6/18** (baseline 1/18) |
| M2 | **AR@k=100** @IoU 0.50 ≥ **6** (baseline 1) — the anti-inflation gate; the ceiling may not be earned by a bigger bank |
| M3 | median proposal size ≥ **300** vertices (baseline 37) |
| M4 | ≤ **2,000** proposals **and** largest proposal ≤ **15%** of mesh vertices |
| M5 | selector v1 recovers ≥ **0.80** of the new ceiling at k=100 |
| M6 | **direction check:** occupied-pixel mask coverage ≥ **40%** (baseline 28.9%, Replica 46.5%) |

All six must pass. As in F1 and R1: **M1 or M2 passing while M6 fails is a
FAIL**, reported as "the ceiling moved for a reason this protocol did not
identify". A result whose mechanism is unconfirmed is not a result.

## Stage 1-anchor — Replica room_2 at the same threshold

Mandatory, same run, written to suffixed paths. **Reported, not gated** — it
is a decision input, not a success criterion.

Report: ceiling @0.50/0.25, occupied-pixel coverage, median mask area,
proposal count, and AR@k=100, each against Replica's own committed 3×3
baseline (25/53 @0.50, 46.5% coverage, 534 proposals).

## Stage 2 — sealed transfer

Only if every Stage-1 gate passes. Identical configuration, once each on
41069025 and 41069042, evaluated only after both banks are final.

| gate | predeclared criterion |
|---|---|
| T1 | ceiling @0.50 improves on **both** versus their own 3×3 baselines, computed in the same run |
| T2 | both satisfy M3 and M4 |
| T3 | both reach M6's coverage floor |

## Budget, stopping rule, and decision

* **Two SAM runs** in Stage 1 (ARKitScenes dev + Replica anchor); two more
  only if Stage 1 passes. Everything downstream is CPU.
* One threshold value. No sweeps, no per-scene tuning, no rescue run.

| outcome | decision |
|---|---|
| V fails | STOP. Nothing claimed. |
| M6 fails | The gate is not what limits coverage. Negative result; the stability hypothesis is refuted and the residual stands unexplained. |
| M6 passes, M1/M2 fail | **Coverage was closed and the ceiling still did not move.** The strongest available negative: mask coverage is *not* the binding constraint either, and the measured chain is exhausted. C1-P1's render-and-lift mechanism does not transfer, and the next candidate is a different proposal source — not another knob. |
| M1–M6 pass, anchor shows Replica improves **by a similar margin** | The pin was suboptimal for both datasets. Adopt it, but the **transfer gap is unexplained and must not be described as closed** — ARKitScenes would still trail Replica by roughly its original margin. |
| M1–M6 pass, anchor shows Replica ~unchanged | The pin was miscalibrated **for real-scan renders specifically**. This is the only outcome that supports "the parameterisation, not the mechanism, failed to transfer". |
| Stage 1 passes, Stage 2 fails | Negative transfer. Dev gain is a 41069021 artifact. |

Note the third row: a pass that also lifts Replica is **not** a
generalization result, and the decision table says so before the data exists.

## Explicitly out of scope

* `points_per_side`, `crop_n_layers` — argued against above from arm A's
  natural experiment, and excluded rather than left implicit.
* `pred_iou_thresh`, `box_nms_thresh`, `min_mask_region_area`, checkpoint,
  commit, seeds.
* Render, fusion, or cut changes; combining this with the refuted `masked`
  denominator or the unadopted 5×5 render.
* Threshold sweeps. If 0.85 is wrong, that is a finding, not an invitation.

## Required artifacts and reporting

* the committed diff and its V3 test;
* per-arm sidecar and bank sha256, with the threshold recorded in each;
* the gate table with measured values, pass or fail;
* the Replica anchor comparison, whatever it shows;
* the verdict committed to this file, negative results included.

## Sign-off

Drafted 2026-08-05. Approved by the owner on all three items, including the
V1 amendment recorded above. Executed 2026-08-08 with two GPU runs, exactly
as budgeted.

---

# VERDICT — refuted, 2026-08-08

## Stage 0

V1 (structural, exactly one keyword argument differs), V2 (85/85 test files,
scorecard 4/27/22/3, six Replica bundle hashes unmoved), V3 (both sidecars
record `stability_score_thresh=0.85` and their own scene string): **all
pass.** No STOP condition.

## Stage 1 — gates

| gate | criterion | measured | |
|---|---|---|---|
| M1 | ceiling @IoU 0.50 ≥ 6/18 | **2/18** (from 1/18) | **FAIL** |
| M2 | AR@k=100 @IoU 0.50 ≥ 6 | **2** (from 1) | **FAIL** |
| M3 | median proposal ≥ 300 vertices | **37** (from 37) | **FAIL** |
| M4 | ≤2000 proposals, largest ≤15% of mesh | 1550, 14.2% | pass |
| M5 | selector recovers ≥0.80 of ceiling at k=100 | 1.00 | pass |
| M6 | direction check: occupied coverage ≥40% | **67.7%** (from 28.9%) | pass |

**Three of six fail. The protocol fails.** Stage 2 is not entered and
41069025 / 41069042 remain sealed, unfused and uninspected, as they have
through all four protocols.

## The hypothesis was wrong in a specific, useful way

H predicted the gate was cutting large whole-object candidates and leaving
part-level siblings. If so, lowering it would return **larger** masks.

It returned **more** masks of the **same** size. Median mask area moved
6,181 → 7,241 px (+17%) while mask count went 457 → 1,272 (+178%). Coverage
rose because the canvas was flooded, not because objects came back whole:
total mask area went 0.34 → 0.99 canvases. The named mechanism is refuted
directly, not merely unsupported.

M6 clearing 40% and landing at 67.7% — above Replica's 46.5% baseline — is
what makes this conclusive rather than inconclusive. Coverage is no longer a
candidate explanation for the transfer gap. It was closed, overshot, and the
ceiling stayed at 2/18.

## What the extra masks did to fusion

Per-cut proposal counts show the bank migrating to the strict cut as
co-membership evidence saturates:

| | lifted masks | cut 0.25 | cut 0.50 | cut 0.75 | total |
|---|---|---|---|---|---|
| ARK 0.95 | 449 | 523 | 625 | 585 | 1733 |
| ARK 0.85 | 1249 | **171** | 392 | **987** | 1550 |
| Replica 0.95 | 619 | 72 | 242 | 220 | 534 |
| Replica 0.85 | 1284 | **3** | 13 | **81** | 97 |

More overlapping masks raise `co-masked / covisible` almost everywhere, so at
loose cuts the confidence graph merges into components too large to survive
the size cap, and they are rejected outright. Replica retains **3 proposals**
at cut 0.25, against 72. The bank stops spanning a range of scales and
collapses onto whatever the strict cut yields. Coverage went up; the scale
diversity that makes a bank useful went down.

## The anchor arm — the result the decision table did not anticipate

Mandatory under H-C, reported not gated:

| quantity | Replica 0.95 | Replica 0.85 |
|---|---|---|
| ceiling @IoU 0.50 | **25/53** | **7/53** |
| ceiling @IoU 0.25 | 39/53 | 11/53 |
| AR@k=100 @IoU 0.50 | — | 7 |
| proposals | 534 | 97 |
| occupied coverage | 46.5% | **87.5%** |
| median mask area | 13,804 px | 12,659 px |

The decision table offered three outcomes for a passing anchor: Replica
improves similarly, Replica unchanged, or Stage-2 negative transfer. It did
not contemplate the anchor **collapsing**, and that is what happened —
Replica lost 18 of 25 entities while its coverage nearly doubled.

So `stability_score_thresh=0.95` was not a miscalibrated pin inherited
carelessly. It is load-bearing, and for Replica it is close to optimal. The
one reading this rules out completely is the one the protocol was written to
test: that the parameterisation, not the mechanism, failed to transfer.

The pin is **not adopted**. Every committed baseline artifact stands
unchanged; the 0.85 arms live at `.stab085` paths beside them.

## Where this leaves C1-P1 on real scans

Four protocols, four measured failures, one attribution each:

| line | protocol | result |
|---|---|---|
| fusion evidence | F1 | refuted — bank got worse (12→9 @IoU 0.10) |
| lifting dilation | R1 arm C | null — +21 pts fill, Δceiling 0 |
| render density | R1 arm A | null — matched Replica on fill and mask count, Δceiling 0 |
| mask coverage | M1 | refuted — overshot Replica's coverage, Δceiling +1 |

Every input quantity that distinguished the ARKitScenes render from the
Replica render has now been closed or overshot, and the ceiling has moved by
one entity in total. **The measured chain is exhausted.** Under this
protocol's own decision table this is the strongest available negative: the
constraint is not fill, not dilation, not density, and not coverage.

The remaining difference between the two datasets is not a render statistic
at all — it is what SAM's masks *mean* on a stippled real-scan render. Mask
count, mask size, and pixel coverage can all be matched while the masks
still fail to align with object boundaries, and none of the four knobs
touches that.

**Stop turning knobs on C1-P1 for ARKitScenes.** The next candidate must be
a different proposal source, not another parameter: Mask3D, which is trained
on ScanNet — real scans, in-distribution for ARKitScenes and
out-of-distribution for Replica, the reverse of C1-P1's bias — and which
already contributed +8 entities on Replica (25/53 → 33/53 pooled).
Infrastructure exists: `notebooks/c1_mask3d_colab.ipynb`, `load_mask3d()`.

## Artifacts

* masks — `c1p1_masks_arkitscenes_41069021.stab085.npz`,
  `c1p1_masks_replica_room_2.stab085.npz`
* banks — `bank_arkitscenes_41069021.stab085.npz`,
  `bank_replica_room_2.stab085.npz`, each with the threshold recorded in its
  sidecar JSON
* reports — `runs/arkitscenes_selector/arkitscenes_41069021_selector_eval.stab085.json`,
  `runs/arkitscenes_selector/replica_room_2_m1_anchor.stab085.json`
* anchor tool — `tools/m1_replica_anchor.py`
