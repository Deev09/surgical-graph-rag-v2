# ARKitScenes F1 — mask-evidence denominator in fusion (draft protocol)

Status: **draft, unexecuted.** Nothing in this document has been run. The
baseline numbers it quotes were measured on the dev scene before it was
written (`runs/arkitscenes_p1/`, commit `7af4164`).

Related: `docs/c1_p1_multiview_proposals_protocol.md` (the frozen generator
this modifies), `docs/selector_v0_results.md`, `docs/frame_and_scale_audit.md`.

---

## Decision this experiment answers

The first end-to-end ARKitScenes run produced `ceiling@IoU0.50 = 1/18` on
the dev scene, against Replica room_2's 25/53. Is that because
`proposal_fusion.edge_confidence` mis-handles *absent* mask evidence — a
defect invisible at Replica's mask coverage — or is the ARKitScenes render
simply too sparse to support multiview fusion at all?

A pass adopts the corrected denominator and keeps the P1 mechanism alive on
real captures. A fail closes this line and moves the investigation upstream
to the renderer, with the fusion exonerated by measurement rather than by
assumption.

## Hypothesis

`edge_confidence` currently computes, for each mesh edge:

    confidence = (# co-visible views where both endpoints share >=1 mask)
                 / (# co-visible views)

A view in which both endpoints are **visible but unmasked** lands in the
denominator and not the numerator, so it counts as evidence *against* the
edge. It is not: it is *absence of evidence*.

That distinction is nearly free on Replica, where 46.5% of occupied pixels
fall inside some mask, and dominant on ARKitScenes, where only **28.9%** do.

**H:** restricting the denominator to views that actually carry mask
evidence for the edge — both endpoints visible **and** both assigned to at
least one mask — will raise co-membership confidence on real-capture
renders, merge the fragments the current cut produces, and lift the ceiling.

**Predicted direction, stated before the run.** Median proposal size rises,
proposal count falls, ceiling rises. If proposal size does not move, the
mechanism is not what this document claims and the result is negative
regardless of what the ceiling does.

## Baseline (measured, dev scene 41069021)

| quantity | value |
|---|---|
| ceiling @IoU 0.50 | **1/18** |
| ceiling @IoU 0.25 | 7/18 |
| ceiling @IoU 0.10 | 12/18 |
| proposals in bank | 1733 |
| median proposal size | 37 vertices |
| median entity size | 10,231 vertices |
| lifted masks | 449 (median 1308 vertices) |
| occupied pixels inside any mask | 28.9% (Replica room_2: 46.5%) |
| mesh: largest connected component per entity | 55–98%, median ~90% |

Ruled out by measurement before this protocol was written, and therefore
**not** what this experiment tests:

* **not a merge-threshold failure** — unioning every proposal ≥50% inside an
  entity still covers <50% for 14/18 entities;
* **not mesh connectivity** — the geometry could support one proposal per
  object;
* **not mask quality** — lifted masks are object-scale.

## Frozen anchors

* The 40-view contract, camera set, splat size, and render constants in
  `segmenter/view_render.py`. **Not touched by this experiment.**
* The SAM 2.1 pin, checkpoint SHA, and automatic-mask-generator
  parameterization. **Not touched.** No new GPU inference is required or
  permitted: the mask sidecars already on disk are reused byte-for-byte.
* The fusion cut thresholds (0.25 / 0.50 / 0.75). **Not swept.**
* Every Replica artifact: banks, bundle hashes, and the 4/27/22/3 scorecard.

## The change

One parameter on the fusion, defaulting to current behaviour:

    evidence_denominator: Literal["covisible", "masked"] = "covisible"

* `"covisible"` — today's rule. Default, so every frozen Replica
  reproduction is bit-identical by construction.
* `"masked"` — denominator counts only views where both endpoints are
  visible **and** each lies in at least one mask. An edge with no such view
  scores 0, matching today's treatment of never-co-visible edges, so the
  conservative case is unchanged.

No other behaviour changes. One variable.

## Isolation boundary

* Annotations are read only by `tools/arkitscenes_eval.py` and
  `tools/arkitscenes_selector_eval.py`, below their ORACLE BOUNDARY
  comments. The fusion never sees them.
* **41069025 and 41069042 are sealed.** Their mask sidecars are on disk and
  must not be fused, evaluated, or inspected during Stage 0 or Stage 1.
* Gates are set in this document before execution and are not adjusted after
  seeing results.

## Stage 0 — implementation validity (no ARKitScenes inference)

| gate | predeclared criterion |
|---|---|
| V1 | with `evidence_denominator="covisible"`, `bank_npz_sha256` for Replica room_2 is **byte-identical** to the committed value |
| V2 | `tools/run_tests.py` stays green and the scorecard stays 4 / 27 / 22 / 3 with all six bundle hashes unmoved |
| V3 | on a synthetic fixture where every endpoint is masked in every view, `"masked"` and `"covisible"` produce **identical** confidences — proving the change is inert where evidence is complete |
| V4 | on a synthetic fixture with a known unmasked view, the two modes differ in the predicted direction and by the hand-computed amount |

Any V-gate failure: **STOP.** The implementation is wrong; no dev run.

## Stage 1 — dev scene 41069021

Re-fuse the existing mask sidecar with `evidence_denominator="masked"`.
One run. No sweeps.

| gate | predeclared criterion |
|---|---|
| F1 | ceiling @IoU 0.50 ≥ **6/18** (baseline 1/18) |
| F2 | ceiling @IoU 0.25 ≥ **12/18** (baseline 7/18; the 0.10 ceiling is already 12/18, so this asks loose recall to convert) |
| F3 | median proposal size ≥ **300** vertices (baseline 37) — the direction check; failing this falsifies the stated mechanism even if F1 passes |
| F4 | bank ≤ **2,000** proposals and ≤ 2 GiB serialized (inherited G6 cap) |
| F5 | selector v1 recovers ≥ **0.80** of the new ceiling at k=100 — the bank must stay rankable, not just larger |

All five must pass. **F1 or F2 passing while F3 fails is a FAIL**, and is
reported as "ceiling moved for a reason this protocol did not identify".

## Stage 2 — sealed transfer

Only if every Stage-1 gate passes. Re-fuse 41069025 and 41069042 once each
with the identical committed code and no intervening change. Evaluate both
only after both banks are final.

| gate | predeclared criterion |
|---|---|
| T1 | ceiling @IoU 0.50 improves on **both** scenes versus their own `"covisible"` baseline, which is computed in the same run |
| T2 | median proposal size rises on both |
| T3 | both stay within the F4 caps |

These are genuinely sealed: neither scene has been fused, evaluated, or
inspected. A pass here is stronger evidence than the Replica Stage-2
transfer was, and may be described as such.

## Budget, stopping rule, and decision

* **No GPU.** Existing mask sidecars only. Fusion is CPU, ~25 s/scene.
* One parameterization. No cut sweeps, no per-scene tuning, no rescue run.
* A run invalidated by a crash is rerun from scratch and reported.

| outcome | decision |
|---|---|
| V fails | STOP. Implementation defect. Nothing is claimed. |
| Stage 1 fails, F3 also fails | Negative result. The absent-evidence hypothesis is **refuted**; the fusion is exonerated and the investigation moves upstream to the renderer. Commit and close. |
| Stage 1 fails, F3 passes | Partial: mechanism confirmed, magnitude insufficient. Negative result for adoption. Do **not** re-gate; record that a renderer change is likely needed *in addition*, and stop. |
| Stage 1 passes, Stage 2 fails | Negative transfer. Not adopted. The dev gain is a 41069021 artifact. |
| Both pass | Adopt `"masked"` as the ARKitScenes default. Replica stays on `"covisible"` unless a separate protocol re-runs its gates. |

The last row matters: a pass does **not** authorize flipping the default on
the Replica path. That would move frozen results and requires its own
experiment.

## Explicitly out of scope

* Renderer changes (`SPLAT_OFFSETS`, canvas size, camera set).
* SAM parameter changes (`points_per_side`, thresholds, checkpoint).
* Fusion cut-threshold sweeps.
* Any change to relation thresholds, the reasoner, or QA.
* Combining this with a renderer change in one run — one variable at a time.

## Required artifacts and reporting

* the committed fusion diff and its synthetic tests (V3, V4);
* per-scene `bank_*.npz` sha256 and `*_bank.json` for both modes;
* the gate table above with measured values filled in, pass or fail;
* the verdict committed to this file, negative results included, in the
  house style of `docs/c1_closeout.md`.

## Sign-off

Drafted 2026-08-03. Gate values approved by the owner before any code was
written. Executed 2026-08-03.

---

# 2026-08-03 verdict — STAGE 1 FAILS. HYPOTHESIS REFUTED. LINE CLOSED.

Stage 2 was **not run**: 41069025 and 41069042 have never been fused,
evaluated, or inspected under either mode, and remain sealed.

## Stage 0 — all validity gates pass

| gate | criterion | measured |
|---|---|---|
| V1 | Replica room_2 `bank_npz_sha256` unmoved | `74b2a9a3…5e07` **identical**, 534 proposals, digests identical |
| V2 | suite green, scorecard and hashes unmoved | 83/83; 4 / 27 / 22 / 3; all six hashes identical |
| V3 | modes identical where every visible vertex is masked | exact, both `1/2` |
| V4 | modes differ in the predicted direction, hand-computed | `covisible` 1/3, `masked` 1/2 |

The implementation is correct and inert by default. Nothing about the frozen
Replica path moved.

## Stage 1 — dev scene 41069021, one run

| gate | criterion | measured | verdict |
|---|---|---|---|
| F1 | ceiling @IoU 0.50 ≥ 6/18 | **1** (baseline 1) | **FAIL** |
| F2 | ceiling @IoU 0.25 ≥ 12/18 | **5** (baseline 7) | **FAIL** |
| F3 | median proposal size ≥ 300 | **41** (baseline 37) | **FAIL** |
| F4 | ≤ 2000 proposals, ≤ 2 GiB | 478, 0.7 MB | pass |
| F5 | selector recovers ≥ 0.80 of ceiling @k=100 | 1.00 | pass |

Full comparison on the dev scene:

| | covisible | masked |
|---|---|---|
| proposals | 1733 | 478 |
| median size | 37 | 41 |
| ceiling @0.50 | 1 | 1 |
| ceiling @0.25 | 7 | **5** |
| ceiling @0.10 | 12 | **9** |
| proposals ≥1000 verts | 82 | **30** |
| proposals ≥5000 verts | 23 | **8** |
| largest proposal | 142,278 | 265,346 |

**The change made the bank worse, not better.** Loose-threshold recall fell
(12→9 at IoU 0.10), and the mid-size proposals that carried it were
destroyed: those ≥1000 vertices fell 82→30 and those ≥5000 fell 23→8, while
the largest single proposal grew to 265k vertices — a quarter of the mesh.

## Decision, per the predeclared table

Stage 1 fails **and F3 fails**. That row reads: *"Negative result. The
absent-evidence hypothesis is refuted; the fusion is exonerated and the
investigation moves upstream to the renderer. Commit and close."*

Executed as written. No re-gating, no cut sweep, no rescue run, no transfer
spend.

## What actually happened, and what this does not refute

The mechanism was the opposite of the prediction. Raising confidence on
evidence-bearing edges let **more** edges survive the frozen cuts, so
components merged — but merged *past object boundaries* into blobs, absorbing
the mid-size proposals that had been the bank's only useful content. The
prediction was "fragments become objects"; the observation is "fragments and
objects alike become blobs".

**Stated confound, because it bounds the claim.** The cut thresholds
(0.25/0.50/0.75) were frozen by this protocol and are calibrated to the
`covisible` confidence distribution. `masked` shifts that whole distribution
upward, so the same cuts necessarily merge more aggressively. This experiment
therefore refutes:

> the absent-evidence correction **alone, at frozen cuts**, improves the
> ARKitScenes bank

and does **not** refute the weaker claim that the denominator is
conceptually wrong. Separating the two would require a joint
denominator-and-cut recalibration, which is a different experiment with its
own gates and its own dev/transfer split. It is not authorized here, and
this document does not claim the answer to it.

## Where the investigation goes

Upstream, as the decision table specifies. The measured chain from the
2026-08-03 dev run stands unchanged, with fusion now exonerated as the
binding constraint:

    render   53.7% pixel fill      (Replica 72.8%)
    SAM      28.9% of occupied px in any mask   (46.5%)
    fusion   ← tested here; not the binding constraint

A renderer experiment (splat density) is the next candidate and needs its own
protocol. Nothing about it is authorized by this one.

## Artifacts

* `segmenter/proposal_fusion.py` — `evidence_denominator`, default `covisible`
* `tests/segmenter/test_fusion_evidence_denominator.py` — V3/V4, 8/8
* `runs/arkitscenes_p1/bank_arkitscenes_41069021.masked.npz` and
  `arkitscenes_41069021_bank.masked.json`
* `runs/arkitscenes_selector/arkitscenes_41069021_selector_eval.masked.json`

The `"masked"` mode is retained in the codebase, off by default, because
deleting it would discard the only evidence that this question was asked and
answered.
