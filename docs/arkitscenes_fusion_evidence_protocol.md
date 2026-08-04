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

Drafted 2026-08-03. **Unexecuted.** Requires owner approval of the gate
values in Stage 1 and Stage 2 before any code is written — the gates are the
experiment, and setting them after seeing results would make this a tuning
exercise wearing a protocol's clothes.
