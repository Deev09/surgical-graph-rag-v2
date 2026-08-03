# Oracle-free proposal selector — measured result

Status: **measured, evaluation-only.** Not wired into the graph pipeline.

## v1 change: `connectivity` dropped from the default score

`DEFAULT_COMPONENTS = ("agreement", "size", "redundancy")`. The v0 ablation
(below) measured `connectivity` as net-harmful on transfer; re-running with it
off improves or ties AR@k on **every scene and both banks, never worse**:

| scene / bank | v0 (with connectivity) | v1 (without) |
|---|---|---|
| room_2 p1 | 10 18 21 25 25 25 | 10 18 **22** 25 25 25 |
| room_2 pooled | 10 18 20 25 32 33 | 10 18 **21** 25 32 33 |
| room_1 p1 | 6 10 13 13 13 13 | **7 11** 13 13 13 13 |
| room_1 pooled | 5 12 14 18 21 26 | **7 13 15** 18 21 26 |
| office_0 p1 | 3 5 6 12 12 12 | **4 9 10** 12 12 12 |
| office_0 pooled | 3 6 7 12 18 19 | **4 9 10** 12 18 19 |

office_0 gains most (+4 entities at k=25 and k=50) and its junk rate at k=10
falls 0.70 -> 0.60 — consistent with connectivity having been a weak proxy that
promoted large contiguous wall patches.

**COMPARABILITY CAVEAT.** v0 was frozen before transfer scenes were read; v1 is
not. This default was chosen using an ablation measured on room_1 and
office_0, so v1 numbers on those scenes are no longer a clean held-out
measurement. v0 remains reproducible via `components=COMPONENTS_V0`, and both
variants are reported side by side in every eval run (`v1_default` / `full`).
The tables below the ablation section are the original v0 measurements and are
left unchanged.

`runs/` is gitignored, so this file is the durable record. Reproduce with
`python3 tools/p1_selector_eval.py`; outputs land under `runs/selector_v0/`.

## The problem this addresses

C1-P1 emits ~534 mask proposals per scene. Choosing which become entities was
done by computing IoU against oracle entities and keeping the best matches —
that step reads the answer key. Consequence: on a scene without annotations the
pipeline cannot produce a graph at all. This is an *executability* gap, not a
quality gap, and it is the standing objection to the C1-P1 result ("the gain
may come partly from admitting a large proposal bank").

## The scorer (`segmenter/selector_free.py`)

    score = agreement
          x gate(connectivity) x gate(size_prior) x gate(exp(-0.5 * n_nested_better))
    gate(x) = 0.3 + 0.7x,  score in [0, 1]

`agreement`: for every view where the proposal is >=50% visible, the best IoU
between its visible part and any lifted 2D mask; score is the mean of the best
3 such views. Reuses `proposal_fusion.lift_mask` and the `edge_confidence`
view-dict contract — the same co-membership evidence that built the bank.

**Oracle-freedom is structurally enforced, not asserted.** The module imports
only `numpy`/`dataclasses`/`__future__` and accepts only in-memory arrays: no
path argument, no config, no I/O. `tests/segmenter/test_selector_free.py`
checks this three ways — an AST import allowlist, an AST scan rejecting
`open`/`np.load`/`fromfile`/`exec`/`eval`/`__import__`, and a
`sys.addaudithook` around a live scoring call that fails on any file, import,
subprocess or socket event (verified to fire on a deliberate `open`). Oracle
IoU is read in exactly one place: `tools/p1_selector_eval.py`, the evaluator.

Ranking is deterministic and byte-identical across runs; ~3-4 s/scene, numpy only.

## AR@k and oracle recovery, IoU 0.50

k = 10 / 25 / 50 / 100 / 200 / all. `recovery` = fraction of the oracle
selection ceiling retained at budget k. `zero_overlap` = fraction of the top-k
overlapping no scored entity.

### P1 bank

| scene | n | ceiling | AR@k | recovery | zero_overlap |
|---|---|---|---|---|---|
| room_2 (dev) | 534 | 25/53 | 10 18 21 **25** 25 25 | .40 .72 .84 **1.00** 1.00 1.00 | .00 .04 .22 .49 .73 .90 |
| room_1 | 406 | 13/45 | 6 10 **13** 13 13 13 | .46 .77 **1.00** 1.00 1.00 1.00 | .10 .24 .42 .61 .81 .90 |
| office_0 | 335 | 12/47 | 3 5 6 **12** 12 12 | .25 .42 .50 **1.00** 1.00 1.00 | .70 .72 .74 .70 .83 .90 |

### Pooled P1 + Mask3D raw

| scene | n | ceiling | AR@k | recovery | zero_overlap |
|---|---|---|---|---|---|
| room_2 (dev) | 705 | 33/53 | 10 18 20 25 32 33 | .30 .55 .61 .76 .97 1.00 | .00 .04 .10 **.15** .18 .72 |
| room_1 | 569 | 26/45 | 5 12 14 18 21 26 | .19 .46 .54 .69 .81 1.00 | .20 .20 .20 **.22** .21 .67 |
| office_0 | 521 | 19/47 | 3 6 7 12 18 19 | .16 .32 .37 .63 .95 1.00 | .70 .72 .68 .53 .47 .71 |

## What this does and does not show

**Shows:** on the P1 bank, an oracle-free ranking retains the entire oracle
selection ceiling at k=50-100 — a 3.4x to 8.1x cut of the bank with zero loss,
on both transfer scenes as well as the tuned one. The oracle was doing no work
at selection time.

**Does not show:** any improvement in entity recall. The ceiling is unchanged
(47% dev, 29% and 26% transfer). This makes the pipeline runnable without an
answer key; it does not make it accurate. Those are separate claims.

**The two banks trade off, and the trade matters.** P1-only gives full recovery
with a dirty head (49-70% of the top-100 matches no entity). Pooled gives
~63-76% recovery with a much cleaner head (15-22% on room_2/room_1). Quote both
together; quoting P1 recovery next to pooled cleanliness, or the reverse,
overstates the result in either direction.

**AR@k alone is not sufficient for deployment.** It measures whether the right
proposal is somewhere in the top k, not what happens when a set is committed to.
A final selection stage (3D NMS) with an instance-precision metric and
end-to-end QA evaluation is the required next increment.

## Signal ablation (AR@k @IoU 0.50, k = 10/25/50/100)

| variant | room_2 | room_1 | office_0 |
|---|---|---|---|
| full | 10 18 21 25 | 6 10 13 13 | 3 5 6 12 |
| -size_prior | 10 17 21 25 | 6 10 13 13 | 2 5 6 12 |
| -connectivity | 10 18 **22** 25 | **7 11** 13 13 | **4 9 10** 12 |
| -redundancy | 9 17 22 25 | 5 10 13 13 | 2 5 6 12 |
| agreement only | 10 18 22 25 | 6 10 13 13 | 3 **6 9** 11 |
| gates only | 4 8 17 20 | 3 8 11 11 | 2 5 7 8 |

Single raw signals (room_2, P1): agreement 10 18 22 25 · support_frac 8 16 21 23
· n_vertices 3 11 16 24 · connectivity 3 7 8 20 · size_prior 4 9 9 9 ·
**sam_quality 1 1 4 7** · random 1 2 2 4.

- `agreement` is essentially the whole scorer; everything else is second-order.
- **`connectivity` is net-harmful and should be dropped in v1** — neutral on
  dev, clearly negative on both transfer scenes. Not removed here: the frozen
  selector was not changed after transfer results were seen.
- `redundancy` is the one real addition, and only on the pooled bank — Mask3D's
  raw output contains blocks of identical masks that survive each other's
  comparison without an index tiebreak.
- **SAM's own predicted-IoU x stability is worse than random** (1 1 4 7 vs
  1 2 2 4). The bank was generated at `pred_iou_thresh=0.8` /
  `stability_thresh=0.95`, so surviving masks span 0.80-0.996 with no
  discriminative range left. The signal was spent upstream.

## Failure modes

1. **Structural surfaces dominate, not bad geometry.** Seven of office_0's top
   ten P1 proposals are flat wall/floor/ceiling patches scoring agreement ~0.97.
   A clean planar patch of wall *is* a stable, multiview-consistent single 2D
   mask; multiview agreement cannot separate stuff from things by construction.
   This fully explains office_0's weak low-k numbers and converges with the C3
   surface-estimator closure from the opposite direction. The next increment
   belongs here — a thing-vs-stuff cue, not more of this signal family.

2. **The scorer is circular with respect to SAM.** Proposals were built from
   co-membership of these same 2D masks, and the score rewards agreeing with
   them. An object SAM never segments whole in any single view — always split
   into parts, or always fused with its neighbour — cannot score highly by
   construction. The selector inherits SAM's 2D biases exactly and cannot
   repair a generator miss.

3. **Constants are dev-tuned with oracle feedback.** The module reads no ground
   truth at inference, but `VIS_MIN`, `TOP_VIEWS`, `GATE_FLOOR`,
   `REDUNDANCY_ALPHA` and the size thresholds were chosen against room_2's
   AR@k. The ablation shows that tuning bought roughly nothing beyond
   `agreement`.

## Dataset portability

`size_prior` is scale-specific: a metric indoor-room-object prior (extent ramps
0.03->0.10 m and 2.0->2.6 m) assuming a metric, gravity-aligned mesh. It would
be wrong on a different-scale capture — and it contributes ~0, so it can be
deleted at no measured cost. See `docs/frame_and_scale_audit.md` for the
scale-variance measurements behind that judgement.

`agreement`, `connectivity` (voxel resolution set by each proposal's own
extent) and `redundancy` are scale-free. `connectivity` does assume a densely
and roughly uniformly sampled surface — true for Replica meshes, not for sparse
point clouds.

## Known gap in the artifacts

`runs/selector_v0/summary.json` contains only `replica_room_1`. The per-scene
`*_selector_eval.json` files carry the complete data and are what the tables
above are read from.
