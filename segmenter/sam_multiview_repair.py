"""SAM 2.1 multi-view instance repair: overlapping mask hypotheses, not a partition.

Design note: `docs/repair_arm_design_note.md` (Checkpoint C section).

WHAT THIS FIXES ABOUT THE PREVIOUS ARM
--------------------------------------
`segmenter/rgb_multiview_repair.py` treated 2D masks as an EXHAUSTIVE per-pixel
partition and fused them with mesh-edge consensus. Measured result: at every
confidence cut the consensus graph was one component covering 68-73% of the
mesh plus dust, because "do these two adjacent vertices land in the same
region" is ~1.0 even across real object boundaries when the regions tile the
image. That arm is a closed negative and is not reused here.

SAM masks are OVERLAPPING, NON-EXHAUSTIVE object hypotheses. Three consequences
are structural in this module, not conventions:

  * no per-pixel partition is ever formed. Each mask is lifted INDEPENDENTLY;
    a pixel in no mask contributes nothing, and a pixel in three masks
    contributes to three hypotheses.
  * background is never a region. There is no complement, no "everything
    else" mask, and no relabelling of unmasked pixels.
  * association happens between MASKS, in 3D, not between vertices. A cluster
    is a set of lifted masks from different views that agree on the same
    surface; the proposal is what those masks vote for.

Nested masks are kept rather than resolved: SAM emits a sofa and its cushions,
and both are legitimate hypotheses to hand to a detection evaluator that scores
a bank. Resolving them is a downstream job this arm deliberately does not do.

ORACLE-FREE. Reads the canonical mesh, the capture stream, the pinned SAM mask
sidecar, and the frozen Mask3D bank. It cannot reach an annotation.

WHAT IS IMPORTED RATHER THAN RESTATED
-------------------------------------
Front-surface visibility, coarse coverage and the emission/classification
constants come from `segmenter.rgb_multiview_repair`. The camera convention
comes from `extractors.arkitscenes_rgb_crops` (OpenCV axes, validated against
sensor depth at 2.5-4.1 cm median error). Reusing them keeps the two arms
comparable: the only thing that changed is the mask source and the fusion rule.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from segmenter.rgb_multiview_repair import (
    ADDITIONAL_MAX_CONTAINMENT, DUPLICATE_IOU, MAX_PROPOSAL_FRAC,
    MAX_REPAIR_PROPOSALS, MIN_MESH_COVERAGE, MIN_PROPOSAL_VERTICES,
    SPLIT_MAX_SIZE_RATIO, SPLIT_MIN_CONTAINMENT, RepairProposal,
    _forward_axis, coarse_coverage, visible_vertices,
)

# ---- stage 1: frame selection -------------------------------------------
# Top of the brief's 16-32 band. Every extra frame buys co-visibility, which
# is the binding resource: see the measurement below.
N_FRAMES = 32
# Minimum angle between any two selected view directions. This is what makes
# the set "angularly diverse"; it is a CONSTRAINT, and coverage multiplicity
# is the objective.
MIN_ANGULAR_SEPARATION_DEG = 8.0
# Selection maximises the number of mesh vertices seen at least this many
# times, not the number seen at all. Measured on the development scene at 32
# frames, share of the 37 frozen Mask3D objects with N views seeing >=50% of
# them:
#
#     selection                     >=2 views   >=3 views   mesh seen
#     farthest-point on direction     18/37        7/37        70.0%
#     multiplicity-greedy, 8 deg      24/37       17/37        75.0%
#
# Association needs an object in two or more views or it cannot cluster at
# all, so maximising spread -- which is what farthest-point does -- optimises
# against the mechanism. Measured before any SAM inference; no annotation.
COVERAGE_MULTIPLICITY_CAP = 3

# ---- stage 3: lifting ----------------------------------------------------
# A lifted mask below this is not an object hypothesis worth clustering.
MIN_MASK_VERTICES = 200
# A mask covering more than this share of the mesh is a wall/floor/whole-room
# hypothesis. Equals the evaluator's giant-mask threshold.
MAX_MASK_FRAC = MAX_PROPOSAL_FRAC

# ---- stage 4: association ------------------------------------------------
# Pairwise IoU is computed on a fixed subsample of mesh vertices; the full
# 1M-vertex sets across a few thousand masks are not tractable pairwise, and
# association only needs to know whether two masks describe the same surface.
# Exact vertex sets are used everywhere a proposal is finally scored.
ASSOC_ANCHOR_STRIDE = 16
# Two masks from different views describe the same surface at or above this.
ASSOC_IOU = 0.30
# ...or when one is almost entirely inside the other at a comparable scale.
# Containment alone chains a cushion to a sofa to a room, so it is admitted
# only with the size guard below.
ASSOC_CONTAINMENT = 0.80
ASSOC_CONTAINMENT_MIN_SIZE_RATIO = 0.50
# Distinct views a cluster needs. Three would be stronger, but the measurement
# above says only 17 of 37 objects are seen three times at the 32-frame budget
# the brief sets, so three would discard half the recoverable objects before
# SAM runs. Two, plus a real angular spread, is the honest operating point.
MIN_SUPPORT_VIEWS = 2
# The supporting views must span at least this angle.
#
# Set EQUAL to the selection floor rather than above it. It was briefly 20 deg
# -- an arbitrary "genuine parallax" figure -- and the loopback self-check
# showed that discarding well-associated objects: three of twelve probes
# associated at 3D IoU 0.60-0.63 and were dropped solely because their two
# supporting views were 10.4 deg apart. Selection already enforces that every
# chosen pair of view directions differs by MIN_ANGULAR_SEPARATION_DEG, so a
# larger per-cluster threshold re-litigates a decision already made, against a
# frame set built not to satisfy it. View diversity is enforced once, at
# selection. Changed on synthetic evidence, before any SAM inference and with
# no annotation open.
MIN_CLUSTER_ANGULAR_SPREAD_DEG = MIN_ANGULAR_SEPARATION_DEG
# A vertex joins the fused proposal when it appears in at least this share of
# the cluster's masks, counted only over views where it is actually visible.
# Normalising by visibility is the point: a vertex seen in one of three views
# must not be penalised for being absent from masks that could never contain
# it.
VERTEX_VOTE_FRACTION = 0.50
# After fusing, members disagreeing with the result this badly are dropped and
# the cluster is fused once more. Bounded single-linkage repair, not a loop.
MIN_MEMBER_IOU = 0.20


@dataclass(frozen=True)
class LiftedMask:
    """One SAM mask lifted onto the mesh, independently of every other mask."""
    view: int
    mask_index: int
    vertices: np.ndarray
    predicted_iou: float
    stability: float


@dataclass(frozen=True)
class MaskCluster:
    """Masks from several views that agree on one surface."""
    members: tuple[LiftedMask, ...]
    vertices: np.ndarray
    support_views: int
    angular_spread_deg: float
    confidence: float
    notes: dict = field(default_factory=dict)


# =========================================================================
# stage 1 — frame selection
# =========================================================================
def select_frames(frames: list, xyz_world: np.ndarray, *,
                  n_frames: int = N_FRAMES,
                  min_coverage: float = MIN_MESH_COVERAGE,
                  min_angle_deg: float = MIN_ANGULAR_SEPARATION_DEG,
                  multiplicity_cap: int = COVERAGE_MULTIPLICITY_CAP,
                  visibility=None) -> tuple[list[int], dict]:
    """Visibility-valid frames chosen to maximise coverage MULTIPLICITY.

    Greedy on `sum_v min(times_seen(v), cap)` subject to every pair of chosen
    view directions differing by at least `min_angle_deg`. Submodular
    objective, so greedy is within (1 - 1/e); determinism comes from breaking
    ties on the frame index.

    `visibility` lets a caller supply precomputed per-frame visible sets.
    """
    coverage = np.array([coarse_coverage(xyz_world, f) for f in frames])
    valid = np.flatnonzero(coverage >= min_coverage)
    if len(valid) == 0:
        raise ValueError(
            f"no frame reaches {min_coverage:.0%} mesh coverage "
            f"(best {coverage.max():.1%})")
    directions = {}
    for i in valid:
        d = _forward_axis(frames[i])
        directions[int(i)] = d / np.linalg.norm(d)

    n_vertices = len(xyz_world)
    seen_count = np.zeros(n_vertices, dtype=np.int16)
    visible: dict[int, np.ndarray] = {}
    for i in valid:
        if visibility is not None:
            visible[int(i)] = visibility[int(i)]
        else:
            vertices, _ = visible_vertices(xyz_world, frames[int(i)])
            visible[int(i)] = vertices

    cos_limit = math.cos(math.radians(min_angle_deg))
    chosen: list[int] = []
    while len(chosen) < n_frames:
        best, best_gain = None, -1
        for i in valid:
            i = int(i)
            if any(float(directions[i] @ directions[j]) > cos_limit
                   for j in chosen):
                continue
            gain = int((seen_count[visible[i]] < multiplicity_cap).sum())
            if gain > best_gain:
                best, best_gain = i, gain
        if best is None or best_gain <= 0:
            break
        chosen.append(best)
        seen_count[visible[best]] += 1

    diagnostics = {
        "n_candidate_frames": len(frames),
        "n_valid_frames": int(len(valid)),
        "n_selected": len(chosen),
        "min_angular_separation_deg": min_angle_deg,
        "multiplicity_cap": multiplicity_cap,
        "mesh_seen_any_view": round(float((seen_count > 0).mean()), 4),
        "mesh_seen_two_views": round(float((seen_count >= 2).mean()), 4),
        "mesh_seen_three_views": round(float((seen_count >= 3).mean()), 4),
        "coverage_per_selected": [round(float(coverage[i]), 4) for i in chosen],
    }
    return chosen, diagnostics


# =========================================================================
# stage 3 — independent lifting
# =========================================================================
def lift_masks(xyz_world: np.ndarray, frame, view: int,
               masks_2d: np.ndarray, scores: np.ndarray) -> list[LiftedMask]:
    """Lift every SAM mask of one frame INDEPENDENTLY onto visible vertices.

    `masks_2d` is [K, H, W] boolean. Masks may overlap and need not cover the
    frame; both are expected. No complement is formed and unmasked pixels are
    dropped, which is the architectural difference from the previous arm.
    """
    vertices, pixel = visible_vertices(xyz_world, frame)
    if len(vertices) == 0:
        return []
    n_pixels = frame.height * frame.width
    out: list[LiftedMask] = []
    for k, mask in enumerate(masks_2d):
        flat = np.asarray(mask, dtype=bool).ravel()
        if flat.size != n_pixels:
            raise ValueError(
                f"view {view} mask {k} has {flat.size} pixels, frame is "
                f"{n_pixels}")
        hit = flat[pixel]
        if not hit.any():
            continue
        lifted = np.unique(vertices[hit])
        if len(lifted) < MIN_MASK_VERTICES:
            continue
        if len(lifted) > MAX_MASK_FRAC * len(xyz_world):
            continue
        predicted_iou = float(scores[k][0]) if len(scores) > k else 1.0
        stability = float(scores[k][1]) if len(scores) > k else 1.0
        out.append(LiftedMask(view, k, lifted, predicted_iou, stability))
    return out


# =========================================================================
# stage 4 — association across views
# =========================================================================
def _anchor_matrix(masks: list[LiftedMask], n_vertices: int,
                   stride: int) -> np.ndarray:
    """[n_masks, n_anchors] float32 indicator over a strided vertex subsample."""
    anchors = np.arange(0, n_vertices, stride, dtype=np.int64)
    lookup = np.full(n_vertices, -1, dtype=np.int64)
    lookup[anchors] = np.arange(len(anchors))
    out = np.zeros((len(masks), len(anchors)), dtype=np.float32)
    for i, mask in enumerate(masks):
        slots = lookup[mask.vertices]
        out[i, slots[slots >= 0]] = 1.0
    return out


def _pairwise_intersections(indicator: np.ndarray,
                            chunk: int = 512) -> np.ndarray:
    """Symmetric [n, n] intersection counts, chunked to bound peak memory."""
    n = len(indicator)
    out = np.zeros((n, n), dtype=np.float32)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        out[start:stop] = indicator[start:stop] @ indicator.T
    return out


def associate(masks: list[LiftedMask], frames: list, n_vertices: int,
              *, anchor_stride: int = ASSOC_ANCHOR_STRIDE) -> list[list[int]]:
    """Single-linkage clusters of masks that describe the same 3D surface.

    Linked when 3D IoU >= ASSOC_IOU, or when one mask is >= ASSOC_CONTAINMENT
    inside the other AND they are within a factor of two in size. The size
    guard is what stops containment from chaining a cushion into a sofa into
    a room; containment without it is the classic failure of this step.

    Masks from the SAME view are never linked. SAM's own overlapping outputs
    at different scales are distinct hypotheses, and merging them here would
    quietly rebuild the partition this arm exists to avoid.
    """
    if not masks:
        return []
    indicator = _anchor_matrix(masks, n_vertices, anchor_stride)
    sizes = indicator.sum(axis=1)
    inter = _pairwise_intersections(indicator)
    union = sizes[:, None] + sizes[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        smaller = np.minimum(sizes[:, None], sizes[None, :])
        containment = np.where(smaller > 0, inter / np.maximum(smaller, 1e-9), 0.0)
        ratio = np.where(
            np.maximum(sizes[:, None], sizes[None, :]) > 0,
            smaller / np.maximum(np.maximum(sizes[:, None], sizes[None, :]), 1e-9),
            0.0)

    views = np.array([m.view for m in masks])
    different_view = views[:, None] != views[None, :]
    link = different_view & (
        (iou >= ASSOC_IOU)
        | ((containment >= ASSOC_CONTAINMENT)
           & (ratio >= ASSOC_CONTAINMENT_MIN_SIZE_RATIO)))
    np.fill_diagonal(link, False)

    parent = list(range(len(masks)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(*np.nonzero(np.triu(link, 1))):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups: dict[int, list[int]] = {}
    for i in range(len(masks)):
        groups.setdefault(find(i), []).append(i)
    return [sorted(v) for _, v in sorted(groups.items())]


# =========================================================================
# stage 5 — fuse a cluster into one proposal
# =========================================================================
def _angular_spread(members: list[LiftedMask], frames: list) -> float:
    directions = []
    for m in {mm.view for mm in members}:
        d = _forward_axis(frames[m])
        directions.append(d / np.linalg.norm(d))
    if len(directions) < 2:
        return 0.0
    worst = 1.0
    for i in range(len(directions)):
        for j in range(i + 1, len(directions)):
            worst = min(worst, float(directions[i] @ directions[j]))
    return math.degrees(math.acos(max(-1.0, min(1.0, worst))))


def _vote(members: list[LiftedMask], visibility: dict[int, np.ndarray],
          n_vertices: int) -> np.ndarray:
    """Vertices in >= VERTEX_VOTE_FRACTION of the masks that could contain them.

    The denominator counts only cluster views where the vertex is visible. A
    vertex hidden in a view is absence of evidence, not evidence against.
    """
    in_mask = np.zeros(n_vertices, dtype=np.int16)
    could = np.zeros(n_vertices, dtype=np.int16)
    for view in {m.view for m in members}:
        could[visibility[view]] += 1
    for member in members:
        in_mask[member.vertices] += 1
    touched = np.flatnonzero(in_mask > 0)
    denominator = np.maximum(could[touched], 1)
    keep = in_mask[touched] / denominator >= VERTEX_VOTE_FRACTION
    return touched[keep]


def fuse_cluster(members: list[LiftedMask], frames: list,
                 visibility: dict[int, np.ndarray],
                 n_vertices: int) -> MaskCluster | None:
    """Vote, prune disagreeing members, vote once more. None if unsupported."""
    views = {m.view for m in members}
    if len(views) < MIN_SUPPORT_VIEWS:
        return None
    spread = _angular_spread(members, frames)
    if spread < MIN_CLUSTER_ANGULAR_SPREAD_DEG:
        return None

    fused = _vote(members, visibility, n_vertices)
    if len(fused) < MIN_PROPOSAL_VERTICES:
        return None

    kept = []
    for member in members:
        inter = np.intersect1d(member.vertices, fused, assume_unique=True).size
        union = len(member.vertices) + len(fused) - inter
        if union and inter / union >= MIN_MEMBER_IOU:
            kept.append(member)
    if len({m.view for m in kept}) < MIN_SUPPORT_VIEWS:
        return None
    if len(kept) != len(members):
        fused = _vote(kept, visibility, n_vertices)
        spread = _angular_spread(kept, frames)
        if (len(fused) < MIN_PROPOSAL_VERTICES
                or spread < MIN_CLUSTER_ANGULAR_SPREAD_DEG):
            return None
    if len(fused) > MAX_PROPOSAL_FRAC * n_vertices:
        return None

    confidence = float(np.mean([m.predicted_iou for m in kept]))
    return MaskCluster(
        tuple(kept), fused, len({m.view for m in kept}), spread, confidence,
        {"n_member_masks": len(kept),
         "mean_stability": round(float(np.mean([m.stability for m in kept])), 4)})


# =========================================================================
# stage 6 — emit beside Mask3D
# =========================================================================
def classify_clusters(clusters: list[MaskCluster], baseline: list[np.ndarray],
                      n_vertices: int) -> tuple[list[RepairProposal], dict]:
    """additional / split, using the SAME thresholds as the previous arm.

    Imported rather than re-declared so the two arms differ only in how a
    candidate region was produced.

    `RepairProposal.cut` is reused as a carrier and is meaningless here: this
    arm has no confidence cut. Cluster support is recorded in
    `consensus_views` and the full cluster record goes to the diagnostics.
    """
    rejected = {"duplicate_of_baseline": 0, "ambiguous_containment": 0,
                "split_piece_too_small": 0, "duplicate_repair": 0}
    # Each candidate remembers which cluster it came from. Emission reorders,
    # caps and dedupes, and a `split` piece is CLIPPED to its parent, so a
    # caller cannot recover the source afterwards by matching vertices -- a
    # detector-guided caller that tried lost the label on every split. The
    # index is reported through diagnostics, which keeps this function's
    # signature and behaviour unchanged.
    sources: dict[int, int] = {}
    candidates: list[RepairProposal] = []
    for cluster_index, cluster in enumerate(clusters):
        verts = cluster.vertices
        best_containment, best_parent, best_iou = 0.0, None, 0.0
        for j, parent in enumerate(baseline):
            inter = np.intersect1d(verts, parent, assume_unique=True).size
            if not inter:
                continue
            containment = inter / len(verts)
            best_iou = max(best_iou, inter / (len(verts) + len(parent) - inter))
            if containment > best_containment:
                best_containment, best_parent = containment, j

        if best_iou >= DUPLICATE_IOU:
            rejected["duplicate_of_baseline"] += 1
            continue
        if best_containment < ADDITIONAL_MAX_CONTAINMENT:
            candidates.append(RepairProposal(
                verts, "additional", cluster.confidence,
                float(cluster.support_views), 0.0, None, best_containment))
            sources[id(candidates[-1])] = cluster_index
            continue
        if best_containment >= SPLIT_MIN_CONTAINMENT and best_parent is not None:
            parent = baseline[best_parent]
            if len(verts) > SPLIT_MAX_SIZE_RATIO * len(parent):
                rejected["ambiguous_containment"] += 1
                continue
            piece = np.intersect1d(verts, parent, assume_unique=True)
            if len(piece) < MIN_PROPOSAL_VERTICES:
                rejected["split_piece_too_small"] += 1
                continue
            candidates.append(RepairProposal(
                piece, "split", cluster.confidence,
                float(cluster.support_views), 0.0, best_parent,
                best_containment))
            sources[id(candidates[-1])] = cluster_index
            continue
        rejected["ambiguous_containment"] += 1

    candidates.sort(key=lambda p: (-p.confidence, -len(p.vertices),
                                   int(p.vertices[0])))
    kept: list[RepairProposal] = []
    for cand in candidates:
        if len(kept) >= MAX_REPAIR_PROPOSALS:
            break
        duplicate = False
        for other in kept:
            inter = np.intersect1d(cand.vertices, other.vertices,
                                   assume_unique=True).size
            if inter and inter / (len(cand.vertices) + len(other.vertices)
                                  - inter) >= DUPLICATE_IOU:
                duplicate = True
                break
        if duplicate:
            rejected["duplicate_repair"] += 1
            continue
        kept.append(cand)

    diagnostics = {
        "source_cluster_indices": [sources[id(p)] for p in kept],
        "n_clusters": len(clusters),
        "n_candidates": len(candidates),
        "n_emitted": len(kept),
        "capped_at": MAX_REPAIR_PROPOSALS,
        "n_dropped_by_cap": max(
            0, len(candidates) - len(kept) - rejected["duplicate_repair"]),
        "rejected": rejected,
        "by_kind": {k: sum(1 for p in kept if p.kind == k)
                    for k in ("additional", "split")},
    }
    return kept, diagnostics


def config_record() -> dict:
    """Every declared constant, for the artifact manifest."""
    return {
        "n_frames": N_FRAMES,
        "min_angular_separation_deg": MIN_ANGULAR_SEPARATION_DEG,
        "coverage_multiplicity_cap": COVERAGE_MULTIPLICITY_CAP,
        "min_mask_vertices": MIN_MASK_VERTICES,
        "max_mask_frac": MAX_MASK_FRAC,
        "assoc_anchor_stride": ASSOC_ANCHOR_STRIDE,
        "assoc_iou": ASSOC_IOU,
        "assoc_containment": ASSOC_CONTAINMENT,
        "assoc_containment_min_size_ratio": ASSOC_CONTAINMENT_MIN_SIZE_RATIO,
        "min_support_views": MIN_SUPPORT_VIEWS,
        "min_cluster_angular_spread_deg": MIN_CLUSTER_ANGULAR_SPREAD_DEG,
        "vertex_vote_fraction": VERTEX_VOTE_FRACTION,
        "min_member_iou": MIN_MEMBER_IOU,
        "additional_max_containment": ADDITIONAL_MAX_CONTAINMENT,
        "split_min_containment": SPLIT_MIN_CONTAINMENT,
        "split_max_size_ratio": SPLIT_MAX_SIZE_RATIO,
        "duplicate_iou": DUPLICATE_IOU,
        "min_proposal_vertices": MIN_PROPOSAL_VERTICES,
        "max_proposal_frac": MAX_PROPOSAL_FRAC,
        "max_repair_proposals": MAX_REPAIR_PROPOSALS,
    }


def generate_from_sidecar(xyz_world: np.ndarray, frames: list,
                          selected: list[int], sidecar_masks: dict,
                          baseline: list[np.ndarray], *,
                          progress=None) -> tuple[list[RepairProposal], dict]:
    """Full local pipeline from a pinned SAM sidecar to emitted proposals."""
    n_vertices = len(xyz_world)
    visibility: dict[int, np.ndarray] = {}
    masks: list[LiftedMask] = []
    per_view = []
    for slot, frame_index in enumerate(selected):
        frame = frames[frame_index]
        vertices, _ = visible_vertices(xyz_world, frame)
        visibility[frame_index] = vertices
        raw = sidecar_masks[slot]["masks"]
        scores = sidecar_masks[slot]["scores"]
        lifted = lift_masks(xyz_world, frame, frame_index, raw, scores)
        masks.extend(lifted)
        per_view.append({
            "slot": slot, "frame_index": frame_index,
            "timestamp": round(frame.timestamp, 3),
            "n_sam_masks": int(len(raw)),
            "n_lifted": len(lifted),
            "n_visible_vertices": int(len(vertices)),
        })
        if progress:
            progress(f"  [{slot + 1}/{len(selected)}] {len(raw)} SAM masks -> "
                     f"{len(lifted)} lifted, {len(vertices)} visible vertices")

    groups = associate(masks, frames, n_vertices)
    if progress:
        progress(f"associated {len(masks)} lifted masks into {len(groups)} clusters")

    clusters, unsupported = [], 0
    for group in groups:
        fused = fuse_cluster([masks[i] for i in group], frames, visibility,
                             n_vertices)
        if fused is None:
            unsupported += 1
        else:
            clusters.append(fused)
    if progress:
        progress(f"{len(clusters)} clusters supported, {unsupported} dropped")

    proposals, diagnostics = classify_clusters(clusters, baseline, n_vertices)
    diagnostics["n_lifted_masks"] = len(masks)
    diagnostics["n_raw_clusters"] = len(groups)
    diagnostics["n_unsupported_clusters"] = unsupported
    diagnostics["per_view"] = per_view
    diagnostics["config"] = config_record()
    return proposals, diagnostics
