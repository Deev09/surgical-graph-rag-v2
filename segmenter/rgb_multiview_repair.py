"""Multi-view 2D-mask → 3D instance repair from the real posed RGB capture.

Design note: `docs/repair_arm_design_note.md`. Constants below were declared
there before this file existed and are NOT swept.

The question: Mask3D segments the mesh directly and never looks at a
photograph. Where it misses an object or merges two, the capture stream still
shows them as separate things from many angles. This arm turns that
disagreement into extra proposals emitted BESIDE Mask3D.

Oracle-free by construction. It reads the canonical mesh, `lowres_wide/`,
`lowres_wide_intrinsics/`, `lowres_wide.traj`, and the frozen Mask3D bank.
Nothing here can reach an annotation.

FIVE STAGES, only two of which are new code
-------------------------------------------
1  frame selection      visibility-valid, angularly diverse
2  2D masks             Felzenszwalb graph segmentation of the real frame
3  lifting              corrected OpenCV convention + mesh z-buffer
4  consensus fusion     segmenter/proposal_fusion.py, UNMODIFIED
5  classification       additional / split, against the frozen Mask3D bank

Stage 4 is the frozen C1-P1 fusion. Reusing it is the point: mesh-edge
consensus across views is already implemented, tested and understood, and a
second copy of that logic is exactly the drift the V3 tests exist to catch.
The only thing this module changes about it is where the views come from --
real photographs instead of point-splat renders of the mesh.

WHAT THE 2D MASK SOURCE IS, AND WHAT THAT COSTS
-----------------------------------------------
Felzenszwalb graph-based segmentation on RGB. Not SAM, not a learned model:
this environment has numpy and PIL, and the arm is meant to test the LIFTING
AND CONSENSUS mechanism, not a segmenter. The consequence is real and is not
hidden anywhere downstream -- colour-coherent regions merge a white curtain
into a white wall and split a patterned sofa into cushions-plus-pattern. So:

  * a NEGATIVE result from this arm does not refute the mechanism, because the
    mask source is weak. It refutes this mask source.
  * a POSITIVE result is the stronger reading, because it survived a weak mask
    source.

`segment_frame` is the single seam where a learned segmenter drops in; nothing
downstream of it knows what produced the labels.

WHY THE CAMERA CONVENTION IS IMPORTED, NOT RESTATED
----------------------------------------------------
`extractors/arkitscenes_rgb_crops.py` established plain OpenCV axes from
Apple's own loader and validated them against synchronized sensor depth
(2.5-4.1 cm median error; the invalidated `[1,-1,-1]` flip gives 37-98 cm).
`Frame`, `load_frames` and `project` come from there. A local reprojection
here would be a second convention that could drift from the validated one,
which is the bug that invalidated every earlier crop measurement.

Note that only 41069021 ships `lowres_depth/`. Visibility therefore uses the
MESH z-buffer, which is the right choice anyway: the mesh is what Mask3D
segmented, so "visible" means "visible in the geometry being repaired".
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from segmenter.proposal_fusion import (
    MAX_COMPONENT_FRAC, MIN_LIFT_VERTICES, build_bank, edge_confidence,
    mesh_edges,
)

# ---- stage 1: frame selection -------------------------------------------
# The capture is 60 Hz and the trajectory ~10 Hz; consecutive posed frames are
# the same viewpoint. Stride is applied AFTER pose matching (see load_frames).
FRAME_STRIDE = 6
# Selected frames.
#
# Originally 24, which was guesswork and was wrong by an order of magnitude. A
# 256x192 frame covers 16k-43k pixels of mesh at 2-4 vertices per pixel, so one
# view sees 2-9% of a 1,008,964-vertex room. Measured co-visibility on the
# development scene, share of mesh vertices seen in at least three selected
# views:
#
#     N frames      24     48     72     96    128    160    200
#     >=3 views    5.6%  37.4%  54.4%  65.7%  71.7%  75.9%  78.9%
#
# At 24 the consensus stage had essentially no evidence: 5.6% of the mesh could
# even be evaluated, and the arm emitted nothing. Set to the smallest N
# reaching 70% (the criterion was fixed before the curve was read); beyond it
# the curve flattens and only cost grows. This is how much evidence the
# mechanism gets, not a decision threshold, and no annotation was involved.
N_FRAMES = 128
# A frame is visibility-valid when at least this share of a coarse pixel grid
# receives a mesh vertex. Rejects frames aimed at unreconstructed space, where
# every lifted region would be empty or wrong.
MIN_MESH_COVERAGE = 0.60
# Coarse grid for that coverage estimate, and the vertex stride feeding it.
COVERAGE_GRID = (24, 32)
COVERAGE_STRIDE = 16
# A vertex is on the visible surface when nothing in the reconstruction sits
# more than this far in front of it at its own pixel. Tight, because the
# buffer is rasterised at full mesh density and leaves few gaps, and well
# above the 2.5-4.1 cm median mesh-versus-sensor-depth error measured in
# extractors/arkitscenes_rgb_crops.py.
SURFACE_DEPTH_TOLERANCE_M = 0.05

# ---- stage 2: 2D masks ---------------------------------------------------
FELZ_SIGMA = 0.8
# Scale and minimum region size. The design note originally declared the
# paper's k=300 / min_size=120; on these 256x192 frames that produced a median
# of 18 regions, with one region covering more than half the frame -- walls and
# floors, not objects. k=300 is calibrated for the paper's larger images, and
# the edge weight here is 0-255 RGB Euclidean distance, so k/|C| dominates for
# the first few hundred merges and everything fuses.
#
# Re-set ONCE, before any annotation was opened for this arm, against a stated
# annotation-free criterion: median regions per frame in 40-120, and the
# largest region below a third of the frame. Measured on five evenly spaced
# development frames (k=40: median 107, max 31%; k=100: median 80, max 50%).
# This is calibration of the mask source, not tuning against a result. It was
# not touched again.
FELZ_K = 40.0
FELZ_MIN_SIZE = 60

# ---- stage 4/5: consensus and classification -----------------------------
# Mean co-visible view count an accepted component's internal edges must have.
# Two views agreeing is a coincidence; the mechanism claims multi-view.
MIN_CONSENSUS_VIEWS = 3.0
# Best containment in any single Mask3D proposal, below which the component is
# largely unexplained by the baseline -> candidate missing object.
ADDITIONAL_MAX_CONTAINMENT = 0.50
# Containment above which the component sits inside one baseline proposal...
SPLIT_MIN_CONTAINMENT = 0.70
# ...and the size ratio below which it is a PIECE of it rather than a copy.
SPLIT_MAX_SIZE_RATIO = 0.70
# A near-copy of a baseline proposal is not a repair.
DUPLICATE_IOU = 0.90
# Junk floor, above the fusion module's own floor of 20.
MIN_PROPOSAL_VERTICES = 200
# Hard size cap. Equals the evaluator's giant-mask threshold, so this arm
# cannot emit a giant mask -- a guardrail, not evidence. See the design note.
MAX_PROPOSAL_FRAC = 0.15
# Emitted proposals, best consensus first. Keeps baseline+repair near 100.
MAX_REPAIR_PROPOSALS = 60


# =========================================================================
# stage 2 — Felzenszwalb graph-based segmentation
# =========================================================================
def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian, reflect-padded. Felzenszwalb's pre-smoothing step."""
    if sigma <= 0:
        return image.astype(np.float64)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    out = image.astype(np.float64)
    for axis in (0, 1):
        padded = np.pad(out, [(radius, radius) if a == axis else (0, 0)
                              for a in range(out.ndim)], mode="reflect")
        stacked = np.stack(
            [np.take(padded, np.arange(out.shape[axis]) + i, axis=axis)
             for i in range(2 * radius + 1)], axis=0)
        out = np.tensordot(kernel, stacked, axes=(0, 0))
    return out


def _neighbour_edges(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """8-connected pixel edges as flat index pairs (a, b), a < b by construction."""
    idx = np.arange(height * width, dtype=np.int64).reshape(height, width)
    pairs = [
        (idx[:, :-1], idx[:, 1:]),          # right
        (idx[:-1, :], idx[1:, :]),          # down
        (idx[:-1, :-1], idx[1:, 1:]),       # down-right
        (idx[:-1, 1:], idx[1:, :-1]),       # down-left
    ]
    a = np.concatenate([p[0].ravel() for p in pairs])
    b = np.concatenate([p[1].ravel() for p in pairs])
    return a, b


def felzenszwalb(rgb: np.ndarray, *, sigma: float = FELZ_SIGMA,
                 k: float = FELZ_K, min_size: int = FELZ_MIN_SIZE
                 ) -> np.ndarray:
    """Efficient graph-based image segmentation (Felzenszwalb & Huttenlocher).

    Returns an int32 label image. Deterministic: edges are sorted by weight
    with the flat pixel index as tiebreak, so equal-weight edges always merge
    in the same order.
    """
    image = _gaussian_blur(np.asarray(rgb, dtype=np.float64), sigma)
    height, width = image.shape[:2]
    n = height * width
    flat = image.reshape(n, -1)
    a, b = _neighbour_edges(height, width)
    weight = np.sqrt(((flat[a] - flat[b]) ** 2).sum(axis=1))
    order = np.lexsort((a, weight))
    a, b, weight = a[order], b[order], weight[order]

    parent = np.arange(n, dtype=np.int64)
    size = np.ones(n, dtype=np.int64)
    internal = np.zeros(n, dtype=np.float64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    for i in range(len(a)):
        ra, rb = find(int(a[i])), find(int(b[i]))
        if ra == rb:
            continue
        w = weight[i]
        if (w <= internal[ra] + k / size[ra]
                and w <= internal[rb] + k / size[rb]):
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]
            internal[ra] = w

    # Post-process: absorb components below min_size across their lightest edge.
    for i in range(len(a)):
        ra, rb = find(int(a[i])), find(int(b[i]))
        if ra != rb and (size[ra] < min_size or size[rb] < min_size):
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]

    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int32).reshape(height, width)


# =========================================================================
# stage 1 + 3 — frame selection and lifting
# =========================================================================
@dataclass(frozen=True)
class LiftedView:
    """One selected frame, lifted onto the mesh."""
    frame_index: int
    timestamp: float
    mesh_coverage: float
    n_regions: int
    visible: np.ndarray            # sorted unique visible vertex indices
    masks: list[np.ndarray]        # per-region vertex sets, >= 20 vertices


def _forward_axis(frame) -> np.ndarray:
    """Camera forward direction in world coordinates (OpenCV +z)."""
    return frame.R_wc.T @ np.array([0.0, 0.0, 1.0])


def coarse_coverage(xyz_world: np.ndarray, frame,
                    stride: int = COVERAGE_STRIDE,
                    grid: tuple[int, int] = COVERAGE_GRID) -> float:
    """Share of a coarse pixel grid that any mesh vertex projects into.

    Cheap enough to run over every candidate frame, which the full
    vertex-index buffer is not.
    """
    pts = xyz_world[::stride]
    cam = (frame.R_wc @ pts.T).T + frame.t_wc
    z = cam[:, 2]
    ok = z > 1e-6
    if not np.any(ok):
        return 0.0
    u = frame.fx * cam[ok, 0] / z[ok] + frame.cx
    v = frame.fy * cam[ok, 1] / z[ok] + frame.cy
    gh, gw = grid
    gu = np.floor(u / frame.width * gw).astype(np.int64)
    gv = np.floor(v / frame.height * gh).astype(np.int64)
    inb = (gu >= 0) & (gu < gw) & (gv >= 0) & (gv < gh)
    if not np.any(inb):
        return 0.0
    cells = np.zeros(gh * gw, dtype=bool)
    cells[gv[inb] * gw + gu[inb]] = True
    return float(cells.mean())


def select_frames(frames: list, xyz_world: np.ndarray, *,
                  n_frames: int = N_FRAMES,
                  min_coverage: float = MIN_MESH_COVERAGE
                  ) -> tuple[list[int], list[float]]:
    """Visibility-valid frames, greedily chosen for angular diversity.

    Farthest-point sampling on the camera forward direction, seeded with the
    best-covered frame. "Angularly diverse" is the requirement, so the metric
    is the angle between view directions -- not camera position, which would
    happily return 24 frames of the same wall from a moving hand.
    """
    coverage = np.array([coarse_coverage(xyz_world, f) for f in frames])
    valid = np.flatnonzero(coverage >= min_coverage)
    if len(valid) == 0:
        raise ValueError(
            f"no frame reaches {min_coverage:.0%} mesh coverage "
            f"(best {coverage.max():.1%}); the mesh and trajectory may not "
            "share a frame")
    directions = np.array([_forward_axis(frames[i]) for i in valid])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    chosen = [int(np.argmax(coverage[valid]))]
    # min angular distance from each candidate to the chosen set, updated
    # incrementally rather than recomputed
    best_cos = directions @ directions[chosen[0]]
    while len(chosen) < min(n_frames, len(valid)):
        candidate = int(np.argmin(best_cos))
        if candidate in chosen:                      # exhausted distinct dirs
            remaining = [i for i in range(len(valid)) if i not in chosen]
            if not remaining:
                break
            candidate = remaining[0]
        chosen.append(candidate)
        best_cos = np.maximum(best_cos, directions @ directions[candidate])
    picked = [int(valid[c]) for c in chosen]
    return picked, [float(coverage[i]) for i in picked]


def visible_vertices(xyz_world: np.ndarray, frame) -> tuple[np.ndarray, np.ndarray]:
    """(vertex indices, their pixel index) for every vertex on the front surface.

    A depth buffer decides occlusion; a vertex is visible when nothing in the
    reconstruction sits more than SURFACE_DEPTH_TOLERANCE_M in front of it at
    its own pixel.

    NOT "the single nearest vertex per pixel". That was the first
    implementation and it is wrong: this mesh has 1,008,964 vertices and a
    frame has 49,152 pixels, so a one-vertex-per-pixel buffer marks at most 5%
    of the mesh visible and reports a vertex as hidden merely because a
    neighbour 2 mm away won the pixel. Measured on the development scene it
    gave 12k-43k visible vertices per view, no mesh edge reached three
    co-visible views, and the arm emitted nothing. Every vertex on the visible
    surface belongs to the region covering that pixel, and this returns all of
    them.
    """
    cam = (frame.R_wc @ xyz_world.T).T + frame.t_wc
    z = cam[:, 2]
    idx = np.flatnonzero(z > 1e-6)
    u = np.rint(frame.fx * cam[idx, 0] / z[idx] + frame.cx).astype(np.int64)
    v = np.rint(frame.fy * cam[idx, 1] / z[idx] + frame.cy).astype(np.int64)
    inb = (u >= 0) & (u < frame.width) & (v >= 0) & (v < frame.height)
    idx, u, v = idx[inb], u[inb], v[inb]
    pixel = v * frame.width + u
    depth = z[idx]
    front = np.full(frame.height * frame.width, np.inf)
    np.minimum.at(front, pixel, depth)
    keep = depth <= front[pixel] + SURFACE_DEPTH_TOLERANCE_M
    return idx[keep], pixel[keep]


def lift_frame(xyz_world: np.ndarray, frame, frame_index: int,
               coverage: float) -> LiftedView:
    """Segment one real RGB frame and lift every region onto the mesh."""
    image = np.asarray(Image.open(frame.png).convert("RGB"), dtype=np.uint8)
    if image.shape[:2] != (frame.height, frame.width):
        raise ValueError(
            f"{frame.png}: image {image.shape[:2]} does not match intrinsics "
            f"{(frame.height, frame.width)}")
    labels = felzenszwalb(image)
    vertices, pixel = visible_vertices(xyz_world, frame)
    per_vertex_label = labels.ravel()[pixel]
    # Group by region in one pass rather than testing every region against the
    # whole visible set: at ~100 regions and several hundred thousand visible
    # vertices the naive loop is the runtime.
    order = np.argsort(per_vertex_label, kind="stable")
    sorted_labels = per_vertex_label[order]
    bounds = np.flatnonzero(np.diff(sorted_labels)) + 1
    masks = []
    for group in np.split(order, bounds):
        if group.size < MIN_LIFT_VERTICES:
            continue
        lifted = np.unique(vertices[group])
        if len(lifted) >= MIN_LIFT_VERTICES:
            masks.append(lifted)
    return LiftedView(frame_index, frame.timestamp, coverage,
                      int(labels.max()) + 1, np.unique(vertices), masks)


# =========================================================================
# stage 5 — classify components against the frozen Mask3D bank
# =========================================================================
@dataclass(frozen=True)
class RepairProposal:
    vertices: np.ndarray
    kind: str                  # "additional" | "split"
    confidence: float          # mean internal mesh-edge consensus
    consensus_views: float     # mean co-visible view count on internal edges
    cut: float
    parent_index: int | None   # baseline proposal it splits, if any
    containment: float


def _component_stats(components: list[dict], edges: np.ndarray,
                     co_vis: np.ndarray, conf: np.ndarray,
                     n_vertices: int) -> list[tuple[float, float]]:
    """(mean internal edge confidence, mean internal co-visible views) each.

    Computed per confidence cut in one pass over the edge list. Components
    within a cut are disjoint (they are connected components), so a single
    vertex->component map is well defined there; across cuts they overlap,
    which is why this cannot be done globally.
    """
    stats: list[tuple[float, float]] = [(0.0, 0.0)] * len(components)
    by_cut: dict[float, list[int]] = {}
    for i, comp in enumerate(components):
        by_cut.setdefault(comp["cut"], []).append(i)

    for cut, indices in by_cut.items():
        owner = np.full(n_vertices, -1, dtype=np.int64)
        for slot, i in enumerate(indices):
            owner[components[i]["vertices"]] = slot
        ea, eb = owner[edges[:, 0]], owner[edges[:, 1]]
        internal = (ea >= 0) & (ea == eb)
        if not np.any(internal):
            continue
        slots = ea[internal]
        counts = np.bincount(slots, minlength=len(indices))
        conf_sum = np.bincount(slots, weights=conf[internal],
                               minlength=len(indices))
        vis_sum = np.bincount(slots, weights=co_vis[internal].astype(np.float64),
                              minlength=len(indices))
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_conf = np.where(counts > 0, conf_sum / np.maximum(counts, 1), 0.0)
            mean_vis = np.where(counts > 0, vis_sum / np.maximum(counts, 1), 0.0)
        for slot, i in enumerate(indices):
            stats[i] = (float(mean_conf[slot]), float(mean_vis[slot]))
    return stats


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.intersect1d(a, b, assume_unique=True).size
    if not inter:
        return 0.0
    return inter / (a.size + b.size - inter)


def classify_components(components: list[dict], stats: list[tuple[float, float]],
                        baseline: list[np.ndarray], n_vertices: int
                        ) -> tuple[list[RepairProposal], dict]:
    """Turn consensus components into additional/split proposals beside Mask3D.

    Emitted proposals are always ADDITIONS. A split piece is clipped to its
    parent so it cannot leak outside the region it claims to be splitting, and
    the parent itself is never modified or removed -- the evaluator asserts
    that the pooled bank still contains every baseline proposal.
    """
    max_vertices = int(MAX_PROPOSAL_FRAC * n_vertices)
    rejected = {"too_small": 0, "too_large": 0, "weak_consensus": 0,
                "duplicate_of_baseline": 0, "ambiguous_containment": 0,
                "split_piece_too_small": 0, "duplicate_repair": 0}
    candidates: list[RepairProposal] = []

    for comp, (mean_conf, mean_views) in zip(components, stats):
        verts = comp["vertices"]
        if len(verts) < MIN_PROPOSAL_VERTICES:
            rejected["too_small"] += 1
            continue
        if len(verts) > max_vertices:
            rejected["too_large"] += 1
            continue
        if mean_views < MIN_CONSENSUS_VIEWS:
            rejected["weak_consensus"] += 1
            continue

        best_containment, best_parent, best_iou = 0.0, None, 0.0
        for j, parent in enumerate(baseline):
            inter = np.intersect1d(verts, parent, assume_unique=True).size
            if not inter:
                continue
            containment = inter / len(verts)
            union = len(verts) + len(parent) - inter
            best_iou = max(best_iou, inter / union)
            if containment > best_containment:
                best_containment, best_parent = containment, j

        if best_iou >= DUPLICATE_IOU:
            rejected["duplicate_of_baseline"] += 1
            continue

        if best_containment < ADDITIONAL_MAX_CONTAINMENT:
            candidates.append(RepairProposal(
                verts, "additional", mean_conf, mean_views, comp["cut"],
                None, best_containment))
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
                piece, "split", mean_conf, mean_views, comp["cut"],
                best_parent, best_containment))
            continue

        rejected["ambiguous_containment"] += 1

    # Best consensus first, then larger, then lowest first vertex id: total
    # and reproducible, and it is the order the emission cap applies to.
    candidates.sort(key=lambda p: (-p.confidence, -len(p.vertices),
                                   int(p.vertices[0])))
    kept: list[RepairProposal] = []
    for cand in candidates:
        if len(kept) >= MAX_REPAIR_PROPOSALS:
            break
        if any(_iou(cand.vertices, k.vertices) >= DUPLICATE_IOU for k in kept):
            rejected["duplicate_repair"] += 1
            continue
        kept.append(cand)

    diagnostics = {
        "n_components": len(components),
        "n_candidates": len(candidates),
        "n_emitted": len(kept),
        "capped_at": MAX_REPAIR_PROPOSALS,
        "n_dropped_by_cap": max(0, len(candidates) - len(kept)
                                - rejected["duplicate_repair"]),
        "rejected": rejected,
        "by_kind": {
            kind: sum(1 for p in kept if p.kind == kind)
            for kind in ("additional", "split")
        },
    }
    return kept, diagnostics


# =========================================================================
# end-to-end
# =========================================================================
def generate_repair_proposals(scene_dir: Path, xyz_canonical: np.ndarray,
                              rotation: np.ndarray, faces: np.ndarray,
                              baseline: list[np.ndarray], *,
                              n_frames: int = N_FRAMES,
                              frame_stride: int = FRAME_STRIDE,
                              progress=None
                              ) -> tuple[list[RepairProposal], dict]:
    """Run the whole arm. Returns (proposals, diagnostics). Never reads annotations.

    `xyz_canonical` and `rotation` come from
    `tools.arkitscenes_eval.load_canonical_geometry`; entities live in the
    canonical frame while the trajectory lives in the original ARKit world
    frame, so the conversion happens once, here.
    """
    from extractors.arkitscenes_rgb_crops import load_frames

    xyz_world = np.asarray(xyz_canonical, dtype=np.float64) @ np.asarray(rotation)
    n_vertices = len(xyz_world)
    frames = load_frames(scene_dir, stride=frame_stride)
    if not frames:
        raise ValueError(f"no pose-matched frames under {scene_dir}")

    picked, coverages = select_frames(frames, xyz_world, n_frames=n_frames)
    if progress:
        progress(f"selected {len(picked)}/{len(frames)} frames "
                 f"(coverage {min(coverages):.0%}–{max(coverages):.0%})")

    views = []
    for slot, (i, cov) in enumerate(zip(picked, coverages)):
        view = lift_frame(xyz_world, frames[i], i, cov)
        views.append(view)
        if progress:
            progress(f"  [{slot + 1}/{len(picked)}] t={view.timestamp:.3f} "
                     f"{view.n_regions} regions -> {len(view.masks)} lifted, "
                     f"{len(view.visible)} visible vertices")

    edges = mesh_edges(faces)
    fusion_views = [{"visible": v.visible, "masks": v.masks} for v in views]
    co_vis, conf = edge_confidence(edges, n_vertices, fusion_views,
                                   evidence_denominator="masked")
    components = build_bank(edges, co_vis, conf, n_vertices)
    if progress:
        progress(f"fusion: {len(edges)} edges, {len(components)} components")

    stats = _component_stats(components, edges, co_vis, conf, n_vertices)
    proposals, diagnostics = classify_components(
        components, stats, baseline, n_vertices)

    diagnostics["views"] = [{
        "frame_index": v.frame_index,
        "timestamp": round(v.timestamp, 3),
        "mesh_coverage": round(v.mesh_coverage, 4),
        "n_regions": v.n_regions,
        "n_lifted_masks": len(v.masks),
        "n_visible_vertices": int(len(v.visible)),
    } for v in views]
    diagnostics["config"] = {
        "frame_stride": frame_stride,
        "n_frames_requested": n_frames,
        "n_frames_used": len(views),
        "min_mesh_coverage": MIN_MESH_COVERAGE,
        "felz_sigma": FELZ_SIGMA, "felz_k": FELZ_K,
        "felz_min_size": FELZ_MIN_SIZE,
        "min_consensus_views": MIN_CONSENSUS_VIEWS,
        "additional_max_containment": ADDITIONAL_MAX_CONTAINMENT,
        "split_min_containment": SPLIT_MIN_CONTAINMENT,
        "split_max_size_ratio": SPLIT_MAX_SIZE_RATIO,
        "duplicate_iou": DUPLICATE_IOU,
        "min_proposal_vertices": MIN_PROPOSAL_VERTICES,
        "max_proposal_frac": MAX_PROPOSAL_FRAC,
        "max_repair_proposals": MAX_REPAIR_PROPOSALS,
        "fusion_cuts_and_dedupe": "frozen in segmenter/proposal_fusion.py",
        "fusion_max_component_frac": MAX_COMPONENT_FRAC,
    }
    return proposals, diagnostics
