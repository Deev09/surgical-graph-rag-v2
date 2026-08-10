"""SAM repair arm: overlapping-hypothesis discipline, association, fusion.

The architectural tests are the load-bearing ones. The previous arm failed
because 2D masks were treated as an exhaustive per-pixel partition, so three
properties are pinned here rather than left to review:

  * lifting is INDEPENDENT per mask — overlapping input masks stay overlapping
    3D hypotheses, and neither is trimmed by the other;
  * unmasked pixels contribute nothing and no complement/background region is
    ever formed;
  * containment-based association cannot chain a small mask into a large one
    into the whole room.

The loopback check against real geometry lives in
`tools/arkitscenes_repair_loopback.py`; this file reads its artifact if one is
on disk and otherwise skips.
"""
from __future__ import annotations

import ast
import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from extractors.arkitscenes_rgb_crops import Frame
from segmenter.rgb_multiview_repair import visible_vertices
from segmenter.sam_multiview_repair import (
    ASSOC_CONTAINMENT_MIN_SIZE_RATIO, MAX_MASK_FRAC, MIN_MASK_VERTICES,
    MIN_SUPPORT_VIEWS, LiftedMask, associate, classify_clusters, config_record,
    fuse_cluster, lift_masks, select_frames,
)

N_VERTICES = 100_000
RUN_ROOT = REPO_ROOT / "runs" / "arkitscenes_repair"


def _frame(R_wc=None, width: int = 32, height: int = 24,
           fx_scale: float = 1.0) -> Frame:
    """`fx_scale` below 1 widens the field of view."""
    return Frame(
        timestamp=0.0, png=Path("/nonexistent.png"),
        R_wc=np.eye(3) if R_wc is None else R_wc, t_wc=np.zeros(3),
        fx=float(width) * fx_scale, fy=float(width) * fx_scale,
        cx=width / 2.0, cy=height / 2.0, width=width, height=height)


def _yaw(degrees: float) -> np.ndarray:
    a = np.radians(degrees)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0],
                     [-np.sin(a), 0, np.cos(a)]])


def _wall(n_side: int = 251, half: float = 3.0, z: float = 2.0) -> np.ndarray:
    grid = np.linspace(-half, half, n_side)
    xx, yy = np.meshgrid(grid, grid)
    return np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, z)], axis=1)


def _shell(n: int = 400_000, radius: float = 2.0) -> np.ndarray:
    """A Fibonacci sphere of points surrounding the camera.

    The selection test needs geometry that fills the frame from EVERY view
    direction. A plane or even a box leaves most of a yawed frame empty, so
    the frame fails the visibility filter for a reason unrelated to what is
    under test. A shell makes coverage orientation-independent.
    """
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return radius * np.stack([np.cos(theta) * np.sin(phi),
                              np.sin(theta) * np.sin(phi),
                              np.cos(phi)], axis=1)


def _pad_to(points: np.ndarray, n_total: int) -> np.ndarray:
    """Embed a visible surface in a larger mesh of never-visible points.

    MAX_MASK_FRAC is a share of the WHOLE mesh, so a fixture whose mesh is
    only the surface under test rejects every mask as room-sized. The padding
    sits behind the camera and can never be lifted.
    """
    if n_total < len(points):
        raise ValueError("n_total must be at least the number of points")
    out = np.zeros((n_total, 3), dtype=np.float64)
    out[:len(points)] = points
    out[len(points):, 2] = -5.0
    return out


def _mask(view: int, index: int, lo: int, hi: int,
          predicted_iou: float = 0.9) -> LiftedMask:
    return LiftedMask(view, index, np.arange(lo, hi, dtype=np.int64),
                      predicted_iou, 0.97)


def test_overlapping_masks_lift_independently() -> None:
    """Two masks sharing pixels must yield two overlapping 3D hypotheses."""
    frame = _frame()
    points = _pad_to(_wall(n_side=120, half=0.9), 200_000)
    height, width = frame.height, frame.width
    left = np.zeros((height, width), dtype=bool)
    left[:, : width // 2 + 6] = True
    right = np.zeros((height, width), dtype=bool)
    right[:, width // 2 - 6:] = True          # deliberately overlapping strips
    lifted = lift_masks(points, frame, 0, np.stack([left, right]),
                        np.ones((2, 2), dtype=np.float32))
    assert len(lifted) == 2, f"expected two hypotheses, got {len(lifted)}"
    shared = np.intersect1d(lifted[0].vertices, lifted[1].vertices,
                            assume_unique=True)
    assert shared.size > 0, \
        "overlapping 2D masks produced disjoint 3D sets; lifting is not independent"


def test_unmasked_pixels_contribute_nothing() -> None:
    """No complement, no background region, no partition."""
    frame = _frame()
    points = _pad_to(_wall(n_side=120, half=0.9), 200_000)
    small = np.zeros((frame.height, frame.width), dtype=bool)
    small[:, :6] = True
    lifted = lift_masks(points, frame, 0, small[None],
                        np.ones((1, 2), dtype=np.float32))
    assert len(lifted) == 1, len(lifted)
    covered = len(lifted[0].vertices)
    assert covered < len(points), \
        "a single narrow mask lifted the whole visible surface"

    # An all-false mask must produce nothing at all rather than a complement.
    empty = np.zeros((1, frame.height, frame.width), dtype=bool)
    assert lift_masks(points, frame, 0, empty,
                      np.ones((1, 2), dtype=np.float32)) == []


def test_lifting_rejects_tiny_and_room_sized_masks() -> None:
    frame = _frame()
    wall = _wall(n_side=120, half=0.9)
    full = np.ones((1, frame.height, frame.width), dtype=bool)

    # Size the mesh so the whole visible surface sits just ABOVE the cap.
    visible, _ = visible_vertices(_pad_to(wall, 200_000), frame)
    n_total = max(len(wall), int(len(visible) / MAX_MASK_FRAC * 0.9))
    assert MAX_MASK_FRAC * n_total < len(visible), "test premise wrong"
    assert lift_masks(_pad_to(wall, n_total), frame, 0, full,
                      np.ones((1, 2), dtype=np.float32)) == [], \
        "a mask covering the whole mesh was accepted"

    tiny = np.zeros((1, frame.height, frame.width), dtype=bool)
    tiny[0, 0, 0] = True
    assert lift_masks(_pad_to(wall, 200_000), frame, 0, tiny,
                      np.ones((1, 2), dtype=np.float32)) == [], \
        f"a mask below {MIN_MASK_VERTICES} vertices was accepted"


def test_masks_from_the_same_view_are_never_associated() -> None:
    """SAM's multi-scale outputs in one frame are distinct hypotheses."""
    frames = [_frame(), _frame(_yaw(30))]
    same_view = [_mask(0, 0, 0, 10_000), _mask(0, 1, 0, 9_500)]
    groups = associate(same_view, frames, N_VERTICES)
    assert len(groups) == 2, f"same-view masks were merged: {groups}"

    across = [_mask(0, 0, 0, 10_000), _mask(1, 0, 0, 9_500)]
    groups = associate(across, frames, N_VERTICES)
    assert len(groups) == 1, f"identical masks across views did not associate: {groups}"


def test_containment_cannot_chain_through_scales() -> None:
    """A cushion inside a sofa inside a room must not become one cluster.

    Containment is admitted only when the two masks are within a factor of
    two in size; without that guard single-linkage walks up the scale ladder
    and returns the whole room as one hypothesis.
    """
    frames = [_frame(), _frame(_yaw(30)), _frame(_yaw(60))]
    cushion = _mask(0, 0, 0, 1_000)
    sofa = _mask(1, 0, 0, 6_000)
    room = _mask(2, 0, 0, 14_000)
    groups = associate([cushion, sofa, room], frames, N_VERTICES)
    assert len(groups) > 1, \
        f"containment chained across scales into one cluster: {groups}"

    # Sanity: within the size guard, containment DOES associate.
    ratio = 6_000 / 10_000
    assert ratio >= ASSOC_CONTAINMENT_MIN_SIZE_RATIO, "test premise wrong"
    pair = associate([_mask(0, 0, 0, 6_000), _mask(1, 0, 0, 10_000)],
                     frames, N_VERTICES)
    assert len(pair) == 1, f"comparable nested masks did not associate: {pair}"


def test_fusion_requires_multi_view_support() -> None:
    frames = [_frame(), _frame(_yaw(30))]
    visibility = {0: np.arange(N_VERTICES), 1: np.arange(N_VERTICES)}
    single = [_mask(0, 0, 0, 5_000)]
    assert fuse_cluster(single, frames, visibility, N_VERTICES) is None, \
        f"a single-view cluster was fused despite MIN_SUPPORT_VIEWS="\
        f"{MIN_SUPPORT_VIEWS}"

    pair = [_mask(0, 0, 0, 5_000), _mask(1, 0, 0, 5_000)]
    fused = fuse_cluster(pair, frames, visibility, N_VERTICES)
    assert fused is not None and fused.support_views == 2, fused
    assert len(fused.vertices) == 5_000, len(fused.vertices)


def test_the_vote_is_normalised_by_visibility() -> None:
    """A vertex hidden in a view is absence of evidence, not evidence against.

    Vertices 5000-6000 are visible only in view 0 and masked there. With a
    naive 1-of-2 denominator they would score 0.5 and sit exactly on the
    threshold by accident; normalised by visibility they score 1.0 and are
    kept for the right reason.
    """
    frames = [_frame(), _frame(_yaw(30))]
    visibility = {0: np.arange(0, 6_000), 1: np.arange(0, 5_000)}
    members = [_mask(0, 0, 0, 6_000), _mask(1, 0, 0, 5_000)]
    fused = fuse_cluster(members, frames, visibility, N_VERTICES)
    assert fused is not None
    assert len(fused.vertices) == 6_000, (
        f"{len(fused.vertices)} vertices; the view-1-invisible tail was "
        "penalised for being absent from a mask that could not contain it")


def test_disagreeing_members_are_pruned_then_refused_if_unsupported() -> None:
    """A cluster held together by one outlier must not survive as that outlier."""
    frames = [_frame(), _frame(_yaw(30)), _frame(_yaw(60))]
    visibility = {i: np.arange(N_VERTICES) for i in range(3)}
    members = [_mask(0, 0, 0, 5_000), _mask(1, 0, 0, 5_000),
               _mask(2, 0, 60_000, 61_000)]
    fused = fuse_cluster(members, frames, visibility, N_VERTICES)
    assert fused is not None, "the agreeing pair should still fuse"
    assert fused.support_views == 2, fused.support_views
    assert 60_000 not in set(fused.vertices.tolist()), \
        "the disagreeing member survived pruning"


def test_classification_emits_beside_the_baseline() -> None:
    frames = [_frame(), _frame(_yaw(30))]
    visibility = {0: np.arange(N_VERTICES), 1: np.arange(N_VERTICES)}
    parent = np.arange(0, 10_000, dtype=np.int64)

    unexplained = fuse_cluster([_mask(0, 0, 50_000, 55_000),
                                _mask(1, 0, 50_000, 55_000)],
                               frames, visibility, N_VERTICES)
    piece = fuse_cluster([_mask(0, 1, 1_000, 4_000),
                          _mask(1, 1, 1_000, 4_000)],
                         frames, visibility, N_VERTICES)
    proposals, diagnostics = classify_clusters(
        [unexplained, piece], [parent], N_VERTICES)
    kinds = sorted(p.kind for p in proposals)
    assert kinds == ["additional", "split"], (kinds, diagnostics)
    split = next(p for p in proposals if p.kind == "split")
    assert split.vertices.max() < 10_000, "split piece leaked outside its parent"
    assert split.parent_index == 0, split.parent_index


def test_a_copy_of_a_baseline_proposal_is_not_a_repair() -> None:
    frames = [_frame(), _frame(_yaw(30))]
    visibility = {0: np.arange(N_VERTICES), 1: np.arange(N_VERTICES)}
    parent = np.arange(0, 10_000, dtype=np.int64)
    copy = fuse_cluster([_mask(0, 0, 0, 10_000), _mask(1, 0, 0, 10_000)],
                        frames, visibility, N_VERTICES)
    proposals, diagnostics = classify_clusters([copy], [parent], N_VERTICES)
    assert proposals == [], [p.kind for p in proposals]
    assert diagnostics["rejected"]["duplicate_of_baseline"] == 1, diagnostics


def test_selection_enforces_diversity_and_prefers_multiplicity() -> None:
    room = _shell()
    # Wide field of view so a 400k-point shell fills the coarse coverage grid
    # from any orientation; sized empirically, not guessed.
    frames = [_frame(_yaw(d), fx_scale=0.5) for d in (0, 1, 2, 20, 40)]
    picked, diagnostics = select_frames(frames, room, n_frames=3)
    directions = []
    for i in picked:
        d = frames[i].R_wc.T @ np.array([0.0, 0.0, 1.0])
        directions.append(d / np.linalg.norm(d))
    for i in range(len(directions)):
        for j in range(i + 1, len(directions)):
            angle = np.degrees(np.arccos(np.clip(directions[i] @ directions[j],
                                                 -1, 1)))
            assert angle >= diagnostics["min_angular_separation_deg"] - 1e-6, \
                f"selected two views {angle:.1f} deg apart"
    assert diagnostics["mesh_seen_two_views"] > 0.0, \
        "multiplicity objective selected no overlapping views at all"


def test_config_record_is_complete() -> None:
    """Every declared constant reaches the artifact manifest."""
    record = config_record()
    for key in ("n_frames", "min_angular_separation_deg", "assoc_iou",
                "assoc_containment", "min_support_views",
                "vertex_vote_fraction", "min_member_iou", "max_proposal_frac",
                "max_repair_proposals"):
        assert key in record, f"{key} missing from config_record()"
    assert record["max_proposal_frac"] == record["max_mask_frac"], \
        "mask and proposal size caps disagree"


def test_module_is_annotation_free() -> None:
    for name in ("segmenter/sam_multiview_repair.py",
                 "tools/arkitscenes_repair_frames.py",
                 "tools/arkitscenes_repair_propose_sam.py"):
        path = REPO_ROOT / name
        text = path.read_text()
        assert "3dod_annotation" not in text, path
        imported: list[str] = []
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import):
                imported += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert "load_oracle_entities" not in text, \
            f"{name} references the oracle loader"


def test_loopback_artifact_if_present() -> None:
    """Dataset-guarded: reads whatever the loopback tool last produced."""
    reports = sorted(RUN_ROOT.glob("*/repair_loopback.json"))
    if not reports:
        print("  skip: no loopback report on disk")
        return
    for path in reports:
        report = json.loads(path.read_text())
        assert report["n_probes_in_two_or_more_views"] > 0, path
        # Necessary condition: given PERFECT masks the geometry must recover a
        # clear majority of multi-view probes. This is not a claim about SAM.
        rate = report["recovery_rate_of_eligible"]
        assert rate is not None and rate >= 0.5, \
            (f"{path}: loopback recovered {rate:.0%} of eligible probes with "
             "perfect masks; lifting/association/fusion is broken")


TESTS = [
    test_overlapping_masks_lift_independently,
    test_unmasked_pixels_contribute_nothing,
    test_lifting_rejects_tiny_and_room_sized_masks,
    test_masks_from_the_same_view_are_never_associated,
    test_containment_cannot_chain_through_scales,
    test_fusion_requires_multi_view_support,
    test_the_vote_is_normalised_by_visibility,
    test_disagreeing_members_are_pruned_then_refused_if_unsupported,
    test_classification_emits_beside_the_baseline,
    test_a_copy_of_a_baseline_proposal_is_not_a_repair,
    test_selection_enforces_diversity_and_prefers_multiplicity,
    test_config_record_is_complete,
    test_module_is_annotation_free,
    test_loopback_artifact_if_present,
]


def main() -> int:
    failed = 0
    for test in TESTS:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
            print()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
