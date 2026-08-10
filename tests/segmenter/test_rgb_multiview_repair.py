"""Multi-view RGB repair arm: mask source, visibility, selection, classification.

All synthetic and always runnable — the dataset run lives in
`runs/arkitscenes_repair/` and is recorded in `docs/repair_arm_design_note.md`.

The tests that matter most are the ones pinning properties the development
result depends on being true:

  * `visible_vertices` returns the whole front SURFACE, not one vertex per
    pixel. The first implementation returned one, which starved the consensus
    stage of evidence and made the arm emit nothing; a regression here would
    reproduce that silently.
  * classification is ADDITIVE and clipped — a split piece never escapes its
    parent, and no baseline proposal is ever consumed.
  * the arm cannot emit a giant mask. This is structural, not measured, and
    the test says so.
"""
from __future__ import annotations

import ast
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from extractors.arkitscenes_rgb_crops import Frame
from segmenter.rgb_multiview_repair import (
    ADDITIONAL_MAX_CONTAINMENT, DUPLICATE_IOU, MAX_PROPOSAL_FRAC,
    MAX_REPAIR_PROPOSALS, MIN_PROPOSAL_VERTICES, SPLIT_MAX_SIZE_RATIO,
    SURFACE_DEPTH_TOLERANCE_M, classify_components, coarse_coverage,
    felzenszwalb, select_frames, visible_vertices,
)

N_VERTICES = 100_000


def _frame(R_wc: np.ndarray | None = None, t_wc: np.ndarray | None = None,
           width: int = 32, height: int = 24) -> Frame:
    return Frame(
        timestamp=0.0, png=Path("/nonexistent.png"),
        R_wc=np.eye(3) if R_wc is None else R_wc,
        t_wc=np.zeros(3) if t_wc is None else t_wc,
        fx=float(width), fy=float(width), cx=width / 2.0, cy=height / 2.0,
        width=width, height=height)


def test_felzenszwalb_separates_regions_and_is_deterministic() -> None:
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[:, 20:] = 255
    labels = felzenszwalb(image, min_size=50)
    assert len(np.unique(labels)) == 2, np.unique(labels)
    assert labels[0, 0] != labels[0, 39], "the two halves merged"
    assert np.array_equal(labels, felzenszwalb(image, min_size=50)), \
        "segmentation is not deterministic"

    flat = np.full((30, 30, 3), 120, dtype=np.uint8)
    assert len(np.unique(felzenszwalb(flat, min_size=50))) == 1, \
        "a uniform image produced more than one region"


def test_min_size_absorbs_small_regions() -> None:
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[0:3, 0:3] = 255                       # 9 px speck
    assert len(np.unique(felzenszwalb(image, min_size=50))) == 1, \
        "a 9 px speck survived min_size=50"


def test_visible_vertices_returns_the_whole_front_surface() -> None:
    """Not one vertex per pixel. This is the bug that made the arm emit nothing.

    A dense patch of vertices at the same depth all project into a handful of
    pixels; every one of them is on the visible surface and must be returned.
    """
    frame = _frame()
    grid = np.linspace(-0.4, 0.4, 40)
    xx, yy = np.meshgrid(grid, grid)
    surface = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, 2.0)], axis=1)
    vertices, pixel = visible_vertices(surface, frame)
    assert len(vertices) == len(surface), \
        f"{len(vertices)}/{len(surface)} of a single flat surface were visible"
    assert len(np.unique(pixel)) < len(surface), \
        "the test surface does not actually share pixels; it proves nothing"


def test_visible_vertices_hides_what_is_occluded() -> None:
    """An occluder in front removes the far point; a co-planar one does not."""
    frame = _frame()
    far = np.array([0.0, 0.0, 4.0])
    near = np.array([0.0, 0.0, 1.0])
    just_behind = np.array([0.0, 0.0, 1.0 + SURFACE_DEPTH_TOLERANCE_M / 2])
    points = np.stack([far, near, just_behind])
    vertices, _ = visible_vertices(points, frame)
    assert set(vertices.tolist()) == {1, 2}, \
        f"expected the near pair to survive, got {vertices.tolist()}"

    # Behind the camera is never visible.
    behind = np.array([[0.0, 0.0, -2.0], [0.0, 0.0, 2.0]])
    vertices, _ = visible_vertices(behind, frame)
    assert vertices.tolist() == [1], vertices.tolist()


def test_coarse_coverage_scores_a_facing_frame_above_an_averted_one() -> None:
    grid = np.linspace(-1.0, 1.0, 60)
    xx, yy = np.meshgrid(grid, grid)
    wall = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, 2.0)], axis=1)
    facing = coarse_coverage(wall, _frame(), stride=1)
    # Rotate 180 degrees about y: the wall falls behind the camera.
    away = np.diag([-1.0, 1.0, -1.0])
    averted = coarse_coverage(wall, _frame(R_wc=away), stride=1)
    assert facing > 0.5, facing
    assert averted == 0.0, averted


def test_select_frames_rejects_invalid_and_spreads_angles() -> None:
    # Dense enough to survive COVERAGE_STRIDE, and a row length coprime with
    # it: select_frames subsamples the mesh 16:1 before scoring coverage, so a
    # grid whose width divides the stride aliases onto a handful of columns
    # and fails validity for a reason that has nothing to do with the frame.
    # Wide enough that a yawed camera still fills its frame, so the validity
    # filter does not quietly remove the very frame diversity is meant to pick.
    grid = np.linspace(-3.0, 3.0, 251)
    xx, yy = np.meshgrid(grid, grid)
    wall = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, 2.0)], axis=1)

    def yaw(degrees: float) -> np.ndarray:
        a = np.radians(degrees)
        return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0],
                         [-np.sin(a), 0, np.cos(a)]])

    frames = [_frame(R_wc=yaw(d)) for d in (0, 2, 4, 25)]
    frames.append(_frame(R_wc=np.diag([-1.0, 1.0, -1.0])))     # faces away
    picked, coverage = select_frames(frames, wall, n_frames=2)
    assert 4 not in picked, "an averted frame passed the validity filter"
    assert all(c >= 0.6 for c in coverage), coverage
    assert 3 in picked, \
        f"the most distinct view direction was not chosen: {picked}"

    try:
        select_frames([frames[4]], wall, n_frames=1)
    except ValueError as exc:
        assert "mesh coverage" in str(exc), exc
    else:
        raise AssertionError("a scene with no valid frame should raise")


def _component(lo: int, hi: int, cut: float = 0.75) -> dict:
    return {"cut": cut, "vertices": np.arange(lo, hi, dtype=np.int64),
            "digest": f"{lo}-{hi}"}


def _stats(n: int, conf: float = 0.9, views: float = 6.0):
    return [(conf, views)] * n


def test_classification_splits_pieces_and_clips_them_to_the_parent() -> None:
    """A piece of an overmerged parent is emitted, clipped, beside it."""
    parent = np.arange(0, 10_000, dtype=np.int64)
    # 3200 vertices, 3000 of them inside the parent and 200 spilling outside:
    # containment 0.94 (>= SPLIT_MIN_CONTAINMENT) at 32% of the parent's size
    # (<= SPLIT_MAX_SIZE_RATIO), which is the split branch.
    component = _component(7_000, 10_200)
    proposals, diag = classify_components(
        [component], _stats(1), [parent], N_VERTICES)
    assert len(proposals) == 1, diag
    piece = proposals[0]
    assert piece.kind == "split", piece.kind
    assert piece.parent_index == 0, piece.parent_index
    assert piece.vertices.max() < 10_000, \
        "the split piece leaked outside its parent"
    assert len(piece.vertices) == 3_000, len(piece.vertices)


def test_classification_emits_unexplained_regions_as_additional() -> None:
    parent = np.arange(0, 10_000, dtype=np.int64)
    component = _component(50_000, 51_000)        # disjoint from the baseline
    proposals, _ = classify_components(
        [component], _stats(1), [parent], N_VERTICES)
    assert len(proposals) == 1 and proposals[0].kind == "additional"
    assert proposals[0].containment == 0.0, proposals[0].containment


def test_classification_rejects_junk_duplicates_and_weak_consensus() -> None:
    parent = np.arange(0, 10_000, dtype=np.int64)
    components = [
        _component(50_000, 50_000 + MIN_PROPOSAL_VERTICES - 1),   # too small
        _component(60_000, 60_000 + int(MAX_PROPOSAL_FRAC * N_VERTICES) + 10),
        _component(0, 10_000),                                    # duplicate
        _component(70_000, 71_000),                               # weak views
    ]
    stats = [(0.9, 6.0), (0.9, 6.0), (0.9, 6.0), (0.9, 1.0)]
    proposals, diag = classify_components(
        components, stats, [parent], N_VERTICES)
    assert proposals == [], [p.kind for p in proposals]
    assert diag["rejected"]["too_small"] == 1, diag["rejected"]
    assert diag["rejected"]["too_large"] == 1, diag["rejected"]
    assert diag["rejected"]["duplicate_of_baseline"] == 1, diag["rejected"]
    assert diag["rejected"]["weak_consensus"] == 1, diag["rejected"]


def test_a_near_copy_of_the_parent_is_not_a_split() -> None:
    """Containment is high but the component is the parent, not a piece of it."""
    parent = np.arange(0, 10_000, dtype=np.int64)
    # 85% of the parent: above SPLIT_MAX_SIZE_RATIO, below DUPLICATE_IOU.
    component = _component(0, int(SPLIT_MAX_SIZE_RATIO * 10_000) + 1_500)
    iou = len(component["vertices"]) / 10_000
    assert iou < DUPLICATE_IOU, "test does not exercise the intended branch"
    proposals, diag = classify_components(
        [component], _stats(1), [parent], N_VERTICES)
    assert proposals == [], [p.kind for p in proposals]
    assert diag["rejected"]["ambiguous_containment"] == 1, diag["rejected"]


def test_emission_is_capped_and_ordered_by_consensus() -> None:
    components = [_component(1_000 * i, 1_000 * i + 500)
                  for i in range(MAX_REPAIR_PROPOSALS + 20)]
    # Ascending confidence, so the cap must keep the LAST ones.
    stats = [(0.5 + 0.001 * i, 6.0) for i in range(len(components))]
    proposals, diag = classify_components(components, stats, [], N_VERTICES)
    assert len(proposals) == MAX_REPAIR_PROPOSALS, len(proposals)
    assert diag["n_dropped_by_cap"] == 20, diag
    confidences = [p.confidence for p in proposals]
    assert confidences == sorted(confidences, reverse=True), \
        "emission is not ordered by consensus"
    assert proposals[0].confidence == max(s[0] for s in stats)


def test_the_arm_cannot_emit_a_giant_mask() -> None:
    """Structural, not measured: MAX_PROPOSAL_FRAC equals the giant threshold.

    Worth pinning precisely because it means the giant-mask gate in the
    evaluator is satisfied by construction and carries no evidential weight.
    """
    from eval.detection_repair import GIANT_FRAC
    assert MAX_PROPOSAL_FRAC <= GIANT_FRAC, (
        f"the arm may emit proposals up to {MAX_PROPOSAL_FRAC} of the mesh "
        f"while the evaluator calls {GIANT_FRAC} giant")
    huge = _component(0, int(GIANT_FRAC * N_VERTICES) + 1)
    proposals, diag = classify_components([huge], _stats(1), [], N_VERTICES)
    assert proposals == [], "a giant component was emitted"
    assert diag["rejected"]["too_large"] == 1, diag["rejected"]


def test_containment_branches_meet_at_the_declared_thresholds() -> None:
    """Nothing falls between `additional` and `split` unnoticed."""
    parent = np.arange(0, 10_000, dtype=np.int64)
    # 60% contained: above ADDITIONAL_MAX_CONTAINMENT, below SPLIT_MIN.
    component = {"cut": 0.75, "digest": "mid",
                 "vertices": np.concatenate([
                     np.arange(0, 600, dtype=np.int64),
                     np.arange(50_000, 50_400, dtype=np.int64)])}
    containment = 600 / 1000
    assert ADDITIONAL_MAX_CONTAINMENT < containment, "branch not exercised"
    proposals, diag = classify_components(
        [component], _stats(1), [parent], N_VERTICES)
    assert proposals == [], [p.kind for p in proposals]
    assert diag["rejected"]["ambiguous_containment"] == 1, diag["rejected"]


def test_repair_module_is_annotation_free() -> None:
    path = REPO_ROOT / "segmenter" / "rgb_multiview_repair.py"
    text = path.read_text()
    assert "3dod_annotation" not in text, path
    imported: list[str] = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    offenders = [m for m in imported if m.startswith("tools")]
    assert not offenders, f"{path.name} imports {offenders}"


TESTS = [
    test_felzenszwalb_separates_regions_and_is_deterministic,
    test_min_size_absorbs_small_regions,
    test_visible_vertices_returns_the_whole_front_surface,
    test_visible_vertices_hides_what_is_occluded,
    test_coarse_coverage_scores_a_facing_frame_above_an_averted_one,
    test_select_frames_rejects_invalid_and_spreads_angles,
    test_classification_splits_pieces_and_clips_them_to_the_parent,
    test_classification_emits_unexplained_regions_as_additional,
    test_classification_rejects_junk_duplicates_and_weak_consensus,
    test_a_near_copy_of_the_parent_is_not_a_split,
    test_emission_is_capped_and_ordered_by_consensus,
    test_the_arm_cannot_emit_a_giant_mask,
    test_containment_branches_meet_at_the_declared_thresholds,
    test_repair_module_is_annotation_free,
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
