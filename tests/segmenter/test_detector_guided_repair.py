"""Detector-guided repair: label discipline, provenance, and no background masks.

The invariants worth pinning are the ones the brief named explicitly:

  * association needs COMPATIBLE LABELS as well as 3D overlap, so two masks the
    detector called different things never merge however much they overlap;
  * no background or complement proposal is ever created;
  * the fixed 41-class vocabulary is used unmodified;
  * detector label, score, box and frame survive into the emitted proposal.
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

from extractors.learned_labels import GLOBAL_INDOOR_VOCABULARY_V1
from extractors.arkitscenes_rgb_crops import Frame
from segmenter.detector_guided_repair import (
    VOCABULARY, associate_by_label_and_overlap, config_record,
    label_from_prompt_phrase, prompt_text,
)
from segmenter.sam_multiview_repair import LiftedMask, lift_masks

N = 100_000


def _frame(R_wc=None, width: int = 32, height: int = 24) -> Frame:
    return Frame(timestamp=0.0, png=Path("/nonexistent.png"),
                 R_wc=np.eye(3) if R_wc is None else R_wc, t_wc=np.zeros(3),
                 fx=float(width), fy=float(width), cx=width / 2.0,
                 cy=height / 2.0, width=width, height=height)


def _yaw(degrees: float) -> np.ndarray:
    a = np.radians(degrees)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0],
                     [-np.sin(a), 0, np.cos(a)]])


def _mask(view: int, index: int, lo: int, hi: int) -> LiftedMask:
    return LiftedMask(view, index, np.arange(lo, hi, dtype=np.int64), 0.9, 0.97)


def test_vocabulary_is_the_repo_list_unmodified() -> None:
    assert VOCABULARY is GLOBAL_INDOOR_VOCABULARY_V1, \
        "the arm copied the vocabulary instead of importing it"
    assert len(VOCABULARY) == 41, len(VOCABULARY)
    assert config_record()["vocabulary_modified"] is False


def test_prompt_uses_spaces_but_labels_keep_hyphens() -> None:
    prompt = prompt_text()
    assert "trash can" in prompt and "trash-can" not in prompt, prompt[:200]
    assert "tv monitor" in prompt, prompt[:200]
    assert prompt.count(" . ") == len(VOCABULARY) - 1
    # ...and the canonical hyphen form is what comes back.
    assert label_from_prompt_phrase("trash can") == "trash-can"
    assert label_from_prompt_phrase("tv monitor") == "tv-monitor"
    assert label_from_prompt_phrase("sofa") == "sofa"


def test_ambiguous_or_unknown_phrases_resolve_to_none() -> None:
    """A guessed label would corrupt association; dropping is the safe failure."""
    assert label_from_prompt_phrase("teapot") is None
    assert label_from_prompt_phrase("") is None
    # 'c' prefixes cabinet, chair, clock, counter, cushion -> ambiguous.
    assert label_from_prompt_phrase("c") is None


def test_different_labels_never_associate_however_much_they_overlap() -> None:
    """The core of label-guided association."""
    frames = [_frame(), _frame()]
    masks = [_mask(0, 0, 0, 10_000), _mask(1, 0, 0, 10_000)]
    same = associate_by_label_and_overlap(masks, ["sofa", "sofa"], N)
    assert len(same) == 1, f"identical masks with one label did not merge: {same}"
    different = associate_by_label_and_overlap(masks, ["sofa", "cushion"], N)
    assert len(different) == 2, \
        f"masks the detector called different classes were merged: {different}"


def test_same_label_still_needs_3d_overlap() -> None:
    frames = [_frame(), _frame()]
    masks = [_mask(0, 0, 0, 5_000), _mask(1, 0, 50_000, 55_000)]
    groups = associate_by_label_and_overlap(masks, ["chair", "chair"], N)
    assert len(groups) == 2, \
        f"a shared label merged two disjoint surfaces: {groups}"


def test_same_view_detections_stay_separate() -> None:
    frames = [_frame(), _frame()]
    masks = [_mask(0, 0, 0, 10_000), _mask(0, 1, 0, 9_800)]
    groups = associate_by_label_and_overlap(masks, ["table", "table"], N)
    assert len(groups) == 2, f"two detections in one frame merged: {groups}"


def test_label_count_must_match_mask_count() -> None:
    frames = [_frame()]
    try:
        associate_by_label_and_overlap([_mask(0, 0, 0, 100)], [], N)
    except ValueError as exc:
        assert "provenance" in str(exc), exc
    else:
        raise AssertionError("misaligned labels should raise")


def test_lifting_creates_no_complement_region() -> None:
    """Boxes cover part of the frame; the rest must produce nothing."""
    frame = _frame()
    grid = np.linspace(-0.4, 0.4, 120)
    xx, yy = np.meshgrid(grid, grid)
    surface = np.stack([xx.ravel(), yy.ravel(),
                        np.full(xx.size, 2.0)], axis=1)
    points = np.zeros((200_000, 3))
    points[:len(surface)] = surface
    points[len(surface):, 2] = -5.0

    # The surface projects into columns ~10..22, so a box has to overlap that
    # range to test anything. Compare it against lifting the WHOLE frame.
    whole = np.ones((1, frame.height, frame.width), dtype=bool)
    everything = lift_masks(points, frame, 0, whole,
                            np.ones((1, 2), dtype=np.float32))
    assert len(everything) == 1, len(everything)

    box_mask = np.zeros((1, frame.height, frame.width), dtype=bool)
    box_mask[0, :, :16] = True
    lifted = lift_masks(points, frame, 0, box_mask,
                        np.ones((1, 2), dtype=np.float32))
    assert len(lifted) == 1, len(lifted)
    assert len(lifted[0].vertices) < len(everything[0].vertices), \
        "a partial box lifted the entire visible surface"
    # And the uncovered remainder produces NO proposal of its own.
    outside = np.setdiff1d(everything[0].vertices, lifted[0].vertices)
    assert len(outside) > 0, "the fixture box covers everything; it proves nothing"


def test_no_module_creates_a_complement() -> None:
    """Structural: no inversion of a mask anywhere in the arm."""
    path = REPO_ROOT / "segmenter" / "detector_guided_repair.py"
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
            raise AssertionError(
                f"{path.name} line {node.lineno}: bitwise inversion of a mask "
                "would create a background/complement region")
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            assert name not in {"logical_not", "invert", "setdiff1d"}, \
                f"{path.name} line {node.lineno}: {name} creates a complement"


def test_a_split_proposal_keeps_its_detector_label() -> None:
    """Regression: split pieces are clipped, and a first-vertex lookup lost
    their label. Provenance must survive clipping."""
    from segmenter.sam_multiview_repair import classify_clusters, fuse_cluster

    # Two distinct view directions: fuse_cluster requires real angular spread.
    frames = [_frame(), _frame(_yaw(30))]
    visibility = {0: np.arange(N), 1: np.arange(N)}
    # Parent starts at 5000, cluster at 4000: clipping removes the HEAD, so
    # the piece's first vertex differs from the cluster's. That is the case a
    # first-vertex lookup got wrong.
    parent = np.arange(5_000, 15_000, dtype=np.int64)
    cluster = fuse_cluster([_mask(0, 0, 4_000, 10_500),
                            _mask(1, 0, 4_000, 10_500)],
                           frames, visibility, N)
    assert cluster is not None
    labels = ["cabinet"]
    proposals, diagnostics = classify_clusters([cluster], [parent], N)
    assert [p.kind for p in proposals] == ["split"], [p.kind for p in proposals]
    attached = [labels[i] for i in diagnostics["source_cluster_indices"]]
    assert attached == ["cabinet"], attached
    assert proposals[0].vertices[0] != cluster.vertices[0], \
        "fixture does not exercise clipping; the test would pass vacuously"


def test_the_arm_is_annotation_free() -> None:
    for name in ("segmenter/detector_guided_repair.py",
                 "tools/arkitscenes_repair_propose_gdino.py"):
        path = REPO_ROOT / name
        text = path.read_text()
        assert "3dod_annotation" not in text, path
        assert "load_oracle_entities" not in text, path


def test_geometric_thresholds_are_shared_with_the_class_agnostic_arm() -> None:
    """The two arms must differ only in the prompt and the label rule."""
    from segmenter import sam_multiview_repair as agnostic
    record = config_record()
    assert record["assoc_iou"] == agnostic.ASSOC_IOU
    assert record["assoc_containment"] == agnostic.ASSOC_CONTAINMENT
    assert record["assoc_anchor_stride"] == agnostic.ASSOC_ANCHOR_STRIDE
    assert record["require_same_label"] is True


TESTS = [
    test_vocabulary_is_the_repo_list_unmodified,
    test_prompt_uses_spaces_but_labels_keep_hyphens,
    test_ambiguous_or_unknown_phrases_resolve_to_none,
    test_different_labels_never_associate_however_much_they_overlap,
    test_same_label_still_needs_3d_overlap,
    test_same_view_detections_stay_separate,
    test_label_count_must_match_mask_count,
    test_lifting_creates_no_complement_region,
    test_no_module_creates_a_complement,
    test_a_split_proposal_keeps_its_detector_label,
    test_the_arm_is_annotation_free,
    test_geometric_thresholds_are_shared_with_the_class_agnostic_arm,
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
