"""Tests for the oracle-free ARKit vertical slice."""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from adapters.arkitscenes import ARKitScenesMesh, write_mesh
from common.types import SceneFrame
from representations.base import (
    GeometryHandle, ReconstructionDiagnostics, RepresentationCapabilities,
    SceneRepresentationBundle,
)
from segmenter.base import SegmentationOutput, save_segmentation_output, sha256_file
from tools.arkit_vertical_slice import build_slice


class _FakeLabeler:
    weights_sha256 = "synthetic-weights"

    def classify(self, images, vocabulary: list[str]) -> list[dict]:
        first = "table" if not hasattr(self, "called") else "chair"
        self.called = True
        ordered = [first] + [label for label in vocabulary if label != first]
        return [
            {"label": label, "score": 0.4 - i * 0.001}
            for i, label in enumerate(ordered)
        ]


def _fixture(root: Path):
    xyz = np.asarray([
        [0.0, 0.0, 0.0], [0.2, 0.0, 0.0],
        [0.2, 0.2, 0.2], [0.0, 0.2, 0.2],
        [0.5, 0.0, 0.0], [0.7, 0.0, 0.0],
        [0.7, 0.2, 0.2], [0.5, 0.2, 0.2],
    ], dtype=np.float64)
    mesh = root / "scene_canonical.ply"
    write_mesh(mesh, ARKitScenesMesh(
        xyz=xyz, rgb=np.full((len(xyz), 3), 127, dtype=np.uint8),
        faces=np.empty((0, 3), dtype=np.int64)))
    frame = SceneFrame(
        gravity=(0.0, 0.0, -1.0), canonical_forward=None,
        canonical_right=None, units="meters", kind="scene_canonical",
        notes="synthetic",
    )
    rep = SceneRepresentationBundle(
        schema_version=1, representation_hash="repr_slice",
        scene_id="arkitscenes_test", frame=frame,
        capabilities=RepresentationCapabilities(
            renderable_channels=frozenset({"rgb"}),
            supports_arbitrary_pose=True, deterministic=True,
            typical_render_ms=1),
        geometry_handle=GeometryHandle(
            kind="mesh_file", uri=str(mesh), notes={}),
        poses=[], diagnostics=ReconstructionDiagnostics(
            loss=None, coverage=None, pose_rmse=None,
            runtime_seconds=0.0, notes="synthetic"),
        notes={"adapter": "arkitscenes", "reads_annotations": False},
    )
    seg = SegmentationOutput(
        input_mesh_sha256=sha256_file(mesh), n_vertices=len(xyz),
        segmenter_name="synthetic_mask3d", segmenter_version="test",
        config_params_json="{}",
        vertex_instance_ids=np.asarray([0, 0, 0, 0, 1, 1, 1, 1]),
        instance_confidence={0: 0.9, 1: 0.8},
    ).finalize()
    seg_dir = root / "seg"
    save_segmentation_output(seg, seg_dir)
    return rep, seg_dir


def test_slice_writes_oracle_free_vertical_outputs() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        rep, seg_dir = _fixture(root)
        out = root / "out"
        manifest = build_slice(rep, seg_dir, out, min_vertices=4)
        for rel in ("entities/manifest.json", "graph/manifest.json",
                    "graph_diagnostics.json", "inspector.html", "manifest.json"):
            if not (out / rel).is_file():
                raise AssertionError(f"missing vertical output: {rel}")
        if manifest["oracle_free"] is not True:
            raise AssertionError("vertical slice did not disclose oracle boundary")
        if manifest["counts"] != {"entities": 2, "edges": 1, "near_edges": 1}:
            raise AssertionError(f"unexpected graph counts: {manifest['counts']}")
        if manifest["question"]["outcome"] != "bindings":
            raise AssertionError(f"near query did not execute: {manifest['question']}")
        if "floor_wall_relations" not in manifest["unavailable_capabilities"]:
            raise AssertionError("slice failed to disclose unavailable structural relations")
        page = (out / "inspector.html").read_text(encoding="utf-8")
        if "arkitscenes_test" not in page or "distance_metric" not in page:
            raise AssertionError("inspector lost ARKit identity or edge evidence")


def test_manifest_is_deterministic_except_paths_are_stable() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        rep, seg_dir = _fixture(root)
        out = root / "out"
        first = build_slice(rep, seg_dir, out, min_vertices=4)
        first_bytes = (out / "manifest.json").read_bytes()
        second = build_slice(rep, seg_dir, out, min_vertices=4)
        if first != second or first_bytes != (out / "manifest.json").read_bytes():
            raise AssertionError("identical slice inputs produced drifting output")


def test_slice_can_attach_oracle_free_learned_labels() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        rep, seg_dir = _fixture(root)
        out = root / "out"
        manifest = build_slice(
            rep, seg_dir, out, min_vertices=4,
            with_learned_labels=True, labeler=_FakeLabeler(),
            with_support_patches=True,
        )
        if manifest["available_capabilities"][
                "learned_semantic_hypotheses"] is not True:
            raise AssertionError("learned label capability was not recorded")
        if "learned_semantic_labels" in manifest["unavailable_capabilities"]:
            raise AssertionError("available learned labels were also marked unavailable")
        if manifest["label_configuration"]["n_promoted_display_labels"] != 2:
            raise AssertionError("label admission was not propagated to the manifest")
        if manifest["oracle_free"] is not True:
            raise AssertionError("label attachment changed the oracle boundary")
        if not (out / "support_patches.json").is_file():
            raise AssertionError("requested horizontal patch evidence was not written")
        if not (out / "entity_patch_rest.json").is_file():
            raise AssertionError("requested target-to-patch evidence was not written")
        if manifest["available_capabilities"][
                "entity_horizontal_patch_evidence"] is not True:
            raise AssertionError("patch-evidence capability was not recorded")
        if manifest["support_patch_summary"]["uses_oracle"] is not False:
            raise AssertionError("support-patch evidence crossed the oracle boundary")
        if manifest["entity_patch_rest_summary"]["uses_oracle"] is not False:
            raise AssertionError("target-to-patch evidence crossed the oracle boundary")


TESTS = [
    test_slice_writes_oracle_free_vertical_outputs,
    test_manifest_is_deterministic_except_paths_are_stable,
    test_slice_can_attach_oracle_free_learned_labels,
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
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
