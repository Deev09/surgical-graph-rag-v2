"""Synthetic tests for the oracle-free ARKitScenes entity bridge."""
from __future__ import annotations

import sys
import tempfile
import traceback
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from adapters.arkitscenes import ARKitScenesMesh, write_mesh
from common.equality import array_aware_equal
from common.types import SceneFrame
from extractors.arkitscenes_segments import build_arkitscenes_segment_artifacts
from extractors.serde import dump_entity_artifacts, load_entity_artifacts
from representations.base import (
    GeometryHandle,
    ReconstructionDiagnostics,
    RepresentationCapabilities,
    SceneRepresentationBundle,
)
from segmenter.base import (
    SegmentationOutput,
    save_segmentation_output,
    sha256_file,
)


def _fixture(root: Path) -> tuple[SceneRepresentationBundle, Path, np.ndarray]:
    # Two separated, non-degenerate components plus one unassigned vertex.
    xyz = np.asarray([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        [3.0, 2.0, 0.2], [4.0, 2.0, 0.2],
        [4.0, 3.0, 0.8], [3.0, 3.0, 0.8],
        [9.0, 9.0, 9.0],
    ], dtype=np.float64)
    mesh_path = root / "410_test_3dod_mesh_canonical.ply"
    write_mesh(mesh_path, ARKitScenesMesh(
        xyz=xyz,
        rgb=np.full((len(xyz), 3), 127, dtype=np.uint8),
        faces=np.empty((0, 3), dtype=np.int64),
    ))
    frame = SceneFrame(
        gravity=(0.0, 0.0, -1.0), canonical_forward=None,
        canonical_right=None, units="meters", kind="scene_canonical",
        notes="synthetic ARKit canonical frame",
    )
    representation = SceneRepresentationBundle(
        schema_version=1,
        representation_hash="repr_ark_test",
        scene_id="arkitscenes_410_test",
        frame=frame,
        capabilities=RepresentationCapabilities(
            renderable_channels=frozenset({"rgb"}),
            supports_arbitrary_pose=True,
            deterministic=True,
            typical_render_ms=1,
        ),
        geometry_handle=GeometryHandle(
            kind="mesh_file", uri=str(mesh_path), notes={"canonical": True}),
        poses=[],
        diagnostics=ReconstructionDiagnostics(
            loss=None, coverage=None, pose_rmse=None,
            runtime_seconds=0.0, notes="synthetic",
        ),
        notes={"adapter": "arkitscenes", "reads_annotations": False},
    )
    ids = np.asarray([10, 10, 10, 10, 42, 42, 42, 42, -1], dtype=np.int64)
    seg = SegmentationOutput(
        input_mesh_sha256=sha256_file(mesh_path),
        n_vertices=len(xyz),
        segmenter_name="synthetic_mask3d",
        segmenter_version="test",
        config_params_json="{}",
        vertex_instance_ids=ids,
        instance_confidence={10: 0.9, 42: 0.75},
    ).finalize()
    seg_dir = root / "segmentation"
    save_segmentation_output(seg, seg_dir)
    return representation, seg_dir, xyz


def test_bridge_is_anonymous_oracle_free_and_in_frame() -> None:
    with tempfile.TemporaryDirectory() as td:
        rep, seg_dir, xyz = _fixture(Path(td))
        artifacts = build_arkitscenes_segment_artifacts(
            rep, seg_dir, min_vertices=4)

        if artifacts.scene_id != rep.scene_id:
            raise AssertionError("scene id did not propagate")
        if artifacts.frame != rep.frame:
            raise AssertionError("representation frame did not propagate exactly")
        if artifacts.representation_hash != rep.representation_hash:
            raise AssertionError("representation hash did not propagate")
        if artifacts.structural_surfaces:
            raise AssertionError("oracle-free bridge must emit no surfaces")
        if artifacts.notes["semantic_source"] != "none":
            raise AssertionError("oracle-free bridge must emit no semantic source")
        if artifacts.notes["surface_source"] != "none":
            raise AssertionError("oracle-free bridge must emit no surface source")
        if artifacts.notes["oracle_free"] is not True:
            raise AssertionError("oracle-free provenance flag missing")
        if artifacts.geometry_store_path != seg_dir:
            raise AssertionError("dense-assignment store was not preserved")

        entities = {e.identity.object_uid: e for e in artifacts.entities}
        if set(entities) != {"obj_10", "obj_42"}:
            raise AssertionError(f"unexpected anonymous entities: {set(entities)}")
        for inst, sl in ((10, slice(0, 4)), (42, slice(4, 8))):
            entity = entities[f"obj_{inst}"]
            if entity.identity.display_label != f"segment_{inst}":
                raise AssertionError("bridge invented a semantic label")
            if entity.semantic_hypotheses:
                raise AssertionError("bridge invented semantic hypotheses")
            # Identity transform: boxes must be in the canonical mesh frame.
            want = (tuple(xyz[sl].min(axis=0)), tuple(xyz[sl].max(axis=0)))
            if not np.allclose(entity.bbox_aabb, want, atol=1e-6):
                raise AssertionError(
                    f"canonical bbox mismatch for {inst}: {entity.bbox_aabb} != {want}")
            if entity.extraction_diagnostics.get("instance_confidence") is None:
                raise AssertionError("segmenter confidence was not preserved")
            if str(seg_dir) not in (entity.geometry_handle or ""):
                raise AssertionError("geometry handle does not name the saved bundle")


def test_hash_is_content_based_and_sensitive_to_inputs() -> None:
    with tempfile.TemporaryDirectory() as td:
        rep, seg_dir, _ = _fixture(Path(td))
        first = build_arkitscenes_segment_artifacts(rep, seg_dir, min_vertices=4)
        again = build_arkitscenes_segment_artifacts(rep, seg_dir, min_vertices=4)
        if first.bundle_hash != again.bundle_hash:
            raise AssertionError("identical inputs produced a drifting bundle hash")

        other_rep = replace(rep, representation_hash="repr_ark_changed")
        changed_rep = build_arkitscenes_segment_artifacts(
            other_rep, seg_dir, min_vertices=4)
        if first.bundle_hash == changed_rep.bundle_hash:
            raise AssertionError("representation hash change did not change entity hash")

        changed_filter = build_arkitscenes_segment_artifacts(
            rep, seg_dir, min_vertices=3)
        if first.bundle_hash == changed_filter.bundle_hash:
            raise AssertionError("extractor config change did not change entity hash")


def test_artifacts_round_trip_for_downstream_graph() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        rep, seg_dir, _ = _fixture(root)
        artifacts = build_arkitscenes_segment_artifacts(
            rep, seg_dir, min_vertices=4)
        artifact_dir = root / "entity_artifacts"
        dump_entity_artifacts(artifacts, artifact_dir)
        loaded = load_entity_artifacts(artifact_dir)
        if not array_aware_equal(artifacts, loaded):
            raise AssertionError("serialized entity bundle did not round-trip")


def test_rejects_noncanonical_or_non_arkit_representation() -> None:
    with tempfile.TemporaryDirectory() as td:
        rep, seg_dir, _ = _fixture(Path(td))
        bad_inputs = (
            replace(rep, frame=replace(rep.frame, kind="world")),
            replace(rep, notes={"adapter": "arkitscenes", "reads_annotations": True}),
            replace(rep, notes={"adapter": "other", "reads_annotations": False}),
            replace(rep, geometry_handle=replace(
                rep.geometry_handle, kind="pointcloud_file")),
        )
        for bad in bad_inputs:
            try:
                build_arkitscenes_segment_artifacts(bad, seg_dir, min_vertices=4)
            except ValueError:
                continue
            raise AssertionError(f"invalid representation was accepted: {bad}")


def test_rejects_wrong_mesh_or_tampered_assignment() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        rep, seg_dir, _ = _fixture(root)

        other_mesh = root / "other_canonical.ply"
        other_mesh.write_bytes(Path(rep.geometry_handle.uri).read_bytes() + b"x")
        wrong_mesh_rep = replace(
            rep, geometry_handle=replace(rep.geometry_handle, uri=str(other_mesh)))
        try:
            build_arkitscenes_segment_artifacts(
                wrong_mesh_rep, seg_dir, min_vertices=4)
        except ValueError as exc:
            if "mesh hash mismatch" not in str(exc):
                raise AssertionError(f"wrong failure for mesh mismatch: {exc}")
        else:
            raise AssertionError("segmentation was accepted against a different mesh")

        ids_path = seg_dir / "vertex_instance_ids.npy"
        tampered = np.load(ids_path)
        tampered[0] = -1
        np.save(ids_path, tampered)
        try:
            build_arkitscenes_segment_artifacts(rep, seg_dir, min_vertices=4)
        except ValueError as exc:
            if "bundle hash mismatch" not in str(exc):
                raise AssertionError(f"wrong failure for sidecar tampering: {exc}")
        else:
            raise AssertionError("tampered segmentation sidecar was accepted")


TESTS = [
    test_bridge_is_anonymous_oracle_free_and_in_frame,
    test_hash_is_content_based_and_sensitive_to_inputs,
    test_artifacts_round_trip_for_downstream_graph,
    test_rejects_noncanonical_or_non_arkit_representation,
    test_rejects_wrong_mesh_or_tampered_assignment,
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
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
