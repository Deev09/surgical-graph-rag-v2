"""Replica oracle ReconstructionAdapter.

Wraps the prior `importers/replica.py` output (a pre-imported
scene_graph.json + capture_meta.json under `scenes/<scene_id>/`) into the
new SceneRepresentationBundle contract.

The raw Habitat `info_semantic.json` is NOT in this repository; the import
was performed previously and the result was committed. This adapter
exists to bridge that legacy oracle export into the new architecture
without rerunning the original import.

Boundary discipline (per Phase 1 batch instructions):
  - This module owns capture metadata, mesh handle, frame normalization,
    and SceneRepresentationBundle construction.
  - It does NOT enumerate semantic instances — that belongs to
    `extractors/oracle_replica.py`, which reads the same pre-imported
    scene_graph.json on its own.

The two stages share only the file path (recorded in `bundle.notes`) and
the z_translation that was applied during the original import. The
extractor never reaches into adapter internals.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from adapters.base import (
    CaptureBundle, ReconstructionAdapter, ReconstructionCapabilities,
    ReconstructionConfig,
)
from common.types import SceneFrame, Vec3
from representations.base import (
    GeometryHandle, ReconstructionDiagnostics, RepresentationCapabilities,
    SceneRepresentationBundle,
)
from representations.serde import CURRENT_SCHEMA_VERSION as REPR_SCHEMA_VERSION


_ADAPTER_NAME = "oracle_replica"
_ADAPTER_VERSION = "0.1"


@dataclass(frozen=True)
class ReplicaCaptureInputs:
    """Convenience holder for the Replica oracle adapter's expected layout.

    scene_dir is expected to contain:
      - scene_graph.json    (produced by importers/replica.py)
      - capture_meta.json   (capture metadata from the same import)
    """
    scene_dir: Path


def _hash_files_for_capture(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def build_replica_capture_bundle(scene_dir: Path) -> CaptureBundle:
    """Construct a CaptureBundle from a pre-imported Replica scene directory.

    The bundle_hash is deterministic over the content of scene_graph.json
    and capture_meta.json so that re-runs against the same on-disk state
    produce identical downstream hashes.
    """
    sg_path = scene_dir / "scene_graph.json"
    cm_path = scene_dir / "capture_meta.json"
    if not sg_path.exists() or not cm_path.exists():
        raise FileNotFoundError(
            f"Replica oracle inputs missing under {scene_dir}; "
            f"expected scene_graph.json and capture_meta.json"
        )
    capture_hash = _hash_files_for_capture([sg_path, cm_path])
    capture_meta = json.loads(cm_path.read_text(encoding="utf-8"))
    scene_id = str(capture_meta["scene_id"])
    return CaptureBundle(
        bundle_hash=f"cap_oracle_replica_{capture_hash}",
        scene_id=scene_id,
        images_dir=None,
        poses=None,
        rgbd_dir=None,
        mesh_path=None,
        semantic_export=sg_path,
        notes={"capture_meta_path": str(cm_path), "source": "replica"},
    )


def _normalize_unit(v: list[float] | tuple[float, float, float]) -> Vec3:
    n = math.sqrt(sum(float(x) * float(x) for x in v))
    if n == 0.0:
        raise ValueError("gravity vector is zero")
    return (float(v[0]) / n, float(v[1]) / n, float(v[2]) / n)


def _compute_representation_hash(capture: CaptureBundle, version: str) -> str:
    payload = json.dumps(
        {
            "capture_bundle_hash": capture.bundle_hash,
            "adapter_name": _ADAPTER_NAME,
            "adapter_version": version,
        },
        sort_keys=True,
    )
    return f"repr_{_ADAPTER_NAME}_{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


@dataclass(frozen=True)
class OracleReplicaAdapter:
    """ReconstructionAdapter that wraps a pre-imported Replica scene
    directory into a SceneRepresentationBundle. No reconstruction is
    performed; runtime_seconds is always 0.
    """
    name: str = _ADAPTER_NAME
    version: str = _ADAPTER_VERSION

    def reconstruct(
        self,
        capture: CaptureBundle,
        config: ReconstructionConfig,
    ) -> SceneRepresentationBundle:
        if capture.semantic_export is None:
            raise ValueError(
                "OracleReplicaAdapter requires capture.semantic_export to point "
                "at a pre-imported scene_graph.json"
            )
        cm_path_str = capture.notes.get("capture_meta_path")
        if not cm_path_str:
            raise ValueError(
                "OracleReplicaAdapter requires capture.notes['capture_meta_path'] "
                "to point at capture_meta.json"
            )
        cm_path = Path(cm_path_str)
        capture_meta = json.loads(cm_path.read_text(encoding="utf-8"))

        axis = capture_meta["axis_convention"]
        gravity_raw = axis["gravity_dir_raw"]
        # importers/replica.py levels a tilted capture rather than refusing it
        # (see its module docstring). When it does, the coordinates in
        # scene_graph.json are in a levelled frame, so the SceneFrame must carry
        # the levelled gravity and say "scene_canonical" — otherwise every edge
        # extracted from this bundle inherits a false frame label, and the
        # gravity-reading predicates compare a tilted up against levelled boxes.
        # Both keys are absent from capture_meta files written before this
        # existed; those describe raw-axes imports, which is exactly the default.
        gravity = _normalize_unit(axis.get("gravity_dir_effective", gravity_raw))
        frame_kind = axis.get("frame_kind", "world")
        z_translation = float(capture_meta["import_notes"]["z_translation_applied"])

        frame = SceneFrame(
            gravity=gravity,
            canonical_forward=None,
            canonical_right=None,
            units="meters",
            notes=(
                f"gravity_dir_raw={gravity_raw}; "
                f"z_translation_applied={z_translation}; "
                f"source={capture_meta.get('source', 'replica')}"
            ),
            kind=frame_kind,
        )

        geometry_handle = GeometryHandle(
            kind="oracle_passthrough",
            uri="habitat://semantic-export",
            notes={
                "reason": "oracle wrapper around pre-imported Replica state; "
                          "no geometry blob loaded in Phase 1",
                "semantic_export_path": str(capture.semantic_export),
            },
        )

        capabilities = RepresentationCapabilities(
            renderable_channels=frozenset(),
            supports_arbitrary_pose=False,
            deterministic=True,
            typical_render_ms=0,
        )

        diagnostics = ReconstructionDiagnostics(
            loss=None,
            coverage=None,
            pose_rmse=None,
            runtime_seconds=0.0,
            notes="oracle replica wrapper; no reconstruction performed",
        )

        return SceneRepresentationBundle(
            schema_version=REPR_SCHEMA_VERSION,
            representation_hash=_compute_representation_hash(capture, self.version),
            scene_id=capture.scene_id,
            frame=frame,
            capabilities=capabilities,
            geometry_handle=geometry_handle,
            poses=[],
            diagnostics=diagnostics,
            notes={
                "semantic_export_path": str(capture.semantic_export),
                "capture_meta_path": str(cm_path),
                "z_translation_applied": z_translation,
                "source": "replica",
                "adapter_config_params": dict(config.params),
                "import_notes": dict(capture_meta.get("import_notes", {})),
            },
        )

    def capabilities(self) -> ReconstructionCapabilities:
        return ReconstructionCapabilities(
            produces_mesh=True,
            produces_pointcloud=False,
            produces_gaussian_splat=False,
            produces_nerf_field=False,
            estimates_poses=False,
            requires_gpu=False,
            typical_runtime_minutes=0,
        )
