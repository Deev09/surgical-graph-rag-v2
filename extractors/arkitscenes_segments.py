"""Oracle-free ARKitScenes segmentation -> EntityArtifacts bridge.

This is the missing vertical seam between the reconstruction adapter and the
graph pipeline.  It consumes only:

* an ARKitScenes ``SceneRepresentationBundle`` whose mesh is already in the
  declared ``scene_canonical`` frame; and
* an immutable ``SegmentationOutput`` directory produced from that exact
  canonical mesh.

No ARKitScenes annotation, label vocabulary, floor, or wall artifact is read.
Entities therefore remain anonymous and ``structural_surfaces`` remains empty.
The bridge delegates vertex membership, AABB/OBB fitting, and small-segment
filtering to :func:`segmenter.candidate.build_candidate_artifacts`.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from extractors.base import EntityArtifacts
from representations.base import SceneRepresentationBundle
from segmenter.base import load_segmentation_output
from segmenter.candidate import build_candidate_artifacts


EXTRACTOR_NAME = "arkitscenes_segmentation"
EXTRACTOR_VERSION = "0.1"

_IDENTITY_ROTATION = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


def _bundle_hash(representation_hash: str, segmentation_hash: str,
                 min_vertices: int) -> str:
    """Content identity for the bridge result, independent of file paths."""
    payload = (
        f"{EXTRACTOR_NAME}\0{EXTRACTOR_VERSION}\0"
        f"{representation_hash}\0{segmentation_hash}\0"
        f"min_vertices={min_vertices}"
    ).encode("utf-8")
    return f"ent_{EXTRACTOR_NAME}_{hashlib.sha256(payload).hexdigest()[:16]}"


def build_arkitscenes_segment_artifacts(
    representation: SceneRepresentationBundle,
    segmentation_dir: Path,
    *,
    min_vertices: int = 20,
) -> EntityArtifacts:
    """Build anonymous graph-ready entities from a saved dense assignment.

    ``representation.geometry_handle.uri`` must name the canonical mesh that
    the segmentation output was produced from.  The segmentation loader and
    generic candidate builder independently verify the immutable sidecar hash,
    mesh hash, vertex count, and instance ids before any artifact is returned.
    """
    if representation.frame.kind != "scene_canonical":
        raise ValueError(
            "ARKitScenes entity bridge requires frame.kind='scene_canonical'; "
            f"got {representation.frame.kind!r}")
    if representation.geometry_handle.kind != "mesh_file":
        raise ValueError(
            "ARKitScenes entity bridge requires a mesh_file geometry handle; "
            f"got {representation.geometry_handle.kind!r}")
    if representation.notes.get("adapter") != "arkitscenes":
        raise ValueError(
            "ARKitScenes entity bridge requires an ARKitScenes representation")
    if representation.notes.get("reads_annotations") is not False:
        raise ValueError(
            "ARKitScenes representation must explicitly record "
            "reads_annotations=False")
    if min_vertices < 1:
        raise ValueError(f"min_vertices must be >= 1, got {min_vertices}")

    segmentation_dir = Path(segmentation_dir)
    seg = load_segmentation_output(segmentation_dir)
    mesh_path = Path(representation.geometry_handle.uri)

    # Coordinates in the geometry handle are already in representation.frame;
    # applying the adapter's rotation again would corrupt the contract.
    candidate = build_candidate_artifacts(
        mesh_path,
        seg,
        representation.scene_id,
        rotation=_IDENTITY_ROTATION,
        z_translation=0.0,
        bundle_dir=segmentation_dir,
        min_vertices=min_vertices,
    )

    entities = []
    for entity in candidate.entities:
        inst = int(entity.identity.source_instance_ref.split(":", 1)[1])
        diagnostics = dict(entity.extraction_diagnostics)
        confidence = seg.instance_confidence.get(inst)
        if confidence is not None:
            diagnostics["instance_confidence"] = float(confidence)
        entities.append(replace(entity, extraction_diagnostics=diagnostics))

    notes = dict(candidate.notes)
    notes.update({
        "source": (
            "ARKitScenes canonical mesh + immutable segmenter dense assignment"
        ),
        "dataset": "arkitscenes",
        "oracle_free": True,
        "semantic_source": "none",
        "surface_source": "none",
        "frame_source": "SceneRepresentationBundle",
        "frame_transform_at_bridge": "identity",
        "representation_hash": representation.representation_hash,
        "input_mesh_sha256": seg.input_mesh_sha256,
        "segmentation_output_sha256": seg.output_sha256,
        "segmentation_bundle_path": str(segmentation_dir),
    })

    return replace(
        candidate,
        bundle_hash=_bundle_hash(
            representation.representation_hash, seg.output_sha256, min_vertices),
        frame=representation.frame,
        representation_hash=representation.representation_hash,
        extractor_name=EXTRACTOR_NAME,
        extractor_version=EXTRACTOR_VERSION,
        entities=entities,
        structural_surfaces=[],
        geometry_store_path=segmentation_dir,
        notes=notes,
    )
