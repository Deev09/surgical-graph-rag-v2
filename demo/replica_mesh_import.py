"""Importer variant that proves the pipeline runs off a raw .PLY MESH, not the
precomputed JSON boxes.

`demo/replica_habitat_import.py` reads object boxes straight from
`habitat/info_semantic.json` (oriented_bbox already computed for you). The honest
question is: does the system work from actual geometry -- a mesh file -- the way a
NeRF / 3DGS / scanner backend would hand you?

Replica ships `habitat/mesh_semantic.ply`: a binary instance mesh where every
FACE carries a `uint16 object_id`. So we can recover each object's box directly
from mesh vertices -- no precomputed bbox used at all:

    object box = axis-aligned bound of every vertex touched by a face tagged
                 with that object_id

Only the CLASS LABEL (table / book / ...) is still read from info_semantic.json's
id->class_name map -- a mesh has geometry, not semantics; a real backend pairs the
mesh with a segmentation/label source exactly the same way.

A/B frame parity (Phase 8): this importer is variant B of the A/B/C ladder --
"boxes from mesh geometry; oracle labels, gravity, and surfaces held identical
to A". It therefore INHERITS the JSON importer's frame wholesale rather than
recomputing it: the rotation (gravity alignment + F1/F3 yaw de-rotation) comes
from the shared `_aligned_structural_surfaces` helper, and the structural
surfaces (including F2 floor calibration, whose voters are A's JSON boxes) are
taken verbatim from `import_habitat_room`. The ONLY thing this importer
computes is the box source -- so an A/B diff attributes every difference to
exactly that one variable.

    arts = import_mesh_room(Path("/.../datasets/replica/room_0"), "replica_room_0")
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from extractors.base import (
    EntityArtifact,
    EntityArtifacts,
    EntityIdentity,
    ExtractionDiagnostics,
    SceneFrame,
    StructuralSurface,
)

from demo.replica_habitat_import import (
    ROOM_0_Z_TRANSLATION,
    STRUCTURAL_CLASSES,
    _aligned_structural_surfaces,
    _gravity_align_matrix,
    import_habitat_room,
)

_VERTEX_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
    ("r", "u1"), ("g", "u1"), ("b", "u1"),
])  # 27 bytes/vertex, matches the mesh_semantic.ply header
# Faces are a `list uint8 uint32 vertex_indices` then `uint16 object_id`. Replica
# meshes are uniform polygons (room_0 is all quads), so we read the polygon size k
# from the first face and build a fixed-stride dtype: uint8 count + k*uint32 + uint16.


def _parse_semantic_ply(ply_path: Path):
    """Return (xyz[V,3] float64, face_vidx[F,k] int64, face_oid[F] int64) from a
    Replica binary instance mesh. Reads the polygon size k from the data and
    asserts every face has the same k, so a format drift fails loudly."""
    raw = ply_path.read_bytes()
    end = raw.find(b"end_header\n")
    if end < 0:
        raise ValueError(f"no end_header in {ply_path}")
    header = raw[:end].decode("ascii", "replace")
    body = raw[end + len(b"end_header\n"):]

    n_vert = n_face = None
    for line in header.splitlines():
        if line.startswith("element vertex"):
            n_vert = int(line.split()[-1])
        elif line.startswith("element face"):
            n_face = int(line.split()[-1])
    if n_vert is None or n_face is None:
        raise ValueError("missing vertex/face element counts")
    if "format binary_little_endian" not in header:
        raise ValueError("expected binary_little_endian PLY")

    verts = np.frombuffer(body, dtype=_VERTEX_DTYPE, count=n_vert)
    xyz = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float64)

    face_off = n_vert * _VERTEX_DTYPE.itemsize
    k = int(body[face_off])  # polygon size of the first face (3=tri, 4=quad, ...)
    fields = [("n", "u1")] + [(f"v{i}", "<u4") for i in range(k)] + [("oid", "<u2")]
    face_dtype = np.dtype(fields)  # packed: 1 + 4k + 2 bytes
    faces = np.frombuffer(body, dtype=face_dtype, count=n_face, offset=face_off)
    if not np.all(faces["n"] == k):
        raise ValueError(f"non-uniform face size (expected all={k}); reader needs "
                         "a variable-length path for this mesh")
    vidx = np.stack([faces[f"v{i}"] for i in range(k)], axis=1).astype(np.int64)
    return xyz, vidx, faces["oid"].astype(np.int64)


def _per_object_aabb(xyz, vidx, oid):
    """Axis-aligned bound of every vertex touched by each object_id.
    Returns {object_id: (lo(3,), hi(3,))}. Pure scatter-reduce, no Python loop
    over faces."""
    k = vidx.shape[1]                           # polygon size (3=tri, 4=quad)
    pts = xyz[vidx.reshape(-1)]                 # (kF, 3) every face-vertex
    poid = np.repeat(oid, k)                     # (kF,)   its object_id
    order = np.argsort(poid, kind="stable")
    spoid, spts = poid[order], pts[order]
    starts = np.searchsorted(spoid, np.unique(spoid), side="left")
    los = np.minimum.reduceat(spts, starts, axis=0)
    his = np.maximum.reduceat(spts, starts, axis=0)
    uids = spoid[starts]
    return {int(o): (los[i], his[i]) for i, o in enumerate(uids)}


def import_mesh_room(
    room_dir: Path,
    scene_id: str,
    *,
    z_translation: float = ROOM_0_Z_TRANSLATION,
    import_surfaces: bool = True,
    drop_classes: tuple[str, ...] = ("undefined", "non-plane", "plane"),
) -> EntityArtifacts:
    """Build EntityArtifacts whose object boxes are derived from the raw
    mesh_semantic.ply geometry. Class labels come from info_semantic.json's
    id->class_name map. Frame and structural surfaces are inherited verbatim
    from the JSON importer (variant A), so an A/B diff isolates box source."""
    info = json.loads((room_dir / "habitat" / "info_semantic.json").read_text())
    g = info["gravity_dir"]
    R_align = _gravity_align_matrix((float(g[0]), float(g[1]), float(g[2])))

    id_to_class = {int(o["id"]): str(o.get("class_name", "")).strip()
                   for o in info["objects"]}

    # A/B parity: rotation from the shared frame helper (bit-identical yaw),
    # surfaces (incl. F2 floor calibration) verbatim from variant A.
    surfaces: list[StructuralSurface] = []
    surf_diag = {"n_floor": 0, "n_wall": 0, "n_ceiling": 0,
                 "walls_without_floor": 0, "walls_dropped_non_vertical": 0}
    yaw_derotation_deg = 0.0
    floor_calibration: dict[str, float] = {}
    if import_surfaces:
        R_align, _, _, yaw_applied = _aligned_structural_surfaces(
            info, R_align, z_translation)
        if yaw_applied:
            yaw_derotation_deg = round(yaw_applied, 4)
        json_arts = import_habitat_room(
            room_dir, scene_id, z_translation=z_translation,
            import_surfaces=True, drop_classes=drop_classes)
        surfaces = json_arts.structural_surfaces
        surf_diag = json_arts.notes["structural_surfaces"]
        floor_calibration = json_arts.notes["floor_calibration"]
        if json_arts.notes["yaw_derotation_deg"] != yaw_derotation_deg:
            raise AssertionError(
                "frame drift between importers: "
                f"{json_arts.notes['yaw_derotation_deg']} != {yaw_derotation_deg}")
    R = np.array(R_align, dtype=np.float64)

    xyz, vidx, oid = _parse_semantic_ply(room_dir / "habitat" / "mesh_semantic.ply")
    # einsum, not `xyz @ R.T`: Accelerate-BLAS matmul intermittently raises
    # spurious divide-by-zero/overflow RuntimeWarnings on these frombuffer
    # views (result is finite either way; einsum is warning-free and equal).
    xyz = np.einsum("ij,nj->ni", R, xyz)  # gravity+yaw canonicalize every vertex
    if not np.isfinite(xyz).all():
        raise ValueError("non-finite vertex coordinates after frame transform")
    xyz[:, 2] += z_translation            # shared floor offset
    boxes = _per_object_aabb(xyz, vidx, oid)

    entities: list[EntityArtifact] = []
    n_unlabeled = 0
    for obj_id, (lo, hi) in sorted(boxes.items()):
        cls = id_to_class.get(obj_id)
        if cls is None:
            n_unlabeled += 1
            continue
        if not cls or cls in drop_classes:
            continue
        if import_surfaces and cls in STRUCTURAL_CLASSES:
            continue
        lo_t, hi_t = (float(lo[0]), float(lo[1]), float(lo[2])), \
                     (float(hi[0]), float(hi[1]), float(hi[2]))
        centroid = tuple((lo_t[i] + hi_t[i]) / 2.0 for i in range(3))
        entities.append(EntityArtifact(
            identity=EntityIdentity(
                object_uid=f"obj_{obj_id}",
                display_label=cls,
                aliases=[],
                source_instance_ref=f"mesh_ply:{obj_id}",
            ),
            bbox_aabb=(lo_t, hi_t),
            bbox_obb=None,
            centroid=centroid,
            geometry_handle=None,
            semantic_hypotheses=[],
            extraction_diagnostics={},
        ))

    raw = (room_dir / "habitat" / "mesh_semantic.ply").read_bytes()
    bundle_hash = "mesh_" + hashlib.sha256(raw).hexdigest()[:16]
    return EntityArtifacts(
        schema_version=2,
        bundle_hash=bundle_hash,
        scene_id=scene_id,
        # Same R_align (and same yaw de-rotation) as variant A by construction,
        # so the same frame kind. A/B parity would be broken by disagreeing here.
        frame=SceneFrame(gravity=(0.0, 0.0, -1.0), canonical_forward=None,
                         canonical_right=None, units="meters",
                         notes="object boxes from mesh_semantic.ply geometry",
                         kind="scene_canonical"),
        representation_hash=bundle_hash,
        extractor_name="replica_mesh_import",
        extractor_version="0.2",
        entities=entities,
        structural_surfaces=surfaces,
        geometry_store_path=None,
        diagnostics=ExtractionDiagnostics(
            n_entities=len(entities),
            n_structural_surfaces=len(surfaces),
            runtime_seconds=0.0,
            coverage_score=1.0,
            notes=(f"boxes from .ply mesh; unlabeled_object_ids={n_unlabeled}; "
                   f"surfaces floor={surf_diag['n_floor']} wall={surf_diag['n_wall']} "
                   f"ceiling={surf_diag['n_ceiling']}; frame inherited from JSON "
                   f"importer (yaw={yaw_derotation_deg:+.2f} deg, "
                   f"floors_calibrated={len(floor_calibration)})"),
        ),
        notes={"source": "habitat/mesh_semantic.ply (geometry) + "
                         "info_semantic.json (labels)",
               "z_translation": z_translation,
               "n_unlabeled_object_ids": n_unlabeled,
               "structural_surfaces": surf_diag,
               "frame_source": "inherited_from_replica_habitat_import",
               "yaw_derotation_deg": yaw_derotation_deg,
               "floor_calibration": floor_calibration},
    )
