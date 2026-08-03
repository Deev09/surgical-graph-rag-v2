"""Reusable importer: raw Replica `habitat/info_semantic.json` -> EntityArtifacts.

This is the real plug seam for a NEW Replica room. It reads the object-level
oriented bounding boxes Replica ships in `habitat/info_semantic.json` and produces
an `EntityArtifacts` bundle the Phase 6 pipeline consumes directly -- object boxes,
class labels, and gravity. No mesh or binary-segmentation decoding required.

Transform (reverse-engineered from room_0 and validated to 2 microns against its
committed enriched_v2 AABBs):

    world_corner = R(obj.orientation.rotation) @ (abb.center +- sizes/2) + obj.translation
    world_corner.z += z_translation          # floor offset; room_0 = 1.52665

The object's AABB is the axis-aligned bound of its 8 world OBB corners. The z
offset is cosmetic for furniture-top support (the relation is invariant to a global
vertical shift -- only relative gaps matter), so a new room imports correctly at any
consistent z_translation; room_0 uses 1.52665 only to byte-match enriched_v2.

Structural surfaces (floor/wall/ceiling) ARE imported here from the Habitat
`floor`/`wall`/`ceiling` semantic classes, so the full QA track -- "what is on
the floor?", "what is against the wall?" -- runs on a new room too, not just the
furniture-top support questions. The OBB->plane+polygon geometry is reused verbatim
from the canonical `importers/replica.py` so the surface shape matches the
committed enriched_v2 records; we only re-express it in this importer's
gravity-aligned frame (the same R_align + z_translation applied to the objects)
so surfaces and objects share one coordinate frame.

Multi-room note: the canonical importer orients every wall by the FIRST floor's
face centroid (it targets single rooms). A real apartment has many rooms, so here
each wall is oriented by its NEAREST floor's face centroid. Walls far from every
floor can still flip; that is a recall-only error (a missed against-the-wall
object), never a false positive, and it is reported, not hidden.

Yaw de-rotation (Phase 8 findings F1/F3): some scenes are captured rotated about
the vertical axis (room_1 ~27 deg, room_2 ~7 deg), which inflates every
axis-aligned bbox downstream (a rotated rug's AABB can cover the whole room).
The dominant room yaw is estimated from the wall normals (length-weighted,
90-deg-symmetric orientation statistics) and composed into the alignment
rotation ONLY when it exceeds a 5 deg guard, so near-axis scenes (room_0,
office_0, frl_apartment_0, apartment_0) import bit-identically. This corrects
the systemic room-level rotation; per-object orientation is still discarded
(bbox_obb=None), so an individual object angled relative to its room keeps an
inflated AABB -- that residual is future work, recorded in the Phase 8 review
guide. Applied yaw is recorded in notes["yaw_derotation_deg"].

Floor calibration (Phase 8 finding F2): some scenes ship a floor instance whose
plane sits well above the objects standing on it (room_1 ~0.11 m, frl_apartment_0
~0.28 m), so nearly everything "penetrates" the floor and ON_SURFACE rejects it.
After import, each floor plane is cross-checked against the lowest objects over
it and snapped down ONLY when the median penetration exceeds a 0.10 m guard --
mild disagreement (room_0 -0.02, apartment_0 -0.07) is deliberately left alone so
previously-imported scenes stay bit-identical. Applied offsets are recorded in
notes["floor_calibration"].

Usage:
    arts = import_habitat_room(Path("/.../datasets/replica/room_1"), "replica_room_1")
"""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

from extractors.base import (
    EntityArtifact,
    EntityArtifacts,
    EntityIdentity,
    ExtractionDiagnostics,
    StructuralSurface,
    SceneFrame,
)

# Reuse the canonical Habitat-OBB surface geometry so floor/wall/ceiling records
# match the committed enriched_v2 shape (single source of truth for the math).
from importers.replica import (
    _dot,
    gravity_align_matrix,
    _obb_face_corners,
    _orient_polygon,
    _orient_tag,
    _quat_axis_in_world,
    _quat_rotate,
    _scale,
    _sub,
    _thin_axis_index,
    _unit,
)
from common.types import Plane

ROOM_0_Z_TRANSLATION = 1.52665  # exact floor offset that reproduces room_0 enriched_v2
STRUCTURAL_CLASSES = ("floor", "wall", "ceiling")  # routed to surfaces, not entities
# A habitat instance labeled "wall" whose interior normal is not roughly
# horizontal is geometrically a slab/ledge, not a vertical wall. Emitting it as
# a surface lets on_surface treat it as a pseudo-shelf (an object "resting on a
# wall"). Drop such degenerate walls and count them.
WALL_MAX_VERTICAL_NORMAL = 0.3
# Yaw de-rotation (F1/F3) guard: only a dominant room yaw beyond this is
# corrected, so near-axis scenes (including the frozen gate scenes) import
# bit-identically. 5 deg keeps office_0 (+3.1) and apartment_0 (+1.7) fixed
# while catching room_1 (+26.6) and room_2 (-7.2).
YAW_DEROTATE_GUARD_DEG = 5.0
# Floor-calibration (F2) knobs. The window keeps other stories of a multi-floor
# scene from voting on a floor they don't stand on; the guard keeps healthy
# scenes (and the frozen gate scenes room_0/apartment_0) bit-identical.
FLOOR_CAL_WINDOW_M = 0.5     # only object bottoms this close to the plane vote
FLOOR_CAL_GUARD_M = 0.10     # correct only median penetration beyond this
FLOOR_CAL_CLUSTER_M = 0.06   # low-cluster width above the lowest voter
FLOOR_CAL_MIN_VOTERS = 3


# Alignment rotation lives in importers/replica.py so BOTH import paths use
# the same one — the JSON importer there now levels tilted scenes instead of
# refusing them, and two importers computing "the same" rotation two ways is
# how the room_2 divergence happened in the first place. Re-exported under the
# old private name; tools/c2_run.py and tools/mvp_viewer.py import it from here.
_gravity_align_matrix = gravity_align_matrix


def _matvec(R, p):
    return (
        R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2],
        R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2],
        R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2],
    )


def _quat_to_matrix(q):
    """Rotation matrix from an [x, y, z, w] quaternion."""
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def _world_aabb(obj, z_translation, R_align):
    """Axis-aligned bound of an object's oriented bbox, in the gravity-aligned
    world frame (R_align maps physical up onto +z)."""
    ob = obj["oriented_bbox"]
    cx, cy, cz = ob["abb"]["center"]
    hx, hy, hz = (s / 2.0 for s in ob["abb"]["sizes"])
    R = _quat_to_matrix(ob["orientation"]["rotation"])
    tx, ty, tz = ob["orientation"]["translation"]
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    for sx, sy, sz in itertools.product((-1, 1), repeat=3):
        lx, ly, lz = cx + sx * hx, cy + sy * hy, cz + sz * hz
        w = (R[0][0] * lx + R[0][1] * ly + R[0][2] * lz + tx,
             R[1][0] * lx + R[1][1] * ly + R[1][2] * lz + ty,
             R[2][0] * lx + R[2][1] * ly + R[2][2] * lz + tz)
        w = _matvec(R_align, w)              # gravity-canonicalize
        w = (w[0], w[1], w[2] + z_translation)
        for i, v in enumerate(w):
            lo[i] = min(lo[i], v)
            hi[i] = max(hi[i], v)
    return (tuple(lo), tuple(hi))


def _surface_geometry_world(inst, interior_hint, interior_ref_world):
    """Interior-facing normal, 4 face corners, and face centroid for one
    structural instance, all in the raw Habitat world frame (pre gravity-align).

    Mirrors importers.replica._build_structural_surface_record's geometry: the
    surface is the OBB face perpendicular to the thin axis; the interior normal's
    sign is picked by an explicit hint (floor/ceiling) or by pointing toward an
    interior reference point (walls -> nearest floor-face centroid)."""
    ob = inst["oriented_bbox"]
    center_local = ob["abb"]["center"]
    quat = [float(x) for x in ob["orientation"]["rotation"]]
    extents = [float(s) / 2.0 for s in ob["abb"]["sizes"]]
    center_world = _quat_rotate(quat, center_local)
    thin_idx = _thin_axis_index(extents)
    thin_world_unit = _unit(_quat_axis_in_world(quat, thin_idx))
    if interior_hint is not None:
        ref_dir = interior_hint
    else:
        ref_dir = _sub(interior_ref_world, center_world)
    sign = 1.0 if _dot(thin_world_unit, ref_dir) >= 0 else -1.0
    interior_normal = _scale(thin_world_unit, sign)
    corners = _obb_face_corners(center_world, extents, quat, thin_idx, sign)
    corners = _orient_polygon(corners, interior_normal)
    face_centroid = [sum(c[k] for c in corners) / 4.0 for k in range(3)]
    return interior_normal, corners, face_centroid


def _to_final_frame_point(p, R_align, z_translation):
    w = _matvec(R_align, p)
    return (w[0], w[1], w[2] + z_translation)


def _structural_surfaces(info, R_align, z_translation):
    """Build StructuralSurface records (floor/wall/ceiling) in this importer's
    gravity-aligned + z-translated frame, the SAME frame the object AABBs use.

    Returns (surfaces, diagnostics_dict). Walls are oriented by the nearest
    floor-face centroid (multi-room robust); the count of walls with no floor
    in the scene is reported so a flip risk is never silent."""
    grouped = {k: [] for k in STRUCTURAL_CLASSES}
    for o in info["objects"]:
        name = str(o.get("class_name", "")).strip()
        if name in grouped:
            grouped[name].append(o)
    for k in grouped:
        grouped[k].sort(key=lambda o: o["id"])

    floor_centroids_world: list[list[float]] = []
    pending: list[tuple[dict, str, list[float] | None, list[float] | None]] = []

    for inst in grouped["floor"]:
        normal, corners, fc = _surface_geometry_world(inst, [0.0, 0.0, 1.0], None)
        floor_centroids_world.append(fc)
        pending.append((inst, "floor", normal, corners))
    for inst in grouped["ceiling"]:
        normal, corners, _ = _surface_geometry_world(inst, [0.0, 0.0, -1.0], None)
        pending.append((inst, "ceiling", normal, corners))

    walls_without_floor = 0
    for inst in grouped["wall"]:
        ob = inst["oriented_bbox"]
        wc = _quat_rotate(
            [float(x) for x in ob["orientation"]["rotation"]], ob["abb"]["center"]
        )
        if floor_centroids_world:
            ref = min(
                floor_centroids_world,
                key=lambda fc: sum((fc[i] - wc[i]) ** 2 for i in range(3)),
            )
        else:
            walls_without_floor += 1
            ref = [wc[0], wc[1], wc[2] + 1.0]  # last-resort: assume up is interior
        normal, corners, _ = _surface_geometry_world(inst, None, ref)
        pending.append((inst, "wall", normal, corners))

    surfaces: list[StructuralSurface] = []
    walls_dropped_non_vertical = 0
    for inst, stype, normal_w, corners_w in pending:
        n = _matvec(R_align, normal_w)  # direction: rotate only, no translation
        nmag = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) or 1.0
        n = (n[0] / nmag, n[1] / nmag, n[2] / nmag)
        if stype == "wall" and abs(n[2]) > WALL_MAX_VERTICAL_NORMAL:
            walls_dropped_non_vertical += 1  # degenerate "wall" (horizontal slab)
            continue
        poly = [_to_final_frame_point(c, R_align, z_translation) for c in corners_w]
        fc = tuple(sum(p[k] for p in poly) / 4.0 for k in range(3))
        d = -(n[0] * fc[0] + n[1] * fc[1] + n[2] * fc[2])
        iid = inst["id"]
        if stype == "floor":
            uid = f"floor_{iid}"
        elif stype == "ceiling":
            uid = f"ceiling_{iid}"
        else:
            uid = f"wall_{iid}_{_orient_tag(list(n))}"
        surfaces.append(StructuralSurface(
            surface_uid=uid,
            surface_type=stype,
            plane=Plane(a=n[0], b=n[1], c=n[2], d=d),
            polygon=[tuple(p) for p in poly],
            confidence=1.0,
            source="habitat_label",
        ))
    diag = {
        "n_floor": len(grouped["floor"]),
        "n_wall": len(grouped["wall"]),
        "n_ceiling": len(grouped["ceiling"]),
        "walls_without_floor": walls_without_floor,
        "walls_dropped_non_vertical": walls_dropped_non_vertical,
    }
    return surfaces, diag


def _rotz_matrix(deg: float):
    """Rotation about +z by deg (right-handed)."""
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def _matmul3(A, B):
    return tuple(
        tuple(sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def _dominant_yaw_deg(walls) -> float:
    """Dominant room yaw from wall normals, in degrees folded to [-45, 45).

    Walls of a rectangular room point along two perpendicular axes, so plain
    angle averaging cancels; instead average unit vectors at 4*theta (the
    standard 90-deg-symmetric orientation statistic), weighted by each wall's
    horizontal length so a long facade outvotes a short angled nook.
    """
    sx = sy = 0.0
    for w in walls:
        theta = math.atan2(w.plane.b, w.plane.a)
        if w.polygon:
            xs = [p[0] for p in w.polygon]
            ys = [p[1] for p in w.polygon]
            weight = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        else:
            weight = 1.0
        sx += weight * math.cos(4 * theta)
        sy += weight * math.sin(4 * theta)
    if sx == 0.0 and sy == 0.0:
        return 0.0
    yaw = math.degrees(math.atan2(sy, sx) / 4.0)
    while yaw >= 45.0:
        yaw -= 90.0
    while yaw < -45.0:
        yaw += 90.0
    return yaw


def _point_in_convex_poly_xy(px: float, py: float, polygon) -> bool:
    """XY containment in a convex polygon (floor faces are OBB rectangles)."""
    sign = 0
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i][0], polygon[i][1]
        x2, y2 = polygon[(i + 1) % n][0], polygon[(i + 1) % n][1]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) < 1e-12:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def _calibrate_floor_planes(surfaces, entities):
    """Snap a floor plane to the objects standing on it when the dataset's
    floor instance is grossly wrong (finding F2; see module docstring).

    Voters are entities whose centroid lies over the floor polygon in XY and
    whose bottom is within FLOOR_CAL_WINDOW_M of the plane (excludes other
    stories). The median of the lowest voter cluster estimates the true
    offset; the plane and polygon move only when that median penetrates
    beyond FLOOR_CAL_GUARD_M. Returns (surfaces, {floor_uid: offset_applied}).
    """
    out = []
    applied: dict[str, float] = {}
    for s in surfaces:
        if s.surface_type != "floor" or not s.polygon:
            out.append(s)
            continue
        p = s.plane
        ds = []
        for e in entities:
            if not _point_in_convex_poly_xy(e.centroid[0], e.centroid[1], s.polygon):
                continue
            lo, hi = e.bbox_aabb
            d = min(
                p.a * x + p.b * y + p.c * z + p.d
                for x in (lo[0], hi[0])
                for y in (lo[1], hi[1])
                for z in (lo[2], hi[2])
            )
            if abs(d) <= FLOOR_CAL_WINDOW_M:
                ds.append(d)
        ds.sort()
        cluster = [d for d in ds if d <= ds[0] + FLOOR_CAL_CLUSTER_M] if ds else []
        if len(cluster) < FLOOR_CAL_MIN_VOTERS:
            out.append(s)
            continue
        offset = statistics.median(cluster)
        if offset >= -FLOOR_CAL_GUARD_M:
            out.append(s)
            continue
        out.append(dataclasses.replace(
            s,
            plane=Plane(a=p.a, b=p.b, c=p.c, d=p.d - offset),
            polygon=[
                (q[0] + offset * p.a, q[1] + offset * p.b, q[2] + offset * p.c)
                for q in s.polygon
            ],
        ))
        applied[s.surface_uid] = round(offset, 4)
    return out, applied


def _aligned_structural_surfaces(info, R_align, z_translation):
    """Shared frame path (F1/F3): build structural surfaces, estimate the
    dominant room yaw from the walls, and — beyond the guard — fold it into
    the alignment rotation and rebuild, so walls AND object boxes share an
    axis-aligned frame. Returns (R_align, surfaces, surf_diag, yaw_applied)
    with yaw_applied at full precision (0.0 when under the guard).

    BOTH importers must obtain their rotation here (the JSON importer and
    demo/replica_mesh_import.py) so the A/B mesh-vs-JSON comparison shares a
    bit-identical frame."""
    surfaces, surf_diag = _structural_surfaces(info, R_align, z_translation)
    yaw = _dominant_yaw_deg([s for s in surfaces if s.surface_type == "wall"])
    yaw_applied = 0.0
    if abs(yaw) >= YAW_DEROTATE_GUARD_DEG:
        yaw_applied = yaw
        R_align = _matmul3(_rotz_matrix(-yaw), R_align)
        surfaces, surf_diag = _structural_surfaces(info, R_align, z_translation)
    return R_align, surfaces, surf_diag, yaw_applied


def import_habitat_room(
    room_dir: Path,
    scene_id: str,
    *,
    z_translation: float = ROOM_0_Z_TRANSLATION,
    import_surfaces: bool = True,
    drop_classes: tuple[str, ...] = ("undefined", "non-plane", "plane"),
) -> EntityArtifacts:
    """Import a Replica room's labeled objects into an EntityArtifacts bundle.

    Reads <room_dir>/habitat/info_semantic.json. Each object becomes an
    EntityArtifact with object_uid='obj_<id>', display_label=class_name, and a
    world-frame AABB. Structural surfaces are left empty (support derives its own).
    """
    info_path = room_dir / "habitat" / "info_semantic.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    g = info["gravity_dir"]
    R_align = _gravity_align_matrix((float(g[0]), float(g[1]), float(g[2])))
    # After alignment the scene is gravity-canonical: up = +z exactly.
    gravity = (0.0, 0.0, -1.0)

    surfaces: list[StructuralSurface] = []
    surf_diag: dict = {"n_floor": 0, "n_wall": 0, "n_ceiling": 0, "walls_without_floor": 0}
    yaw_derotation_deg = 0.0
    if import_surfaces:
        R_align, surfaces, surf_diag, yaw_applied = _aligned_structural_surfaces(
            info, R_align, z_translation)
        if yaw_applied:
            yaw_derotation_deg = round(yaw_applied, 4)

    entities: list[EntityArtifact] = []
    for obj in info["objects"]:
        cls = str(obj.get("class_name", "")).strip()
        if not cls or cls in drop_classes:
            continue
        # floor/wall/ceiling become StructuralSurfaces, never entities
        if import_surfaces and cls in STRUCTURAL_CLASSES:
            continue
        lo, hi = _world_aabb(obj, z_translation, R_align)
        centroid = tuple((lo[i] + hi[i]) / 2.0 for i in range(3))
        uid = f"obj_{obj['id']}"
        entities.append(EntityArtifact(
            identity=EntityIdentity(
                object_uid=uid,
                display_label=cls,
                aliases=[],
                source_instance_ref=f"habitat:{obj.get('node_id', obj['id'])}",
            ),
            bbox_aabb=(lo, hi),
            bbox_obb=None,
            centroid=centroid,
            geometry_handle=None,
            semantic_hypotheses=[],
            extraction_diagnostics={},
        ))

    floor_calibration: dict[str, float] = {}
    if import_surfaces and surfaces:
        surfaces, floor_calibration = _calibrate_floor_planes(surfaces, entities)

    raw = info_path.read_bytes()
    bundle_hash = "habitat_" + hashlib.sha256(raw).hexdigest()[:16]
    return EntityArtifacts(
        schema_version=2,
        bundle_hash=bundle_hash,
        scene_id=scene_id,
        # kind="scene_canonical", NOT "world": R_align has rotated physical up
        # onto +z (every Replica scene is tilted at least 0.11 deg, so this is
        # never the identity), and beyond YAW_DEROTATE_GUARD_DEG a yaw
        # de-rotation is composed on top. Edges extracted from this bundle are
        # computed in THAT frame. See docs/frame_decision.md.
        frame=SceneFrame(gravity=gravity, canonical_forward=None,
                         canonical_right=None, units="meters",
                         notes=(
                             "imported from Replica habitat/info_semantic.json; "
                             "gravity-aligned (up=+z exactly), "
                             f"yaw_derotation_deg={yaw_derotation_deg}"
                         ),
                         kind="scene_canonical"),
        representation_hash=bundle_hash,
        extractor_name="replica_habitat_import",
        extractor_version="0.1",
        entities=entities,
        structural_surfaces=surfaces,
        geometry_store_path=None,
        diagnostics=ExtractionDiagnostics(
            n_entities=len(entities),
            n_structural_surfaces=len(surfaces),
            runtime_seconds=0.0,
            coverage_score=1.0,
            notes=(
                f"objects + structural surfaces; floor={surf_diag['n_floor']} "
                f"wall={surf_diag['n_wall']} ceiling={surf_diag['n_ceiling']} "
                f"walls_without_floor={surf_diag['walls_without_floor']} "
                f"walls_dropped_non_vertical={surf_diag['walls_dropped_non_vertical']} "
                f"floors_calibrated={len(floor_calibration)}"
                if import_surfaces else "objects-only import; surfaces skipped"
            ),
        ),
        notes={
            "source": "habitat/info_semantic.json",
            "z_translation": z_translation,
            "structural_surfaces": surf_diag,
            "floor_calibration": floor_calibration,
            "yaw_derotation_deg": yaw_derotation_deg,
        },
    )
