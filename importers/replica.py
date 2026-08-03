"""Convert a Replica scene's Habitat semantic export into our scene_graph.json.

Reads:
    <replica_scene>/habitat/info_semantic.json

Writes:
    scenes/<scene_id>/scene_graph.json
    scenes/<scene_id>/capture_meta.json
    scenes/<scene_id>/enriched/v2/{scene_graph.json,capture_meta.json}
        when --enriched-v2 is passed

What this does:
    - Filters out class_name == "undefined" instances.
    - Optional structural filter drops {"wall", "floor", "ceiling"} unless --keep-structural.
    - Suffixes duplicate labels (window -> window_1, window_2, ...).
    - Translates xyz so the floor sits at z = 0 (min z of kept objects).
    - Emits zone = null. Replica has no zones.
    - Records gravity_dir, original/translated coord origin, and source provenance
      in capture_meta.json so downstream code can verify axis convention.
    - Optional --enriched-v2 output preserves world-frame OBBs and derives tight
      world-frame AABBs without modifying the Phase 1 replay fixture.

Frame (see docs/frame_decision.md):
    This importer used to REFUSE any scene whose gravity_dir was not within
    ~3 degrees of -Z (`_gravity_is_neg_z`), while demo/replica_habitat_import.py
    imported the same scene happily by rotating it. The two disagreed about
    whether Replica's room_2 (8.72 deg off) is importable at all — on the very
    scene C1-P1, C1-P2 and semantics-v2 were tuned. That divergence is gone:
    tilted scenes are now ACCEPTED and gravity-aligned, matching the demo path.

    The alignment is applied only when the tilt reaches
    GRAVITY_ALIGN_GUARD_DEG. That guard is NOT a claim that a 1 degree tilt is
    negligible — Finding 4 of docs/frame_and_scale_audit.md measures exactly
    how non-negligible a frame change can be. It exists so every scene this
    importer previously ACCEPTED (all five non-room_2 Replica scenes, max tilt
    1.31 deg) still imports byte-identically, which is what keeps the frozen
    Phase 1 v1 fixture under scenes/replica_room_0/ reproducible. The frame
    actually produced is recorded in capture_meta["axis_convention"]["frame_kind"],
    so the difference is declared rather than silent.

What this does NOT do:
    - No relations are emitted. relations/compute.py is the only relation source.
    - No semantic merging across instances.
    - No yaw de-rotation. demo/replica_habitat_import.py does that too; this
      importer only levels gravity. Both produce a "scene_canonical" frame;
      neither claims they produce the SAME scene_canonical frame.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

STRUCTURAL_DROP = {"wall", "floor", "ceiling"}
STRUCTURAL_REQUIRED = ("floor", "wall", "ceiling")
UNDEFINED = "undefined"
ENRICHED_SCHEMA_VERSION = 2

# Gravity tilt at or beyond which this importer levels the scene instead of
# leaving it in the capture's raw axes. Measured Replica tilts:
#   frl_apartment_0 0.11, office_0 0.21, room_1 0.23, room_0 0.27,
#   apartment_0 1.31  |  room_2 8.72
# 5.0 sits in the 1.31 -> 8.72 gap with >3.5x clearance either side, and is the
# same value/idiom as demo/replica_habitat_import.YAW_DEROTATE_GUARD_DEG.
# Below the guard the output is bit-identical to the pre-guard importer.
GRAVITY_ALIGN_GUARD_DEG = 5.0


def _abs(v: float) -> float:
    return v if v >= 0 else -v


def _gravity_is_neg_z(gravity_dir: list[float]) -> bool:
    """Legacy predicate: is gravity already close enough to -Z that the raw
    capture axes ARE the gravity-canonical axes?

    Retained (and still tested) because it names the assumption the whole v1
    path was built on. It no longer gates import — see gravity_tilt_deg and
    GRAVITY_ALIGN_GUARD_DEG, which decide whether to level the scene rather
    than whether to refuse it."""
    return _abs(gravity_dir[0]) < 0.05 and _abs(gravity_dir[1]) < 0.05 and gravity_dir[2] < -0.95


def gravity_tilt_deg(gravity_dir) -> float:
    """Angle in degrees between physical up (-gravity_dir) and world +Z."""
    gx, gy, gz = (float(v) for v in gravity_dir)
    norm = math.sqrt(gx * gx + gy * gy + gz * gz)
    if norm == 0.0:
        raise ValueError("gravity_dir is the zero vector")
    cos = max(-1.0, min(1.0, -gz / norm))
    return math.degrees(math.acos(cos))


def gravity_align_matrix(gravity):
    """Rotation R mapping physical up (=-gravity) onto +z, so the imported
    scene is gravity-canonical (up = +z exactly). Different captures have
    slightly different gravity tilt; the Phase 6 AABB-top derivation requires
    axis-aligned up, so we align here rather than loosen the predicate.
    Rodrigues' formula; identity when already aligned, 180-deg flip handled.

    Single source of the alignment rotation for BOTH import paths: this module
    and demo/replica_habitat_import.py (which re-exports it as
    _gravity_align_matrix). Moved here verbatim from that module so the two
    importers cannot drift to numerically different rotations."""
    gx, gy, gz = gravity
    m = math.sqrt(gx * gx + gy * gy + gz * gz) or 1.0
    ux, uy, uz = -gx / m, -gy / m, -gz / m          # up
    vx, vy, vz = uy * 1.0 - uz * 0.0, uz * 0.0 - ux * 1.0, ux * 0.0 - uy * 0.0  # up x z
    # up x zhat = (uy, -ux, 0); cos = up . zhat = uz
    vx, vy, vz = uy, -ux, 0.0
    c = uz
    if c > 1 - 1e-12:
        return ((1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0))
    if c < -1 + 1e-12:                                # up points at -z: flip about x
        return ((1.0, 0, 0), (0, -1.0, 0), (0, 0, -1.0))
    k = 1.0 / (1.0 + c)
    # R = I + [v]x + [v]x^2 * k
    return (
        (1 + (-(vz * vz) - vy * vy) * k, (-vz) + (vx * vy) * k, (vy) + (vx * vz) * k),
        ((vz) + (vx * vy) * k, 1 + (-(vz * vz) - vx * vx) * k, (-vx) + (vy * vz) * k),
        ((-vy) + (vx * vz) * k, (vx) + (vy * vz) * k, 1 + (-(vy * vy) - vx * vx) * k),
    )


def _matvec(R, p: list[float]) -> list[float]:
    return [
        R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2],
        R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2],
        R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2],
    ]


def _matrix_to_quat(R) -> list[float]:
    """Unit quaternion [x, y, z, w] for a proper rotation matrix R.

    Needed because the enriched-v2 output stores each box's orientation as a
    quaternion: levelling the scene by R means every stored orientation becomes
    R composed with it. Shepperd's method — branch on the largest diagonal term
    so the divisor is never near zero.

    Invariant (asserted in tests/importers/test_replica_gravity_align.py):
    _quat_rotate(_matrix_to_quat(R), v) == _matvec(R, v) for all v."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return [(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s]
    if m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        return [0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s]
    if m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        return [(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s]
    s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
    return [(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s]


def _quat_mul(a: list[float], b: list[float]) -> list[float]:
    """Hamilton product of [x, y, z, w] quaternions: rotate by b, then by a."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _quat_rotate(q: list[float], v: list[float]) -> list[float]:
    """Rotate vector v by unit quaternion q=[x,y,z,w]. Replica's abb.center is
    in the object's local pre-rotation frame; multiplying by the orientation
    quaternion places it in the world (mesh) frame."""
    qxyz = [float(x) for x in q[:3]]
    qw = float(q[3])
    v_xyz = [float(x) for x in v]
    t = [2.0 * x for x in _cross(qxyz, v_xyz)]
    q_cross_t = _cross(qxyz, t)
    return [v_xyz[i] + qw * t[i] + q_cross_t[i] for i in range(3)]


def _aabb_from_obb(
    center: list[float],
    extents: list[float],
    rotation_quat: list[float],
) -> tuple[list[float], list[float]]:
    """Return the tight AABB enclosing an oriented bbox, in whatever frame
    `center` and `rotation_quat` are already expressed in. Levelling is done
    by the caller (rotate the centre, compose the quaternion), so this stays
    frame-agnostic and unchanged."""
    corners: list[list[float]] = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                local_offset = [
                    sx * extents[0],
                    sy * extents[1],
                    sz * extents[2],
                ]
                rotated = _quat_rotate(rotation_quat, local_offset)
                corners.append([center[i] + rotated[i] for i in range(3)])
    return (
        [min(c[i] for c in corners) for i in range(3)],
        [max(c[i] for c in corners) for i in range(3)],
    )


def _translate_z(v: list[float], dz: float) -> list[float]:
    return [float(v[0]), float(v[1]), float(v[2]) + dz]


def _round_vec(v: list[float], digits: int = 6) -> list[float]:
    return [round(float(x), digits) for x in v]


def _dot(a: list[float], b: list[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _scale(v: list[float], s: float) -> list[float]:
    return [s * v[0], s * v[1], s * v[2]]


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(_dot(v, v))
    if n == 0.0:
        raise ValueError("cannot unit-normalize zero vector")
    return [v[0] / n, v[1] / n, v[2] / n]


def _quat_axis_in_world(rotation_quat: list[float], local_axis_idx: int) -> list[float]:
    """World-frame direction of the OBB's local-frame unit axis k ∈ {0,1,2}."""
    basis = [0.0, 0.0, 0.0]
    basis[local_axis_idx] = 1.0
    return _quat_rotate(rotation_quat, basis)


def _thin_axis_index(extents: list[float]) -> int:
    return min(range(3), key=lambda i: extents[i])


def _obb_face_corners(
    center_world: list[float],
    extents: list[float],
    rotation_quat: list[float],
    thin_axis_idx: int,
    sign_along_thin: float,
) -> list[list[float]]:
    """4 corners of the OBB face perpendicular to the thin axis, on the side
    selected by sign_along_thin in {+1,-1}. Initial order is
    fixed (-,-), (+,-), (+,+), (-,+) over the two non-thin local axes;
    callers must run _orient_polygon to enforce the winding convention.

    As in _aabb_from_obb, the frame is whatever `center_world` and
    `rotation_quat` are already in."""
    others = [i for i in range(3) if i != thin_axis_idx]
    i_idx, j_idx = others[0], others[1]
    e_thin = extents[thin_axis_idx]
    e_i = extents[i_idx]
    e_j = extents[j_idx]
    corners: list[list[float]] = []
    for si, sj in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)):
        local = [0.0, 0.0, 0.0]
        local[thin_axis_idx] = sign_along_thin * e_thin
        local[i_idx] = si * e_i
        local[j_idx] = sj * e_j
        rotated = _quat_rotate(rotation_quat, local)
        corners.append([center_world[k] + rotated[k] for k in range(3)])
    return corners


def _winding_dot(corners: list[list[float]], normal: list[float]) -> float:
    """dot(cross(p1-p0, p2-p0), normal). Positive means polygon winding
    matches the interior-facing normal (the agreed-on Phase 2 convention)."""
    a = _sub(corners[1], corners[0])
    b = _sub(corners[2], corners[0])
    return _dot(_cross(a, b), normal)


def _orient_polygon(
    corners: list[list[float]], normal: list[float],
) -> list[list[float]]:
    """Return corners reordered so dot(cross(p1-p0, p2-p0), normal) > 0.
    Keeps p0 fixed and reverses the traversal direction if the test fails."""
    if _winding_dot(corners, normal) > 0:
        return corners
    return [corners[0], corners[3], corners[2], corners[1]]


def _orient_tag(normal_world: list[float]) -> str:
    """Dominant-axis tag for an interior-facing wall normal: e.g. xplus,
    yminus. Used only to make wall surface_uids legible."""
    abs_components = [abs(normal_world[0]), abs(normal_world[1]), abs(normal_world[2])]
    dominant = max(range(3), key=lambda k: abs_components[k])
    axis = ("x", "y", "z")[dominant]
    sign = "plus" if normal_world[dominant] > 0 else "minus"
    return f"{axis}{sign}"


def _build_structural_surface_record(
    inst: dict,
    surface_type: str,
    gravity_aligned_interior_hint: list[float] | None,
    interior_ref_world: list[float] | None,
    z_min: float,
    align=None,
) -> tuple[dict, list[float]]:
    """Build one structural-surface record from a Habitat-labeled instance.

    Returns (record_dict, face_centroid). The face centroid is returned
    untranslated (but aligned, when `align` is given) so the caller can use the
    floor's face centroid as the interior reference for walls (per P2.03
    sign-off point (a): orient walls using the floor-face centroid, NOT the
    object-centroid room_bbox).

    `align` levels the scene. It is applied BEFORE the interior-normal sign
    test, so the floor/ceiling hints [0,0,+-1] are tested against the true
    vertical rather than against raw world +Z.
    """
    instance_id = inst["id"]
    center_local = inst["oriented_bbox"]["abb"]["center"]
    quat = [float(x) for x in inst["oriented_bbox"]["orientation"]["rotation"]]
    sizes = inst["oriented_bbox"]["abb"]["sizes"]
    extents = [float(s) / 2.0 for s in sizes]

    center_world = _quat_rotate(quat, center_local)
    thin_idx = _thin_axis_index(extents)
    thin_world_unit = _unit(_quat_axis_in_world(quat, thin_idx))
    if align is not None:
        center_world = _matvec(align, center_world)
        thin_world_unit = _unit(_matvec(align, thin_world_unit))
        quat = _quat_mul(_matrix_to_quat(align), quat)

    if gravity_aligned_interior_hint is not None:
        ref_dir = gravity_aligned_interior_hint
    else:
        if interior_ref_world is None:
            raise ValueError(
                "wall structural-surface extraction requires interior_ref_world "
                "(floor-face centroid)"
            )
        ref_dir = _sub(interior_ref_world, center_world)

    sign_along_thin = 1.0 if _dot(thin_world_unit, ref_dir) >= 0 else -1.0
    interior_normal = _scale(thin_world_unit, sign_along_thin)

    corners_world = _obb_face_corners(
        center_world, extents, quat, thin_idx, sign_along_thin,
    )
    corners_world = _orient_polygon(corners_world, interior_normal)

    corners_translated = [_translate_z(c, -z_min) for c in corners_world]
    center_translated = _translate_z(center_world, -z_min)

    face_centroid_world = [
        sum(c[k] for c in corners_world) / 4.0 for k in range(3)
    ]
    face_centroid_translated = _translate_z(face_centroid_world, -z_min)

    plane_d = -_dot(interior_normal, face_centroid_translated)

    if surface_type == "floor":
        surface_uid = f"floor_{instance_id}"
    elif surface_type == "ceiling":
        surface_uid = f"ceiling_{instance_id}"
    else:
        tag = _orient_tag(interior_normal)
        surface_uid = f"wall_{instance_id}_{tag}"

    record = {
        "surface_uid": surface_uid,
        "surface_type": surface_type,
        "source": "habitat_label",
        "source_instance_ref": str(instance_id),
        "plane": {
            "normal": _round_vec(interior_normal),
            "d": round(float(plane_d), 6),
        },
        "polygon": [_round_vec(c) for c in corners_translated],
        "source_obb": {
            "center": _round_vec(center_translated),
            "extents": _round_vec(extents),
            "rotation_quat": [round(float(x), 6) for x in quat],
        },
        "confidence": 1.0,
    }
    return record, face_centroid_world


def _extract_structural_surfaces(
    info: dict, z_min: float, align=None,
) -> list[dict]:
    """Extract floor / wall / ceiling surfaces from Habitat semantic labels.

    P2.03 scope: habitat_label path only. Fails explicitly if any of
    {floor, wall, ceiling} is absent from the raw info; RANSAC fallback
    is deferred until a scene actually requires it. Walls are oriented
    using the first floor's face centroid as the interior reference,
    NOT the object-centroid room_bbox.
    """
    grouped: dict[str, list[dict]] = {k: [] for k in STRUCTURAL_REQUIRED}
    for o in info["objects"]:
        name = o.get("class_name", "")
        if name in grouped:
            grouped[name].append(o)

    missing = [k for k in STRUCTURAL_REQUIRED if not grouped[k]]
    if missing:
        raise SystemExit(
            "Refusing to emit enriched-v2 structural surfaces: required "
            f"Habitat classes are absent: {sorted(missing)}. P2.03 implements "
            "the habitat_label path only; RANSAC is out of scope until a "
            "scene needs it."
        )

    for k in grouped:
        grouped[k].sort(key=lambda o: o["id"])

    floor_interior_hint = [0.0, 0.0, 1.0]
    ceiling_interior_hint = [0.0, 0.0, -1.0]

    surfaces: list[dict] = []
    floor_face_centroid_world: list[float] | None = None

    for inst in grouped["floor"]:
        rec, face_centroid = _build_structural_surface_record(
            inst, "floor", floor_interior_hint, None, z_min, align,
        )
        surfaces.append(rec)
        if floor_face_centroid_world is None:
            floor_face_centroid_world = face_centroid

    for inst in grouped["ceiling"]:
        rec, _ = _build_structural_surface_record(
            inst, "ceiling", ceiling_interior_hint, None, z_min, align,
        )
        surfaces.append(rec)

    for inst in grouped["wall"]:
        rec, _ = _build_structural_surface_record(
            inst, "wall", None, floor_face_centroid_world, z_min, align,
        )
        surfaces.append(rec)

    return surfaces


def _suffix_duplicates(records: list[dict]) -> list[dict]:
    counts: dict[str, int] = defaultdict(int)
    label_total: dict[str, int] = defaultdict(int)
    for r in records:
        label_total[r["label"]] += 1
    for r in records:
        if label_total[r["label"]] > 1:
            counts[r["label"]] += 1
            r["label"] = f"{r['label']}_{counts[r['label']]}"
    return records


def import_replica(
    scene_dir: Path,
    scene_id: str,
    out_root: Path,
    keep_structural: bool,
    enriched_v2: bool = False,
) -> dict:
    info = json.loads((scene_dir / "habitat" / "info_semantic.json").read_text())
    objects_raw = info["objects"]
    gravity_dir = info["gravity_dir"]

    # Frame decision. A tilted capture is levelled, not refused — see the
    # module docstring and docs/frame_decision.md. Below the guard nothing is
    # applied and the output is the raw capture axes (plus the z shift), which
    # is what every currently-committed artifact under scenes/ was built from.
    tilt_deg = gravity_tilt_deg(gravity_dir)
    align = gravity_align_matrix(
        tuple(float(v) for v in gravity_dir)
    ) if tilt_deg >= GRAVITY_ALIGN_GUARD_DEG else None
    align_quat = _matrix_to_quat(align) if align is not None else None
    frame_kind = "scene_canonical" if align is not None else "world"

    kept = []
    dropped_undefined = 0
    dropped_structural = 0
    for o in objects_raw:
        name = o["class_name"]
        if name == UNDEFINED:
            dropped_undefined += 1
            continue
        if (not keep_structural) and name in STRUCTURAL_DROP:
            dropped_structural += 1
            continue
        center_local = o["oriented_bbox"]["abb"]["center"]
        quat = [float(x) for x in o["oriented_bbox"]["orientation"]["rotation"]]
        center_world = _quat_rotate(quat, center_local)
        if align is not None:
            # Level the scene: rotate the centre, and compose the levelling
            # rotation into the stored orientation so the OBB stays consistent
            # with it. `sizes` are local box dimensions — rotation-invariant,
            # so they are deliberately left alone.
            center_world = _matvec(align, center_world)
            quat = _quat_mul(align_quat, quat)
        sizes = o["oriented_bbox"]["abb"]["sizes"]
        extents = [float(s) / 2.0 for s in sizes]
        bbox_aabb_raw = _aabb_from_obb(center_world, extents, quat)
        kept.append(
            {
                "instance_id": o["id"],
                "label": name,
                "xyz_raw": [float(center_world[0]), float(center_world[1]), float(center_world[2])],
                "sizes": [float(sizes[0]), float(sizes[1]), float(sizes[2])],
                "bbox_obb_extents": extents,
                "bbox_obb_rotation_quat": [float(x) for x in quat],
                "bbox_aabb_raw": bbox_aabb_raw,
            }
        )

    if enriched_v2:
        z_min = min(r["bbox_aabb_raw"][0][2] for r in kept)
    else:
        # Legacy behavior: local-frame sizes approximate the world-frame AABB.
        z_min = min(r["xyz_raw"][2] - r["sizes"][2] / 2 for r in kept)
    objects_out = []
    for r in _suffix_duplicates(kept):
        x, y, z = r["xyz_raw"]
        centroid = [float(x), float(y), float(z - z_min)]
        obj = {
            "id": f"obj_{r['instance_id']}",
            "label": r["label"],
            "zone": None,
            "xyz": [round(v, 3) for v in centroid],
            "attributes": {
                "type": r["label"],
                "bbox_sizes": [round(s, 3) for s in r["sizes"]],
            },
        }
        if enriched_v2:
            bbox_min_raw, bbox_max_raw = r["bbox_aabb_raw"]
            obj["bbox_obb"] = {
                "center": _round_vec(centroid),
                "extents": _round_vec(r["bbox_obb_extents"]),
                "rotation_quat": _round_vec(r["bbox_obb_rotation_quat"]),
            }
            obj["bbox_aabb"] = [
                _round_vec(_translate_z(bbox_min_raw, -z_min)),
                _round_vec(_translate_z(bbox_max_raw, -z_min)),
            ]
        objects_out.append(obj)

    scene_graph = {"scene": scene_id, "objects": objects_out, "relations": []}
    structural_surfaces: list[dict] = []
    if enriched_v2:
        scene_graph["schema_version"] = ENRICHED_SCHEMA_VERSION
        structural_surfaces = _extract_structural_surfaces(info, z_min, align)
        scene_graph["structural_surfaces"] = structural_surfaces

    xs = [o["xyz"][0] for o in objects_out]
    ys = [o["xyz"][1] for o in objects_out]
    zs = [o["xyz"][2] for o in objects_out]
    capture_meta = {
        "scene_id": scene_id,
        "source": "replica/room_0",
        "axis_convention": {
            "up_axis": "+z",
            "gravity_dir_raw": gravity_dir,
            # What frame the coordinates in scene_graph.json are actually in.
            # "world" = the capture's raw axes (up is within
            # GRAVITY_ALIGN_GUARD_DEG of +z but not exactly +z);
            # "scene_canonical" = levelled, up is exactly +z. Consumed by
            # adapters/oracle_replica.py, which stamps it onto the SceneFrame
            # so every extracted Edge inherits a truthful frame label.
            "frame_kind": frame_kind,
            # Gravity IN THE EMITTED FRAME. Identical to gravity_dir_raw when
            # nothing was applied; exactly (0,0,-1) once levelled. Downstream
            # predicates (on_surface, attached_to, ...) read a SceneFrame's
            # gravity and compare it against the coordinates, so handing them
            # the raw tilted vector alongside levelled coordinates would be
            # incoherent.
            "gravity_dir_effective": (
                [0.0, 0.0, -1.0] if align is not None
                else [float(v) for v in gravity_dir]
            ),
            "gravity_tilt_deg": round(tilt_deg, 4),
            "gravity_align_guard_deg": GRAVITY_ALIGN_GUARD_DEG,
            "gravity_align_applied": align is not None,
        },
        "units": "meters",
        "room_bbox": [
            [round(min(xs), 3), round(min(ys), 3), round(min(zs), 3)],
            [round(max(xs), 3), round(max(ys), 3), round(max(zs), 3)],
        ],
        "object_count": len(objects_out),
        "authored_relation_count": 0,
        "import_notes": {
            "z_translation_applied": round(-z_min, 3),
            "dropped_undefined": dropped_undefined,
            "dropped_structural": dropped_structural,
            "keep_structural": keep_structural,
            "zone_field": "always_null_for_this_scene",
            "abb_center_rotated_by_orientation_quat": True,
            "bbox_sizes_kept_in_local_frame": True,
        },
    }
    if enriched_v2:
        capture_meta["schema_version"] = ENRICHED_SCHEMA_VERSION
        per_type: dict[str, int] = {}
        for s in structural_surfaces:
            per_type[s["surface_type"]] = per_type.get(s["surface_type"], 0) + 1
        capture_meta["import_notes"].update(
            {
                "output_mode": "enriched_v2",
                "bbox_aabb_derived_from_rotated_obb": True,
                "structural_surface_count": len(structural_surfaces),
                "structural_surface_count_per_type": per_type,
                "structural_surface_sources": sorted({s["source"] for s in structural_surfaces}),
            }
        )

    out_dir = out_root / scene_id
    if enriched_v2:
        out_dir = out_dir / "enriched" / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scene_graph.json").write_text(json.dumps(scene_graph, indent=2))
    (out_dir / "capture_meta.json").write_text(json.dumps(capture_meta, indent=2))
    return capture_meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--out-root", default=Path("scenes"), type=Path)
    parser.add_argument("--keep-structural", action="store_true")
    parser.add_argument(
        "--enriched-v2",
        action="store_true",
        help="Write schema-v2 OBB output under <scene>/enriched/v2/ without touching Phase 1 files.",
    )
    args = parser.parse_args()
    meta = import_replica(
        args.scene_dir,
        args.scene_id,
        args.out_root,
        args.keep_structural,
        enriched_v2=args.enriched_v2,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
