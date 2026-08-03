"""Tests for geometry/frame.py — the geometry-only frame + scale estimator.

Synthetic-first: every correctness claim is checked on a hand-built room
whose true up axis, floor height, storey height and footprint are known by
construction, then re-checked after the room is rotated and after it is
uniformly rescaled. Scale equivariance is the property the scale audit
depends on, so it is asserted, not assumed.

One dataset-guarded test compares the estimate against Replica's declared
gravity_dir. It self-skips (exit 0) when the dataset is absent, per the
repo's test-runner contract.

Run: python tests/geometry/test_frame.py
"""
from __future__ import annotations

import math
import struct
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry.frame import (
    DEFAULT_CONFIG,
    FrameEstimatorConfig,
    angle_between_deg,
    estimate_from_ply,
    estimate_scene_frame,
    hemisphere_directions,
    load_mesh,
    object_scale_stats,
)

REPLICA_ROOM = Path("/Users/deevyaswain/Desktop/datasets/replica/room_0")


# ---------------------------------------------------------------- fixtures

def _quad(v: list, f: list, a, b, c, d) -> None:
    s = len(v)
    v.extend([a, b, c, d])
    f.extend([(s, s + 1, s + 2), (s, s + 2, s + 3)])


def _furniture_box(v: list, f: list, lo, hi) -> None:
    """Box with OUTWARD normals and no underside — an object resting on the
    floor, scanned the way a real capture sees it."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    _quad(v, f, (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1))   # +z
    _quad(v, f, (x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0))   # -x
    _quad(v, f, (x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1))   # +x
    _quad(v, f, (x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1))   # -y
    _quad(v, f, (x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0))   # +y


def synthetic_room(width=6.0, depth=4.0, height=2.5):
    """Closed room seen from the inside — floor at z=0 facing +z, ceiling at
    z=height facing -z, four inward-facing walls — plus furniture standing
    on the floor (two tables and a cabinet).

    The furniture is what makes the up-axis SIGN recoverable: an empty box
    is vertically symmetric and no cue could prefer one end of it. Normal
    orientation is deliberate, not cosmetic; the estimator's sign cues read
    signed normals."""
    v: list[tuple[float, float, float]] = []
    f: list[tuple[int, int, int]] = []
    hw, hd, h = width / 2.0, depth / 2.0, height
    _quad(v, f, (-hw, -hd, 0.0), (hw, -hd, 0.0), (hw, hd, 0.0), (-hw, hd, 0.0))
    _quad(v, f, (-hw, -hd, h), (-hw, hd, h), (hw, hd, h), (hw, -hd, h))
    _quad(v, f, (-hw, -hd, 0.0), (-hw, hd, 0.0), (-hw, hd, h), (-hw, -hd, h))
    _quad(v, f, (hw, -hd, 0.0), (hw, -hd, h), (hw, hd, h), (hw, hd, 0.0))
    _quad(v, f, (-hw, -hd, 0.0), (-hw, -hd, h), (hw, -hd, h), (hw, -hd, 0.0))
    _quad(v, f, (-hw, hd, 0.0), (hw, hd, 0.0), (hw, hd, h), (-hw, hd, h))
    _furniture_box(v, f, (-2.0, -1.0, 0.0), (-0.6, 0.6, 0.75))
    _furniture_box(v, f, (0.6, -1.2, 0.0), (2.2, 0.4, 0.72))
    _furniture_box(v, f, (-hw + 0.05, hd - 0.6, 0.0), (-hw + 0.65, hd - 0.05, 1.9))
    return (np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64))


def _rotation(axis, deg: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    t = math.radians(deg)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * k + (1 - math.cos(t)) * (k @ k)


def _write_binary_ply(path: Path, xyz, quads, *, with_object_id: bool) -> None:
    """Binary little-endian quad PLY, optionally with the trailing per-face
    uint16 `object_id` that Replica's Habitat instance meshes carry."""
    head = ["ply", "format binary_little_endian 1.0",
            f"element vertex {len(xyz)}",
            "property float x", "property float y", "property float z",
            f"element face {len(quads)}",
            "property list uint8 uint32 vertex_indices"]
    if with_object_id:
        head.append("property uint16 object_id")
    head.append("end_header")
    with open(path, "wb") as fh:
        fh.write(("\n".join(head) + "\n").encode("ascii"))
        for p in xyz:
            fh.write(struct.pack("<3f", *[float(c) for c in p]))
        for i, q in enumerate(quads):
            fh.write(struct.pack("<B4I", 4, *[int(c) for c in q]))
            if with_object_id:
                fh.write(struct.pack("<H", i % 7))


def _quad_room():
    """Same room as synthetic_room, re-joined into quads for the PLY writer.
    _quad emits (s, s+1, s+2) then (s, s+2, s+3), so consecutive triangle
    pairs recompose exactly."""
    xyz, tris = synthetic_room()
    quads = [(int(tris[i][0]), int(tris[i][1]), int(tris[i][2]),
              int(tris[i + 1][2])) for i in range(0, len(tris), 2)]
    return xyz, quads


# ------------------------------------------------------------------- tests

def test_upright_room_recovers_up_floor_and_height():
    xyz, faces = synthetic_room()
    est = estimate_scene_frame(xyz, faces)
    ang = angle_between_deg(est.up_axis, (0.0, 0.0, 1.0))
    if ang > 0.5:
        raise AssertionError(f"up axis off by {ang:.3f} deg")
    if est.floor is None or est.ceiling is None:
        raise AssertionError("floor/ceiling not found in a closed room")
    if abs(est.floor.offset_m - 0.0) > 0.02:
        raise AssertionError(f"floor offset {est.floor.offset_m}")
    if abs(est.ceiling.offset_m - 2.5) > 0.02:
        raise AssertionError(f"ceiling offset {est.ceiling.offset_m}")
    if abs(est.scale.storey_height_m - 2.5) > 0.03:
        raise AssertionError(f"storey height {est.scale.storey_height_m}")
    expected = math.sqrt(6.0 ** 2 + 4.0 ** 2 + 2.5 ** 2)
    if abs(est.scale.room_diagonal_m - expected) > 0.05:
        raise AssertionError(
            f"room diagonal {est.scale.room_diagonal_m} != {expected}")


def test_sign_is_recovered_not_assumed():
    """The estimator must find up from the furniture, with no +z prior. Turn
    the whole room upside down (a proper 180-degree rotation, so normals
    come along) and it must return the flipped up vector."""
    xyz, faces = synthetic_room()
    r = _rotation((1.0, 0.0, 0.0), 180.0)
    est = estimate_scene_frame(xyz @ r.T, faces)
    ang = angle_between_deg(est.up_axis, (0.0, 0.0, -1.0))
    if ang > 0.5:
        raise AssertionError(
            f"up axis of the upside-down room off by {ang:.3f} deg "
            f"(got {est.up_axis})")
    if abs(est.floor.offset_m - 0.0) > 0.02:
        raise AssertionError(f"flipped floor offset {est.floor.offset_m}")


def test_rotated_room_recovers_rotated_up_axis():
    xyz, faces = synthetic_room()
    for axis, deg in (((1.0, 0.0, 0.0), 12.0), ((0.3, 0.8, 0.0), 25.0)):
        r = _rotation(axis, deg)
        est = estimate_scene_frame(xyz @ r.T, faces)
        expected_up = r @ np.array([0.0, 0.0, 1.0])
        ang = angle_between_deg(est.up_axis, expected_up)
        if ang > 0.5:
            raise AssertionError(
                f"rotation {axis}/{deg}deg: up axis off by {ang:.3f} deg")
        if abs(est.scale.storey_height_m - 2.5) > 0.05:
            raise AssertionError(
                f"rotation {axis}/{deg}deg: storey height "
                f"{est.scale.storey_height_m}")


def test_scale_equivariance():
    """Uniform rescaling must scale every length by exactly that factor and
    leave the axis untouched. The scale audit's fractions are only
    meaningful if this holds."""
    xyz, faces = synthetic_room()
    base = estimate_scene_frame(xyz, faces)
    for k in (0.5, 3.0):
        got = estimate_scene_frame(xyz * k, faces)
        if angle_between_deg(got.up_axis, base.up_axis) > 0.2:
            raise AssertionError(f"scale {k} moved the up axis")
        for name in ("room_diagonal_m", "floor_diagonal_m", "storey_height_m"):
            a = getattr(base.scale, name) * k
            b = getattr(got.scale, name)
            if abs(a - b) > 1e-3 * max(1.0, abs(a)):
                raise AssertionError(f"scale {k}: {name} {b} != {a}")


def test_yaw_tracks_wall_orientation():
    xyz, faces = synthetic_room()
    for yaw in (0.0, 17.0, -31.0):
        r = _rotation((0.0, 0.0, 1.0), yaw)
        est = estimate_scene_frame(xyz @ r.T, faces)
        folded = est.yaw_deg
        # yaw is 90-degree symmetric; compare modulo 90.
        diff = (folded - yaw + 45.0) % 90.0 - 45.0
        if abs(diff) > 1.0:
            raise AssertionError(f"yaw {yaw} -> {folded} (diff {diff})")


def test_multi_level_scene_reports_first_ceiling_not_the_roof():
    """Two stacked rooms: storey height must be one storey, and the
    multi-level flag must fire."""
    xyz, faces = synthetic_room()
    upper = xyz.copy()
    upper[:, 2] += 2.6
    xyz2 = np.vstack([xyz, upper])
    faces2 = np.vstack([faces, faces + len(xyz)])
    est = estimate_scene_frame(xyz2, faces2)
    if est.scale.storey_height_m is None or est.scale.storey_height_m > 3.0:
        raise AssertionError(
            f"storey height {est.scale.storey_height_m} spans both levels")
    if not est.diagnostics["multi_level_suspected"]:
        raise AssertionError("multi-level scene not flagged")


def test_binary_ply_with_per_face_object_id_loads():
    xyz, quads = _quad_room()
    with tempfile.TemporaryDirectory() as td:
        plain = Path(td) / "plain.ply"
        tagged = Path(td) / "tagged.ply"
        _write_binary_ply(plain, xyz, quads, with_object_id=False)
        _write_binary_ply(tagged, xyz, quads, with_object_id=True)
        v_a, f_a, prov_a = load_mesh(plain)
        v_b, f_b, prov_b = load_mesh(tagged)
        if not np.array_equal(f_a, f_b) or not np.allclose(v_a, v_b):
            raise AssertionError(
                "trailing per-face object_id changed the decoded geometry")
        if len(f_a) != 2 * len(quads):
            raise AssertionError("quads were not split into two triangles each")
        if prov_b["n_source_quads"] != len(quads):
            raise AssertionError("quad count not reported")
        est = estimate_from_ply(tagged)
        if angle_between_deg(est.up_axis, (0.0, 0.0, 1.0)) > 0.5:
            raise AssertionError("estimate from PLY disagrees with in-memory")


def test_degenerate_inputs_raise():
    xyz, faces = synthetic_room()
    for bad, why in (
        ((xyz, faces[:2]), "too few triangles"),
        ((xyz, faces[:, :2]), "non-triangle face array"),
    ):
        try:
            estimate_scene_frame(*bad)
        except ValueError:
            continue
        raise AssertionError(f"{why} did not raise")
    try:
        angle_between_deg((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    except ValueError:
        pass
    else:
        raise AssertionError("zero vector angle did not raise")


def test_hemisphere_directions_are_unit_and_upper():
    d = hemisphere_directions(256)
    if d.shape != (256, 3):
        raise AssertionError(f"shape {d.shape}")
    norms = np.linalg.norm(d, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-9):
        raise AssertionError("directions are not unit vectors")
    if float(d[:, 2].min()) < 0.0:
        raise AssertionError("directions leave the upper hemisphere")


def test_object_scale_stats():
    boxes = [((0, 0, 0), (1, 0, 0)),          # diagonal 1
             ((0, 0, 0), (0, 2, 0)),          # diagonal 2
             ((0, 0, 0), (3, 0, 0))]          # diagonal 3
    s = object_scale_stats(boxes)
    if s["n"] != 3 or abs(s["median_diagonal_m"] - 2.0) > 1e-9:
        raise AssertionError(s)
    if abs(s["median_max_extent_m"] - 2.0) > 1e-9:
        raise AssertionError(s)
    if object_scale_stats([])["n"] != 0:
        raise AssertionError("empty input should report n=0")
    try:
        object_scale_stats([((0, 0, 0), (-1, 0, 0))])
    except ValueError:
        pass
    else:
        raise AssertionError("inverted aabb did not raise")


def test_config_is_serializable():
    cfg = FrameEstimatorConfig(cone_deg=8.0)
    d = cfg.to_dict()
    if abs(d["cone_deg"] - 8.0) > 1e-12:
        raise AssertionError(d)
    if not all(isinstance(v, float) for v in DEFAULT_CONFIG.to_dict().values()):
        raise AssertionError("config values must serialize as floats")


def test_replica_room_0_matches_declared_gravity():
    """Dataset-guarded. Skips (exit 0) when Replica is not on disk."""
    import json
    mesh = REPLICA_ROOM / "habitat" / "mesh_semantic.ply"
    info = REPLICA_ROOM / "habitat" / "info_semantic.json"
    if not (mesh.is_file() and info.is_file()):
        print("SKIP test_replica_room_0_matches_declared_gravity "
              f"(no dataset at {REPLICA_ROOM})")
        return
    est = estimate_from_ply(mesh)
    g = json.loads(info.read_text(encoding="utf-8"))["gravity_dir"]
    declared_up = [-float(v) for v in g]
    ang = angle_between_deg(est.up_axis, declared_up)
    if ang > 1.0:
        raise AssertionError(
            f"mesh-derived up is {ang:.3f} deg from Replica's declared up")
    if est.floor is None:
        raise AssertionError("no floor plane found in room_0")


TESTS = [
    test_upright_room_recovers_up_floor_and_height,
    test_sign_is_recovered_not_assumed,
    test_rotated_room_recovers_rotated_up_axis,
    test_scale_equivariance,
    test_yaw_tracks_wall_orientation,
    test_multi_level_scene_reports_first_ceiling_not_the_roof,
    test_binary_ply_with_per_face_object_id_loads,
    test_degenerate_inputs_raise,
    test_hemisphere_directions_are_unit_and_upper,
    test_object_scale_stats,
    test_config_is_serializable,
    test_replica_room_0_matches_declared_gravity,
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
    sys.exit(main())
