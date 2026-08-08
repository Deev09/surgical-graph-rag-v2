"""C3.0-SR Stage 0: real-Replica-format (quad) parser fixtures.

Run: python tests/geometry/test_mesh_surfaces_quads.py

Delta 1+2 of docs/c3_0_sr_mesh_surfaces_protocol.md: the loader accepts
face arity 3 and 4 (fixed v0-v2 diagonal split, the same rule frozen for
Replica quads in the C1-M2 triangulation), with real-layout fixtures.
The synthetic room builder already splits its quads by the identical
rule, so the native-quad fixture must parse to BYTE-IDENTICAL faces and
estimate identical surfaces. The pinned room_2 probe is dataset-guarded.
"""
from __future__ import annotations

import struct
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry.mesh_surfaces import (
    estimate_structural_surfaces,
    load_raw_triangle_mesh,
)
from tests.geometry.test_mesh_surfaces import (
    _room_mesh, _surface_signature, _write_binary,
)


def _room_quads():
    """The same room geometry as _room_mesh(), as native quads.

    _room_mesh builds each quad (a,b,c,d) as (0,1,2)+(0,2,3) over 4 fresh
    vertices — exactly the SR diagonal rule — so parsing the quad form
    must reproduce its triangle list exactly."""
    mesh = _room_mesh()
    quads = []
    for i in range(0, len(mesh.faces), 2):
        t1, t2 = mesh.faces[i], mesh.faces[i + 1]
        # (v0,v1,v2) + (v0,v2,v3) -> (v0,v1,v2,v3)
        quads.append((t1[0], t1[1], t1[2], t2[2]))
    return mesh, quads


def _write_binary_quads(path: Path, xyz, rgb, faces_with_arity,
                        replica_layout: bool = True) -> None:
    """Binary PLY in the ACTUAL Replica raw layout: x y z nx ny nz (float)
    + uchar rgb vertices, uint8-count int32-index faces."""
    vprops = (["property float x", "property float y", "property float z"]
              + (["property float nx", "property float ny",
                  "property float nz"] if replica_layout else [])
              + ["property uchar red", "property uchar green",
                 "property uchar blue"])
    header = "\n".join(
        ["ply", "format binary_little_endian 1.0",
         f"element vertex {len(xyz)}"] + vprops +
        [f"element face {len(faces_with_arity)}",
         "property list uint8 int32 vertex_indices", "end_header", ""]
    ).encode("ascii")
    body = bytearray()
    for i, p in enumerate(xyz):
        if replica_layout:
            body += struct.pack("<ffffffBBB", float(p[0]), float(p[1]),
                                float(p[2]), 0.0, 0.0, 1.0,
                                int(rgb[i][0]), int(rgb[i][1]), int(rgb[i][2]))
        else:
            body += struct.pack("<fffBBB", float(p[0]), float(p[1]),
                                float(p[2]), int(rgb[i][0]), int(rgb[i][1]),
                                int(rgb[i][2]))
    for fa in faces_with_arity:
        body += struct.pack("<B", len(fa))
        body += struct.pack(f"<{len(fa)}i", *[int(v) for v in fa])
    path.write_bytes(header + body)


def test_native_quads_parse_byte_identical_to_triangles():
    mesh, quads = _room_quads()
    with tempfile.TemporaryDirectory() as td:
        tri_p = Path(td) / "tri.ply"
        quad_p = Path(td) / "quad.ply"
        _write_binary(tri_p, mesh)                       # triangle fixture
        _write_binary_quads(quad_p, mesh.xyz, mesh.rgb, quads)
        mt = load_raw_triangle_mesh(tri_p)
        mq = load_raw_triangle_mesh(quad_p)
        if not np.array_equal(mt.faces, mq.faces):
            raise AssertionError("quad split must reproduce the triangle "
                                 "list exactly (fixed v0-v2 diagonal)")
        if not np.allclose(mt.xyz, mq.xyz, atol=1e-6):
            raise AssertionError("vertex geometry mismatch")
        if mt.n_source_quads != 0 or mq.n_source_quads != len(quads):
            raise AssertionError(f"quad accounting wrong: {mt.n_source_quads} "
                                 f"{mq.n_source_quads}")


def test_quad_room_estimates_identical_surfaces():
    mesh, quads = _room_quads()
    with tempfile.TemporaryDirectory() as td:
        quad_p = Path(td) / "quad.ply"
        _write_binary_quads(quad_p, mesh.xyz, mesh.rgb, quads)
        est_tri = estimate_structural_surfaces(mesh)
        est_quad = estimate_structural_surfaces(load_raw_triangle_mesh(quad_p))
        if (_surface_signature(est_tri.surfaces)
                != _surface_signature(est_quad.surfaces)):
            raise AssertionError("quad-parsed room must estimate identical "
                                 "surfaces")
        if est_quad.diagnostics["n_source_quads"] != len(quads):
            raise AssertionError("diagnostics must record source quads")


def test_mixed_arity_parses_and_arity5_hard_fails():
    mesh, quads = _room_quads()
    with tempfile.TemporaryDirectory() as td:
        mixed_p = Path(td) / "mixed.ply"
        # first quad kept as its two triangles, rest as quads
        mixed = [tuple(mesh.faces[0]), tuple(mesh.faces[1])] + quads[1:]
        _write_binary_quads(mixed_p, mesh.xyz, mesh.rgb, mixed)
        mm = load_raw_triangle_mesh(mixed_p)
        if not np.array_equal(mm.faces, mesh.faces):
            raise AssertionError("mixed-arity parse must reproduce the "
                                 "triangle list")
        if mm.n_source_quads != len(quads) - 1:
            raise AssertionError(f"mixed quad count wrong: {mm.n_source_quads}")
        bad_p = Path(td) / "bad.ply"
        penta = [tuple(quads[0]) + (0,)] + quads[1:]
        _write_binary_quads(bad_p, mesh.xyz, mesh.rgb, penta)
        try:
            load_raw_triangle_mesh(bad_p)
            raise AssertionError("arity-5 face must hard-fail")
        except ValueError as e:
            if "arity 3 or 4" not in str(e):
                raise AssertionError(f"wrong failure message: {e}")


def test_pinned_room_2_parses_when_dataset_present():
    mesh_path = Path.home() / "Desktop/datasets/replica/room_2/mesh.ply"
    if not mesh_path.exists():
        print("  (skipped: Replica dataset not present)")
        return
    import hashlib
    digest = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
    if digest != ("e58a7c717c7922e1300ba20ae8053c5dbfdf9bd5f2515e10c"
                  "71edad98bcb7e44"):
        raise AssertionError("pinned room_2 mesh hash drifted")
    mesh = load_raw_triangle_mesh(mesh_path)
    if mesh.n_source_quads != 722398 or len(mesh.faces) != 1444796:
        raise AssertionError(f"expected 722398 quads -> 1444796 triangles, "
                             f"got {mesh.n_source_quads} -> {len(mesh.faces)}")
    if len(mesh.xyz) != 722496:
        raise AssertionError(f"vertex count drifted: {len(mesh.xyz)}")


TESTS = [
    test_native_quads_parse_byte_identical_to_triangles,
    test_quad_room_estimates_identical_surfaces,
    test_mixed_arity_parses_and_arity5_hard_fails,
    test_pinned_room_2_parses_when_dataset_present,
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
