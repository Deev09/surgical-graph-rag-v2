"""Synthetic guards for geometry-only entity-local horizontal patches.

Run: python3 tests/geometry/test_entity_support_patches.py
"""
from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry.entity_support_patches import extract_entity_horizontal_patches


def _box() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = np.asarray([
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    ], dtype=np.float64)
    faces = np.asarray([
        (0, 2, 1), (0, 3, 2),       # bottom
        (4, 5, 6), (4, 6, 7),       # top
        (0, 1, 5), (0, 5, 4),
        (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6),
        (3, 0, 4), (3, 4, 7),
    ], dtype=np.int64)
    return xyz, faces, np.full(len(xyz), 7, dtype=np.int64)


def _assert_close(a: float, b: float, tol: float = 1e-9) -> None:
    if not math.isclose(a, b, rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"{a!r} != {b!r}")


def test_closed_box_yields_top_and_bottom_actual_mesh_patches() -> None:
    xyz, faces, ids = _box()
    estimate = extract_entity_horizontal_patches(xyz, faces, ids)
    if len(estimate.owners) != 1:
        raise AssertionError("one dense instance must yield one owner record")
    owner = estimate.owners[0]
    if owner.owner_instance_id != 7 or owner.n_strict_faces != 12:
        raise AssertionError("owner identity or strict face accounting drifted")
    if owner.n_horizontal_faces != 4 or len(owner.patches) != 2:
        raise AssertionError("box should have two horizontal mesh components")
    top, bottom = owner.patches
    if not top.geometry_qualifies or not bottom.geometry_qualifies:
        raise AssertionError("both planar box patches should qualify")
    _assert_close(top.height_m, 1.0)
    _assert_close(bottom.height_m, 0.0)
    for patch in (top, bottom):
        _assert_close(patch.normal[2], 1.0)
        _assert_close(patch.projected_area_m2, 1.0)
        _assert_close(patch.footprint_area_m2, 1.0)
        _assert_close(patch.coverage_ratio, 1.0)
        _assert_close(patch.roughness_rms_m, 0.0)
        if len(patch.polygon) != 4:
            raise AssertionError("unit-square convex footprint should have 4 vertices")
    if top.patch_uid != "instance_7_hpatch_000":
        raise AssertionError("owner-local patch order is not stable top-first")


def test_mixed_instance_faces_never_enter_either_owner() -> None:
    # Two disjoint unit squares plus a horizontal bridge triangle whose
    # vertices disagree on instance id. The bridge must not connect owners.
    xyz = np.asarray([
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (2, 0, 0), (3, 0, 0), (3, 1, 0), (2, 1, 0),
    ], dtype=np.float64)
    faces = np.asarray([
        (0, 1, 2), (0, 2, 3),
        (4, 5, 6), (4, 6, 7),
        (1, 4, 7),                    # mixed owner ids
    ], dtype=np.int64)
    ids = np.asarray([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    estimate = extract_entity_horizontal_patches(xyz, faces, ids)
    if estimate.diagnostics["n_mixed_or_unassigned_faces"] != 1:
        raise AssertionError("mixed-instance face accounting is wrong")
    if [o.n_strict_faces for o in estimate.owners] != [2, 2]:
        raise AssertionError("mixed face leaked into an owner")
    if [len(o.patches) for o in estimate.owners] != [1, 1]:
        raise AssertionError("bridge face merged two owners")


def test_scale_relative_evidence_is_invariant() -> None:
    # Slightly non-planar connected surface gives non-zero roughness while
    # every individual face remains well inside the horizontal cone.
    xyz = np.asarray([
        (0, 0, 0), (1, 0, 0), (1, 1, 0.02), (0, 1, 0),
    ], dtype=np.float64)
    faces = np.asarray([(0, 1, 2), (0, 2, 3)], dtype=np.int64)
    ids = np.full(4, 4, dtype=np.int64)
    a = extract_entity_horizontal_patches(xyz, faces, ids).owners[0].patches[0]
    b = extract_entity_horizontal_patches(
        xyz * 10.0, faces, ids,
    ).owners[0].patches[0]
    if not a.geometry_qualifies or not b.geometry_qualifies:
        raise AssertionError("uniform scaling changed qualification")
    _assert_close(
        a.projected_area_ratio_owner_bbox,
        b.projected_area_ratio_owner_bbox,
    )
    _assert_close(a.coverage_ratio, b.coverage_ratio)
    _assert_close(
        a.roughness_ratio_owner_diagonal,
        b.roughness_ratio_owner_diagonal,
    )
    _assert_close(b.projected_area_m2, a.projected_area_m2 * 100.0, 1e-8)
    _assert_close(b.roughness_rms_m, a.roughness_rms_m * 10.0, 1e-8)


def test_vertical_mesh_has_no_horizontal_patch() -> None:
    xyz = np.asarray([
        (0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1),
    ], dtype=np.float64)
    faces = np.asarray([(0, 1, 2), (0, 2, 3)], dtype=np.int64)
    ids = np.full(4, 3, dtype=np.int64)
    owner = extract_entity_horizontal_patches(xyz, faces, ids).owners[0]
    if owner.n_horizontal_faces != 0 or owner.patches:
        raise AssertionError("vertical faces were emitted as horizontal patches")


def test_output_is_json_safe_and_records_no_oracle_use() -> None:
    xyz, faces, ids = _box()
    estimate = extract_entity_horizontal_patches(xyz, faces, ids)
    payload = estimate.to_dict()
    json.dumps(payload, sort_keys=True, allow_nan=False)
    if payload["diagnostics"]["uses_semantics"] is not False:
        raise AssertionError("geometry-only output claims semantic use")
    if payload["diagnostics"]["uses_oracle"] is not False:
        raise AssertionError("geometry-only output claims oracle use")
    patch = payload["owners"][0]["patches"][0]
    required = {
        "plane", "normal", "polygon", "mesh_area_m2",
        "projected_area_m2", "footprint_area_m2", "coverage_ratio",
        "roughness_rms_m", "roughness_ratio_owner_diagonal",
    }
    if not required.issubset(patch):
        raise AssertionError(f"patch evidence missing {required - set(patch)}")


def test_rejects_assignment_length_mismatch() -> None:
    xyz, faces, ids = _box()
    try:
        extract_entity_horizontal_patches(xyz, faces, ids[:-1])
    except ValueError as exc:
        if "shape" not in str(exc):
            raise AssertionError(f"wrong validation error: {exc}")
    else:
        raise AssertionError("assignment length mismatch was accepted")


TESTS = [
    test_closed_box_yields_top_and_bottom_actual_mesh_patches,
    test_mixed_instance_faces_never_enter_either_owner,
    test_scale_relative_evidence_is_invariant,
    test_vertical_mesh_has_no_horizontal_patch,
    test_output_is_json_safe_and_records_no_oracle_use,
    test_rejects_assignment_length_mismatch,
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
