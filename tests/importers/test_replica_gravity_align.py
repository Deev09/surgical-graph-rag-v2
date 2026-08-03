"""importers/replica.py levels tilted captures instead of refusing them.

The bug: `_gravity_is_neg_z` required |g_x| < 0.05, so importers/replica.py
raised `SystemExit: Refusing to import` on Replica's room_2 (g_x = -0.1496,
8.72 deg off world +Z) — while demo/replica_habitat_import.py imported the same
scene without complaint by rotating it. Two importers, opposite answers to "is
room_2 importable", on the scene C1-P1, C1-P2 and semantics-v2 were tuned on.

Which behavior the frozen results depend on: ACCEPT-AND-ROTATE. The Phase 8
scorecard builds every scene through demo/replica_habitat_import.py, so making
the demo path refuse tilted scenes would drop room_2 out of it and move the
frozen 4 true_answer / 27 true_empty / 22 miss / 3 false_answer. The legacy
importer is therefore the side that changes.

What is asserted here:
  1. room_2 imports, and comes out levelled (floor normal exactly +z).
  2. Every currently-accepted scene is untouched — byte-identical output, which
     is what keeps the committed scenes/replica_room_0/ fixture reproducible.
     That is the whole reason the alignment is guarded rather than always-on.
  3. Both importers now agree, scene by scene, on what is importable.
  4. The quaternion helpers levelling relies on are exact.

Run: python tests/importers/test_replica_gravity_align.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from importers.replica import (
    GRAVITY_ALIGN_GUARD_DEG,
    _matrix_to_quat,
    _matvec,
    _quat_mul,
    _quat_rotate,
    gravity_align_matrix,
    gravity_tilt_deg,
    import_replica,
)

REPLICA_ROOT = REPO_ROOT / "data" / "replica"
COMMITTED_V2 = REPO_ROOT / "scenes" / "replica_room_0" / "enriched" / "v2"

# Measured Replica gravity tilts, in degrees off world +Z (Finding 1 of
# docs/frame_and_scale_audit.md). room_2 is the only scene past the guard.
TILTS = {
    "frl_apartment_0": 0.1146, "office_0": 0.2060, "room_1": 0.2304,
    "room_0": 0.2709, "apartment_0": 1.3089, "room_2": 8.7246,
}


def _scenes_on_disk() -> list[str]:
    return [name for name in sorted(TILTS)
            if (REPLICA_ROOT / name / "habitat" / "info_semantic.json").exists()]


# --- 1. the maths levelling depends on -----------------------------------


def test_matrix_to_quat_agrees_with_the_matrix_it_came_from() -> None:
    """_matrix_to_quat is only correct if rotating by the quaternion and
    rotating by the matrix are the same operation — the enriched-v2 output
    stores orientation as a quaternion but levels points with the matrix, so a
    mismatch would silently desynchronize the OBB from its own box."""
    probes = [
        (0.0, 0.0, -1.0),                    # already aligned -> identity
        (-0.1496, 0.02506, -0.98843),        # room_2
        (0.00365, 0.00301, -0.99999),        # room_0
        (0.6, -0.3, -0.74),                  # far past anything real
        (0.0, 0.0, 1.0),                     # 180 deg flip branch
    ]
    vs = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
          (0.37, -2.1, 4.4), (-1.0, -1.0, -1.0)]
    for g in probes:
        R = gravity_align_matrix(g)
        q = _matrix_to_quat(R)
        for v in vs:
            a = _matvec(R, list(v))
            b = _quat_rotate(q, list(v))
            if max(abs(a[i] - b[i]) for i in range(3)) > 1e-12:
                raise AssertionError(
                    f"quat/matrix disagree for gravity={g} v={v}: {a} vs {b}")


def test_quat_mul_composes_in_the_documented_order() -> None:
    """_quat_mul(a, b) must mean 'rotate by b, then by a' — the order the
    importer relies on when folding the levelling rotation into a stored
    object orientation."""
    a = _matrix_to_quat(gravity_align_matrix((-0.1496, 0.02506, -0.98843)))
    b = [0.0, 0.0, 0.3826834323650898, 0.9238795325112867]   # 45 deg about +z
    composed = _quat_mul(a, b)
    for v in ((1.0, 0.0, 0.0), (0.0, 2.0, -1.0), (0.3, 0.4, 0.5)):
        step = _quat_rotate(a, _quat_rotate(b, list(v)))
        once = _quat_rotate(composed, list(v))
        if max(abs(step[i] - once[i]) for i in range(3)) > 1e-12:
            raise AssertionError(
                f"composition order wrong for v={v}: {once} vs {step}")


def test_gravity_tilt_matches_the_audit_measurements() -> None:
    """The guard is a threshold on this number, so the number itself is worth
    pinning to the audit that motivated the value."""
    for name, expected in TILTS.items():
        info = REPLICA_ROOT / name / "habitat" / "info_semantic.json"
        if not info.exists():
            continue
        got = gravity_tilt_deg(
            json.loads(info.read_text(encoding="utf-8"))["gravity_dir"])
        if abs(got - expected) > 5e-4:
            raise AssertionError(f"{name}: tilt {got:.4f} != audit {expected}")


def test_guard_sits_strictly_between_the_two_populations() -> None:
    """A guard inside either cluster would be an arbitrary cut. It is not:
    the accepted scenes top out at 1.31 deg and room_2 is at 8.72."""
    below = [t for t in TILTS.values() if t < GRAVITY_ALIGN_GUARD_DEG]
    above = [t for t in TILTS.values() if t >= GRAVITY_ALIGN_GUARD_DEG]
    if not below or not above:
        raise AssertionError(f"guard {GRAVITY_ALIGN_GUARD_DEG} does not split "
                             f"the measured tilts at all: {TILTS}")
    if max(below) > GRAVITY_ALIGN_GUARD_DEG / 3.0:
        raise AssertionError(
            f"guard {GRAVITY_ALIGN_GUARD_DEG} has less than 3x clearance over "
            f"the largest untouched scene ({max(below)} deg)")
    if min(above) < GRAVITY_ALIGN_GUARD_DEG * 1.5:
        raise AssertionError(
            f"guard {GRAVITY_ALIGN_GUARD_DEG} has less than 1.5x clearance "
            f"under the smallest levelled scene ({min(above)} deg)")


# --- 2. the tilted scene the two importers used to disagree about ----------


def test_room_2_imports_instead_of_raising() -> None:
    """The regression this file is named for. Pre-change this raised
    SystemExit('Refusing to import: gravity_dir=... is not approximately -Z')."""
    if "room_2" not in _scenes_on_disk():
        print("  SKIP (raw Replica room_2 not on disk)")
        return
    with tempfile.TemporaryDirectory() as td:
        meta = import_replica(REPLICA_ROOT / "room_2", "replica_room_2",
                              Path(td), keep_structural=False, enriched_v2=True)
        axis = meta["axis_convention"]
        if not axis["gravity_align_applied"]:
            raise AssertionError(
                f"room_2 is {axis['gravity_tilt_deg']} deg off +Z and should "
                f"have been levelled: {axis!r}")
        if axis["frame_kind"] != "scene_canonical":
            raise AssertionError(
                f"a levelled scene is scene_canonical, got {axis['frame_kind']!r}")
        if axis["gravity_dir_effective"] != [0.0, 0.0, -1.0]:
            raise AssertionError(
                f"levelled gravity must be exactly -Z: "
                f"{axis['gravity_dir_effective']!r}")

        scene = json.loads(
            (Path(td) / "replica_room_2" / "enriched" / "v2" /
             "scene_graph.json").read_text(encoding="utf-8"))

    for s in scene["structural_surfaces"]:
        n = s["plane"]["normal"]
        if s["surface_type"] == "floor" and abs(n[2] - 1.0) > 1e-6:
            raise AssertionError(f"levelled floor normal not +z: {n!r}")
        if s["surface_type"] == "ceiling" and abs(n[2] + 1.0) > 1e-6:
            raise AssertionError(f"levelled ceiling normal not -z: {n!r}")
        if s["surface_type"] == "wall" and abs(n[2]) > 1e-6:
            raise AssertionError(
                f"levelled wall normal not gravity-perpendicular: {n!r}")


def test_both_importers_agree_on_what_is_importable() -> None:
    """The actual reconciliation: no scene is importable by one path and
    refused by the other. This is asserted over whatever Replica scenes are on
    disk, not over a hardcoded list, so a newly fetched tilted scene is covered
    automatically."""
    from demo.replica_habitat_import import import_habitat_room

    scenes = _scenes_on_disk()
    if not scenes:
        print("  SKIP (raw Replica data not on disk)")
        return
    disagreements = []
    for name in scenes:
        room = REPLICA_ROOT / name
        try:
            with tempfile.TemporaryDirectory() as td:
                import_replica(room, name, Path(td),
                               keep_structural=False, enriched_v2=True)
            legacy_ok = True
        except SystemExit:
            legacy_ok = False
        try:
            import_habitat_room(room, name)
            demo_ok = True
        except SystemExit:
            demo_ok = False
        if legacy_ok != demo_ok:
            disagreements.append(
                f"{name}: importers/replica.py={legacy_ok} "
                f"demo/replica_habitat_import.py={demo_ok}")
    if disagreements:
        raise AssertionError(
            "the two importers disagree about importability: "
            + "; ".join(disagreements))


# --- 3. nothing that used to import moved --------------------------------


def test_untilted_scenes_are_left_in_the_raw_capture_axes() -> None:
    """Below the guard nothing is applied, and the importer says so rather than
    claiming a canonicalization it did not perform."""
    for name in _scenes_on_disk():
        if TILTS[name] >= GRAVITY_ALIGN_GUARD_DEG:
            continue
        with tempfile.TemporaryDirectory() as td:
            meta = import_replica(REPLICA_ROOT / name, name, Path(td),
                                  keep_structural=False, enriched_v2=True)
        axis = meta["axis_convention"]
        if axis["gravity_align_applied"] or axis["frame_kind"] != "world":
            raise AssertionError(
                f"{name} ({TILTS[name]} deg) is under the "
                f"{GRAVITY_ALIGN_GUARD_DEG} deg guard and must be untouched: "
                f"{axis!r}")
        if axis["gravity_dir_effective"] != axis["gravity_dir_raw"]:
            raise AssertionError(
                f"{name}: nothing was applied, so effective gravity must equal "
                f"raw gravity: {axis!r}")


def test_room_0_enriched_v2_is_byte_identical_to_the_committed_fixture() -> None:
    """The guard's whole justification. scenes/replica_room_0/enriched/v2/ is
    the frozen Phase 1 replay fixture that adapters/oracle_replica.py reads and
    that the v1 benchmark is defined against; if levelling were unconditional,
    room_0's 0.27 deg would move every coordinate and silently redefine that
    baseline. Compares bytes, not parsed values."""
    if "room_0" not in _scenes_on_disk() or not COMMITTED_V2.exists():
        print("  SKIP (raw Replica room_0 or committed v2 fixture not on disk)")
        return
    with tempfile.TemporaryDirectory() as td:
        import_replica(REPLICA_ROOT / "room_0", "replica_room_0", Path(td),
                       keep_structural=False, enriched_v2=True)
        fresh = (Path(td) / "replica_room_0" / "enriched" / "v2" /
                 "scene_graph.json").read_bytes()
    committed = (COMMITTED_V2 / "scene_graph.json").read_bytes()
    if fresh != committed:
        raise AssertionError(
            "importers/replica.py no longer reproduces the committed room_0 "
            "enriched_v2 scene_graph.json byte-for-byte. If that is intended, "
            "it is a benchmark-definition change, not a bug fix.")


TESTS = [
    test_matrix_to_quat_agrees_with_the_matrix_it_came_from,
    test_quat_mul_composes_in_the_documented_order,
    test_gravity_tilt_matches_the_audit_measurements,
    test_guard_sits_strictly_between_the_two_populations,
    test_room_2_imports_instead_of_raising,
    test_both_importers_agree_on_what_is_importable,
    test_untilted_scenes_are_left_in_the_raw_capture_axes,
    test_room_0_enriched_v2_is_byte_identical_to_the_committed_fixture,
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
