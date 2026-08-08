"""Phase 8 A/B parity tests: the mesh importer (variant B) must share the
JSON importer's (variant A) frame exactly, so an A/B diff isolates box source.

Run: python tests/importers/test_mesh_frame_parity.py

Dataset-guarded (self-skip when a scene is not on disk). Properties:
  1. per scene: identical yaw de-rotation, identical floor calibration,
     byte-identical structural surfaces, identical entity uid sets;
  2. B's boxes really come from the mesh (differ from A's JSON boxes),
     so parity is not the result of accidentally reusing A's geometry.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.replica_habitat_import import import_habitat_room
from demo.replica_mesh_import import import_mesh_room

DATA_ROOT = Path.home() / "Desktop/datasets/replica"
SCENES = ("room_0", "room_1", "room_2", "office_0", "frl_apartment_0", "apartment_0")


def _scenes_on_disk():
    return [s for s in SCENES
            if (DATA_ROOT / s / "habitat" / "mesh_semantic.ply").is_file()]


def test_frame_parity_per_scene():
    scenes = _scenes_on_disk()
    if not scenes:
        print("  SKIP (no replica scenes with meshes on disk)")
        return
    failures = []
    for s in scenes:
        A = import_habitat_room(DATA_ROOT / s, f"replica_{s}")
        B = import_mesh_room(DATA_ROOT / s, f"replica_{s}")
        if A.notes["yaw_derotation_deg"] != B.notes["yaw_derotation_deg"]:
            failures.append(f"{s}: yaw {A.notes['yaw_derotation_deg']} != "
                            f"{B.notes['yaw_derotation_deg']}")
        if A.notes["floor_calibration"] != B.notes["floor_calibration"]:
            failures.append(f"{s}: floor calibration differs")
        sa = [(x.surface_uid, x.surface_type, x.plane, tuple(x.polygon))
              for x in A.structural_surfaces]
        sb = [(x.surface_uid, x.surface_type, x.plane, tuple(x.polygon))
              for x in B.structural_surfaces]
        if sa != sb:
            failures.append(f"{s}: structural surfaces not byte-identical")
        ua = {e.identity.object_uid for e in A.entities}
        ub = {e.identity.object_uid for e in B.entities}
        if ua != ub:
            failures.append(f"{s}: entity uid sets differ "
                            f"(A-only={sorted(ua - ub)[:5]}, B-only={sorted(ub - ua)[:5]})")
    if failures:
        raise AssertionError("\n".join(failures))


def test_boxes_actually_from_mesh():
    scenes = _scenes_on_disk()
    if not scenes:
        print("  SKIP (no replica scenes with meshes on disk)")
        return
    s = scenes[0]
    A = import_habitat_room(DATA_ROOT / s, f"replica_{s}")
    B = import_mesh_room(DATA_ROOT / s, f"replica_{s}")
    boxes_a = {e.identity.object_uid: e.bbox_aabb for e in A.entities}
    n_diff = sum(1 for e in B.entities
                 if boxes_a.get(e.identity.object_uid) != e.bbox_aabb)
    if n_diff == 0:
        raise AssertionError(
            f"{s}: every B box equals its A box — B is not reading the mesh")


TESTS = [
    test_frame_parity_per_scene,
    test_boxes_actually_from_mesh,
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
