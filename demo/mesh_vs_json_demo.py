"""Prove the pipeline works from a raw .PLY mesh, not just the precomputed JSON.

Runs the IDENTICAL Phase 6 pipeline twice on room_0 -- once with object boxes from
info_semantic.json (precomputed oriented bboxes), once with boxes recomputed from
mesh_semantic.ply geometry -- and diffs the support answers. Agreement means the
relation is driven by real geometry, not by trusting Replica's bbox numbers.

    python3 demo/mesh_vs_json_demo.py            # room_0 (needs the .ply mesh)
    python3 demo/mesh_vs_json_demo.py /path/to/room replica_room_x
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.replica_habitat_import import import_habitat_room
from demo.replica_mesh_import import import_mesh_room
from graph.builder import ExtractorRun, build_graph
from graph.relations.contacts_surface import ContactsSurfaceConfig, ContactsSurfaceExtractor
from graph.relations.directional import DirectionalConfig, DirectionalExtractor
from graph.relations.on_entity_surface import OnEntitySurfaceConfig, OnEntitySurfaceExtractor
from graph.relations.on_surface import OnSurfaceConfig, OnSurfaceExtractor
from graph.relations.surface import SurfaceProximityConfig, SurfaceProximityExtractor

DEFAULT_ROOM = Path.home() / "Desktop/datasets/replica/room_0"


def _runs():
    return [
        ExtractorRun(DirectionalExtractor(), DirectionalConfig(mode="sparse")),
        ExtractorRun(SurfaceProximityExtractor(),
                     SurfaceProximityConfig(use_polygon_clip=True)),
        ExtractorRun(OnSurfaceExtractor(), OnSurfaceConfig()),
        ExtractorRun(ContactsSurfaceExtractor(), ContactsSurfaceConfig()),
        ExtractorRun(OnEntitySurfaceExtractor(), OnEntitySurfaceConfig()),
    ]


def _answers(arts):
    bundle, _ = build_graph(arts, _runs(), density_policy="phase2_telemetry_only")
    support = {(e.source.uid, e.target.uid)
               for e in bundle.edges if e.type == "ON_ENTITY_SURFACE"}
    floor = {e.source.uid for e in bundle.edges
             if e.type == "ON_SURFACE" and e.target.uid.startswith("floor")}
    wall = {e.source.uid for e in bundle.edges if e.type == "CONTACTS_SURFACE"}
    return support, floor, wall


def _box_delta(a, b):
    """Median + max corner displacement between matching object boxes (m)."""
    ja = {e.identity.object_uid: e.bbox_aabb for e in a.entities}
    jb = {e.identity.object_uid: e.bbox_aabb for e in b.entities}
    shared = sorted(set(ja) & set(jb))
    deltas = []
    for u in shared:
        (alo, ahi), (blo, bhi) = ja[u], jb[u]
        d = max(max(abs(alo[i] - blo[i]), abs(ahi[i] - bhi[i])) for i in range(3))
        deltas.append(d)
    deltas.sort()
    med = deltas[len(deltas) // 2] if deltas else 0.0
    return len(shared), med, (max(deltas) if deltas else 0.0)


def main() -> int:
    room = Path(sys.argv[1]) if len(sys.argv) >= 2 else DEFAULT_ROOM
    scene = sys.argv[2] if len(sys.argv) >= 3 else "replica_room_0"
    ply = room / "habitat" / "mesh_semantic.ply"
    if not ply.exists():
        print(f"Refusing: {ply} not found (this scene is JSON-only; the .ply test "
              "needs the semantic mesh).")
        return 1

    js = import_habitat_room(room, scene)
    mesh = import_mesh_room(room, scene)

    n, med, mx = _box_delta(js, mesh)
    print("=" * 74)
    print(f"BOX SOURCE A/B on {scene}")
    print(f"  JSON importer : {len(js.entities)} objects (precomputed oriented bbox)")
    print(f"  MESH importer : {len(mesh.entities)} objects (AABB from .ply vertices)")
    print(f"  shared objects: {n}   per-object corner delta  median={med*100:.2f}cm "
          f"max={mx*100:.2f}cm")
    print("=" * 74)

    sj, fj, wj = _answers(js)
    sm, fm, wm = _answers(mesh)

    def cmp(name, A, B):
        only_j, only_m = A - B, B - A
        mark = "IDENTICAL" if not only_j and not only_m else "DIFFERS"
        print(f"\n{name}: JSON={len(A)}  MESH={len(B)}  -> {mark}")
        if only_j:
            print(f"   only in JSON: {sorted(only_j)}")
        if only_m:
            print(f"   only in MESH: {sorted(only_m)}")

    cmp("support pairs (ON_ENTITY_SURFACE)", sj, sm)
    cmp("on-floor objects (ON_SURFACE/floor)", fj, fm)
    cmp("against-wall objects (CONTACTS_SURFACE)", wj, wm)

    print("\n" + "=" * 74)
    # overall equality covers ALL compared relations, not just support —
    # a support-only banner over a wall diff misreads as "no differences".
    same = (sj == sm) and (fj == fm) and (wj == wm)
    if same:
        print("[RESULT] Support, on-floor, and against-wall answers are ALL "
              "IDENTICAL whether boxes come from the precomputed JSON or from "
              "raw .ply mesh geometry. A mesh-producing backend plugs in directly.")
    else:
        differing = [name for name, a, b in (
            ("support", sj, sm), ("on-floor", fj, fm), ("against-wall", wj, wm),
        ) if a != b]
        print(f"[RESULT] Box source changes answers for: {', '.join(differing)} "
              "(see above). The delta is the reconstruction-quality margin around "
              "the contact bands.")
    print("=" * 74)
    return 0 if same else 2


if __name__ == "__main__":
    sys.exit(main())
