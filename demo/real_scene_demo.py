"""Real-scene plug-and-chug: import a raw Replica room and run the FULL QA track
through the real Router -- structural surfaces (floor / wall) AND furniture-top
support (table / chair), from one raw habitat/info_semantic.json, no per-scene code.

  python3 demo/real_scene_demo.py                       # room_0 (validates importer)
  python3 demo/real_scene_demo.py /path/to/replica/room_1 replica_room_1

Relations exercised: NEAR_SURFACE, ON_SURFACE (floor), CONTACTS_SURFACE (wall),
ATTACHED_TO (wall-mounted), ON_ENTITY_SURFACE (furniture top), plus directional.

For room_0 it asserts the imported boxes reproduce the committed enriched_v2
support pairs (the 5 table rests + the plant-stand rest) -- proving the importer
is correct. For any other room it just reports what each relation finds, which is
the genuine "does it work on a different real scene?" test.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.replica_habitat_import import import_habitat_room
from graph.builder import ExtractorRun, build_graph
from graph.relations.attached_to import AttachedToConfig, AttachedToExtractor
from graph.relations.contacts_surface import (
    ContactsSurfaceConfig,
    ContactsSurfaceExtractor,
)
from graph.relations.directional import DirectionalConfig, DirectionalExtractor
from graph.relations.on_entity_surface import (
    OnEntitySurfaceConfig,
    OnEntitySurfaceExtractor,
)
from graph.relations.on_surface import OnSurfaceConfig, OnSurfaceExtractor
from graph.relations.surface import SurfaceProximityConfig, SurfaceProximityExtractor
from reasoner.base import CompletenessProfile, ExecutionContext
from reasoner.compiler_rules import RulesCompiler
from reasoner.executor import RulesExecutor
from reasoner.router import Router
from reasoner.verbalizer import StandardVerbalizer

DEFAULT_ROOM = Path.home() / "Desktop/datasets/replica/room_0"

# Ground truth for room_0 (from the committed Phase 6 closeout / enriched_v2).
ROOM0_TABLE_RESTS = {"obj_92", "obj_90", "obj_12", "obj_59", "obj_87"}
ROOM0_PLANTSTAND = ("obj_35", "obj_55")

QUESTIONS = [
    "what is on the floor?",       # on_surface  (structural floor)
    "what is against the wall?",   # contacts_surface (structural wall)
    "what is attached to the wall?",  # attached_to (wall-mounted)
    "what is near the wall?",      # near_surface (structural wall)
    "what is on the table?",       # on_entity_surface (furniture top)
    "what is on the chair?",       # on_entity_surface (expect empty)
    "what is on the shelf?",
    "what is on the sofa?",
]


def run(room_dir: Path, scene_id: str):
    arts = import_habitat_room(room_dir, scene_id)
    bundle, _ = build_graph(
        arts,
        [
            ExtractorRun(DirectionalExtractor(), DirectionalConfig(mode="sparse")),
            ExtractorRun(SurfaceProximityExtractor(),
                         SurfaceProximityConfig(use_polygon_clip=True)),
            ExtractorRun(OnSurfaceExtractor(), OnSurfaceConfig()),
            ExtractorRun(ContactsSurfaceExtractor(), ContactsSurfaceConfig()),
            ExtractorRun(AttachedToExtractor(), AttachedToConfig()),
            ExtractorRun(OnEntitySurfaceExtractor(), OnEntitySurfaceConfig()),
        ],
        density_policy="phase2_telemetry_only",
    )
    router = Router(compiler=RulesCompiler(), executor=RulesExecutor(),
                    verbalizer=StandardVerbalizer())
    ctx = ExecutionContext(completeness=CompletenessProfile(
        source="oracle", entity_recall_by_class={}, edge_recall_by_type={}))

    labels = {e.identity.object_uid: e.identity.display_label for e in arts.entities}
    ent_edges = [e for e in bundle.edges if e.type == "ON_ENTITY_SURFACE"]
    attached_edges = [e for e in bundle.edges if e.type == "ATTACHED_TO"]

    surf_counts: dict[str, int] = {}
    for s in arts.structural_surfaces:
        surf_counts[s.surface_type] = surf_counts.get(s.surface_type, 0) + 1
    edge_counts: dict[str, int] = {}
    for e in bundle.edges:
        edge_counts[e.type] = edge_counts.get(e.type, 0) + 1

    print("=" * 76)
    print(f"REAL SCENE: {scene_id}   (source: {room_dir})")
    print(f"  objects: {len(arts.entities)}   "
          f"surfaces: {surf_counts}   ")
    print(f"  edges by type: {edge_counts}")
    print("=" * 76)
    for q in QUESTIONS:
        ans = router.answer(q, bundle, ctx)
        named = [f"{u}:{labels.get(u, '?')}" for u in ans.cited_uids]
        print(f"\nQ: {q}")
        print(f"   outcome={ans.outcome}  ->  {named if named else '(none)'}")

    print("\n--- all support pairs (supported  ON  supporter) ---")
    for e in sorted(ent_edges, key=lambda e: e.source.uid):
        oc = e.evidence.get("owner_class")
        print(f"   {e.source.uid}:{labels.get(e.source.uid,'?'):16} ON  "
              f"{e.target.uid}:{labels.get(e.target.uid,'?'):16} ({oc})")
    print("\n--- all attached pairs (object  ATTACHED_TO  wall) ---")
    if not attached_edges:
        print("   (none)")
    for e in sorted(attached_edges, key=lambda e: e.source.uid):
        print(f"   {e.source.uid}:{labels.get(e.source.uid,'?'):16} ATTACHED_TO  {e.target.uid}")
    return bundle, ent_edges, labels


def validate_room0(ent_edges):
    pairs = {(e.source.uid, e.target.uid) for e in ent_edges}
    table_supported = {s for s, t in pairs
                       for e in ent_edges
                       if e.source.uid == s and e.target.uid == t
                       and e.evidence.get("owner_class") == "table"}
    ok = True
    if table_supported != ROOM0_TABLE_RESTS:
        ok = False
        print(f"\n[FAIL] table rests {sorted(table_supported)} != "
              f"expected {sorted(ROOM0_TABLE_RESTS)}")
    if ROOM0_PLANTSTAND not in pairs:
        ok = False
        print(f"\n[FAIL] plant-stand rest {ROOM0_PLANTSTAND} missing")
    print("\n" + ("=" * 76))
    if ok:
        print("[VALIDATED] habitat import reproduces enriched_v2 support pairs "
              "(5 table rests + plant-stand). Importer is correct; new rooms plug in.")
    else:
        print("[VALIDATION FAILED] import diverged from enriched_v2 -- do not trust "
              "on new rooms until reconciled.")
    print("=" * 76)
    return ok


def main() -> int:
    if len(sys.argv) >= 3:
        room_dir, scene_id = Path(sys.argv[1]), sys.argv[2]
    elif len(sys.argv) == 2:
        room_dir, scene_id = Path(sys.argv[1]), Path(sys.argv[1]).name
    else:
        room_dir, scene_id = DEFAULT_ROOM, "replica_room_0"

    if not (room_dir / "habitat" / "info_semantic.json").exists():
        print(f"Refusing: {room_dir}/habitat/info_semantic.json not found.")
        return 1

    _, ent_edges, _ = run(room_dir, scene_id)
    if scene_id == "replica_room_0":
        return 0 if validate_room0(ent_edges) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
