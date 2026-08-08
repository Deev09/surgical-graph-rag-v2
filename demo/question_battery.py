"""Plug-and-chug a WIDE question battery on a scene -- every support class plus
every structural relation -- and report each outcome. Reference-object-free, so
the same battery runs unchanged on any room.

  python3 demo/question_battery.py                       # room_0
  python3 demo/question_battery.py /path/to/room scene_id

There is no answer key for an arbitrary scene, so this reports OUTCOME + counts
(answer / empty / defer) for plausibility review, not an accuracy score. room_0 is
the only scene with ground truth (validated elsewhere).
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
from graph.relations.contacts_surface import ContactsSurfaceConfig, ContactsSurfaceExtractor
from graph.relations.directional import DirectionalConfig, DirectionalExtractor
from graph.relations.on_entity_surface import OnEntitySurfaceConfig, OnEntitySurfaceExtractor
from graph.relations.on_surface import OnSurfaceConfig, OnSurfaceExtractor
from graph.relations.surface import SurfaceProximityConfig, SurfaceProximityExtractor
from reasoner.base import CompletenessProfile, ExecutionContext
from reasoner.compiler_rules import RulesCompiler
from reasoner.executor import RulesExecutor
from reasoner.router import Router
from reasoner.verbalizer import StandardVerbalizer

DEFAULT_ROOM = Path.home() / "Desktop/datasets/replica/room_0"

STRUCTURAL_Q = [
    "what is on the floor?",
    "what is against the wall?",
    "what is near the wall?",
    "what is attached to the wall?",
]
# one support question per allowlisted support class
SUPPORT_Q = [f"what is on the {c}?" for c in
             ("table", "desk", "shelf", "counter", "stool",
              "bench", "sofa", "chair", "plant-stand")]


def _runs():
    # exclude_room_scale_flat (Phase 8 F4): wall-to-wall rugs are never
    # "against/near the wall" candidates. Enabled on BOTH wall-relation
    # configs so CONTACTS ⊆ NEAR(wall) holds. Battery-path policy only;
    # phase gates build their own default (exclusion-off) configs.
    return [
        ExtractorRun(DirectionalExtractor(), DirectionalConfig(mode="sparse")),
        ExtractorRun(SurfaceProximityExtractor(),
                     SurfaceProximityConfig(use_polygon_clip=True,
                                            exclude_room_scale_flat=True)),
        ExtractorRun(OnSurfaceExtractor(), OnSurfaceConfig()),
        ExtractorRun(ContactsSurfaceExtractor(),
                     ContactsSurfaceConfig(exclude_room_scale_flat=True)),
        ExtractorRun(OnEntitySurfaceExtractor(), OnEntitySurfaceConfig()),
        ExtractorRun(AttachedToExtractor(), AttachedToConfig()),
    ]


def main() -> int:
    room = Path(sys.argv[1]) if len(sys.argv) >= 2 else DEFAULT_ROOM
    scene = sys.argv[2] if len(sys.argv) >= 3 else "replica_room_0"
    if not (room / "habitat" / "info_semantic.json").exists():
        print(f"Refusing: {room}/habitat/info_semantic.json not found.")
        return 1

    arts = import_habitat_room(room, scene)
    labels = {e.identity.object_uid: e.identity.display_label for e in arts.entities}
    bundle, _ = build_graph(arts, _runs(), density_policy="phase2_telemetry_only")
    router = Router(compiler=RulesCompiler(), executor=RulesExecutor(),
                    verbalizer=StandardVerbalizer())
    ctx = ExecutionContext(completeness=CompletenessProfile(
        source="oracle", entity_recall_by_class={}, edge_recall_by_type={}))

    present = sorted({labels[u] for u in labels})
    print("=" * 78)
    print(f"QUESTION BATTERY: {scene}   ({len(arts.entities)} objects, "
          f"{len(arts.structural_surfaces)} surfaces)")
    print("=" * 78)

    tally = {"bindings": 0, "empty": 0, "abstain": 0, "other": 0}
    for q in STRUCTURAL_Q + SUPPORT_Q:
        ans = router.answer(q, bundle, ctx)
        named = [f"{u}:{labels.get(u, '?')}" for u in ans.cited_uids][:6]
        tally[ans.outcome if ans.outcome in tally else "other"] = \
            tally.get(ans.outcome if ans.outcome in tally else "other", 0) + 1
        # is the asked class even in this scene? (helps read an empty result)
        cls = q.removeprefix("what is on the ").removesuffix("?")
        note = ""
        if q in SUPPORT_Q and cls not in present:
            note = f"  [no '{cls}' in scene]"
        extra = "" if len(ans.cited_uids) <= 6 else f" (+{len(ans.cited_uids)-6} more)"
        print(f"\nQ: {q}")
        print(f"   {ans.outcome:9} -> {named if named else '(none)'}{extra}{note}")

    print("\n" + "-" * 78)
    print(f"outcomes: {tally}")
    print("classes present in scene:", ", ".join(present))
    return 0


if __name__ == "__main__":
    sys.exit(main())
