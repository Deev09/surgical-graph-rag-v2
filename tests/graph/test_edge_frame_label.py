"""Edge.frame must name the frame the edge was ACTUALLY computed in.

The bug this suite exists to catch: every edge constructor in
graph/relations/** hardcoded `frame="world"` while the live importer
(demo/replica_habitat_import.py) handed the extractors a gravity-aligned,
yaw-de-rotated frame. Nothing read the field, so nothing broke — but the field
was wrong, and Finding 4 of docs/frame_and_scale_audit.md measures the size of
the thing it was wrong about: recomputing directional edges in the raw world
frame changes 26.9% of room_1's edges and 9.7% of room_2's. It stayed invisible
because on room_0 — the scene the thresholds were calibrated on — the two frames
agree on 2593 of 2594 edges.

The invariant, and the only one worth asserting, is an EQUALITY rather than a
constant: for every edge e extracted from a bundle b,

    e.frame == b.frame.kind

so this suite is deliberately indifferent to WHICH frame a given importer
produces. It fails if an extractor picks a label instead of reporting one.

test_every_extractor_labels_edges_with_the_bundle_frame is the one that would
have failed before the fix, on the "scene_canonical" and "viewpoint" rounds.
test_extractor_roster_is_complete is what keeps it from rotting: a tenth
extractor added to graph/relations without a row in ROSTER fails the suite
rather than silently going unchecked.

Run: python tests/graph/test_edge_frame_label.py
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.types import Plane, SceneFrame
from extractors.base import (
    EntityArtifact, EntityArtifacts, EntityIdentity, ExtractionDiagnostics,
    SemanticHypothesis, StructuralSurface,
)
from extractors.serde import CURRENT_SCHEMA_VERSION as ENT_SCHEMA_VERSION
from graph.builder import ExtractorRun, build_graph
from graph.relations.attached_to import AttachedToConfig, AttachedToExtractor
from graph.relations.attached_to_v2 import (
    AttachedToV2Config, AttachedToV2Extractor,
)
from graph.relations.base import edge_frame
from graph.relations.contacts_surface import (
    ContactsSurfaceConfig, ContactsSurfaceExtractor,
)
from graph.relations.directional import DirectionalConfig, DirectionalExtractor
from graph.relations.on_entity_surface import (
    OnEntitySurfaceConfig, OnEntitySurfaceExtractor,
)
from graph.relations.on_entity_surface_v2 import (
    OnEntitySurfaceV2Config, OnEntitySurfaceV2Extractor,
)
from graph.relations.on_surface import OnSurfaceConfig, OnSurfaceExtractor
from graph.relations.proximity import ProximityConfig, ProximityExtractor
from graph.relations.surface import (
    SurfaceProximityConfig, SurfaceProximityExtractor,
)
from graph.serde import (
    dump_scene_graph_bundle, load_scene_graph_bundle,
)

REPLICA_ROOMS = REPO_ROOT / "data" / "replica"

# Every frame kind in the schema's domain. The point of running all three is
# that a hardcoded label can only ever match one of them.
ALL_FRAME_KINDS = ("world", "scene_canonical", "viewpoint")


# --- fixtures ------------------------------------------------------------


def _frame(kind: str) -> SceneFrame:
    return SceneFrame(
        gravity=(0.0, 0.0, -1.0), canonical_forward=None,
        canonical_right=None, units="meters", notes="", kind=kind,
    )


def _entity(uid, lo, hi, label=None) -> EntityArtifact:
    label = label or uid
    centroid = tuple((lo[i] + hi[i]) / 2.0 for i in range(3))
    return EntityArtifact(
        identity=EntityIdentity(
            object_uid=uid, display_label=label, aliases=[],
            source_instance_ref=uid,
        ),
        bbox_aabb=(lo, hi), bbox_obb=None, centroid=centroid,
        geometry_handle=None,
        semantic_hypotheses=[
            SemanticHypothesis(label=label, confidence=1.0, source="t")],
        embedding=None, extraction_diagnostics={},
    )


def _surface(uid, stype, plane, polygon) -> StructuralSurface:
    return StructuralSurface(
        surface_uid=uid, surface_type=stype, plane=plane,
        polygon=polygon, confidence=1.0, source="habitat_label",
    )


def _scene(kind: str) -> EntityArtifacts:
    """One synthetic room that gives ALL the extractors something to emit: a
    floor, a wall, a table with a book on it, a wall-mounted elevated sconce,
    a cabinet against the wall, and a far box. Geometry is identical across
    frame kinds — only the LABEL changes — so any difference in the emitted
    labels is attributable to the label alone."""
    floor = _surface(
        "floor_0", "floor", Plane(a=0.0, b=0.0, c=1.0, d=0.0),
        [(-3.0, -3.0, 0.0), (3.0, -3.0, 0.0), (3.0, 3.0, 0.0), (-3.0, 3.0, 0.0)],
    )
    # wall at x = -3, interior normal +x  (x + 3 >= 0 inside the room)
    wall = _surface(
        "wall_0", "wall", Plane(a=1.0, b=0.0, c=0.0, d=3.0),
        [(-3.0, -3.0, 0.0), (-3.0, 3.0, 0.0), (-3.0, 3.0, 2.5), (-3.0, -3.0, 2.5)],
    )
    entities = [
        _entity("table", (-1.0, -1.0, 0.0), (0.0, 0.0, 0.75), label="table"),
        _entity("book", (-0.7, -0.7, 0.75), (-0.5, -0.5, 0.80), label="book"),
        _entity("sconce", (-3.0, 1.0, 1.50), (-2.92, 1.3, 1.80), label="lamp"),
        _entity("cabinet", (-3.0, -2.0, 0.0), (-2.60, -1.0, 1.00), label="cabinet"),
        _entity("far_box", (2.0, 2.0, 0.0), (2.4, 2.4, 0.40), label="box"),
    ]
    surfaces = [floor, wall]
    return EntityArtifacts(
        schema_version=ENT_SCHEMA_VERSION, bundle_hash="ent_frame_t",
        scene_id="t", frame=_frame(kind), representation_hash="rep_t",
        extractor_name="test", extractor_version="0.0",
        entities=entities, structural_surfaces=surfaces,
        geometry_store_path=None,
        diagnostics=ExtractionDiagnostics(
            n_entities=len(entities), n_structural_surfaces=len(surfaces),
            runtime_seconds=0.0, coverage_score=None, notes="",
        ),
        notes={},
    )


def _roster():
    """(label, extractor, config) for every extractor + mode combination that
    can emit an edge. Kept in step with the real roster by
    test_extractor_roster_is_complete."""
    return [
        ("directional_sparse", DirectionalExtractor(),
         DirectionalConfig(mode="sparse")),
        ("directional_compat", DirectionalExtractor(),
         DirectionalConfig(mode="compat")),
        ("proximity_sparse", ProximityExtractor(), ProximityConfig(mode="sparse")),
        ("proximity_compat", ProximityExtractor(), ProximityConfig(mode="compat")),
        ("proximity_sparse_v2", ProximityExtractor(),
         ProximityConfig(mode="sparse", sparse_version=2)),
        ("near_surface", SurfaceProximityExtractor(), SurfaceProximityConfig()),
        ("near_surface_polygon", SurfaceProximityExtractor(),
         SurfaceProximityConfig(use_polygon_clip=True,
                                exclude_room_scale_flat=True)),
        ("on_surface", OnSurfaceExtractor(), OnSurfaceConfig()),
        ("contacts_surface", ContactsSurfaceExtractor(),
         ContactsSurfaceConfig(exclude_room_scale_flat=True)),
        ("on_entity_surface", OnEntitySurfaceExtractor(), OnEntitySurfaceConfig()),
        ("attached_to", AttachedToExtractor(), AttachedToConfig()),
        ("attached_to_v2", AttachedToV2Extractor(), AttachedToV2Config()),
        ("on_entity_surface_v2", OnEntitySurfaceV2Extractor(),
         OnEntitySurfaceV2Config()),
    ]


# One buildable run list. The GraphBuilder rejects mixed compat/sparse runs
# and repeated extractor names, so the compat / sparse_v2 / polygon variants
# are exercised through extractor.extract directly instead of through it.
BUILDABLE = (
    "directional_sparse", "proximity_sparse", "near_surface", "on_surface",
    "contacts_surface", "on_entity_surface", "attached_to",
)


def _sparse_runs():
    return [ExtractorRun(extractor=x, config=c) for label, x, c in _roster()
            if label in BUILDABLE]


# --- tests ---------------------------------------------------------------


def test_every_extractor_labels_edges_with_the_bundle_frame() -> None:
    """THE test. Same geometry, three different declared frames; every emitted
    edge must repeat the bundle's frame back. A hardcoded literal passes at
    most one of the three rounds."""
    emitted_at_least_once: set[str] = set()
    for kind in ALL_FRAME_KINDS:
        arts = _scene(kind)
        for label, extractor, config in _roster():
            edges, _ = extractor.extract(arts, config)
            if edges:
                emitted_at_least_once.add(label)
            wrong = sorted({e.frame for e in edges} - {kind})
            if wrong:
                raise AssertionError(
                    f"{label} emitted frame={wrong!r} from a bundle whose frame "
                    f"is {kind!r}. Edge.frame must report the bundle's frame "
                    f"(graph.relations.base.edge_frame), never a literal."
                )

    # A silent zero-edge extractor would make the loop above vacuous, which is
    # exactly how a hardcoded label would survive unnoticed.
    never = sorted({label for label, _, _ in _roster()} - emitted_at_least_once)
    if never:
        raise AssertionError(
            f"these extractors emitted no edges at all, so the frame assertion "
            f"was vacuous for them: {never}. Fix the fixture, not the assertion."
        )


def test_extractor_roster_is_complete() -> None:
    """Every RelationExtractor class in graph.relations is covered above.

    Without this, adding a tenth extractor that hardcodes frame="world" would
    reintroduce the exact bug this file exists for, and every test here would
    still pass."""
    import graph.relations as pkg

    discovered: set[str] = set()
    for mod_info in pkgutil.iter_modules(pkg.__path__):
        module = importlib.import_module(f"{pkg.__name__}.{mod_info.name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__:
                continue          # imported, not defined here
            if all(hasattr(obj, a) for a in ("name", "version", "edge_types")) \
                    and hasattr(obj, "extract"):
                discovered.add(obj.__name__)

    covered = {type(x).__name__ for _, x, _ in _roster()}
    missing = sorted(discovered - covered)
    if missing:
        raise AssertionError(
            f"relation extractors with no frame-label coverage: {missing}. "
            f"Add them to _roster() in this file."
        )


def test_built_bundle_edges_agree_with_the_bundle_frame() -> None:
    """The same invariant one level up: through the real GraphBuilder, every
    edge in the assembled SceneGraphBundle agrees with bundle.frame.kind."""
    for kind in ALL_FRAME_KINDS:
        arts = _scene(kind)
        runs = _sparse_runs()
        bundle, _diag = build_graph(arts, runs,
                                    density_policy="phase2_telemetry_only")
        if bundle.frame.kind != kind:
            raise AssertionError(
                f"builder lost the frame: {bundle.frame.kind!r} != {kind!r}")
        bad = sorted({e.frame for e in bundle.edges} - {kind})
        if bad:
            raise AssertionError(
                f"bundle frame is {kind!r} but edges carry {bad!r}")
        if not bundle.edges:
            raise AssertionError("no edges built; assertion was vacuous")


def test_frame_label_survives_a_serde_round_trip() -> None:
    """A relabel that does not persist is not a relabel. Round-trip through
    graph/serde.py and re-check the same equality."""
    arts = _scene("scene_canonical")
    bundle, _diag = build_graph(arts, _sparse_runs(),
                                density_policy="phase2_telemetry_only")
    with tempfile.TemporaryDirectory() as td:
        dump_scene_graph_bundle(bundle, Path(td))
        restored = load_scene_graph_bundle(Path(td))
    if restored.frame.kind != "scene_canonical":
        raise AssertionError(
            f"SceneFrame.kind lost in serde: {restored.frame.kind!r}")
    bad = sorted({e.frame for e in restored.edges} - {"scene_canonical"})
    if bad:
        raise AssertionError(f"edge frames lost in serde: {bad!r}")


def test_real_replica_import_declares_and_labels_scene_canonical() -> None:
    """End to end on real data, including room_2 (8.72 deg off world +Z).

    demo/replica_habitat_import.py gravity-aligns and (past its guard)
    yaw-de-rotates, so its bundles are scene_canonical and every edge must say
    so. This is the assertion the pre-fix code failed on live data: the frame
    was canonical and the edges all claimed "world"."""
    from demo.replica_habitat_import import import_habitat_room

    rooms = [d for d in ("room_0", "room_2")
             if (REPLICA_ROOMS / d / "habitat" / "info_semantic.json").exists()]
    if not rooms:
        print("  SKIP (raw Replica data not on disk)")
        return

    runs = _sparse_runs()
    for room in rooms:
        arts = import_habitat_room(REPLICA_ROOMS / room, f"replica_{room}")
        if edge_frame(arts) != "scene_canonical":
            raise AssertionError(
                f"{room}: habitat import gravity-aligns, so its frame is "
                f"scene_canonical, not {edge_frame(arts)!r}")
        bundle, _diag = build_graph(arts, runs,
                                    density_policy="phase2_telemetry_only")
        bad = sorted({e.frame for e in bundle.edges} - {"scene_canonical"})
        if bad:
            raise AssertionError(f"{room}: edges carry {bad!r}")
        if not bundle.edges:
            raise AssertionError(f"{room}: no edges built; assertion vacuous")


def test_oracle_replica_v1_path_still_reports_world() -> None:
    """The other direction, and the reason the invariant is an equality rather
    than a new constant: the v1 path (importers/replica.py under its alignment
    guard, read back through adapters/oracle_replica.py) really is in the
    capture's raw axes. Blanket-relabelling everything "scene_canonical" would
    be the same mistake with a different value."""
    from adapters.base import ReconstructionConfig
    from adapters.oracle_replica import (
        OracleReplicaAdapter, build_replica_capture_bundle,
    )
    from extractors.base import InstanceExtractorConfig
    from extractors.oracle_replica import OracleReplicaExtractor
    from representations.mesh import MeshRepresentation

    scene_dir = REPO_ROOT / "scenes" / "replica_room_0"
    if not (scene_dir / "capture_meta.json").exists():
        print("  SKIP (v1 replica scene artifacts not on disk)")
        return

    capture = build_replica_capture_bundle(scene_dir)
    repr_bundle = OracleReplicaAdapter().reconstruct(
        capture, ReconstructionConfig(name="oracle_replica", version="0.1",
                                      params={}))
    arts = OracleReplicaExtractor(
        enriched_v2_path=scene_dir / "enriched" / "v2",
    ).extract(
        MeshRepresentation(bundle=repr_bundle),
        InstanceExtractorConfig(name="oracle_replica", version="0.1", params={}),
    )

    if arts.frame.kind != "world":
        raise AssertionError(
            f"room_0's v1 import is 0.27 deg off +Z and is NOT levelled by "
            f"importers/replica.py (guard is 5.0 deg), so its frame is the raw "
            f"capture axes; got {arts.frame.kind!r}")
    edges, _ = DirectionalExtractor().extract(arts, DirectionalConfig(mode="sparse"))
    bad = sorted({e.frame for e in edges} - {"world"})
    if bad or not edges:
        raise AssertionError(
            f"v1-path edges should all say 'world' (n={len(edges)}, off={bad!r})")


TESTS = [
    test_every_extractor_labels_edges_with_the_bundle_frame,
    test_extractor_roster_is_complete,
    test_built_bundle_edges_agree_with_the_bundle_frame,
    test_frame_label_survives_a_serde_round_trip,
    test_real_replica_import_declares_and_labels_scene_canonical,
    test_oracle_replica_v1_path_still_reports_world,
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
