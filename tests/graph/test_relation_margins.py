"""Calibration-margin tests: every relation extractor's `emit_margins` flag.

Background. Every extractor in graph/relations computes a continuous quantity
and then thresholds it to a boolean; until now the continuous value was
discarded and every Edge got confidence=1.0, so the system had nothing to
threshold on and no risk-coverage curve. `emit_margins` (default False, per
extractor config) keeps the value: Edge.confidence becomes
margin_confidence(<normalized margin>).

What this suite proves:
  1. ADDITIVE ONLY. With emit_margins=False (the default) every extractor is
     bit-identical to before: same edges in the same order with the same ids
     and evidence, confidence EXACTLY 1.0, and no margin key anywhere in
     rejection evidence. Also: the flag is hash_omit_if_default on all nine
     configs, so default GraphBuilder bundle hashes — and therefore the frozen
     Replica reproductions — cannot move.
  2. The squash is well-formed: range (0,1), the documented anchors, monotone,
     and total on inf/-inf/NaN.
  3. THE RELAXATION INVARIANT, which is what makes these numbers safe to
     threshold on: 0.5 IS the decision boundary. Every emitted edge scores
     >= 0.5 and every measured rejection <= 0.5, on all nine extractors, so
     the score never disagrees with the gate that produced it and any other
     cut trades coverage for risk along a curve that did not previously exist.
     Equality is reserved for a measurement sitting exactly on a threshold —
     see test_exact_axis_tie_scores_a_coin_flip, which is a feature: the
     legacy directional tie-break resolves |dx| == |dy| in favour of x
     arbitrarily, and that arbitrariness is now visible instead of hidden
     behind a hardcoded 1.0.
  4. Confidence is monotone in "how comfortably the test passed" — a nearer
     pair, a flusher contact, a less ambiguous axis all score higher.

EdgeRejection has NO confidence field (graph/schema.py), and graph/schema.py is
out of scope for this change, so rejection margins ride in
evidence["margin_confidence"] instead. Adding `confidence: float = 1.0` to the
EdgeRejection dataclass would let them move onto a real field; nothing else
would need to change, since evidence is already free-form and every producer
here is in graph/relations.

Run: python tests/graph/test_relation_margins.py
"""
from __future__ import annotations

import dataclasses
import math
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.base import ReconstructionConfig
from adapters.oracle_replica import OracleReplicaAdapter, build_replica_capture_bundle
from common.types import Plane, SceneFrame
from extractors.base import (
    EntityArtifact, EntityArtifacts, EntityIdentity, ExtractionDiagnostics,
    InstanceExtractorConfig, SemanticHypothesis, StructuralSurface,
)
from extractors.oracle_replica import OracleReplicaExtractor
from extractors.serde import CURRENT_SCHEMA_VERSION as ENT_SCHEMA_VERSION
from graph.builder import ExtractorRun, _config_hash_payload, build_graph
from graph.relations.attached_to import AttachedToConfig, AttachedToExtractor
from graph.relations.attached_to_v2 import (
    AttachedToV2Config, AttachedToV2Extractor,
)
from graph.relations.base import band_margin, margin_confidence, ratio_margin
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
from representations.mesh import MeshRepresentation


REPLICA_SCENE_DIR = REPO_ROOT / "scenes" / "replica_room_0"
REPLICA_V2_DIR = REPLICA_SCENE_DIR / "enriched" / "v2"

MARGIN_KEY = "margin_confidence"


# --- fixtures ------------------------------------------------------------


def _frame() -> SceneFrame:
    return SceneFrame(
        gravity=(0.0, 0.0, -1.0), canonical_forward=None,
        canonical_right=None, units="meters", notes="",
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


def _artifacts(entities, surfaces) -> EntityArtifacts:
    return EntityArtifacts(
        schema_version=ENT_SCHEMA_VERSION, bundle_hash="ent_t", scene_id="t",
        frame=_frame(), representation_hash="rep_t",
        extractor_name="test", extractor_version="0.0",
        entities=entities, structural_surfaces=surfaces,
        geometry_store_path=None,
        diagnostics=ExtractionDiagnostics(
            n_entities=len(entities), n_structural_surfaces=len(surfaces),
            runtime_seconds=0.0, coverage_score=None, notes="",
        ),
        notes={},
    )


def _scene() -> EntityArtifacts:
    """One synthetic room that gives ALL nine extractors something to emit:
    a floor, a wall, a table with a book on it, a wall-mounted elevated
    sconce, and a floor cabinet shoved against the wall (the ATTACHED_TO
    near-miss)."""
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
    return _artifacts(entities, [floor, wall])


def _runs(emit: bool):
    """(label, extractor, config) for all nine extractors. The battery-path
    policy flags are on so the exercised code paths match production."""
    return [
        ("directional", DirectionalExtractor(),
         DirectionalConfig(mode="sparse", emit_margins=emit)),
        ("directional_compat", DirectionalExtractor(),
         DirectionalConfig(mode="compat", emit_margins=emit)),
        ("proximity", ProximityExtractor(),
         ProximityConfig(mode="sparse", emit_margins=emit)),
        ("proximity_compat", ProximityExtractor(),
         ProximityConfig(mode="compat", emit_margins=emit)),
        ("proximity_sparse_v2", ProximityExtractor(),
         ProximityConfig(mode="sparse", sparse_version=2, emit_margins=emit)),
        ("near_surface", SurfaceProximityExtractor(),
         SurfaceProximityConfig(emit_margins=emit)),
        ("near_surface_polygon", SurfaceProximityExtractor(),
         SurfaceProximityConfig(use_polygon_clip=True,
                                exclude_room_scale_flat=True,
                                emit_margins=emit)),
        ("on_surface", OnSurfaceExtractor(), OnSurfaceConfig(emit_margins=emit)),
        ("contacts_surface", ContactsSurfaceExtractor(),
         ContactsSurfaceConfig(exclude_room_scale_flat=True, emit_margins=emit)),
        ("on_entity_surface", OnEntitySurfaceExtractor(),
         OnEntitySurfaceConfig(emit_margins=emit)),
        ("attached_to", AttachedToExtractor(),
         AttachedToConfig(emit_margins=emit)),
        ("attached_to_v2", AttachedToV2Extractor(),
         AttachedToV2Config(emit_margins=emit)),
        ("on_entity_surface_v2", OnEntitySurfaceV2Extractor(),
         OnEntitySurfaceV2Config(emit_margins=emit)),
    ]


def _replica_present() -> bool:
    return (REPLICA_V2_DIR / "scene_graph.json").exists()


def _replica_artifacts() -> EntityArtifacts:
    capture = build_replica_capture_bundle(REPLICA_SCENE_DIR)
    repr_bundle = OracleReplicaAdapter().reconstruct(
        capture,
        ReconstructionConfig(name="oracle_replica", version="0.1", params={}),
    )
    return OracleReplicaExtractor(enriched_v2_path=REPLICA_V2_DIR).extract(
        MeshRepresentation(bundle=repr_bundle),
        InstanceExtractorConfig(name="oracle_replica", version="0.1", params={}),
    )


# --- 1. additive only ----------------------------------------------------


def _assert_bit_identical(artifacts: EntityArtifacts, where: str) -> None:
    for (label, ex, cfg_off), (_, ex2, cfg_on) in zip(_runs(False), _runs(True)):
        off_edges, off_diag = ex.extract(artifacts, cfg_off)
        on_edges, on_diag = ex2.extract(artifacts, cfg_on)

        for e in off_edges:
            if e.confidence != 1.0:
                raise AssertionError(
                    f"{where}/{label}: emit_margins=False must leave "
                    f"confidence EXACTLY 1.0, got {e.confidence!r}")
        for r in off_diag.rejection_samples:
            if MARGIN_KEY in r.evidence:
                raise AssertionError(
                    f"{where}/{label}: emit_margins=False leaked {MARGIN_KEY} "
                    "into rejection evidence")

        # The margin must be purely additive: identical edge sequence, ids,
        # types, endpoints, weights, provenance and evidence.
        if len(off_edges) != len(on_edges):
            raise AssertionError(
                f"{where}/{label}: edge COUNT moved with the flag "
                f"({len(off_edges)} -> {len(on_edges)})")
        for a, b in zip(off_edges, on_edges):
            if (a.edge_id, a.source, a.type, a.target, a.frame, a.weight,
                    a.extractor, a.extractor_version) != (
                    b.edge_id, b.source, b.type, b.target, b.frame, b.weight,
                    b.extractor, b.extractor_version):
                raise AssertionError(
                    f"{where}/{label}: edge identity moved with the flag")
            if {k: v for k, v in a.evidence.items() if k != MARGIN_KEY} != \
                    {k: v for k, v in b.evidence.items() if k != MARGIN_KEY}:
                raise AssertionError(
                    f"{where}/{label}: edge evidence moved with the flag")
        if off_diag.rejections_per_type != on_diag.rejections_per_type:
            raise AssertionError(
                f"{where}/{label}: rejection counts moved with the flag")
        for a, b in zip(off_diag.rejection_samples, on_diag.rejection_samples):
            if (a.source, a.type, a.target, a.rejected_reason) != \
                    (b.source, b.type, b.target, b.rejected_reason):
                raise AssertionError(
                    f"{where}/{label}: rejection identity moved with the flag")


def test_default_off_is_bit_identical_synthetic() -> None:
    _assert_bit_identical(_scene(), "synthetic")


def test_default_off_is_bit_identical_on_replica() -> None:
    if not _replica_present():
        print("  SKIP (enriched v2 artifact not on disk)")
        return
    _assert_bit_identical(_replica_artifacts(), "replica_room_0")


def test_every_config_omits_default_emit_margins_from_hash() -> None:
    """hash_omit_if_default on all nine configs — this is what keeps default
    bundle hashes, and therefore the frozen Replica reproductions, put."""
    seen = set()
    for label, _ex, cfg in _runs(False):
        name = type(cfg).__name__
        if name in seen:
            continue
        seen.add(name)
        payload = _config_hash_payload(cfg)
        if "emit_margins" in payload:
            raise AssertionError(
                f"{name}: default payload must omit emit_margins, got "
                f"{sorted(payload)}")
        on = _config_hash_payload(dataclasses.replace(cfg, emit_margins=True))
        if on.get("emit_margins") is not True:
            raise AssertionError(
                f"{name}: emit_margins=True must reach the hash payload, else "
                "a calibrated run would share a bundle_hash with a frozen one")
    if len(seen) != 9:
        raise AssertionError(f"expected 9 distinct configs, saw {sorted(seen)}")


def test_builder_bundle_hash_unmoved_by_default_flag() -> None:
    """Belt and braces at the builder level: default vs explicit-False must
    hash the same, and True must hash differently."""
    artifacts = _scene()

    def _hash(cfg) -> str:
        bundle, _ = build_graph(
            artifacts, [ExtractorRun(OnSurfaceExtractor(), cfg)],
            density_policy="phase2_telemetry_only")
        return bundle.bundle_hash

    if _hash(OnSurfaceConfig()) != _hash(OnSurfaceConfig(emit_margins=False)):
        raise AssertionError("bundle_hash drifted on an explicit-False flag")
    if _hash(OnSurfaceConfig()) == _hash(OnSurfaceConfig(emit_margins=True)):
        raise AssertionError(
            "emit_margins=True must change bundle_hash; a calibrated bundle "
            "sharing a hash with a frozen one is worse than the drift risk")


# --- 2. the squash itself ------------------------------------------------


def test_margin_confidence_anchors_and_range() -> None:
    if abs(margin_confidence(0.0) - 0.5) > 1e-12:
        raise AssertionError("margin 0 (on the boundary) must map to 0.5")
    if not (0.98 < margin_confidence(1.0) < 0.983):
        raise AssertionError(
            f"margin +1 anchor moved: {margin_confidence(1.0)!r}")
    if not (0.017 < margin_confidence(-1.0) < 0.02):
        raise AssertionError(
            f"margin -1 anchor moved: {margin_confidence(-1.0)!r}")
    for m in (-1e6, -12.5, -1.0, -0.01, 0.0, 0.01, 1.0, 12.5, 1e6,
              math.inf, -math.inf):
        c = margin_confidence(m)
        if not (0.0 <= c <= 1.0):
            raise AssertionError(f"confidence {c!r} out of [0,1] for {m!r}")
    if margin_confidence(math.inf) != 1.0:
        raise AssertionError("inf (a saturated clause) must read 1.0")
    if margin_confidence(-math.inf) > 1e-9:
        raise AssertionError("-inf must read ~0.0")
    if margin_confidence(math.nan) != 0.5:
        raise AssertionError("NaN (no information) must read 0.5")


def test_margin_confidence_strictly_monotone() -> None:
    xs = [-4.0, -2.0, -0.5, -0.01, 0.0, 0.01, 0.5, 2.0, 4.0]
    ys = [margin_confidence(x) for x in xs]
    for a, b in zip(ys, ys[1:]):
        if not b > a:
            raise AssertionError(f"not monotone: {ys}")


def test_margin_helpers_sign_convention() -> None:
    """>= 0 iff the underlying test passed — the property every extractor's
    relaxation invariant is built on."""
    if ratio_margin(0.0, 1.0) != 1.0 or ratio_margin(1.0, 1.0) != 0.0:
        raise AssertionError("ratio_margin anchors wrong")
    if ratio_margin(1.5, 1.0) >= 0.0:
        raise AssertionError("ratio_margin must be negative past threshold")
    if ratio_margin(9.9, math.inf) != math.inf:
        raise AssertionError("a vacuous threshold must saturate, not NaN")
    if band_margin(0.0, -1.0, 1.0) != 1.0:
        raise AssertionError("band centre must be +1")
    if band_margin(-1.0, -1.0, 1.0) != 0.0 or band_margin(1.0, -1.0, 1.0) != 0.0:
        raise AssertionError("band edges must be 0")
    if band_margin(2.0, -1.0, 1.0) >= 0.0:
        raise AssertionError("outside the band must be negative")
    if band_margin(0.0, 1.0, 1.0) != math.inf:
        raise AssertionError("a zero-width band has no scale; must saturate")


# --- 3. the relaxation invariant -----------------------------------------


def _assert_relaxation(artifacts: EntityArtifacts, where: str) -> None:
    total_edges = 0
    total_rejections = 0
    for label, ex, cfg in _runs(True):
        edges, diag = ex.extract(artifacts, cfg)
        for e in edges:
            total_edges += 1
            if not (0.5 <= e.confidence <= 1.0):
                raise AssertionError(
                    f"{where}/{label}: emitted edge {e.edge_id} has "
                    f"confidence {e.confidence!r}; an emitted edge below 0.5 "
                    "means the score disagrees with the gate that produced it")
        for r in diag.rejection_samples:
            if MARGIN_KEY not in r.evidence:
                continue      # policy rejection: measures nothing comparable
            total_rejections += 1
            m = r.evidence[MARGIN_KEY]
            if not (0.0 <= m <= 0.5):
                raise AssertionError(
                    f"{where}/{label}: rejection ({r.rejected_reason}) scored "
                    f"{m!r}; a rejected pair above 0.5 means the score "
                    "disagrees with the gate that rejected it")
    if total_edges == 0:
        raise AssertionError(f"{where}: fixture emitted no edges at all")
    if total_rejections == 0:
        raise AssertionError(f"{where}: fixture produced no scored rejections")


def test_relaxation_invariant_synthetic() -> None:
    _assert_relaxation(_scene(), "synthetic")


def test_relaxation_invariant_on_replica() -> None:
    if not _replica_present():
        print("  SKIP (enriched v2 artifact not on disk)")
        return
    _assert_relaxation(_replica_artifacts(), "replica_room_0")


def test_confidence_spreads_rather_than_collapsing() -> None:
    """The whole point is a usable continuous score. If a relation's edges all
    land on one value there is still nothing to threshold on."""
    artifacts = _replica_artifacts() if _replica_present() else _scene()
    for label, ex, cfg in _runs(True):
        if label != "directional":
            continue
        edges, _ = ex.extract(artifacts, cfg)
        values = {round(e.confidence, 6) for e in edges}
        if len(values) < 10:
            raise AssertionError(
                f"{label}: only {len(values)} distinct confidences over "
                f"{len(edges)} edges — the score collapsed")


# --- 4. monotone in "how comfortably the test passed" --------------------


def test_near_confidence_falls_with_distance() -> None:
    surfaces = _scene().structural_surfaces
    last = 1.1
    for dx in (0.05, 0.2, 0.4, 0.6, 0.8, 0.95):
        arts = _artifacts([
            _entity("a", (0.0, 0.0, 0.0), (0.01, 0.01, 0.01)),
            _entity("b", (dx, 0.0, 0.0), (dx + 0.01, 0.01, 0.01)),
        ], surfaces)
        edges, _ = ProximityExtractor().extract(
            arts, ProximityConfig(mode="sparse", emit_margins=True))
        if len(edges) != 1:
            raise AssertionError(f"expected one NEAR at dx={dx}")
        if not edges[0].confidence < last:
            raise AssertionError(
                f"NEAR confidence not decreasing in distance at dx={dx}: "
                f"{edges[0].confidence!r} !< {last!r}")
        last = edges[0].confidence


def test_on_surface_confidence_falls_as_the_box_lifts() -> None:
    """Same relation, same surface: a box resting flush must outscore one
    hovering 19 mm up, still inside the 0.02 m contact threshold."""
    floor = _scene().structural_surfaces[0]
    last = 1.1
    for gap in (0.0, 0.005, 0.010, 0.015, 0.019):
        arts = _artifacts(
            [_entity("b", (0.0, 0.0, gap), (0.3, 0.3, gap + 0.3))], [floor])
        edges, _ = OnSurfaceExtractor().extract(
            arts, OnSurfaceConfig(emit_margins=True))
        if len(edges) != 1:
            raise AssertionError(f"expected one ON_SURFACE at gap={gap}")
        if not edges[0].confidence < last:
            raise AssertionError(
                f"ON_SURFACE confidence not decreasing in lift at gap={gap}: "
                f"{edges[0].confidence!r} !< {last!r}")
        last = edges[0].confidence


def test_directional_confidence_falls_as_axes_tie() -> None:
    """A pair separated cleanly along x must outscore one whose x barely beats
    its y — that ambiguity is invisible in a `dominant >= min_delta` test."""
    surfaces = _scene().structural_surfaces
    last = 1.1
    # dy must exceed sparse_min_delta (0.5) before the runner-up axis becomes
    # the binding competitor; below it the threshold binds and the score is
    # flat, which is correct.
    for dy in (0.5, 0.7, 0.9, 1.1, 1.19):
        arts = _artifacts([
            _entity("a", (0.0, 0.0, 0.0), (0.01, 0.01, 0.01)),
            _entity("b", (1.2, dy, 0.0), (1.21, dy + 0.01, 0.01)),
        ], surfaces)
        edges, _ = DirectionalExtractor().extract(
            arts, DirectionalConfig(mode="sparse", emit_margins=True))
        if len(edges) != 1 or edges[0].type != "LEFT_OF":
            raise AssertionError(f"expected one LEFT_OF at dy={dy}")
        if not edges[0].confidence < last:
            raise AssertionError(
                f"directional confidence not decreasing as axes tie at "
                f"dy={dy}: {edges[0].confidence!r} !< {last!r}")
        last = edges[0].confidence


def test_exact_axis_tie_scores_a_coin_flip() -> None:
    """|dx| == |dy| exactly: compat's tie-break picks x arbitrarily. That edge
    is a coin flip and must score exactly 0.5 — the one case where an emitted
    edge sits ON the boundary rather than above it, and precisely the signal
    a hardcoded confidence=1.0 was hiding."""
    arts = _artifacts([
        _entity("a", (0.0, 0.0, 0.0), (0.01, 0.01, 0.01)),
        _entity("b", (2.0, 2.0, 0.0), (2.01, 2.01, 0.01)),
    ], _scene().structural_surfaces)
    edges, _ = DirectionalExtractor().extract(
        arts, DirectionalConfig(mode="compat", emit_margins=True))
    if not edges:
        raise AssertionError("expected compat directional edges for the tie")
    for e in edges:
        if abs(e.evidence["dx"]) != abs(e.evidence["dy"]):
            continue
        if e.confidence != 0.5:
            raise AssertionError(
                f"a perfect axis tie must score exactly 0.5, got "
                f"{e.confidence!r}")
        return
    raise AssertionError("fixture produced no exact axis tie")


def test_attached_to_scores_the_floor_gate_not_just_the_wall() -> None:
    """The cabinet against the wall is a floor-supported near-miss: rejected,
    and its margin must be recorded so it lands in the calibration set."""
    edges, diag = AttachedToExtractor().extract(
        _scene(), AttachedToConfig(emit_margins=True))
    emitted = {e.source.uid for e in edges}
    if "sconce" not in emitted:
        raise AssertionError("elevated sconce must still be ATTACHED_TO")
    if "cabinet" in emitted:
        raise AssertionError("floor-supported cabinet must still be rejected")
    near_miss = [
        r for r in diag.rejection_samples
        if r.source.uid == "cabinet"
        and r.rejected_reason == "attached_to_floor_supported"
    ]
    if not near_miss:
        raise AssertionError("expected a floor_supported rejection for cabinet")
    if MARGIN_KEY not in near_miss[0].evidence:
        raise AssertionError(
            "the floor-supported near-miss carries no margin; near-misses are "
            "half the calibration data")


TESTS = [
    test_default_off_is_bit_identical_synthetic,
    test_default_off_is_bit_identical_on_replica,
    test_every_config_omits_default_emit_margins_from_hash,
    test_builder_bundle_hash_unmoved_by_default_flag,
    test_margin_confidence_anchors_and_range,
    test_margin_confidence_strictly_monotone,
    test_margin_helpers_sign_convention,
    test_relaxation_invariant_synthetic,
    test_relaxation_invariant_on_replica,
    test_confidence_spreads_rather_than_collapsing,
    test_near_confidence_falls_with_distance,
    test_on_surface_confidence_falls_as_the_box_lifts,
    test_directional_confidence_falls_as_axes_tie,
    test_exact_axis_tie_scores_a_coin_flip,
    test_attached_to_scores_the_floor_gate_not_just_the_wall,
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
