"""P2.08 tests: sparse-v2 proximity using exact AABB surface distance.

Two contracts:
  1. Sparse v1 is byte-frozen — same edge set as Phase 1 when sparse_version
     is omitted (default = 1). Real Replica reproduction is checked in the
     Phase 1 gate; this suite locks in:
       - default ProximityConfig still selects v1,
       - v1 edge extractor_version stays "0.1",
       - v1 evidence shape unchanged.
  2. Sparse v2 is the new path:
       - uses aabb_to_aabb_surface (P2.07),
       - records distance_metric="aabb_surface" and sparse_version=2,
       - emits edges with extractor_version=SPARSE_V2_VERSION,
       - threshold semantics: strict-less-than against surface distance.

Run: python tests/relations/test_proximity_sparse_v2.py
"""
from __future__ import annotations

import sys
import traceback
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.types import OrientedBBox, SceneFrame
from extractors.base import (
    EntityArtifact, EntityArtifacts, EntityIdentity, ExtractionDiagnostics,
    SemanticHypothesis,
)
from extractors.serde import CURRENT_SCHEMA_VERSION as ENT_SCHEMA_VERSION
from graph.relations.proximity import (
    SPARSE_V2_VERSION,
    ProximityConfig,
    ProximityExtractor,
    extract_sparse_v2,
)


def _frame() -> SceneFrame:
    return SceneFrame(
        gravity=(0.0, 0.0, -1.0),
        canonical_forward=None, canonical_right=None,
        units="meters", notes="",
    )


def _entity(uid: str, lo: tuple, hi: tuple) -> EntityArtifact:
    cx = (lo[0] + hi[0]) / 2
    cy = (lo[1] + hi[1]) / 2
    cz = (lo[2] + hi[2]) / 2
    return EntityArtifact(
        identity=EntityIdentity(
            object_uid=uid, display_label=uid, aliases=[], source_instance_ref=uid,
        ),
        bbox_aabb=(lo, hi),
        bbox_obb=None,
        centroid=(cx, cy, cz),
        geometry_handle=None,
        semantic_hypotheses=[SemanticHypothesis(label=uid, confidence=1.0, source="t")],
        embedding=None,
        extraction_diagnostics={},
    )


def _artifacts(entities: list[EntityArtifact]) -> EntityArtifacts:
    return EntityArtifacts(
        schema_version=ENT_SCHEMA_VERSION, bundle_hash="ent_test", scene_id="t",
        frame=_frame(), representation_hash="rep_test",
        extractor_name="test", extractor_version="0.0",
        entities=entities, structural_surfaces=[],
        geometry_store_path=None,
        diagnostics=ExtractionDiagnostics(
            n_entities=len(entities), n_structural_surfaces=0,
            runtime_seconds=0.0, coverage_score=None, notes="",
        ),
        notes={},
    )


# --- sparse v1 byte-frozen ------------------------------------------------


def test_default_sparse_config_selects_v1() -> None:
    cfg = ProximityConfig(mode="sparse")
    if cfg.sparse_version != 1:
        raise AssertionError(
            f"default sparse_version drifted from 1: {cfg.sparse_version!r}"
        )


def test_sparse_v1_edges_have_extractor_version_0_1() -> None:
    artifacts = _artifacts([
        _entity("a", (0.0, 0.0, 0.0), (0.4, 0.4, 0.4)),
        _entity("b", (0.5, 0.0, 0.0), (0.9, 0.4, 0.4)),
    ])
    edges, diag = ProximityExtractor().extract(
        artifacts, ProximityConfig(mode="sparse"),
    )
    if diag.version != "0.1":
        raise AssertionError(f"v1 diag.version drifted: {diag.version!r}")
    for e in edges:
        if e.extractor_version != "0.1":
            raise AssertionError(f"v1 edge extractor_version drifted: {e.extractor_version!r}")
        if "distance_metric" in e.evidence:
            raise AssertionError(
                f"v1 edge unexpectedly carries distance_metric: {e.evidence!r}"
            )
        if e.evidence.get("symmetric") is not True:
            raise AssertionError(f"v1 sparse symmetric flag missing: {e.evidence!r}")


# --- sparse v2 opt-in -----------------------------------------------------


def test_explicit_sparse_v2_selects_v2_extractor() -> None:
    artifacts = _artifacts([
        _entity("a", (0.0, 0.0, 0.0), (0.4, 0.4, 0.4)),
        _entity("b", (0.5, 0.0, 0.0), (0.9, 0.4, 0.4)),
    ])
    _edges, diag = ProximityExtractor().extract(
        artifacts, ProximityConfig(mode="sparse", sparse_version=2),
    )
    if diag.version != SPARSE_V2_VERSION:
        raise AssertionError(f"v2 diag.version not truthful: {diag.version!r}")


def test_unknown_sparse_version_raises() -> None:
    artifacts = _artifacts([_entity("a", (0.0,)*3, (1.0,)*3)])
    try:
        ProximityExtractor().extract(
            artifacts, ProximityConfig(mode="sparse", sparse_version=3),  # type: ignore[arg-type]
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError on unknown sparse_version")


# --- v2 math contract -----------------------------------------------------


def test_v2_uses_aabb_surface_distance_for_threshold() -> None:
    """Two unit cubes 0.3m apart surface-to-surface; centroids 1.3m apart.
    At threshold 1.0:
      - v1 (centroid 1.3 >= 1.0) → no edge
      - v2 (surface 0.3 < 1.0) → 1 edge"""
    artifacts = _artifacts([
        _entity("a", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        _entity("b", (1.3, 0.0, 0.0), (2.3, 1.0, 1.0)),
    ])
    v1_edges, _ = ProximityExtractor().extract(
        artifacts, ProximityConfig(mode="sparse", sparse_near_threshold=1.0),
    )
    v2_edges, _ = ProximityExtractor().extract(
        artifacts, ProximityConfig(
            mode="sparse", sparse_version=2, sparse_near_threshold=1.0,
        ),
    )
    if len(v1_edges) != 0:
        raise AssertionError(f"v1 unexpectedly produced edges: {len(v1_edges)}")
    if len(v2_edges) != 1:
        raise AssertionError(f"v2 expected 1 edge, got {len(v2_edges)}")
    dist = v2_edges[0].evidence["distance_m"]
    if abs(dist - 0.3) > 1e-9:
        raise AssertionError(f"v2 distance wrong: {dist}")


def test_v2_edge_evidence_records_distance_metric_and_version() -> None:
    artifacts = _artifacts([
        _entity("a", (0.0,)*3, (1.0,)*3),
        _entity("b", (1.3, 0.0, 0.0), (2.3, 1.0, 1.0)),
    ])
    edges, _ = ProximityExtractor().extract(
        artifacts, ProximityConfig(mode="sparse", sparse_version=2),
    )
    for e in edges:
        ev = e.evidence
        if ev.get("distance_metric") != "aabb_surface":
            raise AssertionError(
                f"v2 evidence missing distance_metric: {ev!r}"
            )
        if ev.get("sparse_version") != 2:
            raise AssertionError(f"v2 evidence missing sparse_version: {ev!r}")
        if ev.get("symmetric") is not True:
            raise AssertionError(f"v2 symmetric flag missing: {ev!r}")
        if "distance_m" not in ev:
            raise AssertionError(f"v2 evidence missing distance_m: {ev!r}")


def test_v2_edge_extractor_version_is_truthful() -> None:
    artifacts = _artifacts([
        _entity("a", (0.0,)*3, (1.0,)*3),
        _entity("b", (1.3, 0.0, 0.0), (2.3, 1.0, 1.0)),
    ])
    edges, _ = ProximityExtractor().extract(
        artifacts, ProximityConfig(mode="sparse", sparse_version=2),
    )
    for e in edges:
        if e.extractor_version != SPARSE_V2_VERSION:
            raise AssertionError(
                f"v2 edge extractor_version not truthful: {e.extractor_version!r}"
            )


def test_v2_threshold_is_strict_less_than() -> None:
    """Surface distance exactly == threshold ⇒ no edge."""
    artifacts = _artifacts([
        _entity("a", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        _entity("b", (2.0, 0.0, 0.0), (3.0, 1.0, 1.0)),  # surface gap = 1.0
    ])
    edges, _ = ProximityExtractor().extract(
        artifacts, ProximityConfig(
            mode="sparse", sparse_version=2, sparse_near_threshold=1.0,
        ),
    )
    if len(edges) != 0:
        raise AssertionError(
            f"strict less-than violated; got {len(edges)} edges at distance==threshold"
        )


def test_v2_overlapping_boxes_have_distance_zero_and_emit() -> None:
    artifacts = _artifacts([
        _entity("a", (0.0,)*3, (2.0,)*3),
        _entity("b", (1.0,)*3, (3.0,)*3),
    ])
    edges, _ = ProximityExtractor().extract(
        artifacts, ProximityConfig(
            mode="sparse", sparse_version=2, sparse_near_threshold=0.1,
        ),
    )
    if len(edges) != 1:
        raise AssertionError(f"overlap should emit 1 NEAR; got {len(edges)}")
    if abs(edges[0].evidence["distance_m"]) > 1e-12:
        raise AssertionError(f"overlap distance should be 0: {edges[0].evidence!r}")


def test_v2_rejections_recorded_with_v2_metadata() -> None:
    artifacts = _artifacts([
        _entity("a", (0.0,)*3, (1.0,)*3),
        _entity("b", (10.0, 0.0, 0.0), (11.0, 1.0, 1.0)),
    ])
    _edges, diag = ProximityExtractor().extract(
        artifacts, ProximityConfig(mode="sparse", sparse_version=2),
    )
    if diag.rejections_per_type.get("NEAR", 0) != 1:
        raise AssertionError(f"v2 rejection count wrong: {diag.rejections_per_type!r}")
    if not diag.rejection_samples:
        raise AssertionError("v2 rejection samples missing")
    sample_ev = diag.rejection_samples[0].evidence
    if sample_ev.get("distance_metric") != "aabb_surface":
        raise AssertionError(f"rejection sample missing distance_metric: {sample_ev!r}")
    if sample_ev.get("sparse_version") != 2:
        raise AssertionError(f"rejection sample missing sparse_version: {sample_ev!r}")


def test_v2_is_deterministic() -> None:
    artifacts = _artifacts([
        _entity("a", (0.0,)*3, (1.0,)*3),
        _entity("b", (1.5,)*3, (2.5,)*3),
        _entity("c", (3.0,)*3, (4.0,)*3),
    ])
    cfg = ProximityConfig(mode="sparse", sparse_version=2)
    e1, _ = ProximityExtractor().extract(artifacts, cfg)
    e2, _ = ProximityExtractor().extract(artifacts, cfg)
    if len(e1) != len(e2):
        raise AssertionError("v2 non-deterministic count")
    for a, b in zip(e1, e2):
        if a.edge_id != b.edge_id or a.evidence != b.evidence:
            raise AssertionError("v2 non-deterministic content")


# --- v1 vs v2 boundary behavior on real-ish geometry ----------------------


def test_v2_yields_at_least_as_many_edges_as_v1_at_same_threshold() -> None:
    """A2: surface distance ≤ centroid distance, so at a fixed threshold
    v2 NEAR is a superset of v1 NEAR. Multi-entity sanity check."""
    artifacts = _artifacts([
        _entity("a", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        _entity("b", (1.5, 0.0, 0.0), (2.5, 1.0, 1.0)),  # surf 0.5; centroid 1.5
        _entity("c", (3.0, 0.0, 0.0), (4.0, 1.0, 1.0)),  # vs b: surf 0.5; centroid 1.5
        _entity("d", (10.0, 0.0, 0.0), (11.0, 1.0, 1.0)),  # far
    ])
    v1_edges, _ = ProximityExtractor().extract(
        artifacts, ProximityConfig(mode="sparse", sparse_near_threshold=1.0),
    )
    v2_edges, _ = ProximityExtractor().extract(
        artifacts, ProximityConfig(
            mode="sparse", sparse_version=2, sparse_near_threshold=1.0,
        ),
    )
    if len(v2_edges) < len(v1_edges):
        raise AssertionError(
            f"v2 edges ({len(v2_edges)}) < v1 edges ({len(v1_edges)}); "
            "surface ≤ centroid invariant violated"
        )


def test_extract_sparse_v2_callable_directly() -> None:
    """Module-level callable should be usable without the dispatcher."""
    ents = [
        _entity("a", (0.0,)*3, (1.0,)*3),
        _entity("b", (1.5, 0.0, 0.0), (2.5, 1.0, 1.0)),
    ]
    edges, counts, rejections, rej_counts = extract_sparse_v2(
        ents, extractor="proximity", frame="world", sparse_near_threshold=1.0,
    )
    if counts["NEAR"] != 1:
        raise AssertionError(f"direct call NEAR count wrong: {counts!r}")
    for e in edges:
        if e.extractor_version != SPARSE_V2_VERSION:
            raise AssertionError(f"direct call edge version wrong: {e.extractor_version!r}")


TESTS = [
    test_default_sparse_config_selects_v1,
    test_sparse_v1_edges_have_extractor_version_0_1,
    test_explicit_sparse_v2_selects_v2_extractor,
    test_unknown_sparse_version_raises,
    test_v2_uses_aabb_surface_distance_for_threshold,
    test_v2_edge_evidence_records_distance_metric_and_version,
    test_v2_edge_extractor_version_is_truthful,
    test_v2_threshold_is_strict_less_than,
    test_v2_overlapping_boxes_have_distance_zero_and_emit,
    test_v2_rejections_recorded_with_v2_metadata,
    test_v2_is_deterministic,
    test_v2_yields_at_least_as_many_edges_as_v1_at_same_threshold,
    test_extract_sparse_v2_callable_directly,
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
