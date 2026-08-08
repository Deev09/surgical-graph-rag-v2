"""P7.01 tests: AttachedToExtractor (wall-mounted object gate).

Run: python tests/relations/test_attached_to.py
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.base import ReconstructionConfig
from adapters.oracle_replica import OracleReplicaAdapter, build_replica_capture_bundle
from common.types import Plane, SceneFrame
from demo.replica_habitat_import import import_habitat_room
from extractors.base import (
    EntityArtifact, EntityArtifacts, EntityIdentity, ExtractionDiagnostics,
    InstanceExtractorConfig, SemanticHypothesis, StructuralSurface,
)
from extractors.oracle_replica import OracleReplicaExtractor
from extractors.serde import CURRENT_SCHEMA_VERSION as ENT_SCHEMA_VERSION
from graph.builder import ExtractorRun, build_graph
from graph.relations.attached_to import (
    ATTACHED_TO_VERSION,
    AttachedToConfig,
    AttachedToExtractor,
)
from representations.mesh import MeshRepresentation


SMOKE_FIXTURE = REPO_ROOT / "eval" / "questions" / "phase7_attachment_smoke.json"
REPLICA_ROOM0 = REPO_ROOT / "scenes" / "replica_room_0"
REPLICA_ROOM0_V2 = REPLICA_ROOM0 / "enriched" / "v2"
APARTMENT0 = Path.home() / "Desktop/datasets/replica/apartment_0"


def _frame() -> SceneFrame:
    return SceneFrame(
        gravity=(0.0, 0.0, -1.0), canonical_forward=None,
        canonical_right=None, units="meters", notes="",
    )


def _entity(uid: str, lo, hi, centroid) -> EntityArtifact:
    return EntityArtifact(
        identity=EntityIdentity(
            object_uid=uid, display_label=uid, aliases=[], source_instance_ref=uid,
        ),
        bbox_aabb=(lo, hi), bbox_obb=None, centroid=centroid,
        geometry_handle=None,
        semantic_hypotheses=[SemanticHypothesis(label=uid, confidence=1.0, source="t")],
        embedding=None, extraction_diagnostics={},
    )


def _surface(uid: str, spec: dict) -> StructuralSurface:
    p = spec["plane"]
    return StructuralSurface(
        surface_uid=uid,
        surface_type=spec["surface_type"],
        plane=Plane(a=p["a"], b=p["b"], c=p["c"], d=p["d"]),
        polygon=[tuple(v) for v in spec["polygon"]],
        confidence=1.0,
        source=spec["source"],
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


def _load_fixture() -> dict:
    return json.loads(SMOKE_FIXTURE.read_text(encoding="utf-8"))


def _case_entity(case: dict) -> EntityArtifact:
    mn = case["entity_aabb"]["min"]
    mx = case["entity_aabb"]["max"]
    return _entity(
        case["entity_uid"], tuple(mn), tuple(mx), tuple(case["entity_centroid"]),
    )


def _build_room0_artifacts() -> EntityArtifacts:
    capture = build_replica_capture_bundle(REPLICA_ROOM0)
    representation = MeshRepresentation(
        bundle=OracleReplicaAdapter().reconstruct(
            capture, ReconstructionConfig(name="oracle_replica", version="0.1", params={}),
        ),
    )
    return OracleReplicaExtractor(enriched_v2_path=REPLICA_ROOM0_V2).extract(
        representation,
        InstanceExtractorConfig(name="oracle_replica", version="0.1", params={}),
    )


def test_config_defaults_match_smoke_fixture() -> None:
    d = _load_fixture()["config_defaults"]
    c = AttachedToConfig()
    if (
        c.contact_threshold_m != d["contact_threshold_m"]
        or c.penetration_tolerance_m != d["penetration_tolerance_m"]
        or c.floor_contact_threshold_m != d["floor_contact_threshold_m"]
        or c.floor_penetration_tolerance_m != d["floor_penetration_tolerance_m"]
    ):
        raise AssertionError("AttachedToConfig defaults drifted from fixture")


def test_synthetic_attachment_smoke_cases() -> None:
    fixture = _load_fixture()
    surfaces = [_surface(uid, spec) for uid, spec in fixture["surfaces"].items()]
    failures: list[str] = []
    for case in fixture["cases"]:
        arts = _artifacts([_case_entity(case)], surfaces)
        edges, diag = AttachedToExtractor().extract(arts, AttachedToConfig())
        if case["expected_attached_to"]:
            if len(edges) != 1:
                failures.append(f"{case['id']}: expected 1 edge, got {len(edges)}")
                continue
            edge = edges[0]
            if edge.type != "ATTACHED_TO":
                failures.append(f"{case['id']}: wrong edge type {edge.type}")
            if edge.source.uid != case["entity_uid"] or edge.target.kind != "surface":
                failures.append(f"{case['id']}: wrong direction")
            if edge.extractor != "attached_to" or edge.extractor_version != ATTACHED_TO_VERSION:
                failures.append(f"{case['id']}: provenance drifted")
            if edge.evidence.get("floor_supported") is not False:
                failures.append(f"{case['id']}: positive must record floor_supported=false")
        else:
            if edges:
                failures.append(f"{case['id']}: expected 0 edges, got {len(edges)}")
            reasons = {r.rejected_reason for r in diag.rejection_samples}
            expected = case["expected_rejection"]
            if expected not in reasons:
                failures.append(f"{case['id']}: expected rejection {expected}, got {sorted(reasons)}")
    if failures:
        raise AssertionError("\n".join(failures))


def test_room0_has_no_attached_to_edges() -> None:
    if not (REPLICA_ROOM0_V2 / "scene_graph.json").exists():
        print("  SKIP (enriched v2 artifact not on disk)")
        return
    edges, _ = AttachedToExtractor().extract(_build_room0_artifacts(), AttachedToConfig())
    if edges:
        raise AssertionError(f"room_0 should have 0 ATTACHED_TO edges, got {len(edges)}")


def test_apartment0_demo_plausibility_edges_if_dataset_present() -> None:
    """Demo-only plausibility check, not a ground-truth gate."""
    if not (APARTMENT0 / "habitat" / "info_semantic.json").exists():
        print("  SKIP (apartment_0 dataset not on disk)")
        return
    arts = import_habitat_room(APARTMENT0, "replica_apartment_0")
    bundle, _ = build_graph(
        arts,
        [ExtractorRun(AttachedToExtractor(), AttachedToConfig())],
        density_policy="phase2_telemetry_only",
    )
    got = sorted(e.source.uid for e in bundle.edges if e.type == "ATTACHED_TO")
    if got != ["obj_176", "obj_260", "obj_309"]:
        raise AssertionError(f"apartment_0 demo positives drifted: {got}")


TESTS = [
    test_config_defaults_match_smoke_fixture,
    test_synthetic_attachment_smoke_cases,
    test_room0_has_no_attached_to_edges,
    test_apartment0_demo_plausibility_edges_if_dataset_present,
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
