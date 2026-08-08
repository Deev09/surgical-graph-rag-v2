"""Phase 7 exit gate (P7.03).

Verifier for ATTACHED_TO plus the P6 compiler freeze. Writes only:

  scenes/replica_room_0/eval/phase7_exit_gate_report.json

Apartment_0 checks are demo/plausibility evidence, not a ground-truth accuracy
score. Missing apartment_0 data is recorded as skipped rather than blocking the
repo-grounded gate.
"""
from __future__ import annotations

# NOTE: the example command strings below embed an absolute dataset path.
# They are reproduced verbatim in the frozen
# scenes/replica_room_0/eval/phase7_exit_gate_report.json, and this tool
# REGENERATES that report, so "tidying" them rewrites a recorded gate
# artifact and fails tests/tools/test_phase7_exit_gate.py. Left as-is
# deliberately; see the v2 path-relativization commit.

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.equality import array_aware_equal
from common.serde import SchemaVersionError
from common.types import Plane, SceneFrame
from demo.replica_habitat_import import import_habitat_room
from eval.router_qa import score_questions
from extractors.base import (
    EntityArtifact, EntityArtifacts, EntityIdentity, ExtractionDiagnostics,
    InstanceExtractorConfig, SemanticHypothesis, StructuralSurface,
)
from extractors.serde import CURRENT_SCHEMA_VERSION as ENT_SCHEMA_VERSION
from graph.builder import ExtractorRun, build_graph
from graph.relations.attached_to import (
    ATTACHED_TO_VERSION,
    AttachedToConfig,
    AttachedToExtractor,
)
from graph.relations.contacts_surface import ContactsSurfaceConfig, ContactsSurfaceExtractor
from graph.relations.directional import DirectionalConfig, DirectionalExtractor
from graph.relations.on_entity_surface import OnEntitySurfaceConfig, OnEntitySurfaceExtractor
from graph.relations.on_surface import OnSurfaceConfig, OnSurfaceExtractor
from graph.relations.surface import SurfaceProximityConfig, SurfaceProximityExtractor
from graph.schema import Edge, GraphRef, Node, SceneGraphBundle, SurfaceRecord
from graph.serde import CURRENT_SCHEMA_VERSION, dump_scene_graph_bundle, load_scene_graph_bundle
from reasoner.ast import EdgeConstraint, SurfaceRef, Variable
from reasoner.base import CompletenessProfile, ExecutionContext
from reasoner.compiler_rules import Phase6RulesCompiler, RulesCompiler
from reasoner.executor import RulesExecutor
from reasoner.router import Router
from reasoner.verbalizer import StandardVerbalizer
from tools.phase2_exit_gate import _phase2_runs, _real_replica_artifacts
from tools.phase6_exit_gate import main as phase6_gate_main


REPLICA_ROOM0 = REPO_ROOT / "scenes" / "replica_room_0"
REPLICA_ROOM0_V2 = REPLICA_ROOM0 / "enriched" / "v2"
EVAL_DIR = REPLICA_ROOM0 / "eval"
ARTIFACT_PATH = EVAL_DIR / "phase7_exit_gate_report.json"
SMOKE_FIXTURE = REPO_ROOT / "eval" / "questions" / "phase7_attachment_smoke.json"
PHASE7_QA = REPO_ROOT / "eval" / "questions" / "phase7_mixed_qa.json"
APARTMENT0 = Path.home() / "Desktop/datasets/replica/apartment_0"


def _frame() -> SceneFrame:
    return SceneFrame(
        gravity=(0.0, 0.0, -1.0), canonical_forward=None,
        canonical_right=None, units="meters", notes="",
    )


def _oracle_ctx() -> ExecutionContext:
    return ExecutionContext(completeness=CompletenessProfile(
        source="oracle", entity_recall_by_class={}, edge_recall_by_type={}))


def _router(compiler=None) -> Router:
    return Router(
        compiler=compiler or RulesCompiler(),
        executor=RulesExecutor(),
        verbalizer=StandardVerbalizer(),
    )


def _node(uid: str, label: str) -> Node:
    return Node(
        id=uid, label=label, label_confidence=1.0, centroid=(0.0, 0.0, 0.0),
        bbox_aabb=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), bbox_obb=None,
        embedding_ref=None, attributes={"display_label": label}, provenance={},
    )


def _wall_surface(uid: str = "wall_1") -> SurfaceRecord:
    return SurfaceRecord(
        uid=uid, surface_type="wall",
        plane=Plane(a=0.0, b=1.0, c=0.0, d=0.0),
        polygon=[(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 2.0), (-1.0, 0.0, 2.0)],
        source="habitat_label", confidence=1.0,
    )


def _synthetic_attached_bundle() -> SceneGraphBundle:
    edge = Edge(
        edge_id="att_demo_1",
        source=GraphRef(kind="entity", uid="obj_sconce"),
        type="ATTACHED_TO",
        target=GraphRef(kind="surface", uid="wall_1"),
        frame="world",
        weight=1.0,
        confidence=1.0,
        extractor="attached_to",
        extractor_version=ATTACHED_TO_VERSION,
        evidence={"floor_supported": False},
    )
    return SceneGraphBundle(
        schema_version=CURRENT_SCHEMA_VERSION,
        bundle_hash="phase7_synth",
        scene_id="phase7_synth",
        frame=_frame(),
        entity_bundle_hash="ent_synth",
        nodes=[_node("obj_sconce", "wall sconce")],
        edges=[edge],
        structural_surface_refs=["wall_1"],
        structural_surfaces=[_wall_surface()],
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


def _case_entity(case: dict) -> EntityArtifact:
    mn = case["entity_aabb"]["min"]
    mx = case["entity_aabb"]["max"]
    return _entity(
        case["entity_uid"], tuple(mn), tuple(mx), tuple(case["entity_centroid"]),
    )


def _phase7_runs() -> list[ExtractorRun]:
    return [
        ExtractorRun(DirectionalExtractor(), DirectionalConfig(mode="sparse")),
        ExtractorRun(SurfaceProximityExtractor(), SurfaceProximityConfig(use_polygon_clip=True)),
        ExtractorRun(OnSurfaceExtractor(), OnSurfaceConfig()),
        ExtractorRun(ContactsSurfaceExtractor(), ContactsSurfaceConfig()),
        ExtractorRun(OnEntitySurfaceExtractor(), OnEntitySurfaceConfig()),
        ExtractorRun(AttachedToExtractor(), AttachedToConfig()),
    ]


def _gate_g1_schema() -> tuple[bool, dict]:
    bundle = _synthetic_attached_bundle()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "bundle"
        dump_scene_graph_bundle(bundle, out)
        loaded = load_scene_graph_bundle(out)
        roundtrip_ok = array_aware_equal(bundle, loaded)
        manifest = out / "manifest.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["schema_version"] = 5
        manifest.write_text(json.dumps(payload))
        v5_rejected = False
        try:
            load_scene_graph_bundle(out)
        except SchemaVersionError:
            v5_rejected = True
    return roundtrip_ok and v5_rejected, {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "attached_to_roundtrip_ok": roundtrip_ok,
        "v5_manifest_strict_rejected": v5_rejected,
    }


def _gate_g2_smoke_fixture() -> tuple[bool, dict]:
    fixture = json.loads(SMOKE_FIXTURE.read_text(encoding="utf-8"))
    surfaces = [_surface(uid, spec) for uid, spec in fixture["surfaces"].items()]
    failures: list[str] = []
    emitted: list[str] = []
    rejection_reasons: dict[str, list[str]] = {}
    for case in fixture["cases"]:
        edges, diag = AttachedToExtractor().extract(
            _artifacts([_case_entity(case)], surfaces), AttachedToConfig(),
        )
        if case["expected_attached_to"]:
            emitted.extend(e.source.uid for e in edges)
            if len(edges) != 1:
                failures.append(f"{case['id']}: expected 1 edge, got {len(edges)}")
            elif edges[0].evidence.get("floor_supported") is not False:
                failures.append(f"{case['id']}: floor_supported evidence drifted")
        else:
            reasons = sorted({r.rejected_reason for r in diag.rejection_samples})
            rejection_reasons[case["id"]] = reasons
            if edges:
                failures.append(f"{case['id']}: expected 0 edges, got {len(edges)}")
            if case["expected_rejection"] not in reasons:
                failures.append(f"{case['id']}: missing rejection {case['expected_rejection']}")
    return not failures, {
        "failures": failures,
        "emitted_entities": sorted(emitted),
        "rejection_reasons": rejection_reasons,
    }


def _gate_g3_room0_empty() -> tuple[bool, dict]:
    if not (REPLICA_ROOM0_V2 / "scene_graph.json").exists():
        return False, {"error": "replica_room_0 enriched v2 missing"}
    artifacts = _real_replica_artifacts()
    bundle, _diag = build_graph(
        artifacts,
        [ExtractorRun(AttachedToExtractor(), AttachedToConfig())],
        density_policy="phase2_telemetry_only",
    )
    edges = [e for e in bundle.edges if e.type == "ATTACHED_TO"]
    ans = _router().answer("what is attached to the wall?", bundle, _oracle_ctx())
    return len(edges) == 0 and ans.outcome == "empty", {
        "attached_to_edge_count": len(edges),
        "answer_outcome": ans.outcome,
        "cited_uids": ans.cited_uids,
    }


def _gate_g4_compiler_freeze() -> tuple[bool, dict]:
    bundle = _synthetic_attached_bundle()
    p7 = RulesCompiler().compile("what is attached to the wall?", bundle)
    p6 = Phase6RulesCompiler().compile("what is attached to the wall?", bundle)
    shape_ok = (
        p7.outcome == "compiled"
        and p7.ast is not None
        and isinstance(p7.ast.where[0], EdgeConstraint)
        and p7.ast.where[0].type == "ATTACHED_TO"
        and isinstance(p7.ast.where[0].source, Variable)
        and isinstance(p7.ast.where[0].target, SurfaceRef)
    )
    p6_ok = p6.outcome == "out_of_schema" and (p6.notes or "").startswith("deferred:")
    return shape_ok and p6_ok, {
        "p7_outcome": p7.outcome,
        "p7_edge_type": p7.ast.where[0].type if p7.ast is not None else None,
        "p6_outcome": p6.outcome,
        "p6_notes": p6.notes,
    }


def _gate_g5_executor_answers() -> tuple[bool, dict]:
    bundle = _synthetic_attached_bundle()
    ans = _router().answer("what is attached to the wall?", bundle, _oracle_ctx())
    return ans.outcome == "bindings" and ans.cited_uids == ["obj_sconce"], {
        "outcome": ans.outcome,
        "cited_uids": ans.cited_uids,
        "cited_edges": ans.cited_edges,
        "text": ans.text,
    }


def _gate_g6_p6_gate_still_passes() -> tuple[bool, dict]:
    rc = phase6_gate_main()
    return rc == 0, {"phase6_exit_gate_rc": rc}


def _gate_g7_default_path_preserved() -> tuple[bool, dict]:
    artifacts = _real_replica_artifacts()
    bundle, _diag = build_graph(
        artifacts, _phase2_runs(), density_policy="phase2_telemetry_only")
    counts = {
        "ON_SURFACE": sum(1 for e in bundle.edges if e.type == "ON_SURFACE"),
        "CONTACTS_SURFACE": sum(1 for e in bundle.edges if e.type == "CONTACTS_SURFACE"),
        "ON_ENTITY_SURFACE": sum(1 for e in bundle.edges if e.type == "ON_ENTITY_SURFACE"),
        "ATTACHED_TO": sum(1 for e in bundle.edges if e.type == "ATTACHED_TO"),
    }
    return all(v == 0 for v in counts.values()), {"default_build_edge_counts": counts}


def _gate_g8_determinism() -> tuple[bool, dict]:
    artifacts = _real_replica_artifacts()
    runs = [ExtractorRun(AttachedToExtractor(), AttachedToConfig())]
    a, _ = build_graph(artifacts, runs, density_policy="phase2_telemetry_only")
    b, _ = build_graph(artifacts, runs, density_policy="phase2_telemetry_only")
    edges_a = sorted(e.edge_id for e in a.edges)
    edges_b = sorted(e.edge_id for e in b.edges)
    return a.bundle_hash == b.bundle_hash and edges_a == edges_b, {
        "bundle_hash_a": a.bundle_hash,
        "bundle_hash_b": b.bundle_hash,
        "edge_ids_a": edges_a,
        "edge_ids_b": edges_b,
    }


def _gate_g9_apartment_demo() -> tuple[bool, dict]:
    if not (APARTMENT0 / "habitat" / "info_semantic.json").exists():
        return True, {"skipped": True, "reason": "apartment_0 dataset not on disk"}
    artifacts = import_habitat_room(APARTMENT0, "replica_apartment_0")
    bundle, _diag = build_graph(
        artifacts, _phase7_runs(), density_policy="phase2_telemetry_only")
    questions = json.loads(PHASE7_QA.read_text(encoding="utf-8"))["questions"]
    scorecard = score_questions(questions, bundle, _router(), _oracle_ctx())
    attached = sorted(e.source.uid for e in bundle.edges if e.type == "ATTACHED_TO")
    agg = scorecard["aggregate"]
    ok = (
        attached == ["obj_176", "obj_260", "obj_309"]
        and agg["all_expected_outcomes_met"]
        and agg["false_answer_count"] == 0
    )
    return ok, {
        "skipped": False,
        "answer_key_type": "plausibility_labels_not_ground_truth",
        "attached_to_entities": attached,
        "question_set": str(PHASE7_QA.relative_to(REPO_ROOT)),
        "aggregate": agg,
        "per_question": scorecard["per_question"],
    }


def main() -> int:
    gates = {
        "G1_schema_v6_attached_to_roundtrip_and_v5_rejection": _gate_g1_schema(),
        "G2_attachment_smoke_fixture": _gate_g2_smoke_fixture(),
        "G3_room0_honest_empty": _gate_g3_room0_empty(),
        "G4_compiler_p7_answer_and_p6_freeze": _gate_g4_compiler_freeze(),
        "G5_executor_answers_attached_to": _gate_g5_executor_answers(),
        "G6_phase6_exit_gate_still_passes": _gate_g6_p6_gate_still_passes(),
        "G7_default_path_preserved": _gate_g7_default_path_preserved(),
        "G8_attached_to_determinism": _gate_g8_determinism(),
        "G9_apartment0_demo_plausibility": _gate_g9_apartment_demo(),
    }
    overall = all(p for p, _ in gates.values())
    payload = {
        "phase": "P7.03",
        "artifact_kind": "phase7_exit_gate_report",
        "schema_version": 1,
        "extractor_version": ATTACHED_TO_VERSION,
        "overall_blocking_pass": overall,
        "gates": {name: {"pass": p, **d} for name, (p, d) in gates.items()},
        "benchmark_semantics": {
            "phase6_q6": "defer under Phase6RulesCompiler",
            "phase7_q6": "answer under RulesCompiler when ATTACHED_TO edges exist",
            "apartment0": "plausibility/demo labels only, not ground-truth accuracy",
            "not_comparable_to_v1": True,
            "not_comparable_to_phase6_scorecard": True,
        },
        "demo_commands": [
            "python3 demo/question_battery.py /Users/deevyaswain/Desktop/datasets/replica/apartment_0 replica_apartment_0",
            "python3 demo/real_scene_demo.py /Users/deevyaswain/Desktop/datasets/replica/apartment_0 replica_apartment_0"
        ],
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\nPhase 7 exit gate report -> {ARTIFACT_PATH.relative_to(REPO_ROOT)}")
    for name, (passed, _details) in gates.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"\nOverall blocking: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
