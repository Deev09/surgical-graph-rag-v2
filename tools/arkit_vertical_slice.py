"""Oracle-free ARKitScenes mesh -> entities -> graph -> answer vertical slice.

Example:

  python3 tools/arkit_vertical_slice.py \
    --scene-dir ~/Desktop/datasets/arkitscenes/Validation/41069021 \
    --segmentation-dir runs/arkit_vertical_slice_ms02/bundle_arkitscenes_41069021 \
    --out runs/arkit_vertical_slice/41069021

The deployable path never reads ARKitScenes annotations.  It intentionally
builds only the relation family supported by the available evidence: NEAR
between delivered anonymous entities.  Horizontal direction is withheld
because the ARKit representation has no declared canonical forward/right;
floor, wall, attachment, and support relations are withheld because the
bundle has no deployable structural or entity-local support surfaces.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.arkitscenes import (
    ARKitScenesAdapter,
    build_arkitscenes_capture_bundle,
)
from adapters.base import ReconstructionConfig
from extractors.arkitscenes_segments import build_arkitscenes_segment_artifacts
from extractors.learned_labels import (
    ImageLabeler,
    LearnedLabelConfig,
    attach_learned_labels,
)
from extractors.serde import dump_entity_artifacts
from graph.builder import ExtractorRun, build_graph
from graph.relations.proximity import ProximityConfig, ProximityExtractor
from graph.serde import dump_build_diagnostics, dump_scene_graph_bundle
from geometry.entity_support_patches import (
    extract_entity_horizontal_patches_from_files,
)
from reasoner.base import CompletenessProfile, ExecutionContext
from reasoner.compiler_rules import RulesCompiler
from reasoner.executor import RulesExecutor
from reasoner.router import Router
from reasoner.verbalizer import StandardVerbalizer
from tools.relation_inspector import write_inspector


SLICE_VERSION = "0.1"
NEAR_THRESHOLD_M = 1.0


def _question_row(question: str, answer) -> dict:
    return {
        "qid": "Q01",
        "question": question,
        "expected_outcome": None,
        "outcome": answer.outcome,
        "answered_by": answer.answered_by,
        "text": answer.text,
        # The v2 margin-calibration attempt was refuted and the Router's
        # default edge confidence is a placeholder 1.0. Do not display that
        # value as if it were calibrated probability.
        "confidence": None,
        "confidence_status": "not_calibrated",
        "raw_router_confidence": answer.confidence,
        "cited_uids": sorted(answer.cited_uids),
        "cited_edges": sorted(answer.cited_edges),
    }


def build_slice(representation, segmentation_dir: Path, out_dir: Path,
                *, question: str | None = None,
                min_vertices: int = 20,
                with_learned_labels: bool = False,
                labeler: ImageLabeler | None = None,
                with_support_patches: bool = False) -> dict:
    """Build one finalized, annotation-free vertical-slice artifact."""
    out_dir = Path(out_dir)
    segmentation_dir = Path(segmentation_dir)
    entities = build_arkitscenes_segment_artifacts(
        representation, segmentation_dir, min_vertices=min_vertices)
    if with_learned_labels:
        entities = attach_learned_labels(
            representation,
            entities,
            segmentation_dir=segmentation_dir,
            config=LearnedLabelConfig(),
            labeler=labeler,
        )

    graph, diagnostics = build_graph(
        entities,
        [ExtractorRun(
            ProximityExtractor(),
            ProximityConfig(
                mode="sparse", sparse_version=2,
                sparse_near_threshold=NEAR_THRESHOLD_M,
            ),
        )],
        density_policy="phase2_telemetry_only",
    )

    entity_dir = out_dir / "entities"
    graph_dir = out_dir / "graph"
    entity_manifest = dump_entity_artifacts(entities, entity_dir)
    graph_manifest = dump_scene_graph_bundle(graph, graph_dir)
    diagnostics_path = dump_build_diagnostics(
        diagnostics, out_dir / "graph_diagnostics.json")

    support_patches_path = None
    support_patch_summary = None
    if with_support_patches:
        patch_evidence = extract_entity_horizontal_patches_from_files(
            Path(representation.geometry_handle.uri),
            segmentation_dir / "vertex_instance_ids.npy",
        )
        support_patches_path = out_dir / "support_patches.json"
        support_patches_path.write_text(
            json.dumps(
                patch_evidence.to_dict(), indent=2, sort_keys=True,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )
        support_patch_summary = patch_evidence.diagnostics

    if not graph.nodes:
        raise ValueError("vertical slice produced no delivered entities")
    if question is None:
        anchor = graph.nodes[0].attributes.get("display_label") or graph.nodes[0].label
        question = f"what is near {anchor}?"

    router = Router(
        compiler=RulesCompiler(), executor=RulesExecutor(),
        verbalizer=StandardVerbalizer(),
    )
    context = ExecutionContext(completeness=CompletenessProfile(
        source="unknown", entity_recall_by_class={}, edge_recall_by_type={}))
    answer = router.answer(question, graph, context)
    qrow = _question_row(question, answer)
    inspector_path = out_dir / "inspector.html"
    write_inspector(
        representation.scene_id, graph, diagnostics, inspector_path,
        questions=[qrow],
    )

    manifest = {
        "schema": "arkit_vertical_slice_v1",
        "slice_version": SLICE_VERSION,
        "scene_id": representation.scene_id,
        "oracle_free": True,
        "input": {
            "representation_hash": representation.representation_hash,
            "segmentation_output_sha256": entities.notes[
                "segmentation_output_sha256"],
        },
        "outputs": {
            "entity_manifest": str(entity_manifest),
            "graph_manifest": str(graph_manifest),
            "graph_diagnostics": str(diagnostics_path),
            "inspector": str(inspector_path),
            "support_patch_evidence": (
                str(support_patches_path) if support_patches_path else None
            ),
        },
        "counts": {
            "entities": len(graph.nodes),
            "edges": len(graph.edges),
            "near_edges": sum(e.type == "NEAR" for e in graph.edges),
        },
        "question": qrow,
        "available_capabilities": {
            "delivered_anonymous_instances": True,
            "learned_semantic_hypotheses": with_learned_labels,
            "entity_horizontal_patch_evidence": with_support_patches,
            "near_relation": True,
            "relation_evidence": True,
            "inspectable_2d_aabb_views": True,
        },
        "unavailable_capabilities": {
            **({} if with_learned_labels else {
                "learned_semantic_labels": (
                    "run with --with-learned-labels to attach the pinned "
                    "global-vocabulary OpenCLIP hypotheses"),
            }),
            "horizontal_direction": (
                "canonical_forward/right are undefined for this representation"),
            "floor_wall_relations": "no oracle-free structural surfaces supplied",
            "wall_attachment": "no oracle-free wall/object attachment evidence",
            "entity_surface_support": (
                "horizontal owner patches are present, but target-relative "
                "resting/contact evidence is not yet implemented"
                if with_support_patches else
                "no entity-local support patches supplied"
            ),
            "hierarchy": "no room/furniture/object hierarchy extractor supplied",
            "conversation": "single-turn deterministic RulesCompiler only",
            "raw_mesh_3d_inspector": "current inspector is an AABB evidence view",
        },
        "relation_configuration": {
            "near_metric": "aabb_surface",
            "near_threshold_m": NEAR_THRESHOLD_M,
            "threshold_status": "provisional Replica-era constant; not a transfer claim",
        },
        "entity_configuration": {"min_vertices": min_vertices},
        "label_configuration": (
            entities.notes.get("label_stage") if with_learned_labels else None
        ),
        "support_patch_summary": support_patch_summary,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--segmentation-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--question", default=None)
    parser.add_argument(
        "--with-learned-labels", action="store_true",
        help=("render each delivered instance and attach pinned OpenCLIP "
              "top-k hypotheses; low-confidence instances stay anonymous"),
    )
    parser.add_argument(
        "--with-support-patches", action="store_true",
        help=("extract geometry-only owner-local horizontal patch evidence; "
              "this does not itself emit support relations"),
    )
    args = parser.parse_args(argv)

    representation = ARKitScenesAdapter().reconstruct(
        build_arkitscenes_capture_bundle(args.scene_dir),
        ReconstructionConfig(name="arkit_vertical_slice", version=SLICE_VERSION),
    )
    manifest = build_slice(
        representation, args.segmentation_dir, args.out,
        question=args.question,
        with_learned_labels=args.with_learned_labels,
        with_support_patches=args.with_support_patches,
    )
    q = manifest["question"]
    print(f"{manifest['scene_id']}: {manifest['counts']['entities']} entities, "
          f"{manifest['counts']['edges']} edges")
    print(f"Q: {q['question']}")
    print(f"A: [{q['outcome']}] {q['text']}")
    print(f"-> {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
