"""Evaluation-only learned-label scoring on a finalized ARKitScenes Lane A run.

Lane A is loaded and integrity-checked in full before the oracle boundary is
crossed.  The evaluator never invokes a labeler and never writes into the
finalized vertical-slice directory: it only scores the committed top-k
``SemanticHypothesis`` records against annotation-box entities matched by
delivered vertex-set IoU.

Usage:

  python3 tools/arkitscenes_learned_label_eval.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extractors.entity_surfaces import normalize_entity_class
from extractors.serde import load_entity_artifacts
from tools.arkitscenes_eval import (
    DEFAULT_DATA_ROOT,
    OracleEntity,
    iou_matrix,
    load_canonical_geometry,
    load_oracle_entities,
)
from tools.arkitscenes_mask3d_eval import load_delivered_partition


DEV_VIDEO_ID = "41069021"
MATCH_IOU = 0.50
DEFAULT_ENTITY_DIR = (
    REPO_ROOT / "runs" / "arkit_vertical_slice" /
    f"{DEV_VIDEO_ID}_labeled" / "entities"
)
DEFAULT_OUT = (
    REPO_ROOT / "runs" / "arkit_vertical_slice_eval" /
    f"{DEV_VIDEO_ID}_learned_label_eval.json"
)


@dataclass(frozen=True)
class LabelPrediction:
    instance_id: int
    object_uid: str
    top_k: tuple[tuple[str, float], ...]
    admitted: bool
    display_label: str


@dataclass(frozen=True)
class GeometryMatch:
    instance_id: int
    oracle_index: int
    iou: float


@dataclass(frozen=True)
class LaneAInputs:
    entity_bundle_hash: str
    entity_manifest_sha256: str
    representation_hash: str
    segmentation_output_sha256: str
    label_stage: dict
    predictions: tuple[LabelPrediction, ...]
    delivered_masks: tuple[np.ndarray, ...]
    delivered_instance_ids: tuple[int, ...]
    mesh_xyz: np.ndarray
    annotation_rotation: np.ndarray


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _instance_id(source_ref: str) -> int:
    prefix = "segmenter:"
    if not source_ref.startswith(prefix):
        raise ValueError(f"unsupported source instance ref {source_ref!r}")
    try:
        value = int(source_ref[len(prefix):])
    except ValueError as exc:
        raise ValueError(f"invalid source instance ref {source_ref!r}") from exc
    if value < 0:
        raise ValueError(f"negative delivered instance id {value}")
    return value


def predictions_from_artifacts(artifacts) -> tuple[LabelPrediction, ...]:
    """Read committed hypotheses and admission decisions; never re-label."""
    stage = artifacts.notes.get("label_stage")
    if not isinstance(stage, dict):
        raise ValueError("EntityArtifacts has no committed label_stage")
    top_k = int(stage.get("top_k", 0))
    threshold = float(stage.get("min_top1_score", float("nan")))
    if top_k < 1 or not math.isfinite(threshold):
        raise ValueError("label_stage top_k/threshold is invalid")

    out: list[LabelPrediction] = []
    seen: set[int] = set()
    for entity in artifacts.entities:
        instance_id = _instance_id(entity.identity.source_instance_ref)
        if instance_id in seen:
            raise ValueError(f"duplicate delivered instance id {instance_id}")
        seen.add(instance_id)
        hypotheses = entity.semantic_hypotheses
        if len(hypotheses) != top_k:
            raise ValueError(
                f"{entity.identity.object_uid}: expected {top_k} committed "
                f"hypotheses, got {len(hypotheses)}")
        rows = tuple((h.label, float(h.confidence)) for h in hypotheses)
        if any(not math.isfinite(score) for _label, score in rows):
            raise ValueError(
                f"{entity.identity.object_uid}: non-finite label score")
        if any(rows[i][1] < rows[i + 1][1]
               for i in range(len(rows) - 1)):
            raise ValueError(
                f"{entity.identity.object_uid}: hypotheses are not ranked")
        admitted = entity.extraction_diagnostics.get("learned_label_admitted")
        if not isinstance(admitted, bool):
            raise ValueError(
                f"{entity.identity.object_uid}: missing committed admission")
        expected_admission = rows[0][1] >= threshold
        if admitted != expected_admission:
            raise ValueError(
                f"{entity.identity.object_uid}: admission {admitted} disagrees "
                f"with committed threshold {threshold} and score {rows[0][1]}")
        out.append(LabelPrediction(
            instance_id=instance_id,
            object_uid=entity.identity.object_uid,
            top_k=rows,
            admitted=admitted,
            display_label=entity.identity.display_label,
        ))
    return tuple(sorted(out, key=lambda p: p.instance_id))


def match_delivered_instances(
    instance_ids: list[int] | tuple[int, ...],
    ious: np.ndarray,
    *,
    threshold: float = MATCH_IOU,
) -> list[GeometryMatch]:
    """Deterministic greedy one-to-one assignment by descending vertex IoU."""
    if ious.ndim != 2 or ious.shape[0] != len(instance_ids):
        raise ValueError("IoU rows do not match delivered instance ids")
    candidates = [
        (float(ious[i, j]), int(instance_ids[i]), j)
        for i in range(len(instance_ids))
        for j in range(ious.shape[1])
        if float(ious[i, j]) >= threshold
    ]
    used_instances: set[int] = set()
    used_oracle: set[int] = set()
    matches: list[GeometryMatch] = []
    for iou, instance_id, oracle_index in sorted(
            candidates, key=lambda row: (-row[0], row[1], row[2])):
        if instance_id in used_instances or oracle_index in used_oracle:
            continue
        used_instances.add(instance_id)
        used_oracle.add(oracle_index)
        matches.append(GeometryMatch(
            instance_id=instance_id,
            oracle_index=oracle_index,
            iou=iou,
        ))
    return matches


def score_label_matches(
    predictions: tuple[LabelPrediction, ...] | list[LabelPrediction],
    oracle_entities: list[OracleEntity],
    matches: list[GeometryMatch],
) -> dict:
    """Top-1/top-3 accuracy and selective admission metrics."""
    by_id = {prediction.instance_id: prediction for prediction in predictions}
    if len(by_id) != len(predictions):
        raise ValueError("duplicate prediction instance ids")

    rows = []
    for match in sorted(matches, key=lambda m: m.instance_id):
        if match.instance_id not in by_id:
            raise ValueError(
                f"matched instance {match.instance_id} has no label prediction")
        if not 0 <= match.oracle_index < len(oracle_entities):
            raise ValueError(f"oracle index out of range: {match.oracle_index}")
        pred = by_id[match.instance_id]
        oracle = oracle_entities[match.oracle_index]
        oracle_class = normalize_entity_class(oracle.label)
        predicted = [normalize_entity_class(label)
                     for label, _score in pred.top_k]
        top1_correct = predicted[0] == oracle_class
        top3_correct = oracle_class in predicted[:3]
        rows.append({
            "instance_id": pred.instance_id,
            "object_uid": pred.object_uid,
            "oracle_uid": oracle.uid,
            "oracle_label": oracle.label,
            "oracle_class_normalized": oracle_class,
            "vertex_iou": round(match.iou, 4),
            "admitted": pred.admitted,
            "display_label": pred.display_label,
            "top_k": [
                {"label": label, "score": score}
                for label, score in pred.top_k
            ],
            "top1_correct": top1_correct,
            "top3_correct": top3_correct,
        })

    n = len(rows)
    n_top1 = sum(row["top1_correct"] for row in rows)
    n_top3 = sum(row["top3_correct"] for row in rows)
    admitted = [row for row in rows if row["admitted"]]
    n_admitted_correct = sum(row["top1_correct"] for row in admitted)
    all_admitted = sum(prediction.admitted for prediction in predictions)

    def ratio(num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None

    return {
        "metric_scope": (
            "delivered instances greedily matched one-to-one to annotation-box "
            f"entities at vertex IoU >= {MATCH_IOU:.2f}"
        ),
        "n_geometry_matches": n,
        "top1": {"n_correct": n_top1, "accuracy": ratio(n_top1, n)},
        "top3": {"n_correct": n_top3, "accuracy": ratio(n_top3, n)},
        "admission": {
            "n_admitted": len(admitted),
            "n_correct": n_admitted_correct,
            "coverage": ratio(len(admitted), n),
            "precision": ratio(n_admitted_correct, len(admitted)),
            "correct_admitted_over_all_matches": ratio(n_admitted_correct, n),
            "definition": (
                "coverage = admitted/matched; precision = top1-correct/admitted"
            ),
        },
        "all_delivered_admission": {
            "n_admitted": all_admitted,
            "n_delivered": len(predictions),
            "rate": ratio(all_admitted, len(predictions)),
            "note": (
                "descriptive only; unmatched delivered instances have no "
                "oracle class and therefore do not enter admission precision"
            ),
        },
        "per_match": rows,
    }


def load_finalized_lane_a(
    scene_dir: Path,
    entity_dir: Path,
    segmentation_dir: Path | None = None,
) -> LaneAInputs:
    """Read and verify finalized deployable artifacts without oracle access."""
    entity_dir = Path(entity_dir)
    manifest_path = entity_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"finalized labeled EntityArtifacts missing at {manifest_path}")
    artifacts = load_entity_artifacts(entity_dir)
    expected_scene_id = f"arkitscenes_{scene_dir.name}"
    if artifacts.scene_id != expected_scene_id:
        raise ValueError(
            f"expected {expected_scene_id!r}, got scene {artifacts.scene_id!r}")
    if artifacts.notes.get("oracle_free") is not True:
        raise ValueError("labeled EntityArtifacts is not marked oracle_free")
    predictions = predictions_from_artifacts(artifacts)

    mesh, rotation, representation = load_canonical_geometry(scene_dir)
    if artifacts.representation_hash != representation.representation_hash:
        raise ValueError(
            "labeled EntityArtifacts representation hash does not match scene")
    store_value = segmentation_dir or artifacts.geometry_store_path
    if store_value is None:
        raise ValueError("labeled EntityArtifacts has no segmentation store")
    masks, instance_ids, provenance = load_delivered_partition(
        Path(store_value), len(mesh.xyz), artifacts.notes["input_mesh_sha256"])
    prediction_ids = {prediction.instance_id for prediction in predictions}
    if prediction_ids != set(instance_ids):
        raise ValueError(
            "labeled entities do not exactly cover the delivered partition: "
            f"labels={sorted(prediction_ids)}, instances={sorted(instance_ids)}")
    if provenance["output_sha256"] != artifacts.notes.get(
            "segmentation_output_sha256"):
        raise ValueError(
            "labeled EntityArtifacts refers to a different segmentation output")

    return LaneAInputs(
        entity_bundle_hash=artifacts.bundle_hash,
        entity_manifest_sha256=_sha256_file(manifest_path),
        representation_hash=artifacts.representation_hash,
        segmentation_output_sha256=provenance["output_sha256"],
        label_stage=dict(artifacts.notes["label_stage"]),
        predictions=predictions,
        delivered_masks=tuple(masks),
        delivered_instance_ids=tuple(instance_ids),
        mesh_xyz=mesh.xyz,
        annotation_rotation=rotation,
    )


def evaluate_labels(
    scene_dir: Path,
    entity_dir: Path,
    segmentation_dir: Path | None = None,
) -> dict:
    scene_dir = Path(scene_dir)
    video_id = scene_dir.name
    if not video_id:
        raise ValueError("scene directory must have a video-id basename")

    # Lane A is finalized, loaded, and integrity-checked before this call
    # returns. Nothing below can alter it.
    lane_a = load_finalized_lane_a(scene_dir, entity_dir, segmentation_dir)

    # ---- ORACLE BOUNDARY: no artifact/ranking/threshold changes below ----
    oracle_entities = load_oracle_entities(
        scene_dir, lane_a.mesh_xyz, lane_a.annotation_rotation)
    ious = iou_matrix(list(lane_a.delivered_masks), oracle_entities)
    matches = match_delivered_instances(
        lane_a.delivered_instance_ids, ious, threshold=MATCH_IOU)
    metrics = score_label_matches(
        lane_a.predictions, oracle_entities, matches)

    manifest_path = Path(entity_dir) / "manifest.json"
    if _sha256_file(manifest_path) != lane_a.entity_manifest_sha256:
        raise RuntimeError(
            "Lane A entity manifest changed during evaluation; refusing report")

    return {
        "schema": "arkitscenes_learned_label_eval_v1",
        "evaluation_only": True,
        "lane_a_immutability": (
            "finalized labeled EntityArtifacts loaded and hash-checked before "
            "annotation access; evaluator never invokes the labeler or writes "
            "to Lane A"
        ),
        "scene_id": f"arkitscenes_{video_id}",
        "video_id": video_id,
        "match_iou_threshold": MATCH_IOU,
        "n_oracle_entities": len(oracle_entities),
        "n_delivered_instances": len(lane_a.delivered_instance_ids),
        "lane_a": {
            "entity_bundle_hash": lane_a.entity_bundle_hash,
            "entity_manifest_sha256": lane_a.entity_manifest_sha256,
            "representation_hash": lane_a.representation_hash,
            "segmentation_output_sha256": lane_a.segmentation_output_sha256,
            "label_stage": lane_a.label_stage,
        },
        "metrics": metrics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scene-dir", type=Path,
        default=DEFAULT_DATA_ROOT / DEV_VIDEO_ID)
    parser.add_argument("--entity-dir", type=Path, default=DEFAULT_ENTITY_DIR)
    parser.add_argument("--segmentation-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    lane_a_root = args.entity_dir.resolve().parent
    try:
        output_is_in_lane_a = args.out.resolve().is_relative_to(lane_a_root)
    except AttributeError:  # pragma: no cover - Python <3.9 compatibility
        output_is_in_lane_a = lane_a_root in args.out.resolve().parents
    if output_is_in_lane_a:
        parser.error(
            f"--out must be outside finalized Lane A directory {lane_a_root}")

    report = evaluate_labels(
        args.scene_dir, args.entity_dir, args.segmentation_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    metrics = report["metrics"]
    admission = metrics["admission"]
    print(f"{report['scene_id']}: {metrics['n_geometry_matches']} geometry matches")
    print(f"top1 {metrics['top1']['n_correct']}/{metrics['n_geometry_matches']} "
          f"({metrics['top1']['accuracy']})")
    print(f"top3 {metrics['top3']['n_correct']}/{metrics['n_geometry_matches']} "
          f"({metrics['top3']['accuracy']})")
    print(f"admission coverage {admission['coverage']}; "
          f"precision {admission['precision']}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
