"""Focused synthetic tests for ARKitScenes learned-label evaluation."""
from __future__ import annotations

import ast
import sys
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.arkitscenes_eval import OracleEntity
from tools.arkitscenes_learned_label_eval import (
    GeometryMatch,
    LabelPrediction,
    MATCH_IOU,
    match_delivered_instances,
    score_label_matches,
)


MODULE = REPO_ROOT / "tools" / "arkitscenes_learned_label_eval.py"


def _oracle(uid: str, label: str) -> OracleEntity:
    return OracleEntity(
        uid=uid, label=label, centroid=np.zeros(3),
        axes_lengths=np.ones(3), axes=np.eye(3),
        vertices=np.asarray([0], dtype=np.int64),
    )


def _prediction(instance_id: int, rows, admitted: bool) -> LabelPrediction:
    return LabelPrediction(
        instance_id=instance_id,
        object_uid=f"obj_{instance_id}",
        top_k=tuple(rows),
        admitted=admitted,
        display_label=(rows[0][0] if admitted else f"segment_{instance_id}"),
    )


def test_matching_is_one_to_one_and_thresholded() -> None:
    # p10 and p11 both prefer oracle 0. Greedy must give oracle 0 to p10,
    # then p11 can take oracle 1. p12's 0.49 is below the frozen floor.
    ious = np.asarray([
        [0.80, 0.00, 0.00],
        [0.70, 0.60, 0.00],
        [0.00, 0.00, 0.49],
    ])
    matches = match_delivered_instances([10, 11, 12], ious)
    got = [(m.instance_id, m.oracle_index, m.iou) for m in matches]
    if got != [(10, 0, 0.8), (11, 1, 0.6)]:
        raise AssertionError(f"unexpected greedy matches: {got}")
    if MATCH_IOU != 0.50:
        raise AssertionError(f"match operating point drifted: {MATCH_IOU}")


def test_metrics_separate_topk_and_admission() -> None:
    oracle = [
        _oracle("a", "tv_monitor"),
        _oracle("b", "chair"),
        _oracle("c", "cabinet"),
    ]
    predictions = [
        _prediction(10, (("tv-monitor", 0.31), ("monitor", 0.29),
                         ("picture", 0.27)), True),
        _prediction(11, (("table", 0.27), ("chair", 0.26),
                         ("stool", 0.25)), False),
        _prediction(12, (("table", 0.32), ("desk", 0.30),
                         ("counter", 0.29)), True),
    ]
    matches = [GeometryMatch(10, 0, 0.8),
               GeometryMatch(11, 1, 0.7),
               GeometryMatch(12, 2, 0.6)]
    report = score_label_matches(predictions, oracle, matches)
    if report["top1"] != {"n_correct": 1, "accuracy": round(1 / 3, 4)}:
        raise AssertionError(report)
    if report["top3"] != {"n_correct": 2, "accuracy": round(2 / 3, 4)}:
        raise AssertionError(report)
    admission = report["admission"]
    if admission["n_admitted"] != 2:
        raise AssertionError(admission)
    if admission["coverage"] != round(2 / 3, 4):
        raise AssertionError(admission)
    if admission["precision"] != 0.5:
        raise AssertionError(admission)
    if report["all_delivered_admission"] != {
            "n_admitted": 2,
            "n_delivered": 3,
            "rate": round(2 / 3, 4),
            "note": (
                "descriptive only; unmatched delivered instances have no "
                "oracle class and therefore do not enter admission precision"
            )}:
        raise AssertionError(report["all_delivered_admission"])
    # Normalization, not a synonym table: tv_monitor and tv-monitor agree.
    if not report["per_match"][0]["top1_correct"]:
        raise AssertionError("underscore/hyphen normalization failed")


def test_zero_admission_precision_is_unknown_not_zero() -> None:
    report = score_label_matches(
        [_prediction(1, (("table", 0.2), ("chair", 0.1),
                         ("sofa", 0.0)), False)],
        [_oracle("a", "chair")],
        [GeometryMatch(1, 0, 0.7)],
    )
    if report["admission"]["coverage"] != 0.0:
        raise AssertionError(report)
    if report["admission"]["precision"] is not None:
        raise AssertionError("zero admitted labels have undefined precision")


def test_lane_a_load_precedes_oracle_boundary_in_source() -> None:
    """Guard the evaluation order without importing any real annotation."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    fn = next(node for node in tree.body
              if isinstance(node, ast.FunctionDef)
              and node.name == "evaluate_labels")
    calls = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"load_finalized_lane_a", "load_oracle_entities"}:
                calls[node.func.id] = node.lineno
    if set(calls) != {"load_finalized_lane_a", "load_oracle_entities"}:
        raise AssertionError(f"boundary calls missing: {calls}")
    if calls["load_finalized_lane_a"] >= calls["load_oracle_entities"]:
        raise AssertionError(f"oracle opens before Lane A is finalized: {calls}")


TESTS = [
    test_matching_is_one_to_one_and_thresholded,
    test_metrics_separate_topk_and_admission,
    test_zero_admission_precision_is_unknown_not_zero,
    test_lane_a_load_precedes_oracle_boundary_in_source,
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
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
