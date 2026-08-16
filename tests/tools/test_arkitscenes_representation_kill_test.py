#!/usr/bin/env python3
"""Synthetic tests for the bounded representation kill test."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extractors.arkitscenes_rgb_crops import Frame
from tools.arkitscenes_representation_kill_test import (
    RESPONSE_SCHEMA,
    decision,
    direct_rows,
    geometry_signature,
    hybrid_rows,
    image_information_score,
    prompt_text,
    public_questions,
    survey_frame_indices,
    summarize,
)


def key() -> dict:
    return {
        "scene_id": "scene_test",
        "independent_questions": [
            {"id": "q_count", "kind": "class_cardinality",
             "question": "How many rugs?", "class": "rug", "expected": 1,
             "review_basis": "must not leak"},
            {"id": "q_present", "kind": "class_presence",
             "question": "Is there a sofa?", "expected": True},
            {"id": "q_support", "kind": "support_relation",
             "question": "What supports the cushion?",
             "expected_relation": "ON_ENTITY_SURFACE",
             "expected_subject_class": "cushion",
             "expected_object_class": "sofa"},
            {"id": "q_extent", "kind": "instance_extent",
             "question": "Is the reported counter object-scale?", "expected": True,
             "room_span_fraction": 0.5, "metric": "object_scale"},
            {"id": "q_uid", "kind": "instance_identity",
             "question": "Which delivered uid?", "expected_uids": ["obj_1"],
             "class": "rug"},
        ],
    }


def packet() -> dict:
    questions = public_questions(key())
    return {
        "scene_id": "scene_test",
        "packet_sha256": "packet-pin",
        "questions": questions,
        "frames": [{"id": "f0"}, {"id": "f1"}, {"id": "f2"}],
    }


def response(*, support="sofa", support_frames=("f0", "f1")) -> dict:
    return {
        "schema": RESPONSE_SCHEMA,
        "scene_id": "scene_test",
        "packet_sha256": "packet-pin",
        "model": {"provider": "synthetic", "name": "fixed", "version": "1"},
        "answers": [
            {"id": "q_count", "outcome": "answer", "answer": 1,
             "confidence": 0.9, "evidence_frame_ids": ["f0", "f1"]},
            {"id": "q_present", "outcome": "answer", "answer": True,
             "confidence": 0.9, "evidence_frame_ids": ["f0", "f2"]},
            {"id": "q_support", "outcome": "answer", "answer": support,
             "confidence": 0.9, "evidence_frame_ids": list(support_frames)},
        ],
    }


def test_public_packet_excludes_answers_and_uid_diagnostic():
    rows = public_questions(key())
    assert [r["id"] for r in rows] == [
        "q_count", "q_present", "q_support"]
    encoded = json.dumps(rows)
    assert "expected" not in encoded
    assert "review_basis" not in encoded
    assert "obj_1" not in encoded
    prompt = prompt_text(packet())
    assert '"packet_sha256": "packet-pin"' in prompt


def test_direct_response_is_structured_and_pinned():
    rows = direct_rows(key(), packet(), response(support="couch"))
    assert [r["outcome"] for r in rows] == ["correct"] * 3
    bad = response()
    bad["packet_sha256"] = "wrong"
    try:
        direct_rows(key(), packet(), bad)
    except ValueError as exc:
        assert "pin" in str(exc)
    else:
        raise AssertionError("un-pinned response accepted")


def test_direct_response_rejects_unknown_frame_and_wrong_types():
    bad = response()
    bad["answers"][0]["evidence_frame_ids"] = ["not-a-frame"]
    try:
        direct_rows(key(), packet(), bad)
    except ValueError as exc:
        assert "invalid evidence" in str(exc)
    else:
        raise AssertionError("unknown evidence frame accepted")
    wrong = response()
    wrong["answers"][0]["answer"] = True  # bool is not a cardinality integer
    rows = direct_rows(key(), packet(), wrong)
    assert rows[0]["outcome"] == "wrong"
    invalid_confidence = response()
    invalid_confidence["answers"][0]["confidence"] = 1.1
    try:
        direct_rows(key(), packet(), invalid_confidence)
    except ValueError as exc:
        assert "confidence" in str(exc)
    else:
        raise AssertionError("invalid confidence accepted")


def test_hybrid_routes_by_kind_and_requires_two_visual_citations():
    visual = direct_rows(key(), packet(), response(support_frames=("f0",)))
    object_rows = [dict(r, source="object") for r in visual]
    graph_rows = [dict(r, source="graph") for r in visual]
    # No materialized support answer: route support to vision, which fails the
    # two-view sufficiency requirement in this fixture.
    graph_rows[2]["answer"] = None
    graph_rows[2]["outcome"] = "unanswered"
    routed = hybrid_rows(packet(), object_rows, graph_rows, visual)
    assert routed[0]["source"] == "direct_vlm"
    assert routed[2]["outcome"] == "unanswered"
    assert routed[2]["insufficiency"]


def test_decision_is_pending_without_direct_and_measured_with_it():
    perfect = {"rows": [], "summary": {
        "n_questions": 4,
        "tally": {"correct": 4, "wrong": 0, "unanswered": 0},
        "coverage": 1.0, "exact_accuracy_all": 1.0,
        "accuracy_when_answered": 1.0, "false_confident_rate": 0.0,
    }}
    weak_rows = [
        {"outcome": "correct"}, {"outcome": "wrong"},
        {"outcome": "wrong"}, {"outcome": "unanswered"},
    ]
    weak = {"rows": weak_rows, "summary": summarize(weak_rows)}
    assert decision({"object_retrieval": weak, "typed_graph": weak})["status"] \
        == "pending_direct_vlm"
    result = decision({"object_retrieval": weak, "typed_graph": weak,
                       "direct_vlm": weak, "hybrid": perfect})
    assert result["status"] == "measured"
    assert result["proceed"] is True


def test_survey_selection_prefers_informative_frame_per_temporal_bin():
    import tempfile
    root = Path(tempfile.mkdtemp(prefix="representation-survey-"))
    frames = []
    for i in range(8):
        path = root / f"{i}.png"
        pixels = np.full((24, 24), 127, dtype=np.uint8)
        if i in {1, 5}:
            pixels[:, ::2] = 0
            pixels[:, 1::2] = 255
        Image.fromarray(pixels).save(path)
        frames.append(Frame(float(i), path, np.eye(3), np.zeros(3),
                            1.0, 1.0, 0.0, 0.0, 24, 24))
    assert image_information_score(frames[1].png) \
        > image_information_score(frames[0].png)
    assert survey_frame_indices(frames, 2) == [1, 5]


def test_geometry_signature_ignores_labels_but_not_geometry():
    manifest = {
        "scene_id": "s", "frame": {"kind": "scene_canonical"},
        "representation_hash": "r",
        "entities": [{
            "identity": {"object_uid": "obj_0", "display_label": "chair"},
            "bbox_aabb": [[0, 0, 0], [1, 1, 1]],
            "bbox_obb": {"center": [0.5, 0.5, 0.5]},
            "centroid": [0.5, 0.5, 0.5],
            "geometry_handle": "ids.npy#0",
        }],
    }
    relabelled = json.loads(json.dumps(manifest))
    relabelled["entities"][0]["identity"]["display_label"] = "stool"
    assert geometry_signature(manifest) == geometry_signature(relabelled)
    moved = json.loads(json.dumps(manifest))
    moved["entities"][0]["centroid"][0] = 0.6
    assert geometry_signature(manifest) != geometry_signature(moved)


def main() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
