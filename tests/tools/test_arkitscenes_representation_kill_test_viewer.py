#!/usr/bin/env python3
"""Synthetic tests for the representation kill test static viewer.

The viewer's contract is narrow and worth pinning: it must render committed
report values verbatim, must never ship a script block, must refuse a report
that was scored against a different packet, and must show unscored arms as
pending rather than as a blank or a zero.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.arkitscenes_representation_kill_test_viewer import (
    SCHEMA,
    build_html,
    fmt_answer,
    fmt_number,
)


def packet() -> dict:
    return {
        "scene_id": "scene_test",
        "packet_sha256": "packet_hash",
        "questions": [
            {"id": "q_count", "kind": "class_cardinality",
             "question": "How many rugs?", "answer_type": "integer"},
            {"id": "q_support", "kind": "support_relation",
             "question": "What supports the cushion?",
             "answer_type": "object_class"},
        ],
        "frames": [{"id": "frame_00_1.000"}, {"id": "frame_01_2.000"}],
        "selection": {"method": "temporal_bins_rgb_information_v1",
                      "n_usable_strided_frames": 40},
    }


def structured_rows() -> list[dict]:
    return [
        {"id": "q_count", "kind": "class_cardinality", "question": "How many rugs?",
         "answer": 0, "expected": 1, "outcome": "wrong", "matched_uids": [],
         "source": "typed_graph"},
        {"id": "q_support", "kind": "support_relation",
         "question": "What supports the cushion?", "answer": None,
         "expected": "cushion ON_ENTITY_SURFACE sofa", "outcome": "unanswered",
         "reason": "the delivered graph emits no ON_ENTITY_SURFACE edge of any kind",
         "source": "typed_graph"},
    ]


def summary(correct: int, wrong: int, unanswered: int) -> dict:
    answered = correct + wrong
    total = correct + wrong + unanswered
    return {
        "n_questions": total,
        "tally": {"correct": correct, "wrong": wrong, "unanswered": unanswered},
        "coverage": round(answered / total, 4),
        "exact_accuracy_all": round(correct / total, 4),
        "accuracy_when_answered": round(correct / answered, 4) if answered else None,
        "false_confident_rate": round(wrong / answered, 4) if answered else None,
    }


def report(arms: dict, decision: dict) -> dict:
    return {
        "schema": SCHEMA,
        "stage": "scored",
        "scene_id": "scene_test",
        "inputs": {
            "key": "key.json", "key_sha256": "key_hash",
            "packet": "packet.json", "packet_sha256": "packet_hash",
            "entities": {"n_entities": 35},
            "graph": {"edge_types": {"NEAR": 151}, "n_edges": 151},
            "graph_geometry_compatibility": {
                "status": "exact_geometry_match",
                "geometry_signature": "geo_sig",
            },
            "direct_responses": None,
        },
        "question_scope": {
            "included": ["q_count", "q_support"],
            "excluded": ["q_uid"],
            "reason": "room-observable questions only",
        },
        "arms": arms,
        "decision": decision,
        "limits": ["One human-reviewed scene is a screening test."],
    }


def partial_report() -> dict:
    rows = structured_rows()
    arms = {
        "object_retrieval": {"rows": rows, "summary": summary(0, 1, 1)},
        "typed_graph": {"rows": rows, "summary": summary(0, 1, 1)},
    }
    out = report(arms, {"status": "pending_direct_vlm", "proceed": None})
    out["stage"] = "partial_pending_direct_vlm"
    return out


def full_report(proceed: bool = False) -> dict:
    rows = structured_rows()
    visual = [
        {"id": "q_count", "kind": "class_cardinality", "question": "How many rugs?",
         "answer": 1, "expected": 1, "outcome": "correct", "confidence": 0.8,
         "evidence_frame_ids": ["frame_00_1.000", "frame_01_2.000"],
         "evidence_sufficient": True, "source": "direct_vlm"},
        {"id": "q_support", "kind": "support_relation",
         "question": "What supports the cushion?", "answer": "sofa",
         "expected": "cushion ON_ENTITY_SURFACE sofa", "outcome": "correct",
         "confidence": 0.9, "evidence_frame_ids": ["frame_00_1.000"],
         "evidence_sufficient": False, "source": "direct_vlm"},
    ]
    routed = [
        dict(visual[0], route="visual_evidence"),
        dict(visual[1], route="visual_evidence", outcome="unanswered", answer=None,
             insufficiency="fewer than two valid cited views"),
    ]
    arms = {
        "object_retrieval": {"rows": rows, "summary": summary(0, 1, 1)},
        "typed_graph": {"rows": rows, "summary": summary(0, 1, 1)},
        "direct_vlm": {"rows": visual, "summary": summary(2, 0, 0)},
        "hybrid": {"rows": routed, "summary": summary(1, 0, 1)},
    }
    return report(arms, {
        "status": "measured",
        "proceed": proceed,
        "absolute_accuracy_gain_vs_best_single": -0.5,
        "wrong_answer_reduction_vs_lowest_single": 1.0,
        "hybrid_coverage": 0.5,
        "rules": {"accuracy": "gain >= 0.10",
                  "safety": "wrong-answer reduction >= 0.30 at coverage >= 0.60",
                  "interpretation": "screening decision only"},
    })


def sheet() -> Path:
    path = Path(tempfile.mkdtemp()) / "contact_sheet.jpg"
    Image.new("RGB", (12, 8), "white").save(path)
    return path


def test_page_is_static_and_self_contained():
    page = build_html(full_report(), packet(), sheet())
    lowered = page.lower()
    assert "<script" not in lowered
    assert "javascript:" not in lowered
    assert "onclick" not in lowered
    # No external fetch of any kind: the only src/href is the inlined sheet.
    assert "http://" not in lowered and "https://" not in lowered
    assert "src=\"data:image/jpeg;base64," in page


def test_partial_report_marks_unscored_arms_pending():
    page = build_html(partial_report(), packet(), sheet())
    assert "pending — arm not scored" in page
    assert "Decision pending" in page
    # A pending arm must not be rendered as a scored zero.
    assert page.count("pending — arm not scored") >= 2


def test_committed_values_render_verbatim():
    rep = full_report()
    page = build_html(rep, packet(), sheet())
    assert "0.500" in page          # hybrid coverage, from the report
    assert "-0.5000" in page        # negative gain shown, not hidden
    assert "gain &gt;= 0.10" in page
    assert "geo_sig" in page
    assert "NEAR: 151" in page
    assert "the delivered graph emits no ON_ENTITY_SURFACE edge of any kind" in page
    assert "fewer than two valid cited views" in page
    assert "frame_00_1.000" in page


def test_decision_verdict_follows_committed_proceed_flag():
    assert "<strong>STOP</strong>" in build_html(full_report(False), packet(), sheet())
    assert "<strong>PROCEED</strong>" in build_html(full_report(True), packet(), sheet())


def test_rejects_report_scored_against_a_different_packet():
    rep = full_report()
    rep["inputs"]["packet_sha256"] = "other_hash"
    try:
        build_html(rep, packet(), sheet())
    except ValueError as exc:
        assert "packet" in str(exc)
    else:
        raise AssertionError("mismatched packet hash must not render")


def test_rejects_foreign_schema_and_scene():
    rep = full_report()
    rep["schema"] = "something_else_v9"
    try:
        build_html(rep, packet(), sheet())
    except ValueError as exc:
        assert "schema" in str(exc)
    else:
        raise AssertionError("foreign schema must not render")
    rep = full_report()
    rep["scene_id"] = "other_scene"
    try:
        build_html(rep, packet(), sheet())
    except ValueError as exc:
        assert "scene" in str(exc)
    else:
        raise AssertionError("scene mismatch must not render")


def test_answer_formatting_preserves_type_semantics():
    assert fmt_answer(None) == '<span class="none">—</span>'
    assert fmt_answer(False) == "<code>false</code>"
    assert fmt_answer(0) == "<code>0</code>"
    assert fmt_answer(True) == "<code>true</code>"
    assert fmt_number(None) == "—"
    assert fmt_number(0.8333) == "0.833"
    assert fmt_number(0) == "0"


def test_escapes_untrusted_report_text():
    rep = full_report()
    rep["limits"] = ["<img src=x onerror=alert(1)>"]
    page = build_html(rep, packet(), sheet())
    assert "<img src=x" not in page
    assert "&lt;img src=x" in page


def test_output_is_deterministic():
    rep, pkt, img = full_report(), packet(), sheet()
    assert build_html(rep, pkt, img) == build_html(rep, pkt, img)


def main() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
