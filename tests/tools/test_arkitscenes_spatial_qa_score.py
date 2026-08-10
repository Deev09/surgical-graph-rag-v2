"""Human spatial-QA key and scorer: independence, counting, and `unanswered`.

Three properties matter more than the score itself:

  * the key never names a delivered instance id, so it survives a perception
    change and cannot be the system scoring itself;
  * `segment_17` is the label stage's placeholder for "not admitted" and must
    not be counted as a label — treating it as one reported 35/35 labelled
    against a stage that admitted 27;
  * a missing relation type is `unanswered`, not `wrong`.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.arkitscenes_spatial_qa_score import (
    DEFAULT_KEY, is_anonymous, matches, normalize, score, scene_footprint,
)

KEY_V2 = (REPO_ROOT / "eval" / "human_feedback"
          / "arkitscenes_41069025_spatial_qa_key_v2.json")

RUN_ROOT = REPO_ROOT / "runs" / "arkit_spatial_qa"


def _entity(uid: str, label: str | None, aabb=None) -> dict:
    return {"uid": uid, "display_label": label,
            "aabb": aabb or [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], "top_k": []}


def _key() -> dict:
    return json.loads(DEFAULT_KEY.read_text())


def test_the_key_never_names_a_delivered_instance() -> None:
    """Independence: a key built on obj_N would be the system judging itself."""
    key = _key()
    for item in key["independent_questions"]:
        blob = json.dumps({k: v for k, v in item.items()
                           if k not in {"review_basis", "scoring"}})
        assert "obj_" not in blob, f"{item['id']} names a delivered uid: {blob}"
    # The uid-referencing facts are quarantined, not silently used.
    assert key["unresolved_uid_mappings"], "no unresolved mappings flagged"
    assert any("obj_" in json.dumps(x) for x in key["deliberately_excluded"]), \
        "the excluded section should record which uid facts were left out"


def test_v2_records_confirmed_cardinalities_and_supersession() -> None:
    """Owner confirmed four counts; v1 stays intact because a score cites it."""
    v2 = json.loads(KEY_V2.read_text())
    assert v2["status"] == "DRAFT_AWAITING_UID_CONFIRMATION", v2["status"]
    assert v2["supersedes"]["file"].endswith("key_v1.json")
    cushion = next(q for q in v2["independent_questions"]
                   if q["id"] == "q5_cushion_cardinality")
    assert cushion["kind"] == "class_cardinality" and cushion["expected"] == 2
    resolved = [m for m in v2["unresolved_uid_mappings"]
                if m["status"] == "RESOLVED"]
    assert [m["item"] for m in resolved] == ["cushion cardinality"], resolved
    # The three genuine UID mappings stay open, with cardinality noted as fixed.
    still_open = [m for m in v2["unresolved_uid_mappings"]
                  if m["status"] == "UNRESOLVED"]
    assert len(still_open) == 3, still_open
    for m in still_open:
        assert "CONFIRMED by owner" in m["cardinality_status"], m
    # v2 must still name no delivered instance in any question.
    for item in v2["independent_questions"]:
        blob = json.dumps({k: v for k, v in item.items()
                           if k not in {"review_basis", "scoring"}})
        assert "obj_" not in blob, item["id"]


def test_key_declares_draft_status_and_limits() -> None:
    key = _key()
    assert key["status"] == "DRAFT_PENDING_OWNER_CONFIRMATION", key["status"]
    assert "not a benchmark" in key["interpretation_limit"]
    assert key["provenance"]["derived_from"].endswith(
        "arkitscenes_sealed_visual_review_2026-08-09.json")


def test_anonymous_placeholders_are_not_labels() -> None:
    for placeholder in ("segment_0", "segment_21", "SEGMENT_7", "segment-3", ""):
        assert is_anonymous(placeholder), placeholder
        assert normalize(placeholder) is None, placeholder
    for real in ("chair", "trash-can", "tv-monitor"):
        assert not is_anonymous(real), real
    assert normalize("trash_can") == "trash-can"
    assert normalize("TV Monitor") == "tv-monitor"


def test_sofa_accepts_couch_and_nothing_else() -> None:
    assert matches("sofa", "sofa") and matches("couch", "sofa")
    assert not matches("cushion", "sofa")
    assert not matches("armchair", "sofa")
    assert matches("cushion", "cushion")


def test_cardinality_counts_only_admitted_labels() -> None:
    key = _key()
    entities = [_entity("obj_0", "rug"), _entity("obj_1", "segment_1"),
                _entity("obj_2", "trash-can"), _entity("obj_3", "trash-can")]
    results = {r["id"]: r for r in score(key, entities, [])}
    assert results["q1_rug_cardinality"]["answer"] == 1
    assert results["q1_rug_cardinality"]["outcome"] == "correct"
    # Two trash cans against an expected one.
    assert results["q2_trash_can_cardinality"]["answer"] == 2
    assert results["q2_trash_can_cardinality"]["outcome"] == "wrong"
    # The anonymous instance contributed to nothing.
    assert "obj_1" not in json.dumps(results)


def test_a_missing_relation_type_is_unanswered_not_wrong() -> None:
    key = _key()
    entities = [_entity("obj_0", "cushion"), _entity("obj_1", "sofa")]
    near_only = [{"type": "NEAR", "source": {"uid": "obj_0"},
                  "target": {"uid": "obj_1"}}]
    row = {r["id"]: r for r in score(key, entities, near_only)}["q6_cushion_on_sofa"]
    assert row["outcome"] == "unanswered", row
    assert "no ON_ENTITY_SURFACE edge" in row["reason"], row

    # Present but pointing the wrong way round is WRONG, not unanswered.
    backwards = [{"type": "ON_ENTITY_SURFACE", "source": {"uid": "obj_1"},
                  "target": {"uid": "obj_0"}}]
    row = {r["id"]: r for r in score(key, entities, backwards)}["q6_cushion_on_sofa"]
    assert row["outcome"] == "wrong", row

    correct = [{"type": "ON_ENTITY_SURFACE", "source": {"uid": "obj_0"},
                "target": {"uid": "obj_1"}}]
    row = {r["id"]: r for r in score(key, entities, correct)}["q6_cushion_on_sofa"]
    assert row["outcome"] == "correct", row


def test_room_spanning_counter_is_caught() -> None:
    key = _key()
    small = _entity("obj_0", "counter", [[0.0, 0.0, 0.0], [1.0, 1.0, 0.2]])
    far = _entity("obj_1", "table", [[0.0, 0.0, 0.0], [10.0, 10.0, 1.0]])
    ok = {r["id"]: r for r in score(key, [small, far], [])}
    assert ok["q7_counter_is_not_room_spanning"]["outcome"] == "correct"

    spanning = _entity("obj_2", "counter", [[0.0, 0.0, 0.0], [9.0, 9.0, 0.2]])
    bad = {r["id"]: r for r in score(key, [spanning, far], [])}
    assert bad["q7_counter_is_not_room_spanning"]["outcome"] == "wrong"
    assert bad["q7_counter_is_not_room_spanning"]["room_spanning"], bad

    assert scene_footprint([small, far]) == 100.0


def test_recorded_report_is_evaluation_only() -> None:
    """Dataset-guarded: whatever the scorer last wrote must say so."""
    reports = sorted(RUN_ROOT.glob("*_human_spatial_qa.json"))
    if not reports:
        print("  skip: no spatial-QA report on disk")
        return
    for path in reports:
        report = json.loads(path.read_text())
        assert report["evaluation_only"] is True, path
        assert report["perception_changed"] is False, path
        assert len(report["key_sha256"]) == 64, path
        assert report["tally"]["correct"] + report["tally"]["wrong"] \
            + report["tally"]["unanswered"] == len(report["results"]), path
        # The two headline scores must differ whenever anything is unanswered.
        if report["tally"]["unanswered"]:
            assert (report["score_excluding_unanswered"]
                    != report["score_counting_unanswered_as_failure"]), path


TESTS = [
    test_the_key_never_names_a_delivered_instance,
    test_v2_records_confirmed_cardinalities_and_supersession,
    test_key_declares_draft_status_and_limits,
    test_anonymous_placeholders_are_not_labels,
    test_sofa_accepts_couch_and_nothing_else,
    test_cardinality_counts_only_admitted_labels,
    test_a_missing_relation_type_is_unanswered_not_wrong,
    test_room_spanning_counter_is_caught,
    test_recorded_report_is_evaluation_only,
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
