"""Support-truth ingestion: keep every judgement, flag nothing away.

The ingest exists between a human and a calibration target, which is exactly
where data quietly goes missing. Four properties guard that:

  * a judgement is never altered or dropped, however odd it looks;
  * a pre-confirmed pair the form omitted is carried forward, not treated as a
    retraction -- the form only emits checked rows, so silence is a UI defect;
  * the consistency checks flag DIRECTION problems only, and specifically do
    not flag a target resting on a lower shelf;
  * nothing here changes support logic or a threshold.
"""
from __future__ import annotations

import ast
import json
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.arkitscenes_support_truth_ingest import (
    FLOOR_MARGIN_M, consistency_flags,
)

TOOL = REPO_ROOT / "tools" / "arkitscenes_support_truth_ingest.py"
KEY = (REPO_ROOT / "eval" / "human_feedback"
       / "arkitscenes_41069025_support_relation_key_v1.json")


def _boxes(**spec) -> dict:
    """uid -> aabb from (z_low, z_high); xy is irrelevant to these checks."""
    return {uid: [[0.0, 0.0, lo], [1.0, 1.0, hi]] for uid, (lo, hi) in spec.items()}


def test_a_book_on_a_middle_shelf_is_not_flagged() -> None:
    """The case the naive 'target above owner top' test would wrongly flag."""
    boxes = _boxes(book=(1.10, 1.35), shelf=(0.00, 2.00))
    flags = consistency_flags("book", "shelf", boxes, floor_z=0.0)
    assert flags == [], flags


def test_a_floor_standing_target_is_flagged() -> None:
    boxes = _boxes(column=(0.00, 2.30), bracket=(1.47, 1.84))
    flags = [f["flag"] for f in consistency_flags(
        "column", "bracket", boxes, floor_z=0.0)]
    assert "floor_standing_target" in flags, flags
    assert "target_below_owner" in flags, flags


def test_a_target_beneath_its_owner_is_flagged() -> None:
    """Usually the pair is right and the direction is reversed."""
    boxes = _boxes(below=(1.46, 2.29), slab=(2.29, 2.48))
    flags = [f["flag"] for f in consistency_flags(
        "below", "slab", boxes, floor_z=0.0)]
    assert flags == ["target_below_owner"], flags


def test_floor_margin_boundary() -> None:
    just_off = _boxes(t=(FLOOR_MARGIN_M + 0.01, 0.5), o=(0.0, 1.0))
    assert not any(f["flag"] == "floor_standing_target"
                   for f in consistency_flags("t", "o", just_off, 0.0))
    just_on = _boxes(t=(FLOOR_MARGIN_M - 0.01, 0.5), o=(0.0, 1.0))
    assert any(f["flag"] == "floor_standing_target"
               for f in consistency_flags("t", "o", just_on, 0.0))


def test_the_tool_changes_no_support_logic() -> None:
    tree = ast.parse(TOOL.read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    offenders = [m for m in imported
                 if m.startswith(("relations", "geometry", "graph"))]
    assert not offenders, f"the ingest imports support machinery: {offenders}"


def test_recorded_key_preserves_every_judgement() -> None:
    """Dataset-guarded: read whatever the ingest last wrote."""
    if not KEY.is_file():
        print("  skip: no support-relation key on disk")
        return
    key = json.loads(KEY.read_text())
    assert key["provenance"]["logic_changed"] is False
    assert key["provenance"]["thresholds_changed"] is False

    records = key["human_relation_truth"]
    counts = key["judgement_counts"]
    for value, expected in counts.items():
        got = sum(1 for r in records if str(r["judgement"]) == value)
        assert got == expected, (value, got, expected)

    # Flags never change a judgement: everything flagged is still a positive.
    flagged = {r["pair_id"] for r in key["needs_owner_recheck"]}
    by_id = {r["pair_id"]: r for r in records}
    for pair_id in flagged:
        assert by_id[pair_id]["judgement"] == "supports", by_id[pair_id]

    # Omitted pre-confirmed pairs are carried, and say so.
    for pair_id in key["coverage"]["rows_omitted"]:
        record = by_id[pair_id]
        assert record["judgement"] == "supports", record
        assert record["source"] == "owner_confirmed_prior_step", record
    assert "not a retraction" in key["coverage"]["omitted_reason"]


def test_recorded_key_reports_algorithm_agreement_without_tuning() -> None:
    if not KEY.is_file():
        print("  skip: no support-relation key on disk")
        return
    key = json.loads(KEY.read_text())
    agreement = key["algorithm_agreement"]
    positives = {r["pair_id"] for r in key["human_relation_truth"]
                 if r["judgement"] == "supports"}
    assert set(agreement["candidates_owner_confirms"]) <= positives
    assert set(agreement["owner_positives_algorithm_misses"]) <= positives
    assert (set(agreement["candidates_owner_confirms"])
            & set(agreement["owner_positives_algorithm_misses"])) == set()
    assert "no logic or threshold changed" in agreement["note"]


TESTS = [
    test_a_book_on_a_middle_shelf_is_not_flagged,
    test_a_floor_standing_target_is_flagged,
    test_a_target_beneath_its_owner_is_flagged,
    test_floor_margin_boundary,
    test_the_tool_changes_no_support_logic,
    test_recorded_key_preserves_every_judgement,
    test_recorded_key_reports_algorithm_agreement_without_tuning,
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
