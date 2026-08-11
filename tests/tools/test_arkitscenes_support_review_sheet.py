"""Support-relation review sheet: sampling honesty, separation, inertness.

Four properties, each guarding a way this sheet could quietly become useless:

  * owner-confirmed positives appear even when the algorithm REJECTS them.
    `obj_23 -> obj_13` sits at footprint overlap 0.4545 against a 0.50 gate; a
    sheet built from candidates alone would omit the one pair that proves the
    gate is wrong, and the resulting key could only ever agree with the code.
  * human judgement and system metrics live in separate blocks joined by
    pair_id. Mixing them is how a threshold gets fitted to the evidence that
    produced it.
  * sampling is deterministic, so two runs produce the same sheet.
  * the tool changes nothing: no relations import, no threshold written.
"""
from __future__ import annotations

import ast
import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.arkitscenes_support_review_sheet import (
    BANDS, DEFINITION, OWNER_CONFIRMED, best_patch, select_rows,
)

TOOL = REPO_ROOT / "tools" / "arkitscenes_support_review_sheet.py"
SHEET = (REPO_ROOT / "eval" / "human_feedback"
         / "arkitscenes_41069025_support_review.json")
REST = (REPO_ROOT / "runs" / "arkit_vertical_slice" / "sealed_pair_rest"
        / "41069025" / "entity_patch_rest.json")


def _pair(target: str, owner: str, overlap: float | None,
          candidate: bool = False, patch: str = "p0") -> dict:
    patches = []
    if overlap is not None:
        patches = [{"patch_uid": patch, "overlap_ratio_target": overlap,
                    "vertical_gap_at_overlap_centroid_m": 0.01,
                    "patch_footprint_xy_area_m2": 0.3,
                    "target_footprint_area_m2": 0.2,
                    "rejection_reasons": []}]
    return {"target_entity_uid": target, "owner_entity_uid": owner,
            "relation_candidate": candidate,
            "rejection_reasons": [] if candidate else ["rejected"],
            "selected_patch_uid": patch if candidate else None,
            "evaluated_patches": patches}


def test_owner_confirmed_pairs_survive_algorithm_rejection() -> None:
    """The load-bearing property: a rejected positive must still be reviewed."""
    pairs = [_pair("obj_9", "obj_13", 0.82, candidate=True),
             _pair("obj_23", "obj_13", 0.4545),          # rejected
             _pair("obj_1", "obj_2", 0.9)]
    rows, _ = select_rows(pairs)
    by_id = {r["pair_id"]: r for r in rows}
    for target, owner in OWNER_CONFIRMED:
        pair_id = f"{target}->{owner}"
        assert pair_id in by_id, f"{pair_id} was dropped from the sheet"
        assert "owner_confirmed" in by_id[pair_id]["sources"], by_id[pair_id]
    assert by_id["obj_23->obj_13"]["system"]["relation_candidate"] is False, \
        "the fixture no longer exercises a REJECTED confirmed positive"


def test_a_missing_confirmed_pair_is_an_error_not_a_silent_drop() -> None:
    try:
        select_rows([_pair("obj_1", "obj_2", 0.5)])
    except ValueError as exc:
        assert "absent from the resting artifact" in str(exc), exc
    else:
        raise AssertionError("a missing owner-confirmed pair should raise")


def test_sampling_is_deterministic() -> None:
    pairs = [_pair("obj_9", "obj_13", 0.82, candidate=True),
             _pair("obj_23", "obj_13", 0.4545)]
    pairs += [_pair(f"obj_{i}", "obj_30", 0.9 - i * 0.02) for i in range(40)]
    first, bands_a = select_rows(pairs)
    second, bands_b = select_rows(pairs)
    assert [r["pair_id"] for r in first] == [r["pair_id"] for r in second]
    assert bands_a == bands_b


def test_band_budgets_are_respected() -> None:
    pairs = [_pair("obj_9", "obj_13", 0.82, candidate=True),
             _pair("obj_23", "obj_13", 0.4545)]
    # 30 rejects sitting in the top band, against a budget of 10.
    pairs += [_pair(f"obj_{i}", "obj_31", 0.95) for i in range(100, 130)]
    _, bands = select_rows(pairs)
    top_name, _, _, budget = BANDS[0]
    assert bands[top_name] <= budget, (bands, budget)


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
    assert not offenders, \
        f"the review sheet imports support machinery: {offenders}"


def test_emitted_form_keeps_truth_and_evidence_apart() -> None:
    if not SHEET.is_file():
        print("  skip: support review form not generated yet")
        return
    form = json.loads(SHEET.read_text())
    assert form["definition"] == DEFINITION, form["definition"]
    assert form["provenance"]["logic_changed"] is False
    assert form["provenance"]["thresholds_changed"] is False

    metric_keys = {"overlap_ratio_target", "vertical_gap_m", "patch_footprint_m2",
                   "relation_candidate", "rejection_reasons", "best_patch_uid"}
    for entry in form["human_relation_truth"]:
        assert not (set(entry) & metric_keys), \
            f"a metric leaked into human truth: {entry}"
        assert set(entry) <= {"pair_id", "judgement", "source"}, entry
    for entry in form["system_candidate_evidence"]:
        assert "judgement" not in entry, \
            f"a human judgement leaked into system evidence: {entry}"

    truth_ids = {e["pair_id"] for e in form["human_relation_truth"]}
    evidence_ids = {e["pair_id"] for e in form["system_candidate_evidence"]}
    assert truth_ids == evidence_ids, "the two blocks do not share a pair set"

    # The confirmed positives are pre-filled; everything else is blank.
    prefilled = {e["pair_id"] for e in form["human_relation_truth"]
                 if e["judgement"] is not None}
    assert prefilled == {f"{t}->{o}" for t, o in OWNER_CONFIRMED}, prefilled


def test_every_qualifying_patch_is_represented() -> None:
    """Dataset-guarded. Patches collapse to fewer pairs; none may vanish."""
    if not (SHEET.is_file() and REST.is_file()):
        print("  skip: dataset artifacts not present")
        return
    form = json.loads(SHEET.read_text())
    artifact = json.loads(REST.read_text())
    all_patches = {p["patch_uid"] for pair in artifact["pairs"]
                   for p in pair["evaluated_patches"]}
    assert form["provenance"]["n_qualifying_patches"] == len(all_patches)

    reviewed = {(e["target_uid"], e["owner_uid"])
                for e in form["system_candidate_evidence"]}
    for patch_uid in all_patches:
        best = None
        for pair in artifact["pairs"]:
            for patch in pair["evaluated_patches"]:
                if patch["patch_uid"] != patch_uid:
                    continue
                score = patch["overlap_ratio_target"]
                if best is None or score > best[0]:
                    best = (score, (pair["target_entity_uid"],
                                    pair["owner_entity_uid"]))
        assert best[1] in reviewed, \
            f"patch {patch_uid} has no reviewed pair"


TESTS = [
    test_owner_confirmed_pairs_survive_algorithm_rejection,
    test_a_missing_confirmed_pair_is_an_error_not_a_silent_drop,
    test_sampling_is_deterministic,
    test_band_budgets_are_respected,
    test_the_tool_changes_no_support_logic,
    test_emitted_form_keeps_truth_and_evidence_apart,
    test_every_qualifying_patch_is_represented,
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
