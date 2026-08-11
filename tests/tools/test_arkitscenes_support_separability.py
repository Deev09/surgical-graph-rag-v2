"""Separability analysis: classify honestly, change nothing.

The analysis exists to stop a threshold being moved for the wrong reason, so
the tests are about the distinctions that decide that:

  * a positive with no patch at the contact height is `evidence_missing`, not a
    threshold failure. Calling it a threshold failure would aim the next change
    at a gate that cannot possibly recover it.
  * separability is reported for one feature and for two, because the one that
    matters is whether OVERLAP ALONE separates -- that is the gate under
    discussion.
  * the tool refuses a key that is still under re-check.
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

from tools.arkitscenes_support_separability import CONTACT_BAND_M, patch_views

TOOL = REPO_ROOT / "tools" / "arkitscenes_support_separability.py"
REPORT = (REPO_ROOT / "runs" / "arkit_support_calibration"
          / "arkitscenes_41069025_support_separability.json")


def _pair(*patches) -> dict:
    return {"evaluated_patches": [
        {"patch_uid": f"p{i}", "overlap_ratio_target": ov,
         "vertical_gap_at_overlap_centroid_m": gap}
        for i, (ov, gap) in enumerate(patches)]}


def test_contact_band_census_distinguishes_the_two_failure_modes() -> None:
    # A patch right at the contact height: threshold-reachable.
    reachable = patch_views(_pair((0.45, -0.01), (0.07, -0.33)))
    assert reachable["n_in_contact_band"] == 1, reachable
    # Every patch far from contact: no gate can recover this.
    missing = patch_views(_pair((0.39, -0.81), (0.08, -0.80), (0.02, -0.80)))
    assert missing["n_in_contact_band"] == 0, missing


def test_contact_and_overlap_selection_can_disagree() -> None:
    """The stage picks the biggest overlap; that need not be the contact."""
    view = patch_views(_pair((0.90, -0.80), (0.10, -0.005)))
    assert view["by_overlap"]["patch_uid"] == "p0", view
    assert view["by_contact"]["patch_uid"] == "p1", view
    assert view["by_overlap"]["patch_uid"] != view["by_contact"]["patch_uid"]


def test_patches_without_overlap_or_gap_are_ignored() -> None:
    view = patch_views(_pair((0.0, None), (0.5, None)))
    assert view["by_contact"] is None and view["n_in_contact_band"] == 0, view
    assert patch_views({"evaluated_patches": []})["by_overlap"] is None


def test_contact_band_boundary() -> None:
    inside = patch_views(_pair((0.5, -(CONTACT_BAND_M - 0.001))))
    outside = patch_views(_pair((0.5, -(CONTACT_BAND_M + 0.001))))
    assert inside["n_in_contact_band"] == 1
    assert outside["n_in_contact_band"] == 0


def test_the_tool_changes_nothing() -> None:
    tree = ast.parse(TOOL.read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    offenders = [m for m in imported
                 if m.startswith(("relations", "geometry", "graph"))]
    assert not offenders, f"the analysis imports support machinery: {offenders}"
    # It must not write into the relations config either.
    assert "min_target_overlap_ratio" not in TOOL.read_text().split(
        '"""', 2)[-1], "the analysis names the gate outside its docstring"


def test_the_pair_test_in_the_relation_module_is_existential() -> None:
    """The withdrawn claim, pinned so it cannot come back.

    A patch-SELECTION bug can only exist if the module tests one selected
    patch. It does not: it evaluates every qualifying patch and accepts the
    pair when any passes. If this ever changes, the separability tool's
    reporting caveat becomes wrong and should be revisited.
    """
    module = (REPO_ROOT / "graph" / "relations" / "entity_patch_rest.py")
    tree = ast.parse(module.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and any(isinstance(s, ast.Assign)
                      and any(getattr(t, "id", "") == "evaluated"
                              for t in s.targets)
                      for s in ast.walk(n)))
    src = ast.get_source_segment(module.read_text(), fn)
    # Every qualifying patch is evaluated...
    assert "for patch in qualifying" in src, src[:400]
    # ...and the pair is a candidate when ANY evaluated result is one.
    assert "relation_candidate=selected is not None" in src
    assert "for result in evaluated if result.relation_candidate" in src


def test_report_does_not_call_a_reporting_artefact_a_selection_bug() -> None:
    if not REPORT.is_file():
        print("  skip: no separability report on disk")
        return
    report = json.loads(REPORT.read_text())
    assert "patch_selection" not in report, \
        "the withdrawn patch_selection claim is back in the report"
    block = report["patch_reporting"]
    assert "existential" in block["pair_test"], block
    assert "not_a_selection_bug" in block, block


def test_recorded_report_separates_one_feature_from_two() -> None:
    if not REPORT.is_file():
        print("  skip: no separability report on disk")
        return
    report = json.loads(REPORT.read_text())
    assert report["read_only"] is True
    assert report["logic_changed"] is False
    assert report["thresholds_changed"] is False
    assert report["key_status"] == "FINAL"

    sep = report["separability"]
    # Both questions must be answered, not conflated.
    assert "separable_on_overlap_alone" in sep
    assert "separable_in_2d" in sep
    # If overlap alone separated, the gate discussion would be trivial; the
    # recorded run says it does not, and the count of intruders proves it.
    if not sep["separable_on_overlap_alone"]:
        assert sep["negatives_above_min_positive_overlap"] > 0, sep

    modes = {c["mode"] for c in report["positives_by_failure_mode"]}
    assert modes <= {"found", "threshold_reachable", "evidence_missing"}, modes
    n_reach = sum(1 for c in report["positives_by_failure_mode"]
                  if c["mode"] != "evidence_missing")
    assert n_reach == sep["n_reachable_positives"], report
    assert "not a calibration" in report["caution"]


def test_recorded_report_precision_recall_agree_with_its_own_lists() -> None:
    if not REPORT.is_file():
        print("  skip: no separability report on disk")
        return
    stage = json.loads(REPORT.read_text())["current_stage"]
    tp, fp, fn = stage["true_positives"], stage["false_positives"], \
        stage["false_negatives"]
    assert not (set(tp) & set(fn)), (tp, fn)
    if tp or fp:
        assert abs(stage["precision"] - len(tp) / (len(tp) + len(fp))) < 1e-9
    if tp or fn:
        assert abs(stage["recall"] - len(tp) / (len(tp) + len(fn))) < 1e-9


TESTS = [
    test_contact_band_census_distinguishes_the_two_failure_modes,
    test_contact_and_overlap_selection_can_disagree,
    test_patches_without_overlap_or_gap_are_ignored,
    test_contact_band_boundary,
    test_the_tool_changes_nothing,
    test_the_pair_test_in_the_relation_module_is_existential,
    test_report_does_not_call_a_reporting_artefact_a_selection_bug,
    test_recorded_report_separates_one_feature_from_two,
    test_recorded_report_precision_recall_agree_with_its_own_lists,
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
