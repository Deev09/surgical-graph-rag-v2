"""Evidence probe: locate the failure by vertices, not by patches.

The probe decides whether a missed support pair is worth fixing inside the
relation module at all. Getting that wrong wastes the next change, so the
tests pin the three verdicts and the statistic they turn on.

Coverage, not raw count, is the deciding statistic: a handful of fringe
vertices at the right height is not a supporting surface, and counting them
would send the next change to extraction when the real gap is segmentation.
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

import numpy as np

from tools.arkitscenes_support_evidence_probe import (
    COVERAGE_CELL_M, OWNER_COVERAGE_MIN, coverage, probe,
)

TOOL = REPO_ROOT / "tools" / "arkitscenes_support_evidence_probe.py"
REPORT = (REPO_ROOT / "runs" / "arkit_support_calibration"
          / "arkitscenes_41069025_obj_1_obj_14_evidence_probe.json")


def _scene(target_xy_span=0.30, owner_frac=0.0, unassigned_frac=0.0,
           height=0.60):
    """Target block above a plate whose ownership is split as requested."""
    grid = np.arange(0.0, target_xy_span, COVERAGE_CELL_M / 2)
    xx, yy = np.meshgrid(grid, grid)
    plate = np.stack([xx.ravel(), yy.ravel(),
                      np.full(xx.size, height)], axis=1)
    target = np.stack([xx.ravel(), yy.ravel(),
                       np.full(xx.size, height + 0.01)], axis=1)
    target = np.concatenate([target, target + [0, 0, 0.2]])

    n = len(plate)
    order = np.argsort(plate[:, 0] * 1000 + plate[:, 1])   # deterministic
    # Filler must be a POSITIVE instance id. Using a negative one made every
    # unclaimed plate vertex read as `unassigned`, so the "no surface at all"
    # fixture silently became a segmentation-failure fixture.
    ids = np.full(n, 99, dtype=np.int64)
    n_owner = int(owner_frac * n)
    n_unassigned = int(unassigned_frac * n)
    ids[order[:n_owner]] = 14
    ids[order[n_owner:n_owner + n_unassigned]] = -1
    xyz = np.concatenate([target, plate])
    all_ids = np.concatenate([np.full(len(target), 1, dtype=np.int64), ids])
    return xyz, all_ids


def test_owner_owns_the_surface_means_extraction_failed() -> None:
    xyz, ids = _scene(owner_frac=0.9)
    result = probe(xyz, ids, target_id=1, owner_id=14)
    assert result["verdict"] == "patch_extraction_failure", result
    assert result["by_owner"]["owner"]["footprint_coverage"] >= OWNER_COVERAGE_MIN


def test_unassigned_owns_the_surface_means_segmentation_failed() -> None:
    xyz, ids = _scene(owner_frac=0.03, unassigned_frac=0.9)
    result = probe(xyz, ids, target_id=1, owner_id=14)
    assert result["verdict"] == "segmentation_evidence_missing", result
    assert "no change inside the relation module" in result["because"]


def test_no_surface_at_all_is_its_own_verdict() -> None:
    xyz, ids = _scene(owner_frac=0.0, unassigned_frac=0.0)
    result = probe(xyz, ids, target_id=1, owner_id=14)
    assert result["verdict"] == "no_supporting_surface_in_mesh", result


def test_a_few_fringe_vertices_do_not_count_as_a_surface() -> None:
    """Raw count would say 'present'; coverage says otherwise."""
    xyz, ids = _scene(owner_frac=0.04, unassigned_frac=0.9)
    result = probe(xyz, ids, target_id=1, owner_id=14)
    owner = result["by_owner"]["owner"]
    assert owner["n_vertices"] > 0, owner
    assert owner["footprint_coverage"] < OWNER_COVERAGE_MIN, owner
    assert result["verdict"] != "patch_extraction_failure", result


def test_coverage_counts_cells_not_points() -> None:
    origin, span = np.array([0.0, 0.0]), np.array([0.10, 0.10])
    piled = np.tile(np.array([[0.01, 0.01, 0.0]]), (500, 1))
    cells, total, ratio = coverage(piled, origin, span)
    assert cells == 1 and ratio < 0.1, (cells, total, ratio)
    spread = np.stack([np.repeat(np.arange(0, 0.10, COVERAGE_CELL_M), 5),
                       np.tile(np.arange(0, 0.10, COVERAGE_CELL_M), 5),
                       np.zeros(25)], axis=1)
    cells2, _, ratio2 = coverage(spread, origin, span)
    assert cells2 > cells and ratio2 > ratio, (cells2, ratio2)


def test_the_probe_changes_nothing() -> None:
    tree = ast.parse(TOOL.read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    offenders = [m for m in imported
                 if m.startswith(("relations", "graph.relations"))]
    assert not offenders, f"the probe imports the relation module: {offenders}"


def test_recorded_probe_records_its_verdict_and_inertness() -> None:
    if not REPORT.is_file():
        print("  skip: no evidence probe on disk")
        return
    report = json.loads(REPORT.read_text())
    assert report["read_only"] is True
    assert report["logic_changed"] is False
    assert report["thresholds_changed"] is False
    assert report["verdict"] in {
        "patch_extraction_failure", "segmentation_evidence_missing",
        "no_supporting_surface_in_mesh"}, report["verdict"]
    # The coverage figures must actually support the verdict printed.
    owner = report["by_owner"]["owner"]["footprint_coverage"]
    if report["verdict"] == "patch_extraction_failure":
        assert owner >= report["owner_coverage_min"], report
    else:
        assert owner < report["owner_coverage_min"], report
    assert "existential" in report["pair_test_is_existential"]


TESTS = [
    test_owner_owns_the_surface_means_extraction_failed,
    test_unassigned_owns_the_surface_means_segmentation_failed,
    test_no_surface_at_all_is_its_own_verdict,
    test_a_few_fringe_vertices_do_not_count_as_a_surface,
    test_coverage_counts_cells_not_points,
    test_the_probe_changes_nothing,
    test_recorded_probe_records_its_verdict_and_inertness,
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
