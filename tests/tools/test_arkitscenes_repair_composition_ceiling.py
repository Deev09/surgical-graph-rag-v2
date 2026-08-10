"""Oracle-guided composition ceiling: arithmetic, and the diagnostic firewall.

Two kinds of test. The arithmetic ones check that a union of parts is scored as
a union (not a sum of intersections, which double-counts overlap). The firewall
ones check that this tool cannot be mistaken for, or promoted into, a result:
it consults annotations to pick parts, so its numbers are an upper bound that
no oracle-free method can reach.
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

from tools.arkitscenes_repair_composition_ceiling import (
    PART_BUDGETS, TARGET_IOU, exhaustive_pair, greedy_union,
)

N = 10_000
RUN_ROOT = REPO_ROOT / "runs" / "arkitscenes_repair"


def _p(lo: int, hi: int) -> np.ndarray:
    return np.arange(lo, hi, dtype=np.int64)


def test_two_halves_compose_into_a_whole() -> None:
    """The case the diagnostic exists to detect."""
    entity = _p(0, 1000)
    parts = [_p(0, 500), _p(500, 1000)]
    out = greedy_union(parts, entity, N)
    assert out["1"]["iou"] == 0.5, out["1"]
    assert out["2"]["iou"] == 1.0, out["2"]
    assert out["2"]["n_parts"] == 2, out["2"]
    assert out["2"]["precision"] == 1.0 and out["2"]["recall"] == 1.0, out["2"]


def test_overlapping_parts_are_unioned_not_summed() -> None:
    """Summing intersections would report recall 1.5 and IoU above one."""
    entity = _p(0, 1000)
    parts = [_p(0, 750), _p(250, 1000)]
    out = greedy_union(parts, entity, N)
    assert out["2"]["iou"] == 1.0, out["2"]
    assert out["2"]["recall"] == 1.0, out["2"]
    assert out["2"]["union_vertices"] == 1000, out["2"]


def test_a_part_that_hurts_iou_is_not_added() -> None:
    """Greedy maximises IoU, so coverage that costs precision is refused."""
    entity = _p(0, 1000)
    # Adds 100 vertices of recall and 5000 of junk.
    parts = [_p(0, 900), _p(900, 6000)]
    out = greedy_union(parts, entity, N)
    assert out["16"]["n_parts"] == 1, out["16"]
    assert out["16"]["recall"] == 0.9, out["16"]
    assert out["16"]["iou"] == 0.9, out["16"]


def test_budgets_are_monotone_and_at_most_k() -> None:
    entity = _p(0, 1000)
    parts = [_p(i * 100, (i + 1) * 100) for i in range(10)]
    out = greedy_union(parts, entity, N)
    ious = [out[str(k)]["iou"] for k in PART_BUDGETS]
    assert ious == sorted(ious), ious
    for k in PART_BUDGETS:
        assert out[str(k)]["n_parts"] <= k, (k, out[str(k)])
    assert out["16"]["iou"] == 1.0, out["16"]


def test_parts_touching_nothing_are_ignored() -> None:
    entity = _p(0, 1000)
    parts = [_p(0, 1000), _p(5000, 6000)]
    out = greedy_union(parts, entity, N)
    assert out["n_candidates_touching_entity"] == 1, out
    assert out["16"]["iou"] == 1.0, out["16"]


def test_exhaustive_pair_finds_what_greedy_can_miss() -> None:
    """Greedy takes the best single first; the best PAIR may exclude it."""
    entity = _p(0, 1000)
    # Best single is C (IoU 0.6). But A+B tile the entity exactly (IoU 1.0),
    # and after taking C greedy cannot reach that.
    parts = [_p(0, 500), _p(500, 1000), np.concatenate([_p(0, 600), _p(2000, 2400)])]
    exact = exhaustive_pair(parts, entity, N)
    assert exact == 1.0, exact
    greedy = greedy_union(parts, entity, N)["2"]["iou"]
    assert greedy <= exact + 1e-9, (greedy, exact)


def test_the_report_is_marked_undeployable() -> None:
    """Dataset-guarded: whatever the tool last wrote must carry the firewall."""
    reports = sorted(RUN_ROOT.glob("*/composition_ceiling.json"))
    if not reports:
        print("  skip: no composition_ceiling.json on disk")
        return
    for path in reports:
        report = json.loads(path.read_text())
        assert report["oracle_guided"] is True, path
        assert report["deployable"] is False, path
        assert "upper bound" in report["interpretation_limit"], path
        assert report["target_iou"] == TARGET_IOU, path


def test_the_ceiling_never_feeds_a_proposal_artifact() -> None:
    """The firewall: this tool must not be able to mint or pool a bank."""
    path = REPO_ROOT / "tools" / "arkitscenes_repair_composition_ceiling.py"
    tree = ast.parse(path.read_text())

    # AST, not substring: the module docstring explains WHY it never touches a
    # ProposalArtifact, and that sentence is worth keeping. What must not exist
    # is a use of these names in code.
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            used.update(a.name for a in node.names)
    forbidden = {"ProposalArtifact", "Proposal", "development_gates",
                 "compare_banks", "score_bank"} & used
    assert not forbidden, (
        f"{path.name} uses {sorted(forbidden)}; the oracle-guided ceiling must "
        "never mint a finalized bank or reach the gate sheet")

    imported = [n.module for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) and n.module]
    assert "eval.detection_repair" not in imported, \
        "the ceiling tool imports the evaluator"


TESTS = [
    test_two_halves_compose_into_a_whole,
    test_overlapping_parts_are_unioned_not_summed,
    test_a_part_that_hurts_iou_is_not_added,
    test_budgets_are_monotone_and_at_most_k,
    test_parts_touching_nothing_are_ignored,
    test_exhaustive_pair_finds_what_greedy_can_miss,
    test_the_report_is_marked_undeployable,
    test_the_ceiling_never_feeds_a_proposal_artifact,
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
