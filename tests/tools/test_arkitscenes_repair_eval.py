"""Detection-repair CLI: proposal/oracle ordering, and baseline reproduction.

The load-bearing test is `test_oracle_is_imported_after_proposals_are_finalized`.
It AST-scans the CLI to prove that the annotation loader is reached only after
`build_proposals` has returned, and that it is not a module-level import that
could run first. `eval/detection_repair.py` re-verifies digests at scoring
time, so this is the second of two independent locks on the same ordering.

The reproduction checks are dataset-guarded: they read whatever
`tools/arkitscenes_repair_eval.py` last wrote and compare it against the
numbers recorded in `docs/arkitscenes_mask3d_contract.md`. They do not re-run
the evaluation.
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

CLI = REPO_ROOT / "tools" / "arkitscenes_repair_eval.py"
RUN_ROOT = REPO_ROOT / "runs" / "arkitscenes_repair"

# From docs/arkitscenes_mask3d_contract.md and the Checkpoint 2 record in
# docs/arkit_vertical_slice_72h.md. The repair evaluator is independent code;
# it has to land on the same numbers from the same frozen bundle or one of the
# two is wrong.
RECORDED = {
    "arkitscenes_41069021": {"n_proposals": 37, "n_entities": 18,
                             "recovered_050": 7, "recovered_025": 11},
    "arkitscenes_41069025": {"n_proposals": 42, "n_entities": 20,
                             "recovered_050": 9},
}


def _oracle_import_nodes(tree: ast.AST) -> list[ast.ImportFrom]:
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom)
            and n.module == "tools.arkitscenes_eval"
            and any(a.name == "load_oracle_entities" for a in n.names)]


def test_oracle_is_imported_after_proposals_are_finalized() -> None:
    """The annotation loader must be reached only inside `evaluate`, and only
    after `build_proposals` has returned finalized artifacts."""
    tree = ast.parse(CLI.read_text())
    imports = _oracle_import_nodes(tree)
    assert len(imports) == 1, \
        f"expected exactly one load_oracle_entities import, found {len(imports)}"
    node = imports[0]

    module_level = [n for n in tree.body if isinstance(n, ast.ImportFrom)]
    assert node not in module_level, \
        "load_oracle_entities is imported at module level; it must sit below " \
        "the oracle boundary inside evaluate()"

    evaluate = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "evaluate")
    assert any(n is node for n in ast.walk(evaluate)), \
        "load_oracle_entities is not imported inside evaluate()"

    build_calls = [n.lineno for n in ast.walk(evaluate)
                   if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Name)
                   and n.func.id == "build_proposals"]
    assert build_calls, "evaluate() never calls build_proposals"
    assert node.lineno > max(build_calls), (
        f"annotations are opened on line {node.lineno}, before the last "
        f"build_proposals call on line {max(build_calls)}")


def test_proposal_builder_half_never_reaches_the_oracle() -> None:
    """`build_proposals` and its loaders stay annotation-free."""
    tree = ast.parse(CLI.read_text())
    for name in ("build_proposals", "load_baseline_bank", "load_repair_bank"):
        fn = next(n for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == name)
        for sub in ast.walk(fn):
            if isinstance(sub, (ast.Import, ast.ImportFrom)):
                raise AssertionError(
                    f"{name}() contains an import; the annotation-free half "
                    "must not pull anything in at call time")
            if isinstance(sub, ast.Name) and sub.id == "load_oracle_entities":
                raise AssertionError(f"{name}() references the oracle loader")


def test_reports_carry_the_interpretation_limit() -> None:
    """Every comparison must state that these are detection metrics only."""
    reports = sorted(RUN_ROOT.glob("*/detection_repair_eval.json"))
    if not reports:
        print("  skip: no detection_repair_eval.json written yet")
        return
    for path in reports:
        report = json.loads(path.read_text())
        comparison = report.get("comparison")
        if comparison is None:
            continue
        assert "not measure labelling" in comparison["interpretation_limit"], \
            f"{path} lost its interpretation limit"


def test_baseline_reproduces_the_recorded_contract_numbers() -> None:
    """Dataset-guarded: the independent evaluator must agree with the record."""
    checked = 0
    for scene_id, expected in RECORDED.items():
        path = RUN_ROOT / scene_id / "detection_repair_eval.json"
        if not path.is_file():
            continue
        report = json.loads(path.read_text())
        baseline = report["baseline"]
        assert report["n_entities"] == expected["n_entities"], \
            f"{scene_id}: {report['n_entities']} entities, expected " \
            f"{expected['n_entities']}"
        assert baseline["n_proposals"] == expected["n_proposals"], \
            f"{scene_id}: {baseline['n_proposals']} proposals, expected " \
            f"{expected['n_proposals']}"
        assert baseline["n_recovered"]["0.50"] == expected["recovered_050"], \
            f"{scene_id}: {baseline['n_recovered']['0.50']} recovered @0.50, " \
            f"expected {expected['recovered_050']}"
        if "recovered_025" in expected:
            assert baseline["n_recovered"]["0.25"] == expected["recovered_025"], \
                f"{scene_id}: {baseline['n_recovered']['0.25']} recovered " \
                f"@0.25, expected {expected['recovered_025']}"
        assert baseline["giant_mask_rate"] == 0.0, \
            f"{scene_id}: baseline giant-mask rate is not zero"
        checked += 1
    if not checked:
        print("  skip: no baseline reports on disk")


def test_baseline_bank_manifest_declares_no_annotations() -> None:
    """The finalized artifact records that it predates the oracle."""
    manifests = sorted(RUN_ROOT.glob("*/baseline_bank.manifest.json"))
    if not manifests:
        print("  skip: no finalized baseline bank on disk")
        return
    for path in manifests:
        manifest = json.loads(path.read_text())
        assert manifest["annotations_read"] is False, path
        assert len(manifest["proposal_sha256"]) == 64, path
        assert manifest["provenance"]["bank"] == "mask3d_ms02", path
        assert manifest["provenance"]["min_score"] == 0.2, path


TESTS = [
    test_oracle_is_imported_after_proposals_are_finalized,
    test_proposal_builder_half_never_reaches_the_oracle,
    test_reports_carry_the_interpretation_limit,
    test_baseline_reproduces_the_recorded_contract_numbers,
    test_baseline_bank_manifest_declares_no_annotations,
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
