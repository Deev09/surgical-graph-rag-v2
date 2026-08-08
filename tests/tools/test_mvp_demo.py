"""Tests for tools/mvp_demo.py + tools/mvp_report_html.py (MVP-v0 slice).

Run: python tests/tools/test_mvp_demo.py

Synthetic coverage: variant runner schema + determinism + C1 uid handling
on the two-cube fixture; HTML rendered purely from a hand-assembled JSON
report (no key files, no network). End-to-end on real room_2 (including
the committed-reference consistency assertion) runs only when the dataset
and the frozen ms02 bundle are present.
"""
from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.segmenter.test_c1_pipeline import PERFECT, _scene


def _key():
    return {
        "answer_key_type": "human_verified",
        "fixture_id": "synthetic_mvp_key",
        "questions": [{
            "question_id": "Q01",
            "question": "what is on the table?",
            "relation": "ON_ENTITY_SURFACE",
            "expected_outcome": "empty",
            "expected_must_contain": [],
            "expected_must_not_contain": [],
            "exhaustive": True,
            "candidate_labels": {},
        }],
    }


def _row(tmp: Path):
    room, bundle = _scene(tmp, PERFECT)
    return {"scene_id": "synthetic_scene", "short": "synthetic",
            "variants": ["A", "C1"], "room_dir": room,
            "bundle_dir": bundle, "key": _key()}


def _router_ctx():
    from reasoner.base import CompletenessProfile, ExecutionContext
    from reasoner.compiler_rules import RulesCompiler
    from reasoner.executor import RulesExecutor
    from reasoner.router import Router
    from reasoner.verbalizer import StandardVerbalizer
    router = Router(compiler=RulesCompiler(), executor=RulesExecutor(),
                    verbalizer=StandardVerbalizer())
    ctx = ExecutionContext(completeness=CompletenessProfile(
        source="oracle", entity_recall_by_class={}, edge_recall_by_type={}))
    return router, ctx


def test_variant_schema_and_determinism():
    from tools.mvp_demo import run_variant
    with tempfile.TemporaryDirectory() as td:
        row = _row(Path(td))
        router, ctx = _router_ctx()
        a1 = run_variant("A", row, router, ctx, min_vertices=4)
        a2 = run_variant("A", row, router, ctx, min_vertices=4)
        if json.dumps(a1, sort_keys=True) != json.dumps(a2, sort_keys=True):
            raise AssertionError("variant run must be deterministic")
        for field in ("graph_bundle_hash", "micro_precision", "micro_recall",
                      "per_relation", "questions", "n_graph_edges"):
            if field not in a1:
                raise AssertionError(f"missing field {field}")
        q = a1["questions"][0]
        if q["expected_outcome"] != "empty" or q["actual_outcome"] != "empty":
            raise AssertionError(f"empty table expected: {q}")
        if not isinstance(q["verbalized"], str) or not q["verbalized"]:
            raise AssertionError("verbalized text must be present")


def test_c1_variant_reports_matches():
    from tools.mvp_demo import run_variant
    with tempfile.TemporaryDirectory() as td:
        row = _row(Path(td))
        router, ctx = _router_ctx()
        c1 = run_variant("C1", row, router, ctx, min_vertices=4)
        if c1["entity_matches_at_05"] != "2/2":
            raise AssertionError(f"perfect synthetic pred must match 2/2: {c1}")


def test_html_renders_from_json_only():
    from tools.mvp_demo import run_variant
    from tools.mvp_report_html import build_html
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        row = _row(tmp)
        router, ctx = _router_ctx()
        variants = {v: run_variant(v, row, router, ctx, min_vertices=4) for v in ("A", "C1")}
        out = tmp / "out"
        out.mkdir()
        report = {"schema": "mvp_v0_report_v1", "scene_id": "synthetic_scene",
                  "c1_status": "frozen Mask3D @0.2 reference (synthetic)",
                  "key_questions": {"Q01": _key()["questions"][0]},
                  "provenance": {"git_commit": "test", "note": "synthetic"},
                  "variants": variants}
        (out / "synthetic_scene_mvp.json").write_text(
            json.dumps(report, indent=1, sort_keys=True))
        (out / "aggregate.json").write_text(json.dumps({
            "comparability": "synthetic", "git_commit": "testtesttest",
            "reference_check": "n/a (synthetic)",
            "headline": [{"scene_id": "synthetic_scene", "variant": v,
                          "micro_precision": variants[v]["micro_precision"],
                          "micro_recall": variants[v]["micro_recall"],
                          "support_hits": None,
                          "n_graph_edges": variants[v]["n_graph_edges"]}
                         for v in ("A", "C1")]}))
        html_path = build_html(out)
        html = html_path.read_text()
        if "synthetic_scene" not in html or "Disclosures" not in html:
            raise AssertionError("HTML missing expected sections")
        if "src=\"http" in html or "src='http" in html or "href='http" in html:
            raise AssertionError("HTML must make no external requests")


def test_e2e_room_2_reproduces_reference():
    """Dataset-guarded: full room_2 run incl. the hard reference check."""
    data = Path.home() / "Desktop/datasets/replica/room_2"
    bundle = REPO_ROOT / "runs" / "phase8_c1" / "bundles_ms02" / "room_2"
    if not data.exists() or not bundle.exists():
        print("  (skipped: dataset or ms02 bundle not present)")
        return
    from tools.mvp_demo import main as mvp_main
    with tempfile.TemporaryDirectory() as td:
        rc = mvp_main(["--scene", "replica_room_2", "--out-dir", td,
                       "--no-html"])
        if rc != 0:
            raise AssertionError(f"mvp_demo failed on room_2: rc={rc}")
        r = json.loads((Path(td) / "replica_room_2_mvp.json").read_text())
        c1 = r["variants"]["C1"]
        if c1["micro_precision"] != 1.0 or c1["micro_recall"] != 0.2449:
            raise AssertionError(f"reference drift: {c1['micro_precision']} "
                                 f"{c1['micro_recall']}")


def test_e2e_office_0_reproduces_reference():
    """Dataset-guarded: office human key + full A/B/C1/C2 reference check."""
    data = Path.home() / "Desktop/datasets/replica/office_0"
    bundle = REPO_ROOT / "runs" / "phase8_c1" / "bundles_ms02" / "office_0"
    sidecar = (REPO_ROOT / "eval" / "predictions" / "phase8_c2"
               / "replica_office_0_c2_labels.json")
    if not data.exists() or not bundle.exists() or not sidecar.exists():
        print("  (skipped: office dataset, ms02 bundle, or C2 sidecar not present)")
        return
    from tools.mvp_demo import main as mvp_main
    with tempfile.TemporaryDirectory() as td:
        rc = mvp_main(["--scene", "replica_office_0", "--out-dir", td,
                       "--no-html"])
        if rc != 0:
            raise AssertionError(f"mvp_demo failed on office_0: rc={rc}")
        r = json.loads((Path(td) / "replica_office_0_mvp.json").read_text())
        c2 = r["variants"]["C2"]
        if c2["micro_recall"] != 0.0:
            raise AssertionError(f"office C2 reference drift: {c2}")
        if c2["semantic_citation"]["accuracy"] != 0.3125:
            raise AssertionError(f"office C2 semantic drift: {c2}")


TESTS = [
    test_variant_schema_and_determinism,
    test_c1_variant_reports_matches,
    test_html_renders_from_json_only,
    test_e2e_room_2_reproduces_reference,
    test_e2e_office_0_reproduces_reference,
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
