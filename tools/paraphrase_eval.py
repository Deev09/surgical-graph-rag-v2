"""Phase 8 E4 — paraphrase robustness measurement for the rules compiler.

Runs eval/questions/paraphrase_bank.json through RulesCompiler and reports,
per group and aggregate:

  - parse coverage: compiled / parser_failure / out_of_schema rates,
  - parse_target_match: does a compiled paraphrase hit the SAME relation +
    anchor as the group's expected parse? (e.g. "what's close to the wall?"
    may compile through the generic NEAR-entity template instead of
    NEAR_SURFACE - compiled, but the WRONG parse),
  - answer_consistent: on a real scene (default room_0), does the compiled
    paraphrase cite the same UIDs as the canonical phrasing?

Gate vs measurement (do not confuse them):
  - HARD (exit code): every canonical phrasing compiles to its expected
    parse. That is regression protection for current behavior.
  - REPORTED ONLY: the paraphrase compiled-rate. It quantifies the regex
    compiler's brittleness (the known weakest link); it is a benchmark-
    definition measurement, never a pass/fail claim.

If the scene data is not on disk, the parse-coverage half still runs
(RulesCompiler never reads the bundle) and answer-consistency is recorded as
skipped - keeps the tool runnable without the dataset.

Output: runs/phase8_paraphrase/report.json (new path; touches nothing
existing). NOT comparable to eval_paraphrase.py (legacy v1 track) or any
phase scorecard.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reasoner.ast import EdgeConstraint, EntityClassRef, EntityRef, SurfaceRef, Variable
from reasoner.base import CompileResult
from reasoner.compiler_rules import RulesCompiler

BANK_PATH = REPO_ROOT / "eval" / "questions" / "paraphrase_bank.json"
DEFAULT_ROOM = Path.home() / "Desktop/datasets/replica/room_0"
DEFAULT_OUT = REPO_ROOT / "runs" / "phase8_paraphrase" / "report.json"


def parse_target(cr: CompileResult) -> dict | None:
    """Extract {edge_type, anchor_kind, anchor} from a compiled result's
    first constraint. The anchor is whichever operand is not the bind
    variable (SUPPORTS anchors on source; entity->surface relations anchor
    on target)."""
    if cr.outcome != "compiled" or cr.ast is None or not cr.ast.where:
        return None
    constraint = cr.ast.where[0]
    if not isinstance(constraint, EdgeConstraint):
        return None
    anchors = [op for op in (constraint.source, constraint.target)
               if not isinstance(op, Variable)]
    if len(anchors) != 1:
        return None
    anchor = anchors[0]
    if isinstance(anchor, SurfaceRef):
        return {"edge_type": constraint.type, "anchor_kind": "surface",
                "anchor": anchor.surface_type}
    if isinstance(anchor, EntityClassRef):
        return {"edge_type": constraint.type, "anchor_kind": "entity_class",
                "anchor": anchor.entity_class}
    if isinstance(anchor, EntityRef):
        return {"edge_type": constraint.type, "anchor_kind": "entity",
                "anchor": anchor.label}
    return None


def _row(question: str, compiler: RulesCompiler, expected: dict,
         bundle, router, ctx) -> dict:
    cr = compiler.compile(question, bundle)
    target = parse_target(cr)
    row = {
        "question": question,
        "compile_outcome": cr.outcome,
        "parse_target": target,
        "parse_target_match": (target == expected) if target is not None else None,
        "cited_uids": None,
    }
    if cr.outcome == "compiled" and router is not None:
        ans = router.answer(question, bundle, ctx)
        row["cited_uids"] = sorted(ans.cited_uids)
    return row


def run_group(group: dict, compiler: RulesCompiler, bundle, router, ctx) -> dict:
    expected = group["expected_parse"]
    canonical = _row(group["canonical"], compiler, expected, bundle, router, ctx)
    rows = [_row(p, compiler, expected, bundle, router, ctx)
            for p in group["paraphrases"]]
    for r in rows:
        r["answer_consistent"] = (
            r["cited_uids"] == canonical["cited_uids"]
            if r["cited_uids"] is not None and canonical["cited_uids"] is not None
            else None
        )
    n = len(rows)
    compiled = [r for r in rows if r["compile_outcome"] == "compiled"]
    return {
        "group_id": group["group_id"],
        "expected_parse": expected,
        "canonical": canonical,
        "canonical_ok": bool(canonical["compile_outcome"] == "compiled"
                             and canonical["parse_target_match"]),
        "paraphrases": rows,
        "summary": {
            "n": n,
            "compiled": len(compiled),
            "parser_failure": sum(r["compile_outcome"] == "parser_failure" for r in rows),
            "out_of_schema": sum(r["compile_outcome"] == "out_of_schema" for r in rows),
            "parse_target_match": sum(bool(r["parse_target_match"]) for r in compiled),
            "answer_consistent": sum(r["answer_consistent"] is True for r in compiled),
        },
    }


def _build_room(room_dir: Path, scene_id: str):
    """Import + build the full-relation bundle; None-triple if data absent."""
    if not (room_dir / "habitat" / "info_semantic.json").exists():
        return None, None, None
    from demo.question_battery import _runs
    from demo.replica_habitat_import import import_habitat_room
    from graph.builder import build_graph
    from reasoner.base import CompletenessProfile, ExecutionContext
    from reasoner.executor import RulesExecutor
    from reasoner.router import Router
    from reasoner.verbalizer import StandardVerbalizer

    arts = import_habitat_room(room_dir, scene_id)
    bundle, _ = build_graph(arts, _runs(), density_policy="phase2_telemetry_only")
    router = Router(compiler=RulesCompiler(), executor=RulesExecutor(),
                    verbalizer=StandardVerbalizer())
    ctx = ExecutionContext(completeness=CompletenessProfile(
        source="oracle", entity_recall_by_class={}, edge_recall_by_type={}))
    return bundle, router, ctx


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--room", type=Path, default=DEFAULT_ROOM)
    parser.add_argument("--scene-id", default="replica_room_0")
    parser.add_argument("--bank", type=Path, default=BANK_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    bank = json.loads(args.bank.read_text(encoding="utf-8"))
    bundle, router, ctx = _build_room(args.room, args.scene_id)
    if router is None:
        print(f"[note] scene data not found at {args.room}; "
              "running parse-coverage only (answer consistency skipped).")

    compiler = RulesCompiler()
    groups = [run_group(g, compiler, bundle, router, ctx) for g in bank["groups"]]

    total = sum(g["summary"]["n"] for g in groups)
    compiled = sum(g["summary"]["compiled"] for g in groups)
    agg = {
        "groups": len(groups),
        "paraphrases_total": total,
        "compiled": compiled,
        "parser_failure": sum(g["summary"]["parser_failure"] for g in groups),
        "out_of_schema": sum(g["summary"]["out_of_schema"] for g in groups),
        "compiled_rate": round(compiled / total, 4) if total else None,
        "parse_target_match": sum(g["summary"]["parse_target_match"] for g in groups),
        "answer_consistent": sum(g["summary"]["answer_consistent"] for g in groups),
        "canonicals_ok": sum(g["canonical_ok"] for g in groups),
        "canonicals_total": len(groups),
    }
    report = {
        "schema": "phase8_paraphrase_report",
        "schema_version": 1,
        "bank_fixture_id": bank["fixture_id"],
        "scene_id": args.scene_id if router is not None else None,
        "answer_consistency_available": router is not None,
        "interpretation": (
            "compiled_rate measures regex-template brittleness; it is a "
            "benchmark-definition measurement, NOT an accuracy claim and NOT "
            "gated. Only canonical phrasings gate. Not comparable to the "
            "legacy v1 paraphrase eval or any phase scorecard."
        ),
        "aggregate": agg,
        "groups": groups,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"paraphrases: {compiled}/{total} compiled "
          f"({agg['parser_failure']} parser_failure, {agg['out_of_schema']} out_of_schema)")
    print(f"parse-target match among compiled: {agg['parse_target_match']}/{compiled}")
    if router is not None:
        print(f"answer-consistent among compiled: {agg['answer_consistent']}/{compiled}")
    bad = [g["group_id"] for g in groups if not g["canonical_ok"]]
    if bad:
        print(f"FAILED: canonical phrasing broken for: {bad}")
        return 1
    print(f"canonicals: {agg['canonicals_ok']}/{agg['canonicals_total']} ok")
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
