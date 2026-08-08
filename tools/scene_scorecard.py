"""Phase 8 E3 — unified multi-scene scorecard.

Runs every scene in eval/questions/phase8/scene_manifest.json (plus,
optionally, the Phase 7 apartment_0 key) through the real Router and reports
what the per-phase gates never did: per-relation and aggregate metrics.

Per scene:
  - eval/router_qa.py::score_questions UNCHANGED for outcome categories,
  - precision / recall / F1 over cited UIDs — computed ONLY for questions
    with expected_outcome=="answer" AND "exhaustive": true in a
    human_verified key. must_contain in an unreviewed key is a lower bound,
    so recall from it would be dishonest; such rows stay membership pass/fail,
  - defer rate, expected-vs-actual outcome confusion matrix,
  - per-question latency (separate timing pass; the router is deterministic
    per gate G8, so the passes agree).

Honesty split: the headline aggregate covers human_verified keys only.
Plausibility keys (unreviewed drafts, the Phase 7 apartment_0 labels) are
reported in a separate, explicitly labeled section — a fresh draft scoring
100% is CIRCULAR, not evidence.

Phase 8 is a NEW eval track: numbers here are not comparable to the v1
benchmark or the per-phase scorecards.

Usage:
  python3 tools/scene_scorecard.py [--manifest PATH] [--include-phase7]
                                   [--out DIR] [--scene SCENE_ID]
Outputs: runs/phase8_scorecard/<scene_id>.json + aggregate.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.router_qa import score_questions
from reasoner.base import CompletenessProfile, ExecutionContext
from reasoner.compiler_rules import RulesCompiler
from reasoner.executor import RulesExecutor
from reasoner.router import Router
from reasoner.verbalizer import StandardVerbalizer

MANIFEST_PATH = REPO_ROOT / "eval" / "questions" / "phase8" / "scene_manifest.json"
PHASE7_QA = REPO_ROOT / "eval" / "questions" / "phase7_mixed_qa.json"
PHASE7_ROOM = Path.home() / "Desktop/datasets/replica/apartment_0"
DEFAULT_OUT_DIR = REPO_ROOT / "runs" / "phase8_scorecard"
HUMAN_VERIFIED = "human_verified"


def _router() -> Router:
    return Router(compiler=RulesCompiler(), executor=RulesExecutor(),
                  verbalizer=StandardVerbalizer())


def _ctx() -> ExecutionContext:
    return ExecutionContext(completeness=CompletenessProfile(
        source="oracle", entity_recall_by_class={}, edge_recall_by_type={}))


def prf_for_question(q: dict, cited: set[str], answer_key_type: str) -> dict | None:
    """P/R/F1 vs an EXHAUSTIVE expected set; None when the key can't honestly
    support it (non-answer row, non-exhaustive row, or unverified key)."""
    if answer_key_type != HUMAN_VERIFIED:
        return None
    if q.get("expected_outcome") != "answer" or not q.get("exhaustive"):
        return None
    expected = set(q.get("expected_must_contain", []) or [])
    tp = len(cited & expected)
    precision = tp / len(cited) if cited else (1.0 if not expected else 0.0)
    recall = tp / len(expected) if expected else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "tp": tp,
            "n_cited": len(cited), "n_expected": len(expected)}


def timing_pass(questions: list[dict], bundle, router, ctx) -> dict[str, float]:
    out = {}
    for q in questions:
        t0 = time.perf_counter()
        router.answer(q["question"], bundle, ctx)
        out[q["question_id"]] = round((time.perf_counter() - t0) * 1000.0, 3)
    return out


def _confusion(rows: list[dict]) -> dict:
    conf: dict[str, dict[str, int]] = {}
    for r in rows:
        exp = r["expected_outcome"]
        got = "deferred" if r["deferred"] else r["actual_outcome"]
        conf.setdefault(exp, {})
        conf[exp][got] = conf[exp].get(got, 0) + 1
    return conf


def scorecard_for_key(key: dict, bundle, labels: dict[str, str]) -> dict:
    """Pure core (unit-testable on synthetic bundles)."""
    answer_key_type = key.get("answer_key_type", "unknown")
    questions = key["questions"]
    router, ctx = _router(), _ctx()
    card = score_questions(questions, bundle, router, ctx)
    latency = timing_pass(questions, bundle, router, ctx)

    by_id = {q["question_id"]: q for q in questions}
    per_relation: dict[str, dict] = {}
    prf_rows = 0
    for row in card["per_question"]:
        q = by_id[row["question_id"]]
        cited = set(row["cited_uids"])
        prf = prf_for_question(q, cited, answer_key_type)
        row["prf"] = prf
        row["latency_ms"] = latency[row["question_id"]]
        rel = q.get("relation", "UNSPECIFIED")
        bucket = per_relation.setdefault(rel, {
            "n": 0, "exhaustive_n": 0, "categories": {},
            "_tp": 0, "_cited": 0, "_expected": 0,
        })
        bucket["n"] += 1
        bucket["categories"][row["category"]] = \
            bucket["categories"].get(row["category"], 0) + 1
        if prf is not None:
            prf_rows += 1
            bucket["exhaustive_n"] += 1
            bucket["_tp"] += prf["tp"]
            bucket["_cited"] += prf["n_cited"]
            bucket["_expected"] += prf["n_expected"]

    for rel, b in per_relation.items():
        tp, cited_n, exp_n = b.pop("_tp"), b.pop("_cited"), b.pop("_expected")
        if b["exhaustive_n"]:
            # Match prf_for_question's empty-prediction convention. A relation
            # with expected positives but zero citations has precision 0, not
            # vacuous precision 1; otherwise the per-question and relation
            # rollups contradict one another on complete misses.
            p = tp / cited_n if cited_n else (1.0 if exp_n == 0 else 0.0)
            r = tp / exp_n if exp_n else 1.0
            b["precision"] = round(p, 4)
            b["recall"] = round(r, 4)
            b["f1"] = round(2 * p * r / (p + r), 4) if (p + r) else 0.0
        else:
            b["precision"] = b["recall"] = b["f1"] = None

    lat_values = sorted(latency.values())
    deferred_n = sum(1 for r in card["per_question"] if r["deferred"])
    return {
        "schema": "phase8_scene_scorecard",
        "schema_version": 1,
        "scene_id": key.get("scene_id"),
        "fixture_id": key.get("fixture_id"),
        "answer_key_type": answer_key_type,
        "bundle_hash": bundle.bundle_hash,
        "router_qa": card,
        "defer_rate": round(deferred_n / len(questions), 4) if questions else None,
        "latency_ms": {
            "mean": round(statistics.fmean(lat_values), 3) if lat_values else None,
            "p50": round(statistics.median(lat_values), 3) if lat_values else None,
            "max": max(lat_values) if lat_values else None,
        },
        "confusion": _confusion(card["per_question"]),
        "per_relation": {k: per_relation[k] for k in sorted(per_relation)},
        "prf_scope_note": (
            "P/R/F1 computed only over exhaustive answer rows of human_verified "
            "keys; all other rows are membership pass/fail via eval/router_qa.py. "
            f"prf_rows={prf_rows}."
        ),
    }


def _build_scene(room_dir: Path, scene_id: str):
    from demo.question_battery import _runs
    from demo.replica_habitat_import import import_habitat_room
    from graph.builder import build_graph
    arts = import_habitat_room(room_dir, scene_id)
    labels = {e.identity.object_uid: e.identity.display_label for e in arts.entities}
    bundle, _ = build_graph(arts, _runs(), density_policy="phase2_telemetry_only")
    return bundle, labels


def _load_key(entry: dict) -> tuple[dict, Path]:
    reviewed = REPO_ROOT / entry["key_path"]
    if reviewed.exists():
        return json.loads(reviewed.read_text(encoding="utf-8")), reviewed
    draft = REPO_ROOT / entry["draft_path"]
    return json.loads(draft.read_text(encoding="utf-8")), draft


def aggregate(per_scene: list[dict]) -> dict:
    verified = [s for s in per_scene if s["answer_key_type"] == HUMAN_VERIFIED]
    plausibility = [s for s in per_scene if s["answer_key_type"] != HUMAN_VERIFIED]

    def _rollup(cards: list[dict]) -> dict:
        counts: dict[str, int] = {}
        for c in cards:
            for k, v in c["router_qa"]["aggregate"]["category_counts"].items():
                counts[k] = counts.get(k, 0) + v
        return {
            "scenes": [c["scene_id"] for c in cards],
            "total_questions": sum(
                c["router_qa"]["aggregate"]["total_questions"] for c in cards),
            "category_counts": dict(sorted(counts.items())),
            "all_expected_outcomes_met": all(
                c["router_qa"]["aggregate"]["all_expected_outcomes_met"]
                for c in cards) if cards else None,
        }

    return {
        "schema": "phase8_scorecard_aggregate",
        "schema_version": 1,
        "comparability": (
            "NEW Phase 8 track — not comparable to the v1 benchmark or the "
            "per-phase scorecards. The plausibility section is NOT evidence: "
            "unreviewed drafts score 100% against themselves by construction."
        ),
        "headline_human_verified": _rollup(verified),
        "plausibility_not_ground_truth": _rollup(plausibility),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--include-phase7", action="store_true",
                        help="Also score eval/questions/phase7_mixed_qa.json on apartment_0.")
    parser.add_argument("--scene", default=None,
                        help="Limit to one scene_id from the manifest.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    entries: list[dict] = []
    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        entries = manifest["scenes"]
    if args.scene:
        entries = [e for e in entries if e["scene_id"] == args.scene]

    cards: list[dict] = []
    skipped: list[str] = []
    for entry in entries:
        room_dir = Path(entry["room_dir"])
        if not (room_dir / "habitat" / "info_semantic.json").exists():
            skipped.append(f"{entry['scene_id']} (no data at {room_dir})")
            continue
        key, key_path = _load_key(entry)
        bundle, labels = _build_scene(room_dir, entry["scene_id"])
        card = scorecard_for_key(key, bundle, labels)
        card["key_path"] = str(key_path.relative_to(REPO_ROOT))
        cards.append(card)

    if args.include_phase7:
        if (PHASE7_ROOM / "habitat" / "info_semantic.json").exists():
            key = json.loads(PHASE7_QA.read_text(encoding="utf-8"))
            bundle, labels = _build_scene(PHASE7_ROOM, "replica_apartment_0")
            card = scorecard_for_key(key, bundle, labels)
            card["key_path"] = str(PHASE7_QA.relative_to(REPO_ROOT))
            cards.append(card)
        else:
            skipped.append("replica_apartment_0 (no data)")

    args.out.mkdir(parents=True, exist_ok=True)
    for card in cards:
        p = args.out / f"{card['scene_id']}.json"
        p.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    agg = aggregate(cards)
    agg["skipped_scenes"] = skipped
    (args.out / "aggregate.json").write_text(
        json.dumps(agg, indent=2) + "\n", encoding="utf-8")

    print("=" * 78)
    print("PHASE 8 SCORECARD (new track; not comparable to v1 / phase gates)")
    print("=" * 78)
    for card in cards:
        a = card["router_qa"]["aggregate"]
        tag = "VERIFIED " if card["answer_key_type"] == HUMAN_VERIFIED else "plausible"
        print(f"  [{tag}] {card['scene_id']:24} q={a['total_questions']:3} "
              f"cats={a['category_counts']} defer_rate={card['defer_rate']} "
              f"lat_mean={card['latency_ms']['mean']}ms")
        for rel, b in card["per_relation"].items():
            if b["precision"] is not None:
                print(f"      {rel:18} P={b['precision']} R={b['recall']} "
                      f"F1={b['f1']} (exhaustive_n={b['exhaustive_n']})")
    for s in skipped:
        print(f"  [skipped ] {s}")
    print(f"reports -> {args.out}")
    headline = agg["headline_human_verified"]
    if not headline["scenes"]:
        print("NOTE: no human_verified keys yet — headline is empty until review "
              "(tools/draft_answer_key.py docstring).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
