#!/usr/bin/env python3
"""StageReach3D track runner — traces, ladders and matrices as artifacts.

Reads ONLY tracked evidence (the packed ARKit relation-challenge report,
the packed Phase-8 Replica scorecards + QA keys, and the committed fault
fixture). Runs no model, touches no threshold, and produces no new
measurement — every number is a re-reading of outcomes that were already
scored and committed, resolved through the frozen StageReach3D schema.

    tools/stagereach_eval.py --track arkit     # eval/results/stagereach/arkit_stagereach_v1.json
    tools/stagereach_eval.py --track replica   # eval/results/stagereach/replica_stagereach_v1.json
    tools/stagereach_eval.py --track fixtures  # regenerate the fault fixture + run the battery
    tools/stagereach_eval.py [--track all] --check

`--check` recomputes and fails if the committed outputs would change.
Artifacts are written sorted-keys, schema-stamped, with a trailing newline.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eval.stagereach import faults, metrics  # noqa: E402
from eval.stagereach.adapters import arkit, replica  # noqa: E402
from eval.stagereach.schema import PATHS, trace_to_dict  # noqa: E402

OUT_DIR = REPO / "eval" / "results" / "stagereach"
OUT_ARKIT = OUT_DIR / "arkit_stagereach_v1.json"
OUT_REPLICA = OUT_DIR / "replica_stagereach_v1.json"
OUT_FIXTURE = REPO / faults.FIXTURE_RELPATH

RESULT_SCHEMA = "stagereach_track_result"
RESULT_SCHEMA_VERSION = 1


# -------------------------------------------------------------- arkit track
def build_arkit() -> dict:
    report = arkit.load_report(REPO)
    traces_by_arm = arkit.derive_traces(report)
    arms_out: dict[str, dict] = {}
    for arm, path_id, scope in arkit.ARMS:
        traces = traces_by_arm[arm]
        path = PATHS[path_id]
        arms_out[arm] = {
            "path_id": path_id,
            "scope": scope,
            "ladder": metrics.survival_ladder(traces, path),
            "stage_report": metrics.stage_report(traces, path),
            "outcome_matrix": metrics.matrix_to_json(
                metrics.outcome_matrix(traces)),
            "raw_category_counts": metrics.raw_category_counts(traces),
            "traces": [trace_to_dict(t) for t in traces],
        }
    n_scored = sum(1 for t in traces_by_arm["delivered_graph"]
                   if t.status("key_eligibility") == "pass")
    return {
        "schema": RESULT_SCHEMA,
        "schema_version": RESULT_SCHEMA_VERSION,
        "track": "arkit",
        "source": arkit.REPORT_RELPATH,
        "purpose": ("Per-arm StageReach traces and causal survival ladders "
                    "for the ARKit relation challenge, derived "
                    "independently of the legacy statistics tool."),
        "n_questions": len(traces_by_arm["delivered_graph"]),
        "n_scored": n_scored,
        "arms": arms_out,
        "legacy_compat": arkit.legacy_reachability_block(traces_by_arm),
    }


# ------------------------------------------------------------ replica track
def build_replica() -> dict:
    scorecards = replica.load_scorecards(REPO)
    keys = replica.load_keys(REPO)
    traces = replica.derive_traces(scorecards, keys)
    aggregate = replica.load_aggregate(REPO)
    cross = replica.aggregate_cross_check(aggregate, traces)

    scenes_out: dict[str, dict] = {}
    for scene in replica.SCENES:
        scene_traces = [t for t in traces if t.scene_id == scene]
        scenes_out[scene] = {
            "n": len(scene_traces),
            "raw_category_counts": metrics.raw_category_counts(scene_traces),
            "normalized_matrix": metrics.matrix_to_json(
                metrics.outcome_matrix(scene_traces)),
            "relation_exhaustive": replica.relation_exhaustive_map(
                keys[scene]),
        }
    nonexhaustive = sorted({rel
                            for scene in replica.SCENES
                            for rel, ex in replica.relation_exhaustive_map(
                                keys[scene]).items() if not ex})
    return {
        "schema": RESULT_SCHEMA,
        "schema_version": RESULT_SCHEMA_VERSION,
        "track": "replica",
        "sources": ([f"{replica.PACK_RELPATH}/phase8_scorecard_{s}.json"
                     for s in ("aggregate",) + replica.SCENES]
                    + [f"{replica.KEYS_RELPATH}/{s}_qa.json"
                       for s in replica.SCENES]),
        "purpose": ("Schema/outcome transfer of the frozen Phase-8 Replica "
                    "scorecards. Internal stages are unknown by "
                    "construction; only the final router outcome is "
                    "restated, raw and normalized."),
        "scenes": scenes_out,
        "totals": {
            "n": len(traces),
            "raw_category_counts": metrics.raw_category_counts(traces),
            "normalized_matrix": metrics.matrix_to_json(
                metrics.outcome_matrix(traces)),
        },
        "guards": {
            "scene_allowlist": list(replica.SCENES),
            "forbidden_scene_substring": replica.FORBIDDEN_SCENE_SUBSTRING,
            "answer_key_type": "human_verified",
            "aggregate_cross_check": cross,
            "precision_recall_refused_for_nonexhaustive": nonexhaustive,
            "internal_stages_unknown": ["object_delivery",
                                        "relation_applicability",
                                        "relation_correctness",
                                        "serialization_consistency",
                                        "referent_grounding"],
        },
        "traces": [trace_to_dict(t) for t in traces],
    }


# ----------------------------------------------------------- fixtures track
def build_fixture_bytes() -> bytes:
    return faults.fixture_bytes(faults.build_fixture())


def run_fixture_battery() -> dict:
    return faults.run_battery(faults.build_fixture())


# -------------------------------------------------------------------- shell
def _artifact_bytes(doc: dict) -> bytes:
    return (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--track", choices=("arkit", "replica", "fixtures",
                                        "all"), default="all")
    ap.add_argument("--out", type=Path, default=None,
                    help="override the output PATH (single-track runs only)")
    ap.add_argument("--check", action="store_true",
                    help="recompute and fail if committed outputs would "
                         "change")
    args = ap.parse_args(argv)

    jobs: list[tuple[str, Path, bytes]] = []
    if args.track in ("arkit", "all"):
        jobs.append(("arkit", OUT_ARKIT, _artifact_bytes(build_arkit())))
    if args.track in ("replica", "all"):
        jobs.append(("replica", OUT_REPLICA,
                     _artifact_bytes(build_replica())))
    if args.track in ("fixtures", "all"):
        jobs.append(("fixtures", OUT_FIXTURE, build_fixture_bytes()))

    if args.out is not None:
        if len(jobs) != 1:
            print("--out needs a single --track")
            return 2
        jobs = [(jobs[0][0], args.out, jobs[0][2])]

    if args.check:
        stale = []
        for name, path, data in jobs:
            before = path.read_bytes() if path.is_file() else b""
            if before != data:
                stale.append(str(path.relative_to(REPO)))
        if stale:
            print("committed stagereach artifacts are stale; re-run "
                  f"tools/stagereach_eval.py ({', '.join(stale)})")
            return 1
        print("stagereach artifacts are current")
        return 0

    for name, path, data in jobs:
        _write(path, data)
        print(f"wrote {path.relative_to(REPO) if path.is_relative_to(REPO) else path}")
        if name == "arkit":
            doc = json.loads(data)
            for arm, block in doc["arms"].items():
                rungs = "->".join(str(r["survivors"])
                                  for r in block["ladder"])
                print(f"  {arm}: {rungs}")
        elif name == "replica":
            doc = json.loads(data)
            print(f"  raw: {doc['totals']['raw_category_counts']}")
            print(f"  normalized: {doc['totals']['normalized_matrix']}")
        elif name == "fixtures":
            b = run_fixture_battery()
            print(f"  fault battery: {b['n_localized']}/{b['n_total']} "
                  f"localized, clean failures {b['clean_failures']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
