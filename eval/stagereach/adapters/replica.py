"""Replica Phase-8 scorecard adapter — schema/outcome transfer ONLY.

Reads ONLY the packed copies in eval/results/project_census_v1/
(phase8_scorecard_{aggregate,replica_office_0,replica_room_0,replica_room_1,
replica_room_2}.json) plus the QA keys in eval/questions/phase8/ for
exhaustive flags and relations. Never `runs/`.

Machine-checked guards (freeze doc §6):
- every trace carries oracle scope labeling, and only human_verified
  scorecards are accepted (plausibility-only scenes are refused, reported
  separately upstream; replica_frl_apartment_0 is NOT in the pack and any
  appearance raises);
- NEAR_SURFACE (and any other non-exhaustive key): the evaluator REFUSES
  precision/recall/true-negative metrics (metrics.precision_recall raises,
  invariant 5);
- definition_change results cannot pool with the frozen track (pooling
  raises, invariant 6, enforced inside metrics.outcome_matrix);
- every internal stage the scorecards cannot support is UNKNOWN — delivery,
  relation correctness, serialization and grounding are NEVER inferred from
  the final answer.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..evaluator import evaluate_trace
from ..metrics import normalize_router_record
from ..schema import Trace

PACK_RELPATH = "eval/results/project_census_v1"
KEYS_RELPATH = "eval/questions/phase8"

SCENES: tuple[str, ...] = ("replica_office_0", "replica_room_0",
                           "replica_room_1", "replica_room_2")
FORBIDDEN_SCENE_SUBSTRING = "frl_apartment"

ARM = "router_delivered_graph"
PATH_ID = "graph_deployable_delivered"
SCOPE = "delivered"

_UNSUPPORTED = ("object_delivery", "relation_applicability",
                "relation_correctness", "serialization_consistency",
                "referent_grounding")


class ReplicaGuardError(RuntimeError):
    """A packed-scorecard guard was violated."""


def guard_scene(scene_id: str) -> None:
    if FORBIDDEN_SCENE_SUBSTRING in scene_id:
        raise ReplicaGuardError(
            f"{scene_id}: plausibility-only scene — not human-verified, "
            "not in the pack, and never poolable with the headline scenes")
    if scene_id not in SCENES:
        raise ReplicaGuardError(
            f"{scene_id}: not one of the four packed human-verified scenes "
            f"{SCENES}")


def guard_scorecard(scorecard: dict) -> None:
    """Oracle scope labeling: the answer key's provenance must be stated
    and human-verified before any trace is derived from the scorecard."""
    scene = scorecard.get("scene_id", "<missing scene_id>")
    guard_scene(scene)
    akt = scorecard.get("answer_key_type")
    if akt != "human_verified":
        raise ReplicaGuardError(
            f"{scene}: answer_key_type is {akt!r}; only human_verified "
            "scorecards may enter the frozen track (plausibility results "
            "are reported separately, never pooled)")
    if "router_qa" not in scorecard:
        raise ReplicaGuardError(f"{scene}: scorecard has no router_qa block")


def load_scorecards(repo_root: Path) -> dict[str, dict]:
    pack = Path(repo_root) / PACK_RELPATH
    out: dict[str, dict] = {}
    for scene in SCENES:
        doc = json.loads((pack / f"phase8_scorecard_{scene}.json")
                         .read_text())
        guard_scorecard(doc)
        out[scene] = doc
    return out


def load_aggregate(repo_root: Path) -> dict:
    pack = Path(repo_root) / PACK_RELPATH
    return json.loads((pack / "phase8_scorecard_aggregate.json").read_text())


def load_keys(repo_root: Path) -> dict[str, dict]:
    keys = Path(repo_root) / KEYS_RELPATH
    return {scene: json.loads((keys / f"{scene}_qa.json").read_text())
            for scene in SCENES}


def relation_exhaustive_map(key_doc: dict) -> dict[str, bool]:
    """Per relation: True only if EVERY key row for that relation is
    exhaustive. Anything less and precision/recall are refused."""
    out: dict[str, bool] = {}
    for q in key_doc["questions"]:
        rel = q["relation"]
        out[rel] = out.get(rel, True) and bool(q.get("exhaustive"))
    return out


def derive_traces(scorecards: dict[str, dict],
                  keys: dict[str, dict]) -> list[Trace]:
    """One trace per question. Internal stages are UNKNOWN: the packed
    scorecards record only the router's final outcome per question, and no
    internal stage is ever inferred from that outcome."""
    traces: list[Trace] = []
    for scene in SCENES:
        scorecard = scorecards[scene]
        guard_scorecard(scorecard)
        key_doc = keys[scene]
        key_rel = {q["question_id"]: q["relation"]
                   for q in key_doc["questions"]}
        source_note = (f"phase8_scorecard_{scene}.json router_qa "
                       "records only the final router outcome; this stage "
                       "is unmeasured there and is never inferred from the "
                       "final answer")
        for rec in sorted(scorecard["router_qa"]["per_question"],
                          key=lambda r: r["question_id"]):
            expected, result = normalize_router_record(rec)
            answer_status = {"correct": "pass", "wrong": "fail",
                             "abstain": "abstain",
                             "excluded": "abstain"}[result]
            obs = {s: ("unknown", source_note) for s in _UNSUPPORTED}
            obs["key_eligibility"] = (
                "pass",
                f"{KEYS_RELPATH}/{scene}_qa.json: human_verified key row "
                f"({key_rel.get(rec['question_id'], 'unknown-relation')})")
            obs["answer_generation"] = (
                answer_status,
                f"phase8_scorecard_{scene}.json router_qa.per_question "
                f"category={rec['category']!r}")
            traces.append(evaluate_trace(
                question_id=rec["question_id"], scene_id=scene,
                arm=ARM, path_id=PATH_ID, scope=SCOPE,
                expected_outcome=expected,
                observations=obs, result=result,
                raw_category=rec["category"]))
    return traces


def precision_recall_for(scene: str, relation: str,
                         scorecards: dict[str, dict],
                         keys: dict[str, dict]) -> dict:
    """P/R for one relation on one scene — refused (invariant 5) unless the
    key is exhaustive for that relation. This is the ONLY route to P/R
    numbers in this adapter; the packed per_relation block is never
    re-emitted without passing through the guard."""
    from ..metrics import require_exhaustive_key
    exhaustive = relation_exhaustive_map(keys[scene]).get(relation, False)
    require_exhaustive_key(relation, exhaustive)
    block = scorecards[scene]["per_relation"].get(relation, {})
    return {"scene": scene, "relation": relation, "exhaustive": True,
            "n": block.get("n"), "exhaustive_n": block.get("exhaustive_n"),
            "precision": block.get("precision"),
            "recall": block.get("recall"), "f1": block.get("f1")}


def aggregate_cross_check(aggregate: dict,
                          traces: list[Trace]) -> dict:
    """The packed aggregate's headline (human-verified) block must restate
    exactly what the per-scene traces sum to, and its plausibility block
    must stay disjoint from the headline scenes."""
    headline = aggregate["headline_human_verified"]
    if sorted(headline["scenes"]) != sorted(SCENES):
        raise ReplicaGuardError(
            f"aggregate headline scenes {headline['scenes']} != packed "
            f"scenes {SCENES}")
    for scene in aggregate.get("plausibility_not_ground_truth",
                               {}).get("scenes", []):
        if scene in SCENES:
            raise ReplicaGuardError(
                f"{scene}: appears in BOTH the headline and plausibility "
                "blocks")
        if FORBIDDEN_SCENE_SUBSTRING not in scene:
            raise ReplicaGuardError(
                f"{scene}: unexpected plausibility scene in the pack")
    from collections import Counter
    ours = Counter(t.raw_category for t in traces)
    theirs = headline["category_counts"]
    if dict(ours) != dict(theirs):
        raise ReplicaGuardError(
            f"per-scene traces {dict(ours)} do not sum to the aggregate "
            f"headline {dict(theirs)}")
    if headline["total_questions"] != len(traces):
        raise ReplicaGuardError(
            f"aggregate total {headline['total_questions']} != "
            f"{len(traces)} traces")
    return {"headline_scenes": sorted(headline["scenes"]),
            "total_questions": headline["total_questions"],
            "category_counts_match": True}
