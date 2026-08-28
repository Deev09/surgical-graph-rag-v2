"""Artifact-based, evaluator-masked fault injection (freeze doc §9 G-FAULT).

The fixture is a chain of REAL intermediate artifacts:

    evaluation_key + entity_artifact -> relation_artifact
        -> serialized_graph -> grounded_candidates -> answer_artifact

built once, deterministically, by an actual reference pipeline over AABB
geometry, covering three relation types (NEAR, ON_ENTITY_SURFACE,
ATTACHED_TO) and a 9-question key mixing expected answer / expected empty.

Eight injection functions each mutate EXACTLY ONE artifact. The evaluator
(`evaluate_artifacts`) derives stage statuses via independent per-stage
checkers — delivery checks the entity artifact against the key's
references; relation applicability/correctness check the relation artifact
against the fixture's frozen ground truth; serialization checks the
serialized graph against the relation artifact; grounding checks the
grounded candidates against entity identity; the answer checks against the
key — and is NEVER told which fault was injected. The hidden expected
labels live in EXPECTED_LOCALIZATION, which only the test (and the battery
runner's comparison step) reads, strictly AFTER evaluation.

The design distinguishes, by construction:
- a relation absent/incorrect BEFORE serialization -> relation_correctness
  (first fail in path order; the serialized graph also disagrees, but
  serialization_consistency does not gate on relation_correctness and
  attribution reports the first gating fail);
- a correct computed relation omitted/corrupted AT serialization ->
  serialization_consistency.
"""
from __future__ import annotations

import copy
import json
from math import sqrt
from pathlib import Path

from .evaluator import attribute, evaluate_trace
from .schema import Trace

FIXTURE_SCHEMA = "stagereach_fault_fixture"
FIXTURE_SCHEMA_VERSION = 1
FIXTURE_RELPATH = "eval/fixtures/stagereach/fault_fixture_v1.json"

RELATION_VOCAB = ("NEAR", "ON_ENTITY_SURFACE", "ATTACHED_TO")

NEAR_GAP_M = 1.0
ON_SURFACE_TOL_M = 0.02
ATTACH_GAP_M = 0.01
ATTACH_MIN_OVERLAP_M = 0.05

# ------------------------------------------------------------ the entities
# Three well-separated clusters, one per relation type. Positions are AABB
# center + size; every relation below is COMPUTED from this geometry by the
# reference pipeline, never hand-asserted.
_ENTITIES = [
    # NEAR cluster (around x=0)
    ("e01", "reading lamp", (0.0, 0.0, 0.4), (0.3, 0.3, 0.8)),
    ("e02", "side table", (0.8, 0.0, 0.25), (0.5, 0.5, 0.5)),
    ("e03", "armchair", (3.0, 3.0, 0.4), (0.8, 0.8, 0.8)),
    ("e04", "ottoman", (3.9, 3.0, 0.2), (0.6, 0.6, 0.4)),
    ("e05", "floor safe", (0.0, 8.0, 0.3), (0.5, 0.5, 0.6)),
    # ON_ENTITY_SURFACE cluster (around x=100)
    ("e06", "writing desk", (100.0, 0.0, 0.4), (1.2, 0.6, 0.8)),
    ("e07", "notebook", (100.0, 0.0, 0.83), (0.2, 0.15, 0.06)),
    ("e08", "kitchen counter", (104.0, 0.0, 0.45), (1.0, 0.6, 0.9)),
    ("e09", "coffee mug", (104.0, 0.0, 0.95), (0.1, 0.1, 0.1)),
    ("e10", "bar stool", (100.0, 8.0, 0.35), (0.4, 0.4, 0.7)),
    # ATTACHED_TO cluster (around x=200)
    ("e11", "wall panel", (200.0, 0.0, 1.5), (2.0, 0.1, 3.0)),
    ("e12", "coat hook", (200.0, 0.08, 1.5), (0.06, 0.06, 0.1)),
    ("e13", "support pillar", (204.0, 0.0, 1.5), (0.3, 0.3, 3.0)),
    ("e14", "handrail", (204.0, 0.16, 1.2), (0.4, 0.06, 0.06)),
    ("e15", "wall mirror", (200.0, 8.0, 1.5), (0.8, 0.05, 1.2)),
]

# qid, relation, anchor name, expected_outcome, expected uids
_QUESTIONS = [
    ("q_near_answer", "NEAR", "side table", "answer", ["e01"]),
    ("q_near_empty", "NEAR", "floor safe", "empty", []),
    ("q_near_extra", "NEAR", "ottoman", "answer", ["e03"]),
    ("q_on_answer", "ON_ENTITY_SURFACE", "writing desk", "answer", ["e07"]),
    ("q_on_empty", "ON_ENTITY_SURFACE", "bar stool", "empty", []),
    ("q_on_extra", "ON_ENTITY_SURFACE", "kitchen counter", "answer",
     ["e09"]),
    ("q_att_answer", "ATTACHED_TO", "wall panel", "answer", ["e12"]),
    ("q_att_empty", "ATTACHED_TO", "wall mirror", "empty", []),
    ("q_att_extra", "ATTACHED_TO", "support pillar", "answer", ["e14"]),
]

# The canonical injection target per relation type: the primary answer
# question of that cluster. Used only by the INJECTORS and the test — the
# evaluator never reads these.
TARGET_QID = {"NEAR": "q_near_answer",
              "ON_ENTITY_SURFACE": "q_on_answer",
              "ATTACHED_TO": "q_att_answer"}
_TARGET_MEMBER = {"NEAR": "e01", "ON_ENTITY_SURFACE": "e07",
                  "ATTACHED_TO": "e12"}
_TARGET_ANCHOR = {"NEAR": "e02", "ON_ENTITY_SURFACE": "e06",
                  "ATTACHED_TO": "e11"}
_ISOLATED = {"NEAR": "e05", "ON_ENTITY_SURFACE": "e10",
             "ATTACHED_TO": "e15"}
_OTHER_PREDICATE = {"NEAR": "ATTACHED_TO", "ON_ENTITY_SURFACE": "NEAR",
                    "ATTACHED_TO": "NEAR"}


# ------------------------------------------------------ geometry (aabb ops)
def _aabb(ent: dict) -> tuple[list[float], list[float]]:
    lo = [c - s / 2 for c, s in zip(ent["center"], ent["size"])]
    hi = [c + s / 2 for c, s in zip(ent["center"], ent["size"])]
    return lo, hi


def _axis_gap(alo, ahi, blo, bhi) -> float:
    return max(0.0, alo - bhi, blo - ahi)


def _gap(a: dict, b: dict, axes=(0, 1, 2)) -> float:
    (alo, ahi), (blo, bhi) = _aabb(a), _aabb(b)
    return sqrt(sum(_axis_gap(alo[i], ahi[i], blo[i], bhi[i]) ** 2
                    for i in axes))


def _vertical_overlap(a: dict, b: dict) -> float:
    (alo, ahi), (blo, bhi) = _aabb(a), _aabb(b)
    return min(ahi[2], bhi[2]) - max(alo[2], blo[2])


def _volume(ent: dict) -> float:
    return ent["size"][0] * ent["size"][1] * ent["size"][2]


def compute_relations(entities: list[dict]) -> list[list[str]]:
    """Deterministic relation extraction from AABB geometry. NEAR tuples
    are canonicalized subject<object; ON/ATTACHED are directed
    member->anchor."""
    by_uid = {e["uid"]: e for e in entities}
    uids = sorted(by_uid)
    rels: list[list[str]] = []
    for i, ua in enumerate(uids):
        for ub in uids[i + 1:]:
            a, b = by_uid[ua], by_uid[ub]
            if _gap(a, b) <= NEAR_GAP_M:
                rels.append(["NEAR", ua, ub])
    for ux in uids:
        for ubase in uids:
            if ux == ubase:
                continue
            x, base = by_uid[ux], by_uid[ubase]
            (xlo, xhi), (blo, bhi) = _aabb(x), _aabb(base)
            cx, cy = x["center"][0], x["center"][1]
            if (abs(xlo[2] - bhi[2]) <= ON_SURFACE_TOL_M
                    and blo[0] <= cx <= bhi[0] and blo[1] <= cy <= bhi[1]):
                rels.append(["ON_ENTITY_SURFACE", ux, ubase])
    for ux in uids:
        for uhost in uids:
            if ux == uhost:
                continue
            x, host = by_uid[ux], by_uid[uhost]
            if (_gap(x, host, axes=(0, 1)) <= ATTACH_GAP_M
                    and _vertical_overlap(x, host) >= ATTACH_MIN_OVERLAP_M
                    and _volume(x) < _volume(host)):
                rels.append(["ATTACHED_TO", ux, uhost])
    return sorted(rels)


# --------------------------------------------------- the reference pipeline
def _answer_question(q: dict, edges: list[dict],
                     grounded: dict[str, dict]) -> dict:
    anchor_uid = grounded[q["question_id"]]["uid"]
    hits: set[str] = set()
    for e in edges:
        if e["relation"] != q["relation"]:
            continue
        if q["relation"] == "NEAR":
            if e["subject"] == anchor_uid:
                hits.add(e["object"])
            elif e["object"] == anchor_uid:
                hits.add(e["subject"])
        elif e["object"] == anchor_uid:
            hits.add(e["subject"])
    if hits:
        return {"outcome": "answer", "uids": sorted(hits)}
    return {"outcome": "empty", "uids": []}


def build_fixture() -> dict:
    """Build the full clean artifact chain, deterministically."""
    entities = [{"uid": u, "name": n, "center": list(c), "size": list(s)}
                for u, n, c, s in _ENTITIES]
    key = {"questions": [
        {"question_id": qid, "relation": rel, "anchor": anchor,
         "expected_outcome": exp, "expected_uids": uids,
         "exhaustive": True, "human_marked_ambiguous": False}
        for qid, rel, anchor, exp, uids in _QUESTIONS]}
    relations = compute_relations(entities)
    relation_artifact = {"vocabulary": list(RELATION_VOCAB),
                         "relations": relations}
    serialized_graph = {"edges": [
        {"relation": r, "subject": s, "object": o} for r, s, o in relations]}
    name_to_uid = {e["name"]: e["uid"] for e in entities}
    grounded = {q["question_id"]: {"anchor": q["anchor"],
                                   "uid": name_to_uid[q["anchor"]]}
                for q in key["questions"]}
    answers = {q["question_id"]:
               _answer_question(q, serialized_graph["edges"], grounded)
               for q in key["questions"]}
    return {
        "schema": FIXTURE_SCHEMA,
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "evaluation_key": key,
        "entity_artifact": {"entities": entities},
        "relation_artifact": relation_artifact,
        "serialized_graph": serialized_graph,
        "grounded_candidates": grounded,
        "answer_artifact": answers,
        # Frozen ground truth for the independent per-stage checkers: the
        # relations the clean geometry supports, and each question's anchor
        # identity. Injections never touch this block.
        "ground_truth": {
            "relations": relations,
            "anchor_uids": {q["question_id"]: grounded[q["question_id"]]["uid"]
                            for q in key["questions"]},
        },
    }


ARTIFACT_KEYS = ("evaluation_key", "entity_artifact", "relation_artifact",
                 "serialized_graph", "grounded_candidates",
                 "answer_artifact")


def fixture_bytes(fixture: dict) -> bytes:
    return (json.dumps(fixture, indent=1, sort_keys=True) + "\n").encode()


def load_fixture(repo_root: Path) -> dict:
    return json.loads((Path(repo_root) / FIXTURE_RELPATH).read_text())


# -------------------------------------------------------------- injections
def _mutated(fixture: dict) -> dict:
    return copy.deepcopy(fixture)


def _target_tuple(relation: str) -> list[str]:
    m, a = _TARGET_MEMBER[relation], _TARGET_ANCHOR[relation]
    if relation == "NEAR":
        m, a = sorted((m, a))
    return [relation, m, a]


def inject_entity_removal(fixture: dict, relation: str) -> dict:
    """Mutates entity_artifact: the delivered partition loses the entity
    the target question's expected answer names."""
    out = _mutated(fixture)
    member = _TARGET_MEMBER[relation]
    ents = out["entity_artifact"]["entities"]
    out["entity_artifact"]["entities"] = [e for e in ents
                                         if e["uid"] != member]
    return out


def inject_identity_swap(fixture: dict, relation: str) -> dict:
    """Mutates grounded_candidates: the anchor binds to a different real
    entity (the cluster's isolated one)."""
    out = _mutated(fixture)
    out["grounded_candidates"][TARGET_QID[relation]]["uid"] = \
        _ISOLATED[relation]
    return out


def inject_relation_deletion_pre_serialization(fixture: dict,
                                               relation: str) -> dict:
    """Mutates relation_artifact: the computed relation is absent BEFORE
    serialization (the serialized graph still carries it)."""
    out = _mutated(fixture)
    target = _target_tuple(relation)
    rels = out["relation_artifact"]["relations"]
    out["relation_artifact"]["relations"] = [r for r in rels if r != target]
    return out


def inject_serialization_corruption(fixture: dict, relation: str) -> dict:
    """Mutates serialized_graph: a correctly computed relation is omitted
    at serialization."""
    out = _mutated(fixture)
    r, s, o = _target_tuple(relation)
    edges = out["serialized_graph"]["edges"]
    out["serialized_graph"]["edges"] = [
        e for e in edges
        if not (e["relation"] == r and e["subject"] == s
                and e["object"] == o)]
    return out


def inject_predicate_change(fixture: dict, relation: str) -> dict:
    """Mutates serialized_graph: the serializer wrote the wrong predicate
    for a correctly computed relation."""
    out = _mutated(fixture)
    r, s, o = _target_tuple(relation)
    for e in out["serialized_graph"]["edges"]:
        if e["relation"] == r and e["subject"] == s and e["object"] == o:
            e["relation"] = _OTHER_PREDICATE[relation]
    return out


def inject_answer_corruption(fixture: dict, relation: str) -> dict:
    """Mutates answer_artifact: a confidently wrong final answer."""
    out = _mutated(fixture)
    out["answer_artifact"][TARGET_QID[relation]] = {
        "outcome": "answer", "uids": [_ISOLATED[relation]]}
    return out


def inject_forced_abstention(fixture: dict, relation: str) -> dict:
    """Mutates answer_artifact: the system abstains on an answerable item."""
    out = _mutated(fixture)
    out["answer_artifact"][TARGET_QID[relation]] = {"outcome": "abstain",
                                                    "uids": []}
    return out


def inject_ambiguous_key(fixture: dict, relation: str) -> dict:
    """Mutates the KEY artifact: the item is owner-marked ambiguous, which
    must localize to key_eligibility and nowhere downstream."""
    out = _mutated(fixture)
    for q in out["evaluation_key"]["questions"]:
        if q["question_id"] == TARGET_QID[relation]:
            q["human_marked_ambiguous"] = True
    return out


INJECTIONS = {
    "entity_removal": inject_entity_removal,
    "identity_swap": inject_identity_swap,
    "relation_deletion_pre_serialization":
        inject_relation_deletion_pre_serialization,
    "serialization_corruption": inject_serialization_corruption,
    "predicate_change": inject_predicate_change,
    "answer_corruption": inject_answer_corruption,
    "forced_abstention": inject_forced_abstention,
    "ambiguous_key": inject_ambiguous_key,
}

# The hidden expected label per fault class: (stage, status). Read ONLY by
# the test and by the battery's post-hoc comparison — evaluate_artifacts
# never sees it.
EXPECTED_LOCALIZATION = {
    "entity_removal": ("object_delivery", "fail"),
    "identity_swap": ("referent_grounding", "fail"),
    "relation_deletion_pre_serialization": ("relation_correctness", "fail"),
    "serialization_corruption": ("serialization_consistency", "fail"),
    "predicate_change": ("serialization_consistency", "fail"),
    "answer_corruption": ("answer_generation", "fail"),
    "forced_abstention": ("answer_generation", "abstain"),
    "ambiguous_key": ("key_eligibility", "fail"),
}


# ------------------------------------------------- the masked evaluator
def _touching(tuples: list[list[str]], uid: str) -> set[tuple[str, ...]]:
    return {tuple(t) for t in tuples if uid in (t[1], t[2])}


def _question_observations(q: dict, fixture: dict) -> dict:
    """Independent per-stage checkers. Nothing here reads the injection
    tables or any final-answer field to infer an internal stage."""
    qid = q["question_id"]
    key_ok = (not q.get("human_marked_ambiguous")
              and (q["expected_outcome"] != "answer" or q["expected_uids"]))
    entities = fixture["entity_artifact"]["entities"]
    uids = {e["uid"] for e in entities}
    names = {e["name"] for e in entities}
    by_uid = {e["uid"]: e for e in entities}
    delivered = (q["anchor"] in names
                 and all(u in uids for u in q["expected_uids"]))
    applicable = q["relation"] in fixture["relation_artifact"]["vocabulary"]

    anchor_uid = fixture["ground_truth"]["anchor_uids"][qid]
    gt = {tuple(t) for t in fixture["ground_truth"]["relations"]
          if t[0] == q["relation"] and anchor_uid in (t[1], t[2])}
    computed = {tuple(t) for t in fixture["relation_artifact"]["relations"]
                if t[0] == q["relation"] and anchor_uid in (t[1], t[2])}
    correct = computed == gt

    rel_touch = _touching(fixture["relation_artifact"]["relations"],
                          anchor_uid)
    ser_touch = {(e["relation"], e["subject"], e["object"])
                 for e in fixture["serialized_graph"]["edges"]
                 if anchor_uid in (e["subject"], e["object"])}
    serialized = ser_touch == rel_touch

    cand = fixture["grounded_candidates"][qid]
    grounded = (cand["uid"] in by_uid
                and by_uid[cand["uid"]]["name"] == q["anchor"])

    ans = fixture["answer_artifact"][qid]
    if ans["outcome"] == "abstain":
        answer_status = "abstain"
    elif q["expected_outcome"] == "answer":
        answer_status = ("pass" if ans["outcome"] == "answer"
                         and set(ans["uids"]) == set(q["expected_uids"])
                         else "fail")
    else:
        answer_status = "pass" if ans["outcome"] == "empty" else "fail"

    return {
        "key_eligibility": (
            "pass" if key_ok else "fail",
            "evaluation_key: unambiguous item with a usable expected "
            "outcome"),
        "object_delivery": (
            "pass" if delivered else "fail",
            "entity_artifact contains every entity the question "
            "references"),
        "relation_applicability": (
            "pass" if applicable else "fail",
            "relation_artifact.vocabulary expresses the question's "
            "relation"),
        "relation_correctness": (
            "pass" if correct else "fail",
            "relation_artifact vs fixture ground-truth relations at the "
            "question's anchor"),
        "serialization_consistency": (
            "pass" if serialized else "fail",
            "serialized_graph edges vs relation_artifact at the "
            "question's anchor"),
        "referent_grounding": (
            "pass" if grounded else "fail",
            "grounded_candidates uid resolves to an entity whose name is "
            "the question's anchor"),
        "answer_generation": (
            answer_status,
            "answer_artifact vs evaluation_key expected outcome"),
    }


def evaluate_artifacts(fixture: dict) -> dict[str, Trace]:
    """Masked evaluation: derive per-question traces from the artifact
    chain alone. The caller may have injected any fault into any single
    artifact; this function is never told which (or whether)."""
    traces: dict[str, Trace] = {}
    for q in fixture["evaluation_key"]["questions"]:
        obs = _question_observations(q, fixture)
        key_failed = obs["key_eligibility"][0] == "fail"
        answer_status = obs["answer_generation"][0]
        result = ("excluded" if key_failed else
                  {"pass": "correct", "fail": "wrong",
                   "abstain": "abstain"}[answer_status])
        traces[q["question_id"]] = evaluate_trace(
            question_id=q["question_id"], scene_id="fixture_scene",
            arm="fixture_pipeline", path_id="fixture_diagnostic",
            scope="bug_diagnostic",
            expected_outcome=q["expected_outcome"],
            observations=obs, result=result,
            raw_category=fixture["answer_artifact"][q["question_id"]]
            ["outcome"])
    return traces


def attributions(traces: dict[str, Trace]) -> dict[str, dict | None]:
    return {qid: attribute(t) for qid, t in sorted(traces.items())}


# ------------------------------------------------------------- the battery
def run_battery(fixture: dict) -> dict:
    """8 fault classes x 3 relation types, evaluator-masked. A cell is
    localized iff the target question's attribution names exactly the
    expected (stage, status) AND every other question stays clean. The
    expected label is consulted only AFTER evaluation."""
    clean = attributions(evaluate_artifacts(fixture))
    clean_failures = sum(1 for a in clean.values() if a is not None)

    cells = []
    n_localized = 0
    for name, inject in sorted(INJECTIONS.items()):
        for relation in RELATION_VOCAB:
            traces = evaluate_artifacts(inject(fixture, relation))
            attr = attributions(traces)  # masked evaluation ends here
            expected_stage, expected_status = EXPECTED_LOCALIZATION[name]
            target = TARGET_QID[relation]
            got = attr[target]
            others_clean = all(a is None for qid, a in attr.items()
                               if qid != target)
            localized = (got is not None
                         and got["stage"] == expected_stage
                         and got["status"] == expected_status
                         and others_clean)
            n_localized += localized
            cells.append({
                "fault": name, "relation": relation,
                "expected": {"stage": expected_stage,
                             "status": expected_status},
                "attributed": (None if got is None
                               else {"stage": got["stage"],
                                     "status": got["status"]}),
                "other_questions_clean": others_clean,
                "localized": localized,
            })
    return {
        "schema": "stagereach_fault_battery",
        "schema_version": 1,
        "n_total": len(cells),
        "n_localized": n_localized,
        "n_fault_classes": len(INJECTIONS),
        "n_relation_types": len(RELATION_VOCAB),
        "clean_failures": clean_failures,
        "cells": cells,
    }
