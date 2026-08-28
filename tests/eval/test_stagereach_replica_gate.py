#!/usr/bin/env python3
"""G-REPLICA gate — Phase-8 scorecard transfer, raw AND normalized.

The gate (freeze doc §9): BOTH the raw categories
{true_answer: 4, true_empty: 27, miss: 22, false_answer: 3} summing to 56
AND the normalized matrix {(answer,correct): 4, (answer,wrong): 20,
(answer,abstain): 4, (empty,correct): 27, (empty,wrong): 1}; per-scene n
13/14/16/13; and every §6 guard raises when violated.

The adapter reads ONLY the packed copies in
eval/results/project_census_v1/ (never runs/), and every internal stage the
scorecards cannot support is UNKNOWN — never inferred from the final
answer.
"""
from __future__ import annotations

import copy
import sys
import traceback
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.stagereach import metrics  # noqa: E402
from eval.stagereach.adapters import replica  # noqa: E402
from eval.stagereach.evaluator import InvariantViolation  # noqa: E402
from eval.stagereach.schema import StageRecord, Trace  # noqa: E402

_CACHE: dict = {}


def _data():
    if "traces" not in _CACHE:
        _CACHE["scorecards"] = replica.load_scorecards(REPO_ROOT)
        _CACHE["keys"] = replica.load_keys(REPO_ROOT)
        _CACHE["traces"] = replica.derive_traces(_CACHE["scorecards"],
                                                 _CACHE["keys"])
    return _CACHE["scorecards"], _CACHE["keys"], _CACHE["traces"]


# ------------------------------------------------------------ gate numbers
def test_gate_replica_raw_categories():
    _, _, traces = _data()
    assert len(traces) == 56
    raw = metrics.raw_category_counts(traces)
    assert raw == {"true_answer": 4, "true_empty": 27, "miss": 22,
                   "false_answer": 3}
    assert sum(raw.values()) == 56


def test_gate_replica_normalized_matrix():
    _, _, traces = _data()
    m = metrics.outcome_matrix(traces)
    assert m == {("answer", "correct"): 4, ("answer", "wrong"): 20,
                 ("answer", "abstain"): 4, ("empty", "correct"): 27,
                 ("empty", "wrong"): 1}
    # both margins close over the same 56 questions
    assert sum(m.values()) == 56


def test_gate_replica_per_scene_populations():
    _, _, traces = _data()
    per_scene = Counter(t.scene_id for t in traces)
    assert per_scene == {"replica_office_0": 13, "replica_room_0": 14,
                         "replica_room_1": 16, "replica_room_2": 13}


def test_gate_replica_aggregate_cross_check():
    scorecards, _, traces = _data()
    agg = replica.load_aggregate(REPO_ROOT)
    out = replica.aggregate_cross_check(agg, traces)
    assert out["category_counts_match"] is True
    assert out["total_questions"] == 56


# ------------------------------------------------------------------ guards
def test_guard_every_trace_carries_oracle_scope_labeling():
    _, _, traces = _data()
    for t in traces:
        assert t.scope == "delivered"
        assert "human_verified" in t.record("key_eligibility").source, \
            t.question_id


def test_guard_internal_stages_are_unknown_never_inferred():
    """Delivery, relation correctness, serialization and grounding are
    unmeasured by the packed scorecards: unknown on every trace, with a
    source that says so, regardless of the final answer."""
    _, _, traces = _data()
    for t in traces:
        for stage in ("object_delivery", "relation_applicability",
                      "relation_correctness", "serialization_consistency",
                      "referent_grounding"):
            r = t.record(stage)
            assert r.status == "unknown", (t.question_id, stage, r.status)
            assert "never inferred from the final answer" in r.source
    # concretely: a correct answer did NOT promote any internal stage
    correct = [t for t in traces if t.result == "correct"]
    assert correct and all(t.status("object_delivery") == "unknown"
                           for t in correct)


def test_guard_forbidden_scene_raises():
    try:
        replica.guard_scene("replica_frl_apartment_0")
        raise AssertionError("frl_apartment_0 accepted")
    except replica.ReplicaGuardError:
        pass
    try:
        replica.guard_scene("replica_room_9")
        raise AssertionError("unknown scene accepted")
    except replica.ReplicaGuardError:
        pass
    # and the packed pack itself contains no forbidden scene
    scorecards, _, _ = _data()
    for scene, doc in scorecards.items():
        assert replica.FORBIDDEN_SCENE_SUBSTRING not in scene
        assert replica.FORBIDDEN_SCENE_SUBSTRING not in doc["scene_id"]


def test_guard_plausibility_scorecard_refused():
    scorecards, _, _ = _data()
    fake = copy.deepcopy(scorecards["replica_room_0"])
    fake["answer_key_type"] = "plausibility_not_ground_truth"
    try:
        replica.guard_scorecard(fake)
        raise AssertionError("plausibility scorecard accepted")
    except replica.ReplicaGuardError:
        pass


def test_guard_near_surface_refuses_precision_recall():
    """NEAR_SURFACE keys are non-exhaustive on every packed scene, so
    requesting P/R for them raises invariant 5; an exhaustive relation on
    the same scene passes through the same gate."""
    scorecards, keys, _ = _data()
    for scene in replica.SCENES:
        assert replica.relation_exhaustive_map(keys[scene])[
            "NEAR_SURFACE"] is False, scene
        try:
            replica.precision_recall_for(scene, "NEAR_SURFACE", scorecards,
                                         keys)
            raise AssertionError(f"{scene}: NEAR_SURFACE P/R emitted")
        except InvariantViolation as e:
            assert e.invariant == 5
    ok = replica.precision_recall_for("replica_room_0", "ON_ENTITY_SURFACE",
                                      scorecards, keys)
    assert ok["exhaustive"] is True and ok["relation"] == "ON_ENTITY_SURFACE"


def test_guard_definition_change_cannot_pool_with_frozen_track():
    _, _, traces = _data()
    changed = Trace(
        question_id="Q99", scene_id="replica_room_0", arm="retuned_router",
        path_id="graph_deployable_delivered", scope="definition_change",
        expected_outcome="answer",
        stages=tuple(StageRecord(s, "unknown", "hypothetical retune")
                     if s != "answer_generation"
                     else StageRecord(s, "pass", "hypothetical retune")
                     for s in ("key_eligibility", "object_delivery",
                               "relation_applicability",
                               "relation_correctness",
                               "serialization_consistency",
                               "referent_grounding", "answer_generation")),
        result="correct", raw_category="true_answer")
    try:
        metrics.outcome_matrix(list(traces) + [changed])
        raise AssertionError("definition_change pooled with frozen track")
    except InvariantViolation as e:
        assert e.invariant == 6


def test_replica_reads_only_the_packed_copies():
    """The adapter's source constants point at the pack and the QA keys —
    never at runs/."""
    assert replica.PACK_RELPATH == "eval/results/project_census_v1"
    assert replica.KEYS_RELPATH == "eval/questions/phase8"
    for rel in (replica.PACK_RELPATH, replica.KEYS_RELPATH):
        assert not rel.startswith("runs"), rel


TESTS = [
    test_gate_replica_raw_categories,
    test_gate_replica_normalized_matrix,
    test_gate_replica_per_scene_populations,
    test_gate_replica_aggregate_cross_check,
    test_guard_every_trace_carries_oracle_scope_labeling,
    test_guard_internal_stages_are_unknown_never_inferred,
    test_guard_forbidden_scene_raises,
    test_guard_plausibility_scorecard_refused,
    test_guard_near_surface_refuses_precision_recall,
    test_guard_definition_change_cannot_pool_with_frozen_track,
    test_replica_reads_only_the_packed_copies,
]


def main() -> int:
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
            print()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
