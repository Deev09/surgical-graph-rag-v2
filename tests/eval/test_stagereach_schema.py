#!/usr/bin/env python3
"""Tests for eval/stagereach/schema.py — the frozen vocabulary and paths.

Every assertion here restates docs/stagereach_schema_freeze.md sections 1-4
and 7-8 as code, so a drift in the declarations is a test failure, not a
review comment.
"""
from __future__ import annotations

import dataclasses
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.stagereach import metrics, schema  # noqa: E402
from eval.stagereach.schema import (  # noqa: E402
    FROZEN_ARKIT_PATH_IDS,
    PATHS,
    SCOPES,
    STAGE_UNIVERSE,
    STATUSES,
    StageDecl,
    StagePath,
    StageRecord,
    Trace,
    trace_to_dict,
)


def test_vocabularies_are_frozen():
    assert STAGE_UNIVERSE == (
        "key_eligibility", "object_delivery", "relation_applicability",
        "relation_correctness", "serialization_consistency",
        "identity_injection", "referent_grounding", "answer_generation")
    assert STATUSES == ("pass", "fail", "unknown", "not_applicable",
                        "abstain", "not_reached")
    assert len(STATUSES) == 6
    assert SCOPES == ("deployable", "identity_oracle", "delivered",
                      "oracle_free_component_eval", "proposal_ceiling",
                      "definition_change", "bug_diagnostic")
    assert len(SCOPES) == 7


def test_frozen_paths_match_the_freeze_table():
    """Section 4 of the freeze doc, row by row."""
    for pid in FROZEN_ARKIT_PATH_IDS:
        assert pid in PATHS, pid
    full = ("key_eligibility", "object_delivery", "relation_applicability",
            "relation_correctness", "serialization_consistency",
            "referent_grounding", "answer_generation")
    for pid in ("graph_deployable_delivered", "graph_deployable_grounded"):
        p = PATHS[pid]
        assert p.stage_names() == full, pid
        assert p.bypassed == ()
        assert set(p.allowed_scopes) == {"deployable", "delivered"}
    p = PATHS["graph_identity_oracle"]
    assert p.stage_names() == (
        "key_eligibility", "object_delivery", "relation_applicability",
        "relation_correctness", "serialization_consistency",
        "identity_injection", "answer_generation")
    assert p.bypassed == ("referent_grounding",)
    assert p.allowed_scopes == ("identity_oracle",)
    p = PATHS["geometry_ceiling"]
    assert p.stage_names() == (
        "key_eligibility", "object_delivery", "relation_applicability",
        "identity_injection", "answer_generation")
    assert set(p.bypassed) == {"relation_correctness",
                               "serialization_consistency",
                               "referent_grounding"}
    assert set(p.allowed_scopes) == {"proposal_ceiling", "identity_oracle"}
    p = PATHS["direct_rgb"]
    assert p.stage_names() == ("key_eligibility", "answer_generation")
    assert set(p.bypassed) == {"object_delivery", "relation_applicability",
                               "relation_correctness",
                               "serialization_consistency",
                               "referent_grounding"}
    assert p.allowed_scopes == ("deployable",)


def test_stage_attributes_are_frozen():
    """key_eligibility is answer_path=False, oracle_fed=True everywhere;
    identity_injection is answer_path=True, oracle_fed=True."""
    for p in PATHS.values():
        d = p.decl("key_eligibility")
        assert d.answer_path is False and d.oracle_fed is True, p.path_id
        if "identity_injection" in p.stage_names():
            d = p.decl("identity_injection")
            assert d.answer_path is True and d.oracle_fed is True, p.path_id


def test_serialization_never_depends_on_relation_correctness():
    for p in PATHS.values():
        if "serialization_consistency" in p.stage_names():
            deps = p.decl("serialization_consistency").gating_deps
            assert "relation_correctness" not in deps, p.path_id
    # and constructing a path that violates this refuses outright
    try:
        StagePath(
            path_id="bad",
            stages=(
                StageDecl("key_eligibility", False, True),
                StageDecl("relation_correctness", False, True,
                          ("key_eligibility",)),
                StageDecl("serialization_consistency", True, False,
                          ("relation_correctness",)),
            ),
            allowed_scopes=("bug_diagnostic",))
        raise AssertionError("a serialization->correctness dep was accepted")
    except ValueError:
        pass


def test_relation_correctness_is_non_gating_on_arkit_paths():
    """It appears in no other stage's gating deps on any frozen ARKit path
    (it is a NON-GATING audit stage there); on the fixture path it DOES
    gate the answer, which is what makes relation faults attributable."""
    for pid in FROZEN_ARKIT_PATH_IDS:
        assert "relation_correctness" not in PATHS[pid].gating_stages(), pid
    assert "relation_correctness" in PATHS["fixture_diagnostic"].gating_stages()


def test_ladder_stages_per_path():
    """The causal ladder is the gating closure of answer_generation."""
    full_chain = ("key_eligibility", "object_delivery",
                  "relation_applicability", "serialization_consistency",
                  "referent_grounding", "answer_generation")
    assert PATHS["graph_deployable_delivered"].ladder_stages() == full_chain
    assert PATHS["graph_deployable_grounded"].ladder_stages() == full_chain
    # oracle arms: human identity injection makes the deployable
    # intermediate stages audit-only, so the causal ladder is scored->answer
    assert PATHS["graph_identity_oracle"].ladder_stages() == (
        "key_eligibility", "answer_generation")
    assert PATHS["geometry_ceiling"].ladder_stages() == (
        "key_eligibility", "answer_generation")
    assert PATHS["direct_rgb"].ladder_stages() == (
        "key_eligibility", "answer_generation")


def test_path_construction_rejects_misdeclarations():
    k = StageDecl("key_eligibility", False, True)
    # out-of-order subset
    try:
        StagePath("bad", (StageDecl("answer_generation", True, False), k),
                  allowed_scopes=("deployable",))
        raise AssertionError("out-of-order stages accepted")
    except ValueError:
        pass
    # gating dep on a later stage
    try:
        StagePath("bad", (StageDecl("key_eligibility", False, True,
                                    ("answer_generation",)),
                          StageDecl("answer_generation", True, False)),
                  allowed_scopes=("deployable",))
        raise AssertionError("forward gating dep accepted")
    except ValueError:
        pass
    # a stage both on-path and bypassed
    try:
        StagePath("bad", (k,), bypassed=("key_eligibility",),
                  allowed_scopes=("deployable",))
        raise AssertionError("on-path bypass accepted")
    except ValueError:
        pass
    # unknown scope
    try:
        StagePath("bad", (k,), allowed_scopes=("headline",))
        raise AssertionError("unknown scope accepted")
    except ValueError:
        pass


def test_declarations_are_frozen_dataclasses():
    d = PATHS["direct_rgb"].decl("key_eligibility")
    try:
        d.oracle_fed = False  # type: ignore[misc]
        raise AssertionError("StageDecl is mutable")
    except dataclasses.FrozenInstanceError:
        pass
    r = StageRecord("key_eligibility", "pass", "test")
    try:
        r.status = "fail"  # type: ignore[misc]
        raise AssertionError("StageRecord is mutable")
    except dataclasses.FrozenInstanceError:
        pass


def _tiny_trace() -> Trace:
    return Trace(
        question_id="q1", scene_id="s1", arm="blinded_rgb_vlm",
        path_id="direct_rgb", scope="deployable", expected_outcome="answer",
        stages=(
            StageRecord("key_eligibility", "pass", "key"),
            StageRecord("answer_generation", "pass", "arm outcome"),
            StageRecord("object_delivery", "not_applicable", "bypass"),
            StageRecord("relation_applicability", "not_applicable", "bypass"),
            StageRecord("relation_correctness", "not_applicable", "bypass"),
            StageRecord("serialization_consistency", "not_applicable", "bypass"),
            StageRecord("referent_grounding", "not_applicable", "bypass"),
        ),
        result="correct", raw_category="correct")


def test_trace_serialization_carries_schema_and_final_outcome():
    d = trace_to_dict(_tiny_trace())
    assert d["schema"] == "stagereach_trace"
    assert d["schema_version"] == 1
    assert d["final_outcome"] == {"result": "correct",
                                  "positive_expected": True}
    assert d["raw_category"] == "correct"
    assert [s["stage"] for s in d["stages"]][:2] == [
        "key_eligibility", "answer_generation"]


def test_trace_rejects_bad_vocabulary():
    for kwargs in (
        {"path_id": "nope"},
        {"scope": "headline"},
        {"expected_outcome": "maybe"},
        {"result": "great"},
    ):
        base = dict(question_id="q", scene_id="s", arm="a",
                    path_id="direct_rgb", scope="deployable",
                    expected_outcome="answer", stages=(), result="correct")
        base.update(kwargs)
        try:
            Trace(**base)  # type: ignore[arg-type]
            raise AssertionError(f"accepted {kwargs}")
        except ValueError:
            pass


def test_arkit_outcome_mapping_is_total():
    outcomes = ("correct", "wrong", "unanswered", "excluded_no_human_answer")
    for o in outcomes:
        e, r = metrics.normalize_arkit_outcome(o)
        assert e == "answer" and r in ("correct", "wrong", "abstain",
                                       "excluded")
    assert metrics.normalize_arkit_outcome("unanswered") == ("answer",
                                                             "abstain")
    assert metrics.normalize_arkit_outcome("excluded_no_human_answer") == (
        "answer", "excluded")
    try:
        metrics.normalize_arkit_outcome("kinda_right")
        raise AssertionError("unknown ARKit outcome accepted")
    except ValueError:
        pass


def test_router_outcome_mapping_is_total_over_nine_categories():
    """Every router_qa category maps to a valid (expected, result) pair,
    and the miss/false_answer splits use the per-question record."""
    assert len(metrics.ROUTER_CATEGORIES) == 9
    rep = {
        "true_answer": {"expected_outcome": "answer"},
        "true_empty": {"expected_outcome": "empty"},
        "correct_defer": {"expected_outcome": "defer"},
        "true_unknown": {"expected_outcome": "unknown"},
        "true_parser_failure": {"expected_outcome": "parser_failure"},
        "true_execution_error": {"expected_outcome": "execution_error"},
        "miss": {"expected_outcome": "answer", "actual_outcome": "bindings",
                 "deferred": False},
        "false_answer": {"expected_outcome": "empty",
                         "actual_outcome": "bindings", "deferred": False},
        "unexpected": {"expected_outcome": "empty", "actual_outcome":
                       "abstain", "deferred": False},
    }
    for cat, rec in rep.items():
        rec = dict(rec, category=cat)
        e, r = metrics.normalize_router_record(rec)
        assert e in ("answer", "empty", "defer"), (cat, e)
        assert r in ("correct", "wrong", "abstain", "excluded"), (cat, r)
    # the record-level splits the freeze doc pins down:
    assert metrics.normalize_router_record(
        {"category": "miss", "expected_outcome": "answer",
         "actual_outcome": "abstain", "deferred": True}) == ("answer",
                                                             "abstain")
    assert metrics.normalize_router_record(
        {"category": "miss", "expected_outcome": "answer",
         "actual_outcome": "empty", "deferred": False}) == ("answer", "wrong")
    assert metrics.normalize_router_record(
        {"category": "miss", "expected_outcome": "answer",
         "actual_outcome": "bindings", "deferred": False}) == ("answer",
                                                               "wrong")
    assert metrics.normalize_router_record(
        {"category": "false_answer", "expected_outcome": "empty",
         "actual_outcome": "bindings"}) == ("empty", "wrong")
    assert metrics.normalize_router_record(
        {"category": "false_answer", "expected_outcome": "answer",
         "actual_outcome": "bindings"}) == ("answer", "wrong")
    try:
        metrics.normalize_router_record({"category": "vibes",
                                         "expected_outcome": "answer"})
        raise AssertionError("unknown category accepted")
    except ValueError:
        pass


TESTS = [
    test_vocabularies_are_frozen,
    test_frozen_paths_match_the_freeze_table,
    test_stage_attributes_are_frozen,
    test_serialization_never_depends_on_relation_correctness,
    test_relation_correctness_is_non_gating_on_arkit_paths,
    test_ladder_stages_per_path,
    test_path_construction_rejects_misdeclarations,
    test_declarations_are_frozen_dataclasses,
    test_trace_serialization_carries_schema_and_final_outcome,
    test_trace_rejects_bad_vocabulary,
    test_arkit_outcome_mapping_is_total,
    test_router_outcome_mapping_is_total_over_nine_categories,
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
