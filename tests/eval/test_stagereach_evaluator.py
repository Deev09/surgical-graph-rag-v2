#!/usr/bin/env python3
"""Tests for eval/stagereach/{evaluator,metrics}.py — gating, survival,
attribution, and the six frozen invariants (each one raises, and each raise
is exercised here).

Every expected number is computed by hand from the semantics in
docs/stagereach_schema_freeze.md section 5, on tiny synthetic traces.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.stagereach import metrics  # noqa: E402
from eval.stagereach.evaluator import (  # noqa: E402
    InvariantViolation,
    attribute,
    evaluate_trace,
    unmeasured_stages,
    validate_trace,
)
from eval.stagereach.schema import (  # noqa: E402
    PATHS,
    StageDecl,
    StagePath,
    StageRecord,
    Trace,
)

DEPLOYABLE = PATHS["graph_deployable_delivered"]
FULL = DEPLOYABLE.stage_names()


def _obs(**overrides) -> dict[str, tuple[str, str]]:
    """A fully-passing deployable-path observation set, then overrides."""
    base = {s: ("pass", "test") for s in FULL}
    base["relation_correctness"] = ("unknown", "no annotation")
    base.update(overrides)
    return base


def _trace(observations=None, *, result="correct", raw="correct",
           arm="delivered_graph", path_id="graph_deployable_delivered",
           scope="deployable", qid="q1") -> Trace:
    return evaluate_trace(
        question_id=qid, scene_id="s1", arm=arm, path_id=path_id,
        scope=scope, expected_outcome="answer",
        observations=observations or _obs(), result=result,
        raw_category=raw)


# ------------------------------------------------------------------- gating
def test_fail_gates_downstream_to_not_reached():
    t = _trace(_obs(object_delivery=("fail", "no uid mapping")))
    assert t.status("object_delivery") == "fail"
    for s in ("relation_applicability", "serialization_consistency",
              "referent_grounding", "answer_generation"):
        assert t.status(s) == "not_reached", s
    # relation_correctness gates on the chain too (its own reachability),
    # so it is also not_reached after a delivery failure
    assert t.status("relation_correctness") == "not_reached"


def test_unknown_never_zeroes_downstream():
    """The delivered-graph shape: grounding is unmeasured, the answer stage
    is still reached and its abstention is recorded."""
    t = _trace(_obs(referent_grounding=("unknown", "arm is silent"),
                    answer_generation=("abstain", "unanswered")),
               result="abstain", raw="unanswered")
    assert t.status("referent_grounding") == "unknown"
    assert t.status("answer_generation") == "abstain"
    assert unmeasured_stages(t) == ("relation_correctness",
                                    "referent_grounding")


def test_audit_fail_does_not_reduce_reachability():
    """relation_correctness is non-gating on ARKit paths: a fail there
    stays visible but everything downstream is still reached."""
    t = _trace(_obs(relation_correctness=("fail", "hypothetical audit")))
    assert t.status("relation_correctness") == "fail"
    assert t.status("serialization_consistency") == "pass"
    assert t.status("answer_generation") == "pass"
    # ... and it is never attributed (freeze doc section 5)
    assert attribute(t) is None


def test_bypassed_stages_are_not_applicable():
    obs = {"key_eligibility": ("pass", "key"),
           "answer_generation": ("pass", "arm")}
    t = _trace(obs, arm="blinded_rgb_vlm", path_id="direct_rgb")
    for s in PATHS["direct_rgb"].bypassed:
        assert t.status(s) == "not_applicable", s


def test_observations_must_cover_the_path():
    try:
        _trace({"key_eligibility": ("pass", "key")})
        raise AssertionError("missing observations accepted")
    except ValueError:
        pass
    try:
        _trace(dict(_obs(), identity_injection=("pass", "off-path")))
        raise AssertionError("off-path observation accepted")
    except ValueError:
        pass
    try:
        _trace(_obs(object_delivery=("not_reached", "adapter lied")))
        raise AssertionError("adapter-supplied not_reached accepted")
    except ValueError:
        pass


def test_scope_must_be_allowed_on_the_path():
    try:
        _trace(scope="proposal_ceiling")
        raise AssertionError("disallowed scope accepted")
    except ValueError:
        pass


# -------------------------------------------------------------- attribution
def test_attribution_reports_first_gating_fail():
    t = _trace(_obs(serialization_consistency=("fail", "edge drifted"),
                    referent_grounding=("fail", "would also fail")))
    a = attribute(t)
    assert a is not None and a["stage"] == "serialization_consistency"
    assert a["status"] == "fail"


def test_attribution_never_names_unknown():
    t = _trace(_obs(referent_grounding=("unknown", "arm is silent"),
                    answer_generation=("fail", "wrong answer")),
               result="wrong", raw="wrong")
    a = attribute(t)
    assert a is not None and a["stage"] == "answer_generation"


def test_attribution_of_terminal_abstention():
    t = _trace(_obs(answer_generation=("abstain", "unanswered")),
               result="abstain", raw="unanswered")
    a = attribute(t)
    assert a == {"stage": "answer_generation", "status": "abstain",
                 "source": "unanswered"}


def test_clean_trace_attributes_nothing():
    assert attribute(_trace()) is None


# ------------------------------------------------------- the six invariants
def test_invariant_one_unknown_is_never_pass_in_survival():
    """Accounting check: a cooked stage row where an unknown was folded
    into pass raises; and the ladder counts an all-unknown stage as
    unmeasured (no rung), never as survivors."""
    row = {"stage": "referent_grounding", "n": 10, "reached": 8, "pass": 8,
           "fail": 0, "unknown": 2, "abstain": 0, "not_reached": 2,
           "not_applicable": 0}
    try:
        metrics.verify_stage_accounting(row)
        raise AssertionError("unknown folded into pass was accepted")
    except InvariantViolation as e:
        assert e.invariant == 1
    traces = [_trace(_obs(referent_grounding=("unknown", "silent"),
                          answer_generation=("abstain", "unanswered")),
                     result="abstain", raw="unanswered", qid=f"q{i}")
              for i in range(3)]
    rungs = metrics.survival_ladder(traces, DEPLOYABLE)
    assert "referent_grounding" not in [r["stage"] for r in rungs]
    assert not metrics.counts_as_pass("unknown")


def test_invariant_two_deployable_never_pass_consumes_oracle_stage():
    """CLAUDE.md deployable-lane rule 3 as code. allowed_scopes already
    blocks this at evaluate_trace time, so exercise the deeper check on a
    directly-constructed trace that claims deployable scope while
    pass-consuming identity_injection."""
    oracle = PATHS["graph_identity_oracle"]
    records = [StageRecord(s, "pass", "test")
               for s in oracle.stage_names()] + [
               StageRecord("referent_grounding", "not_applicable", "bypass")]
    t = Trace(question_id="q1", scene_id="s1", arm="stored",
              path_id="graph_identity_oracle", scope="deployable",
              expected_outcome="answer", stages=tuple(records),
              result="correct", raw_category="correct")
    try:
        validate_trace(t)
        raise AssertionError("deployable trace pass-consumed "
                             "identity_injection")
    except InvariantViolation as e:
        assert e.invariant == 2


def test_invariant_three_no_silent_bypass():
    """A reached status downstream of a fail on a depended-on stage raises,
    and not_applicable off the declared bypass list raises."""
    names = DEPLOYABLE.stage_names()
    records = {s: "pass" for s in names}
    records["object_delivery"] = "fail"
    records["answer_generation"] = "pass"  # silently bypassed the fail
    records["relation_applicability"] = "not_reached"
    records["relation_correctness"] = "not_reached"
    records["serialization_consistency"] = "not_reached"
    records["referent_grounding"] = "not_reached"
    t = Trace(question_id="q1", scene_id="s1", arm="delivered_graph",
              path_id="graph_deployable_delivered", scope="deployable",
              expected_outcome="answer",
              stages=tuple(StageRecord(s, records[s], "test") for s in names),
              result="correct", raw_category="correct")
    try:
        validate_trace(t)
        raise AssertionError("silent bypass accepted")
    except InvariantViolation as e:
        assert e.invariant == 3
    # not_applicable on a stage the path does not declare bypassed
    records["answer_generation"] = "not_reached"
    records["referent_grounding"] = "not_applicable"
    t2 = Trace(question_id="q1", scene_id="s1", arm="delivered_graph",
               path_id="graph_deployable_delivered", scope="deployable",
               expected_outcome="answer",
               stages=tuple(StageRecord(s, records[s], "test")
                            for s in names),
               result="correct", raw_category="correct")
    try:
        validate_trace(t2)
        raise AssertionError("undeclared not_applicable accepted")
    except InvariantViolation as e:
        assert e.invariant == 3


def test_invariant_four_no_pooled_accuracy():
    try:
        metrics.pooled_accuracy([_trace()])
        raise AssertionError("pooled accuracy was computed")
    except InvariantViolation as e:
        assert e.invariant == 4
    # and the matrix keeps the expected axis: positives never pool with
    # true-empties
    m = metrics.outcome_matrix([_trace()])
    assert m == {("answer", "correct"): 1}


def test_invariant_five_non_exhaustive_keys_refuse_prf():
    try:
        metrics.precision_recall(3, 1, 2, relation="NEAR_SURFACE",
                                 exhaustive=False)
        raise AssertionError("P/R emitted for a non-exhaustive key")
    except InvariantViolation as e:
        assert e.invariant == 5
    ok = metrics.precision_recall(3, 1, 2, relation="ON_ENTITY_SURFACE",
                                  exhaustive=True)
    assert ok["precision"] == 0.75 and ok["recall"] == 0.6


def test_invariant_six_definition_change_never_pools():
    frozen = _trace()
    # a definition_change-scope trace, constructed directly (no frozen path
    # allows the scope, which is the point)
    records = tuple(StageRecord(s, "pass", "t") for s in FULL)
    changed = Trace(question_id="q9", scene_id="s1", arm="retuned",
                    path_id="graph_deployable_delivered",
                    scope="definition_change", expected_outcome="answer",
                    stages=records, result="correct", raw_category="correct")
    try:
        metrics.outcome_matrix([frozen, changed])
        raise AssertionError("definition_change pooled with frozen track")
    except InvariantViolation as e:
        assert e.invariant == 6
    # a pure definition_change set is reportable on its own
    assert metrics.outcome_matrix([changed]) == {("answer", "correct"): 1}


# ------------------------------------------------------------------ ladders
def test_ladder_shape_on_synthetic_arm():
    """3 scored, 1 delivery fail, then 1 grounding fail, 1 correct:
    ladder over the gating chain = 3, 2, 2, 2, 1, 1."""
    traces = [
        _trace(qid="qa"),
        _trace(_obs(object_delivery=("fail", "missing")), result="abstain",
               raw="unanswered", qid="qb"),
        _trace(_obs(referent_grounding=("fail", "bridge abstained")),
               result="abstain", raw="unanswered", qid="qc"),
    ]
    rungs = metrics.survival_ladder(traces, DEPLOYABLE)
    assert [(r["stage"], r["survivors"]) for r in rungs] == [
        ("key_eligibility", 3), ("object_delivery", 2),
        ("relation_applicability", 2), ("serialization_consistency", 2),
        ("referent_grounding", 1), ("answer_generation", 1)]
    # reached is reported separately at every rung
    assert [r["reached"] for r in rungs] == [3, 3, 2, 2, 2, 1]


def test_stage_report_reports_unknown_separately():
    traces = [
        _trace(_obs(referent_grounding=("unknown", "silent"),
                    answer_generation=("abstain", "unanswered")),
               result="abstain", raw="unanswered", qid=f"q{i}")
        for i in range(2)
    ] + [_trace(qid="q3")]
    rows = {r["stage"]: r for r in metrics.stage_report(traces, DEPLOYABLE)}
    g = rows["referent_grounding"]
    assert (g["reached"], g["pass"], g["unknown"]) == (3, 1, 2)
    rc = rows["relation_correctness"]
    assert (rc["unknown"], rc["pass"]) == (3, 0)


TESTS = [
    test_fail_gates_downstream_to_not_reached,
    test_unknown_never_zeroes_downstream,
    test_audit_fail_does_not_reduce_reachability,
    test_bypassed_stages_are_not_applicable,
    test_observations_must_cover_the_path,
    test_scope_must_be_allowed_on_the_path,
    test_attribution_reports_first_gating_fail,
    test_attribution_never_names_unknown,
    test_attribution_of_terminal_abstention,
    test_clean_trace_attributes_nothing,
    test_invariant_one_unknown_is_never_pass_in_survival,
    test_invariant_two_deployable_never_pass_consumes_oracle_stage,
    test_invariant_three_no_silent_bypass,
    test_invariant_four_no_pooled_accuracy,
    test_invariant_five_non_exhaustive_keys_refuse_prf,
    test_invariant_six_definition_change_never_pools,
    test_ladder_shape_on_synthetic_arm,
    test_stage_report_reports_unknown_separately,
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
