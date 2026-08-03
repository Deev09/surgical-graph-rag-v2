"""Tests for answer-level confidence + the tau gate (v2-calibration).

Run: python tests/reasoner/test_answer_confidence.py

Synthetic bundles only -- no Replica dataset, so this runs anywhere.

What is pinned here:
  - DEFAULT BEHAVIOR IS UNCHANGED. answer_tau defaults to 0.0, rejections
    default to empty, and no confidence is ever below 0.0, so the gate can
    never fire on a default context. This is the property the frozen
    scene_scorecard gate depends on.
  - The rejection ceiling is `1 - max(margin_confidence)` over rejections
    scoped to the query, and policy rejections (no margin) are skipped
    rather than scored as 0.
  - Scope discipline: a rejection in a different relation family, or one
    that touches neither anchor, is not in scope.
  - `empty` and `bindings` both consult the ceiling; refusals score 0.0.
  - min / mean / product actually differ, so the aggregation choice is a
    real parameter rather than a comment.
  - The gate rewrites a low-confidence claim to `unknown` with the
    "not enough evidence" text, and leaves refusals alone.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.types import SceneFrame
from graph.schema import (
    Edge, EdgeRejection, GraphRef, Node, SceneGraphBundle, SurfaceRecord,
)
from common.types import Plane
from reasoner.base import (
    CandidateScope, CompileResult, CompletenessProfile, ExecutionContext,
    ExecutionResult,
)
from reasoner.compiler_rules import RulesCompiler
from reasoner.confidence import (
    AGGREGATIONS, aggregate, rejection_ceiling, score_answer, scoped_rejections,
)
from reasoner.executor import RulesExecutor
from reasoner.router import Router
from reasoner.verbalizer import StandardVerbalizer


FRAME = SceneFrame(gravity=(0.0, 0.0, -1.0), canonical_forward=(0.0, 1.0, 0.0),
                   canonical_right=(1.0, 0.0, 0.0), units="meters", notes="test")


def _node(uid: str, label: str, label_confidence: float = 1.0) -> Node:
    return Node(
        id=uid, label=label, label_confidence=label_confidence,
        centroid=(0.0, 0.0, 0.0),
        bbox_aabb=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        bbox_obb=None, embedding_ref=None,
        attributes={"display_label": label, "aliases": []},
    )


def _edge(edge_id: str, src: str, type_: str, tgt: str, confidence: float,
          tgt_kind: str = "entity") -> Edge:
    return Edge(
        edge_id=edge_id, source=GraphRef(kind="entity", uid=src), type=type_,
        target=GraphRef(kind=tgt_kind, uid=tgt), frame="world",
        weight=1.0, confidence=confidence,
        extractor="test", extractor_version="0.1",
    )


def _rejection(src: str, type_: str, tgt: str, margin: float | None,
               tgt_kind: str = "entity") -> EdgeRejection:
    evidence: dict = {}
    if margin is not None:
        evidence["margin_confidence"] = margin
    return EdgeRejection(
        source=GraphRef(kind="entity", uid=src), type=type_,
        target=GraphRef(kind=tgt_kind, uid=tgt),
        extractor="test",
        rejected_reason="below_threshold" if margin is not None else "policy_excluded",
        evidence=evidence,
    )


def _bundle(nodes: list[Node], edges: list[Edge],
            surfaces: list[SurfaceRecord] | None = None) -> SceneGraphBundle:
    surfaces = surfaces or []
    return SceneGraphBundle(
        schema_version=1, bundle_hash="graph_test", scene_id="test",
        frame=FRAME, entity_bundle_hash="ent_test",
        nodes=nodes, edges=edges,
        structural_surface_refs=[s.uid for s in surfaces],
        structural_surfaces=surfaces,
    )


def _ctx(*, tau: float = 0.0, rejections=()) -> ExecutionContext:
    return ExecutionContext(
        completeness=CompletenessProfile(
            source="oracle", entity_recall_by_class={}, edge_recall_by_type={}),
        answer_tau=tau, rejections=tuple(rejections),
    )


# ---------------------------------------------------------------- defaults

def test_default_context_is_frozen_behavior() -> None:
    ctx = ExecutionContext(completeness=CompletenessProfile(
        source="oracle", entity_recall_by_class={}, edge_recall_by_type={}))
    assert ctx.answer_tau == 0.0, ctx.answer_tau
    assert ctx.rejections == (), ctx.rejections


def test_confidence_is_never_below_zero_so_default_tau_never_gates() -> None:
    """The property the frozen scorecard gate rests on: every outcome scores
    >= 0.0, and the default tau is 0.0, so `value < tau` is unsatisfiable."""
    bundle = _bundle([_node("a", "cup")], [])
    ctx = _ctx()
    cases = [
        ExecutionResult(outcome="bindings", bindings=[], evidence=[],
                        coverage_floor=1.0),
        ExecutionResult(outcome="empty", bindings=[], evidence=[],
                        coverage_floor=1.0),
        ExecutionResult(outcome="unknown", bindings=[], evidence=[],
                        coverage_floor=1.0),
        ExecutionResult(outcome="abstain", bindings=[], evidence=[],
                        coverage_floor=1.0),
        ExecutionResult(outcome="execution_error", bindings=[], evidence=[],
                        coverage_floor=0.0),
        None,
    ]
    for er in cases:
        r = score_answer(er, bundle, ctx)
        assert r.value >= 0.0, (er, r)
        assert not (r.value < ctx.answer_tau), (er, r)


# ---------------------------------------------------------------- scoping

def test_scoped_rejections_require_type_and_anchor_match() -> None:
    scope = CandidateScope(edge_type="ON_ENTITY_SURFACE", anchor_uids=("table",))
    rejections = (
        _rejection("cup", "ON_ENTITY_SURFACE", "table", 0.40),   # in scope
        _rejection("cup", "ON_SURFACE", "table", 0.90),          # wrong family
        _rejection("cup", "ON_ENTITY_SURFACE", "shelf", 0.90),   # wrong anchor
    )
    got = scoped_rejections(scope, rejections)
    assert len(got) == 1, got
    assert got[0].evidence["margin_confidence"] == 0.40, got


def test_empty_scope_means_no_candidates_at_all() -> None:
    """A query anchored on a class the scene does not contain has no
    candidate neighborhood, so no rejection can be a near-miss of it."""
    scope = CandidateScope(edge_type="ON_ENTITY_SURFACE")
    rejections = (_rejection("cup", "ON_ENTITY_SURFACE", "table", 0.49),)
    assert scoped_rejections(scope, rejections) == []
    assert scoped_rejections(None, rejections) == []


def test_rejection_ceiling_uses_max_margin_and_skips_policy_rejections() -> None:
    scope = CandidateScope(edge_type="ON_ENTITY_SURFACE", anchor_uids=("table",))
    rejections = (
        _rejection("cup", "ON_ENTITY_SURFACE", "table", 0.01),
        _rejection("pen", "ON_ENTITY_SURFACE", "table", 0.47),   # the near-miss
        _rejection("rug", "ON_ENTITY_SURFACE", "table", None),   # policy: no margin
    )
    ceiling, n_scoped, n_margin, worst = rejection_ceiling(scope, rejections)
    assert n_scoped == 3, n_scoped
    assert n_margin == 2, n_margin           # the policy rejection is not scored
    assert abs(worst - 0.47) < 1e-12, worst
    assert abs(ceiling - 0.53) < 1e-12, ceiling


def test_no_scoped_rejection_yields_ceiling_one_but_flags_absence() -> None:
    """Ceiling 1.0 with n_scoped_with_margin == 0 is ABSENCE of evidence, not
    evidence of confidence. Callers must be able to tell them apart."""
    scope = CandidateScope(edge_type="ON_ENTITY_SURFACE", anchor_uids=("table",))
    ceiling, n_scoped, n_margin, worst = rejection_ceiling(scope, ())
    assert (ceiling, n_scoped, n_margin, worst) == (1.0, 0, 0, None)


# ---------------------------------------------------------------- scoring

def test_empty_confidence_is_one_minus_worst_near_miss() -> None:
    bundle = _bundle([_node("table", "table")], [])
    scope = CandidateScope(edge_type="ON_ENTITY_SURFACE", anchor_uids=("table",))
    er = ExecutionResult(outcome="empty", bindings=[], evidence=[],
                         coverage_floor=1.0, scope=scope)

    confident = score_answer(er, bundle, _ctx(rejections=(
        _rejection("cup", "ON_ENTITY_SURFACE", "table", 0.001),)))
    assert abs(confident.value - 0.999) < 1e-9, confident

    uncertain = score_answer(er, bundle, _ctx(rejections=(
        _rejection("cup", "ON_ENTITY_SURFACE", "table", 0.001),
        _rejection("pen", "ON_ENTITY_SURFACE", "table", 0.49),)))
    assert abs(uncertain.value - 0.51) < 1e-9, uncertain
    assert uncertain.value < confident.value


def test_bindings_confidence_is_capped_by_the_rejection_ceiling() -> None:
    """Recall risk applies to non-empty answers too: an answer that cited two
    solid edges while a third candidate nearly passed is not trustworthy just
    because what it DID cite was clean."""
    bundle = _bundle([_node("cup", "cup"), _node("table", "table")],
                     [_edge("e1", "cup", "ON_ENTITY_SURFACE", "table", 0.95)])
    scope = CandidateScope(edge_type="ON_ENTITY_SURFACE", anchor_uids=("table",))
    er = ExecutionResult(
        outcome="bindings",
        bindings=[{"x": GraphRef(kind="entity", uid="cup")}],
        evidence=[_edge("e1", "cup", "ON_ENTITY_SURFACE", "table", 0.95)],
        coverage_floor=1.0, scope=scope)

    clean = score_answer(er, bundle, _ctx())
    assert abs(clean.value - 0.95) < 1e-9, clean

    capped = score_answer(er, bundle, _ctx(rejections=(
        _rejection("pen", "ON_ENTITY_SURFACE", "table", 0.6),)))
    assert abs(capped.value - 0.4) < 1e-9, capped
    assert capped.rejection_ceiling < capped.evidence_confidence


def test_label_confidence_participates() -> None:
    bundle = _bundle([_node("cup", "cup", label_confidence=0.3),
                      _node("table", "table")],
                     [_edge("e1", "cup", "ON_ENTITY_SURFACE", "table", 0.95)])
    er = ExecutionResult(
        outcome="bindings",
        bindings=[{"x": GraphRef(kind="entity", uid="cup")}],
        evidence=[_edge("e1", "cup", "ON_ENTITY_SURFACE", "table", 0.95)],
        coverage_floor=1.0, scope=CandidateScope(edge_type="ON_ENTITY_SURFACE"))
    r = score_answer(er, bundle, _ctx())
    assert abs(r.value - 0.3) < 1e-9, r
    assert r.min_label_confidence == 0.3, r


def test_refusals_score_zero() -> None:
    bundle = _bundle([_node("a", "cup")], [])
    for outcome in ("unknown", "abstain", "execution_error"):
        er = ExecutionResult(outcome=outcome, bindings=[], evidence=[],
                             coverage_floor=1.0)
        r = score_answer(er, bundle, _ctx())
        assert r.value == 0.0, (outcome, r)
        assert r.basis.startswith("declined:"), r
    assert score_answer(None, bundle, _ctx()).value == 0.0


def test_aggregations_differ_and_are_validated() -> None:
    vals = [0.5, 0.8, 1.0]
    assert aggregate(vals, "min") == 0.5
    assert abs(aggregate(vals, "mean") - 0.7666666666666667) < 1e-12
    assert abs(aggregate(vals, "product") - 0.4) < 1e-12
    assert aggregate([], "min") is None
    assert len(set(aggregate(vals, a) for a in AGGREGATIONS)) == 3
    try:
        aggregate(vals, "median")  # type: ignore[arg-type]
    except ValueError:
        pass
    else:
        raise AssertionError("unknown aggregation must raise")


def test_min_aggregation_falls_with_citation_count() -> None:
    """Documents the confound the eval found: min() over cited edges drops as
    an answer cites more, so on a recall-limited set it reads as a cardinality
    detector rather than a calibration signal. mean() does not have this
    property, which is why both are selectable."""
    many = [0.99] * 20 + [0.51]
    few = [0.9]
    assert aggregate(many, "min") < aggregate(few, "min")
    assert aggregate(many, "mean") > aggregate(few, "mean")


# ---------------------------------------------------------------- tau gate

def _support_bundle() -> SceneGraphBundle:
    """table with a cup on it, plus a floor surface so the graph is legal."""
    floor = SurfaceRecord(uid="s_floor", surface_type="floor",
                          plane=Plane(a=0.0, b=0.0, c=1.0, d=0.0),
                          polygon=None, source="habitat_label", confidence=1.0)
    return _bundle(
        [_node("cup", "cup"), _node("table", "table")],
        [_edge("e1", "cup", "ON_ENTITY_SURFACE", "table", 0.60)],
        [floor],
    )


def _router() -> Router:
    return Router(compiler=RulesCompiler(), executor=RulesExecutor(),
                  verbalizer=StandardVerbalizer())


def test_router_stamps_confidence_and_leaves_answer_alone_at_default_tau() -> None:
    bundle = _support_bundle()
    ans = _router().answer("what is on the table?", bundle, _ctx())
    assert ans.outcome == "bindings", ans
    assert ans.cited_uids == ["cup"], ans
    assert ans.confidence is not None and abs(ans.confidence - 0.60) < 1e-9, ans
    assert ans.confidence_parts["gated_by_tau"] is False, ans.confidence_parts


def test_router_downgrades_low_confidence_claim_to_unknown() -> None:
    bundle = _support_bundle()
    ans = _router().answer("what is on the table?", bundle, _ctx(tau=0.7))
    assert ans.outcome == "unknown", ans
    assert ans.cited_uids == [] and ans.cited_edges == [], ans
    assert "enough evidence" in ans.text, ans.text
    assert ans.confidence_parts["gated_by_tau"] is True, ans.confidence_parts
    # The score itself is reported unchanged; only the outcome moved.
    assert abs(ans.confidence - 0.60) < 1e-9, ans


def test_tau_is_a_parameter_not_a_constant() -> None:
    """Same graph, same question, three taus, three different outcomes."""
    bundle = _support_bundle()
    outcomes = [
        _router().answer("what is on the table?", bundle, _ctx(tau=t)).outcome
        for t in (0.0, 0.59, 0.61)
    ]
    assert outcomes == ["bindings", "bindings", "unknown"], outcomes


def test_tau_gate_also_downgrades_an_uncertain_empty() -> None:
    """The empty path is the one the design is actually aimed at: an empty
    whose neighborhood held a near-miss becomes `unknown`."""
    floor = SurfaceRecord(uid="s_floor", surface_type="floor",
                          plane=Plane(a=0.0, b=0.0, c=1.0, d=0.0),
                          polygon=None, source="habitat_label", confidence=1.0)
    bundle = _bundle([_node("table", "table"), _node("cup", "cup")], [], [floor])
    q = "what is on the table?"

    confident = _router().answer(q, bundle, _ctx(tau=0.6, rejections=(
        _rejection("cup", "ON_ENTITY_SURFACE", "table", 0.001),)))
    assert confident.outcome == "empty", confident

    uncertain = _router().answer(q, bundle, _ctx(tau=0.6, rejections=(
        _rejection("cup", "ON_ENTITY_SURFACE", "table", 0.49),)))
    assert uncertain.outcome == "unknown", uncertain
    assert abs(uncertain.confidence - 0.51) < 1e-9, uncertain


def test_gate_does_not_overwrite_a_refusal() -> None:
    """An abstain must stay an abstain: relabelling an unparsed question as
    'not enough evidence' would hide a compiler failure behind a calibration
    story."""
    bundle = _support_bundle()
    ans = _router().answer("why is the cup sad?", bundle, _ctx(tau=1.0))
    assert ans.outcome in ("abstain", "parser_failure"), ans
    assert ans.confidence == 0.0, ans
    assert ans.confidence_parts["gated_by_tau"] is False, ans.confidence_parts


def test_executor_records_scope_for_entity_support_query() -> None:
    bundle = _support_bundle()
    cr: CompileResult = RulesCompiler().compile("what is on the table?", bundle)
    assert cr.outcome == "compiled", cr
    er = RulesExecutor().execute(cr.ast, bundle, _ctx())
    assert er.scope is not None, er
    assert er.scope.edge_type == "ON_ENTITY_SURFACE", er.scope
    assert er.scope.anchor_uids == ("table",), er.scope


TESTS = [
    test_default_context_is_frozen_behavior,
    test_confidence_is_never_below_zero_so_default_tau_never_gates,
    test_scoped_rejections_require_type_and_anchor_match,
    test_empty_scope_means_no_candidates_at_all,
    test_rejection_ceiling_uses_max_margin_and_skips_policy_rejections,
    test_no_scoped_rejection_yields_ceiling_one_but_flags_absence,
    test_empty_confidence_is_one_minus_worst_near_miss,
    test_bindings_confidence_is_capped_by_the_rejection_ceiling,
    test_label_confidence_participates,
    test_refusals_score_zero,
    test_aggregations_differ_and_are_validated,
    test_min_aggregation_falls_with_citation_count,
    test_router_stamps_confidence_and_leaves_answer_alone_at_default_tau,
    test_router_downgrades_low_confidence_claim_to_unknown,
    test_tau_is_a_parameter_not_a_constant,
    test_tau_gate_also_downgrades_an_uncertain_empty,
    test_gate_does_not_overwrite_a_refusal,
    test_executor_records_scope_for_entity_support_query,
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
