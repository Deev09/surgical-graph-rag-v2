"""Answer-level confidence for the rules reasoner (v2-calibration).

WHAT THIS IS
------------
`graph/relations/*.py` now emit a real per-edge confidence behind each
extractor's `emit_margins` flag: sigmoid(4 * normalized_margin), with 0.5 as
the decision boundary (emitted edges >= 0.5, rejected pairs <= 0.5). Those
are EDGE-level scores. A user gets an ANSWER, not an edge, so something has
to aggregate. This module is that something, and nothing else in the
reasoner knows how confidence is computed.

THE TWO HALVES
--------------
An answer can be wrong in two directions, and they need different evidence:

  precision risk (the answer says something)
      Aggregate the confidences the graph already carries for what was
      cited: Edge.confidence over `exec_result.evidence`, and
      Node.label_confidence over the bound entities. Low = the system is
      standing on edges that barely cleared their thresholds.

  recall risk (the answer says nothing, or says too little)
      An empty answer is only trustworthy if NOTHING CAME CLOSE. Every
      extractor records the pairs it threw away (EdgeRejection), and with
      `emit_margins` on, a rejection carries `evidence["margin_confidence"]`
      -- how close that pair came to passing. So:

          rejection_ceiling = 1 - max(margin_confidence over scoped rejections)

      A neighborhood whose best near-miss scored 0.001 yields ceiling 0.999
      ("nothing was close"); one whose best near-miss scored 0.49 yields
      0.51 ("this empty is a coin flip"). Policy rejections carry no margin
      and are skipped -- they measure nothing comparable and no score is
      fabricated for them.

The ceiling applies to BOTH outcomes, not just empties. A `miss` in this
repo's scorecard is usually an answer that returned SOME bindings but left
required ones out; that is a recall failure inside a non-empty answer, and
the near-miss evidence for it lives in exactly the same place.

    bindings : min(aggregate(cited confidences), rejection_ceiling)
    empty    : rejection_ceiling
    anything else (unknown / abstain / execution_error / no compile)
             : 0.0 -- the system already declined; it gets no credit for it.

AGGREGATION CHOICE
------------------
`min` is the default because an answer is a conjunction of citations and is
only as good as its weakest one, matching how the extractors themselves
combine clause margins (graph/relations/base.py::rest_contact_margin).
`mean` and `product` are selectable so the choice is testable rather than
asserted; tools/rules_selective_eval.py sweeps all three.

MEASURED RESULT -- READ BEFORE TRUSTING THIS SCORE
--------------------------------------------------
On the 56 human-verified Phase 8 questions none of the three aggregations
carries usable signal, and `min` is actively anti-correlated with
correctness on non-empty answers:

  * min over cited edges is a CARDINALITY DETECTOR. The correct answers on
    this set cite 1, 30, 37 and 40 edges while the wrong ones mostly cite
    1-3, so the minimum of many draws is mechanically lower for the correct
    answers. AUROC on the 21 bindings answers is 0.118 for `min` and 0.279
    for `mean` -- both BELOW the 0.5 chance line, i.e. anti-correlated --
    while the raw cited-edge COUNT scores 0.794.
  * the rejection ceiling is ~1.0 for 28 of 31 empty answers, because 24 of
    them anchored on an entity class absent from the scene (no candidates,
    hence no rejections) and the ON_ENTITY_SURFACE family's sampled
    rejections top out at margin_confidence 0.003.

So this module is wired, parameterized and measured, and the measurement
says the score is not yet a usable selective-prediction signal. See
tools/rules_selective_eval.py and runs/rules_selective/ for the full
comparison against random-permutation controls.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from graph.schema import EdgeRejection, SceneGraphBundle
from reasoner.base import CandidateScope, ExecutionContext, ExecutionResult


Aggregation = Literal["min", "mean", "product"]

AGGREGATIONS: tuple[Aggregation, ...] = ("min", "mean", "product")

DEFAULT_AGGREGATION: Aggregation = "min"

# Key under which graph/relations/*.py park a rejection's margin. The
# EdgeRejection schema has no `confidence` field, so this ride-along is the
# only place the number exists. Absent = a policy rejection, which measures
# nothing comparable and is skipped rather than scored as 0.
MARGIN_KEY = "margin_confidence"

# Outcomes that make a claim and therefore get scored. Everything else is
# already a refusal.
_SCORED_OUTCOMES = frozenset({"bindings", "empty"})


@dataclass(frozen=True)
class ConfidenceReport:
    """The score plus every input that produced it.

    Kept verbose on purpose: a single float that cannot be decomposed is not
    auditable, and the decomposition is what showed this score to be a
    cardinality artifact rather than a calibration signal.
    """
    value: float
    aggregation: Aggregation
    basis: str
    evidence_confidence: float | None = None
    n_cited_edges: int = 0
    n_cited_nodes: int = 0
    min_edge_confidence: float | None = None
    min_label_confidence: float | None = None
    rejection_ceiling: float = 1.0
    n_scoped_rejections: int = 0
    n_scoped_with_margin: int = 0
    max_rejection_margin: float | None = None
    scope_had_anchors: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "value": self.value,
            "aggregation": self.aggregation,
            "basis": self.basis,
            "evidence_confidence": self.evidence_confidence,
            "n_cited_edges": self.n_cited_edges,
            "n_cited_nodes": self.n_cited_nodes,
            "min_edge_confidence": self.min_edge_confidence,
            "min_label_confidence": self.min_label_confidence,
            "rejection_ceiling": self.rejection_ceiling,
            "n_scoped_rejections": self.n_scoped_rejections,
            "n_scoped_with_margin": self.n_scoped_with_margin,
            "max_rejection_margin": self.max_rejection_margin,
            "scope_had_anchors": self.scope_had_anchors,
        }


def aggregate(values: list[float], aggregation: Aggregation) -> float | None:
    """Combine per-citation confidences into one number. None on an empty
    list -- "no citations" is not confidence 1.0, and the caller decides
    what it means for its outcome."""
    if not values:
        return None
    if aggregation == "min":
        return min(values)
    if aggregation == "mean":
        return sum(values) / len(values)
    if aggregation == "product":
        out = 1.0
        for v in values:
            out *= v
        return out
    raise ValueError(f"unknown aggregation {aggregation!r}; expected one of {AGGREGATIONS}")


def scoped_rejections(
    scope: CandidateScope | None,
    rejections: tuple[EdgeRejection, ...] | list[EdgeRejection],
) -> list[EdgeRejection]:
    """Rejections that were candidates for the query this scope describes.

    Match rule: same relation family, and at least one endpoint in the
    query's anchor set. A scope with no anchors at all returns nothing --
    the query named something the scene does not contain, so there was no
    candidate neighborhood to have near-misses in.
    """
    if scope is None or scope.edge_type is None:
        return []
    keys = set(scope.anchor_uids) | set(scope.surface_uids)
    if not keys:
        return []
    return [
        r for r in rejections
        if r.type == scope.edge_type and {r.source.uid, r.target.uid} & keys
    ]


def rejection_ceiling(
    scope: CandidateScope | None,
    rejections: tuple[EdgeRejection, ...] | list[EdgeRejection],
) -> tuple[float, int, int, float | None]:
    """`1 - max(margin_confidence)` over the scoped rejections.

    Returns (ceiling, n_scoped, n_with_margin, max_margin). No scoped
    rejection with a margin -> ceiling 1.0, which is NOT evidence of
    confidence; it is absence of evidence, and `n_scoped_with_margin == 0`
    is what distinguishes the two. Callers that report this must carry that
    count, because on the Phase 8 set it is zero for most empty answers.
    """
    scoped = scoped_rejections(scope, rejections)
    margins = [
        float(r.evidence[MARGIN_KEY])
        for r in scoped
        if MARGIN_KEY in r.evidence and r.evidence[MARGIN_KEY] is not None
    ]
    if not margins:
        return 1.0, len(scoped), 0, None
    worst = max(margins)
    return max(0.0, 1.0 - worst), len(scoped), len(margins), worst


def score_answer(
    exec_result: ExecutionResult | None,
    graph: SceneGraphBundle,
    ctx: ExecutionContext,
    *,
    aggregation: Aggregation = DEFAULT_AGGREGATION,
) -> ConfidenceReport:
    """Confidence for one executed query. Pure; no I/O, no graph mutation."""
    if aggregation not in AGGREGATIONS:
        raise ValueError(
            f"unknown aggregation {aggregation!r}; expected one of {AGGREGATIONS}"
        )
    if exec_result is None:
        return ConfidenceReport(value=0.0, aggregation=aggregation,
                                basis="no_execution_result")
    if exec_result.outcome not in _SCORED_OUTCOMES:
        return ConfidenceReport(value=0.0, aggregation=aggregation,
                                basis=f"declined:{exec_result.outcome}")

    ceiling, n_scoped, n_margin, worst = rejection_ceiling(
        exec_result.scope, ctx.rejections
    )
    scope = exec_result.scope
    had_anchors = bool(
        scope is not None and (scope.anchor_uids or scope.surface_uids)
    )

    if exec_result.outcome == "empty":
        return ConfidenceReport(
            value=ceiling, aggregation=aggregation, basis="empty:rejection_ceiling",
            rejection_ceiling=ceiling, n_scoped_rejections=n_scoped,
            n_scoped_with_margin=n_margin, max_rejection_margin=worst,
            scope_had_anchors=had_anchors,
        )

    # bindings
    edge_confs = [float(e.confidence) for e in exec_result.evidence]
    by_id = {n.id: n for n in graph.nodes}
    label_confs: list[float] = []
    seen: set[str] = set()
    for binding in exec_result.bindings:
        for ref in binding.values():
            if ref.kind != "entity" or ref.uid in seen:
                continue
            seen.add(ref.uid)
            node = by_id.get(ref.uid)
            if node is not None:
                label_confs.append(float(node.label_confidence))

    ev = aggregate(edge_confs + label_confs, aggregation)
    if ev is None:
        # bindings with zero citable evidence: the answer asserts something
        # it cannot point at.
        return ConfidenceReport(
            value=0.0, aggregation=aggregation, basis="bindings:no_evidence",
            rejection_ceiling=ceiling, n_scoped_rejections=n_scoped,
            n_scoped_with_margin=n_margin, max_rejection_margin=worst,
            scope_had_anchors=had_anchors,
        )
    return ConfidenceReport(
        value=min(ev, ceiling), aggregation=aggregation,
        basis="bindings:min(evidence,ceiling)",
        evidence_confidence=ev,
        n_cited_edges=len(edge_confs), n_cited_nodes=len(label_confs),
        min_edge_confidence=min(edge_confs) if edge_confs else None,
        min_label_confidence=min(label_confs) if label_confs else None,
        rejection_ceiling=ceiling, n_scoped_rejections=n_scoped,
        n_scoped_with_margin=n_margin, max_rejection_margin=worst,
        scope_had_anchors=had_anchors,
    )
