"""ASTExecutor — phase0_design.md §5.5.

Phase 1: handles single-Aggregation ASTs with a single EdgeConstraint
whose operands are one Variable and one EntityRef. Returns bindings
plus the empty/unknown decision per the CompletenessProfile.

Storage-vs-query indirection:
  The graph may store edges in only the canonical direction (sparse
  mode) or in both (compat mode). The executor handles either by
  consulting the inverse mapping: a question for RIGHT_OF(?x, sink) is
  satisfied by stored LEFT_OF(sink, ?x). Symmetric types (NEAR) are
  matched on both directions.

empty vs unknown decision (per §5.5):
  - completeness.source == 'oracle'   → empty when no bindings
  - completeness.source == 'unknown'  → unknown when no bindings
  - completeness.source == 'measured' → empty iff min(touched recall priors)
                                        >= ctx.empty_recall_threshold,
                                        else unknown
"""
from __future__ import annotations

import re

from graph.relations.base import (
    CANONICAL_INVERSE_PAIRS, INVERSE_TO_CANONICAL, SYMMETRIC_EDGE_TYPES,
)
from graph.schema import Edge, EdgeType, GraphRef, Node, SceneGraphBundle
from extractors.entity_surfaces import normalize_entity_class
from graph.views.support import entity_support_facts, support_facts
from reasoner.ast import (
    Aggregation, EdgeConstraint, EntityClassRef, EntityRef, Operand, QueryAST,
    SurfaceRef, Variable,
)
from reasoner.base import CandidateScope, ExecutionContext, ExecutionResult


def _norm_label(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).replace("_", " ").strip().lower())


def _node_match_labels(node: Node) -> list[str]:
    """All labels under which the node can be matched: `node.label`,
    `attributes.display_label`, plus any aliases."""
    cands: list[str] = [str(node.label)]
    display = node.attributes.get("display_label")
    if display:
        cands.append(str(display))
    for a in node.attributes.get("aliases", []) or []:
        cands.append(str(a))
    return cands


def _node_class_candidates(node: Node) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for label in _node_match_labels(node):
        cls = normalize_entity_class(label)
        if cls and cls not in seen:
            out.append(cls)
            seen.add(cls)
    return out


def _resolve_entity_ref(ref: EntityRef, graph: SceneGraphBundle) -> list[GraphRef]:
    target = _norm_label(ref.label)
    matches: list[GraphRef] = []
    for node in graph.nodes:
        for cand in _node_match_labels(node):
            if _norm_label(cand) == target:
                matches.append(GraphRef(kind="entity", uid=node.id))
                break
    return matches


def _resolve_entity_class_ref(ref: EntityClassRef, graph: SceneGraphBundle) -> list[GraphRef]:
    target = normalize_entity_class(ref.entity_class)
    matches: list[GraphRef] = []
    for node in graph.nodes:
        if target in _node_class_candidates(node):
            matches.append(GraphRef(kind="entity", uid=node.id))
    return matches


def _edges_matching(
    graph: SceneGraphBundle,
    *,
    edge_type: EdgeType,
    source_ref: GraphRef | None,
    target_ref: GraphRef | None,
) -> list[Edge]:
    """Find edges matching (source_ref, edge_type, target_ref), handling
    canonical-vs-inverse storage and symmetric types."""
    out: list[Edge] = []
    seen_ids: set[str] = set()

    def _push(e: Edge) -> None:
        if e.edge_id not in seen_ids:
            out.append(e)
            seen_ids.add(e.edge_id)

    def _scan(*, want_type: EdgeType, src: GraphRef | None, tgt: GraphRef | None) -> None:
        for e in graph.edges:
            if e.type != want_type:
                continue
            if src is not None and e.source != src:
                continue
            if tgt is not None and e.target != tgt:
                continue
            _push(e)

    # Direct storage of the requested type (compat mode has both; sparse
    # has only canonical).
    _scan(want_type=edge_type, src=source_ref, tgt=target_ref)

    # If the requested type is non-canonical, the canonical inverse may
    # be stored with endpoints swapped.
    if edge_type in INVERSE_TO_CANONICAL:
        canonical = INVERSE_TO_CANONICAL[edge_type]
        _scan(want_type=canonical, src=target_ref, tgt=source_ref)
    # If the requested type IS canonical, its non-canonical inverse may
    # also be present (compat mode).
    if edge_type in CANONICAL_INVERSE_PAIRS:
        inverse = CANONICAL_INVERSE_PAIRS[edge_type]
        _scan(want_type=inverse, src=target_ref, tgt=source_ref)

    # Symmetric types: also try swapped endpoints with the same type.
    if edge_type in SYMMETRIC_EDGE_TYPES:
        _scan(want_type=edge_type, src=target_ref, tgt=source_ref)

    return out


def _coverage_floor_for_query(ctx: ExecutionContext, edge_type: EdgeType) -> float:
    """Min of touched-class entity recall and the edge-type recall.
    Phase 1 approximation: treat all entity classes as touched because
    we cannot know which class would have answered the query."""
    cp = ctx.completeness
    if cp.source == "oracle":
        return 1.0
    if cp.source == "unknown":
        return 0.0
    # measured
    entity_recalls = list(cp.entity_recall_by_class.values())
    entity_floor = min(entity_recalls) if entity_recalls else 1.0
    edge_floor = cp.edge_recall_by_type.get(edge_type, 1.0)
    return min(entity_floor, edge_floor)


def _empty_or_unknown(ctx: ExecutionContext, edge_type: EdgeType, floor: float) -> str:
    cp = ctx.completeness
    if cp.source == "oracle":
        return "empty"
    if cp.source == "unknown":
        return "unknown"
    # measured
    return "empty" if floor >= ctx.empty_recall_threshold else "unknown"


def _operand_role(constraint: EdgeConstraint) -> tuple[Variable, EntityRef, str]:
    """Identify which operand is the Variable and which is the EntityRef,
    plus a 'role' tag: 'var_is_source' or 'var_is_target'. Raises if
    the constraint shape is unsupported in Phase 1."""
    if isinstance(constraint.source, Variable) and isinstance(constraint.target, EntityRef):
        return constraint.source, constraint.target, "var_is_source"
    if isinstance(constraint.target, Variable) and isinstance(constraint.source, EntityRef):
        return constraint.target, constraint.source, "var_is_target"
    raise ValueError(
        "Phase 1 executor requires exactly one Variable and one EntityRef "
        f"in the EdgeConstraint; got source={type(constraint.source).__name__}, "
        f"target={type(constraint.target).__name__}"
    )


_SURFACE_RELATION_TYPES: frozenset[EdgeType] = frozenset(
    {"NEAR_SURFACE", "CONTACTS_SURFACE", "ATTACHED_TO"}
)


def _scope(
    edge_type: EdgeType,
    *,
    anchor_uids: set[str] | None = None,
    surface_uids: set[str] | None = None,
) -> CandidateScope:
    """Record what this query searched over, for reasoner/confidence.py.

    `edge_type` is the STORED relation consulted, which is not always the
    constraint type: SUPPORTS is answered from ON_SURFACE / ON_ENTITY_SURFACE
    edges, so those are the rejection families whose near-misses bear on the
    answer. uid tuples are sorted so the scope is deterministic.
    """
    return CandidateScope(
        edge_type=edge_type,
        anchor_uids=tuple(sorted(anchor_uids or ())),
        surface_uids=tuple(sorted(surface_uids or ())),
    )


class RulesExecutor:
    """Implements the ASTExecutor Protocol."""
    name: str = "executor_v1"
    version: str = "0.1"

    def _execute_surface_relation(
        self,
        constraint: EdgeConstraint,
        graph: SceneGraphBundle,
        ctx: ExecutionContext,
    ) -> ExecutionResult:
        """RELATION(?x, SurfaceRef(type)) over a STORED entity->surface
        relation (NEAR_SURFACE / CONTACTS_SURFACE / ATTACHED_TO). Resolves the SurfaceRef
        against structural_surfaces by type, scans stored edges of
        constraint.type, binds the entity source, and cites the stored edge.
        No inverse/canonical lookup -- these relations are stored one-
        direction only. Coverage/empty-unknown use constraint.type."""
        if not (
            isinstance(constraint.source, Variable)
            and isinstance(constraint.target, SurfaceRef)
        ):
            return ExecutionResult(
                outcome="execution_error", bindings=[], evidence=[],
                coverage_floor=0.0,
                notes=(
                    f"surface relation {constraint.type} requires a Variable "
                    f"source and a SurfaceRef target; got source="
                    f"{type(constraint.source).__name__}, target="
                    f"{type(constraint.target).__name__}"
                ),
            )
        var = constraint.source
        surface_type = constraint.target.surface_type
        floor = _coverage_floor_for_query(ctx, constraint.type)

        surface_uids = {
            s.uid for s in graph.structural_surfaces
            if s.surface_type == surface_type
        }
        scope = _scope(constraint.type, surface_uids=surface_uids)
        if not surface_uids:
            outcome = _empty_or_unknown(ctx, constraint.type, floor)
            return ExecutionResult(
                outcome=outcome, bindings=[], evidence=[],
                coverage_floor=floor,
                notes=f"no {surface_type!r} surface in graph",
                scope=scope,
            )

        bindings: list[dict[str, GraphRef]] = []
        evidence: list[Edge] = []
        seen: set[str] = set()
        for edge in graph.edges:
            if edge.type != constraint.type:
                continue
            if edge.source.kind != "entity" or edge.target.kind != "surface":
                continue
            if edge.target.uid not in surface_uids:
                continue
            if edge.source.uid in seen:
                continue
            seen.add(edge.source.uid)
            bindings.append({var.name: edge.source})
            evidence.append(edge)

        if not bindings:
            outcome = _empty_or_unknown(ctx, constraint.type, floor)
            return ExecutionResult(
                outcome=outcome, bindings=[], evidence=[],
                coverage_floor=floor,
                notes=f"no {constraint.type} edges touch a {surface_type!r} surface",
                scope=scope,
            )
        return ExecutionResult(
            outcome="bindings", bindings=bindings, evidence=evidence,
            coverage_floor=floor,
            notes=f"matched {len(bindings)} binding(s) via stored {constraint.type}",
            scope=scope,
        )

    def _execute_supports(
        self,
        constraint: EdgeConstraint,
        graph: SceneGraphBundle,
        ctx: ExecutionContext,
    ) -> ExecutionResult:
        if isinstance(constraint.source, EntityClassRef):
            return self._execute_entity_supports(constraint, graph, ctx)
        """SUPPORTS(SurfaceRef(type), ?x) -> entities resting on a surface
        of that type, derived from support_facts (no stored SUPPORTS edges).
        Evidence cites the originating ON_SURFACE edges. Coverage/empty-
        unknown uses the ON_SURFACE recall, since the derived view depends
        on stored ON_SURFACE edges."""
        if not (
            isinstance(constraint.source, SurfaceRef)
            and isinstance(constraint.target, Variable)
        ):
            return ExecutionResult(
                outcome="execution_error", bindings=[], evidence=[],
                coverage_floor=0.0,
                notes=(
                    "SUPPORTS query requires SurfaceRef source and Variable "
                    f"target; got source={type(constraint.source).__name__}, "
                    f"target={type(constraint.target).__name__}"
                ),
            )
        var = constraint.target
        surface_type = constraint.source.surface_type
        # ON_SURFACE is the stored relation the derived view depends on.
        floor = _coverage_floor_for_query(ctx, "ON_SURFACE")

        surface_uids = {
            s.uid for s in graph.structural_surfaces
            if s.surface_type == surface_type
        }
        # SUPPORTS is derived from stored ON_SURFACE edges, so ON_SURFACE is
        # also the rejection family that bears on this query.
        scope = _scope("ON_SURFACE", surface_uids=surface_uids)
        if not surface_uids:
            outcome = _empty_or_unknown(ctx, "ON_SURFACE", floor)
            return ExecutionResult(
                outcome=outcome, bindings=[], evidence=[],
                coverage_floor=floor,
                notes=f"no {surface_type!r} surface in graph",
                scope=scope,
            )

        try:
            facts = support_facts(graph)
        except ValueError as e:
            # Strict view raised (materialized SUPPORTS or malformed
            # ON_SURFACE) — surface as an execution error, not a wrong answer.
            return ExecutionResult(
                outcome="execution_error", bindings=[], evidence=[],
                coverage_floor=floor, notes=str(e),
            )

        edge_by_id = {e.edge_id: e for e in graph.edges}
        bindings: list[dict[str, GraphRef]] = []
        evidence: list[Edge] = []
        seen: set[str] = set()
        for fact in facts:
            if fact.supporter.uid not in surface_uids:
                continue
            if fact.supported.uid in seen:
                continue
            seen.add(fact.supported.uid)
            bindings.append({var.name: fact.supported})
            edge = edge_by_id.get(fact.derived_from_edge_id)
            if edge is not None:
                evidence.append(edge)

        if not bindings:
            outcome = _empty_or_unknown(ctx, "ON_SURFACE", floor)
            return ExecutionResult(
                outcome=outcome, bindings=[], evidence=[],
                coverage_floor=floor,
                notes=f"nothing rests on a {surface_type!r} surface",
                scope=scope,
            )
        return ExecutionResult(
            outcome="bindings", bindings=bindings, evidence=evidence,
            coverage_floor=floor,
            notes=f"matched {len(bindings)} binding(s) via support view",
            scope=scope,
        )

    def _execute_entity_supports(
        self,
        constraint: EdgeConstraint,
        graph: SceneGraphBundle,
        ctx: ExecutionContext,
    ) -> ExecutionResult:
        """SUPPORTS(EntityClassRef(class), ?x) over ON_ENTITY_SURFACE edges."""
        if not (
            isinstance(constraint.source, EntityClassRef)
            and isinstance(constraint.target, Variable)
        ):
            return ExecutionResult(
                outcome="execution_error", bindings=[], evidence=[],
                coverage_floor=0.0,
                notes=(
                    "entity SUPPORTS query requires EntityClassRef source and "
                    f"Variable target; got source={type(constraint.source).__name__}, "
                    f"target={type(constraint.target).__name__}"
                ),
            )
        var = constraint.target
        entity_class = normalize_entity_class(constraint.source.entity_class)
        floor = _coverage_floor_for_query(ctx, "ON_ENTITY_SURFACE")
        owner_refs = _resolve_entity_class_ref(constraint.source, graph)
        owner_uids = {ref.uid for ref in owner_refs}
        scope = _scope("ON_ENTITY_SURFACE", anchor_uids=owner_uids)
        if not owner_uids:
            # No owner entity of this class exists. The scope is empty, which
            # is the strongest form of empty available: there was never a
            # candidate pair for a rejection to be a near-miss of.
            outcome = _empty_or_unknown(ctx, "ON_ENTITY_SURFACE", floor)
            return ExecutionResult(
                outcome=outcome, bindings=[], evidence=[],
                coverage_floor=floor,
                notes=f"no {entity_class!r} owner entity in graph",
                scope=scope,
            )

        try:
            facts = entity_support_facts(graph)
        except ValueError as e:
            return ExecutionResult(
                outcome="execution_error", bindings=[], evidence=[],
                coverage_floor=floor, notes=str(e),
            )

        edge_by_id = {e.edge_id: e for e in graph.edges}
        bindings: list[dict[str, GraphRef]] = []
        evidence: list[Edge] = []
        seen: set[str] = set()
        for fact in facts:
            if fact.supporter.uid not in owner_uids:
                continue
            if fact.supported.uid in seen:
                continue
            seen.add(fact.supported.uid)
            bindings.append({var.name: fact.supported})
            edge = edge_by_id.get(fact.derived_from_edge_id)
            if edge is not None:
                evidence.append(edge)

        if not bindings:
            outcome = _empty_or_unknown(ctx, "ON_ENTITY_SURFACE", floor)
            return ExecutionResult(
                outcome=outcome, bindings=[], evidence=[],
                coverage_floor=floor,
                notes=f"nothing rests on a {entity_class!r} entity top",
                scope=scope,
            )
        return ExecutionResult(
            outcome="bindings", bindings=bindings, evidence=evidence,
            coverage_floor=floor,
            notes=(
                f"matched {len(bindings)} binding(s) via entity support view "
                f"for {entity_class!r}"
            ),
            scope=scope,
        )

    def execute(
        self,
        ast: QueryAST,
        graph: SceneGraphBundle,
        ctx: ExecutionContext,
    ) -> ExecutionResult:
        if not isinstance(ast, Aggregation):
            return ExecutionResult(
                outcome="execution_error", bindings=[], evidence=[],
                coverage_floor=0.0,
                notes=f"expected Aggregation, got {type(ast).__name__}",
            )
        if len(ast.where) != 1 or not isinstance(ast.where[0], EdgeConstraint):
            return ExecutionResult(
                outcome="execution_error", bindings=[], evidence=[],
                coverage_floor=0.0,
                notes="Phase 1 executor handles exactly one EdgeConstraint",
            )

        constraint: EdgeConstraint = ast.where[0]

        # P4.04: SUPPORTS is a DERIVED view, not stored. Special-case it
        # BEFORE the generic stored-edge path (which would otherwise try to
        # satisfy SUPPORTS via stored ON_TOP_OF edges through the inverse
        # map). Routes to support_facts(graph) over ON_SURFACE edges.
        if constraint.type == "SUPPORTS":
            return self._execute_supports(constraint, graph, ctx)

        # P5.03/P7: SurfaceRef-anchored stored entity->surface relations
        # (NEAR_SURFACE, CONTACTS_SURFACE, ATTACHED_TO). One parameterized branch; the
        # relation is constraint.type. No inverse/canonical lookup (these are
        # stored one-direction only).
        if constraint.type in _SURFACE_RELATION_TYPES:
            return self._execute_surface_relation(constraint, graph, ctx)

        try:
            var, anchor_ref, role = _operand_role(constraint)
        except ValueError as e:
            return ExecutionResult(
                outcome="execution_error", bindings=[], evidence=[],
                coverage_floor=0.0, notes=str(e),
            )

        floor = _coverage_floor_for_query(ctx, constraint.type)
        anchor_refs = _resolve_entity_ref(anchor_ref, graph)
        scope = _scope(
            constraint.type, anchor_uids={r.uid for r in anchor_refs}
        )
        if not anchor_refs:
            outcome = _empty_or_unknown(ctx, constraint.type, floor)
            return ExecutionResult(
                outcome=outcome, bindings=[], evidence=[],
                coverage_floor=floor,
                notes=f"anchor {anchor_ref.label!r} not found in graph",
                scope=scope,
            )

        bindings: list[dict[str, GraphRef]] = []
        evidence: list[Edge] = []
        seen_binding_uids: set[str] = set()

        for anchor in anchor_refs:
            if role == "var_is_source":
                source_ref = None
                target_ref = anchor
            else:  # var_is_target
                source_ref = anchor
                target_ref = None
            matches = _edges_matching(
                graph,
                edge_type=constraint.type,
                source_ref=source_ref,
                target_ref=target_ref,
            )
            for edge in matches:
                bound_ref = edge.source if role == "var_is_source" else edge.target
                # The matched edge might be the inverse stored with swapped
                # endpoints — re-derive which endpoint corresponds to the
                # variable by excluding the anchor uid.
                if bound_ref == anchor:
                    bound_ref = edge.target if bound_ref == edge.source else edge.source
                if bound_ref.uid in seen_binding_uids:
                    continue
                seen_binding_uids.add(bound_ref.uid)
                bindings.append({var.name: bound_ref})
                evidence.append(edge)

        if not bindings:
            outcome = _empty_or_unknown(ctx, constraint.type, floor)
            return ExecutionResult(
                outcome=outcome, bindings=[], evidence=[],
                coverage_floor=floor,
                notes=f"no edges of type {constraint.type} touch {anchor_ref.label!r}",
                scope=scope,
            )

        return ExecutionResult(
            outcome="bindings", bindings=bindings, evidence=evidence,
            coverage_floor=floor,
            notes=f"matched {len(bindings)} binding(s)",
            scope=scope,
        )
