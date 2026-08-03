"""Reasoner stage Protocols and result types — phase0_design.md §5.5.

Routing (Phase 4-final, Phase 1 only ships rules + executor + verbalizer):

    QueryCompiler(rules) →
        compiled →           ASTExecutor → (bindings|empty|unknown) → Verbalizer
        parser_failure →     QueryCompiler(llm) →
                                 compiled →       ASTExecutor → Verbalizer
                                 parser_failure → Verbalizer abstains
                                 out_of_schema →  Verbalizer abstains
        out_of_schema →      Verbalizer abstains

`empty` and `unknown` are both valid executor outcomes; neither triggers
escalation. The LLM compiles NL to AST; it never sees the graph as JSON.

`empty` vs `unknown` is decided in the executor by reading the
CompletenessProfile in ExecutionContext, NOT by reading extractor
diagnostics. Stages do not score themselves.

Answer-level confidence (v2-calibration) adds a SECOND, independent route
from `bindings`/`empty` to `unknown`: the Router scores the answer (see
reasoner/confidence.py) and downgrades it when the score is below
ExecutionContext.answer_tau. This does not move the empty/unknown decision
out of the executor -- the executor still decides what the graph asserts;
tau decides whether the assertion is trustworthy enough to say out loud.
With the default answer_tau=0.0 the downgrade never fires.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from graph.schema import Edge, EdgeRejection, EdgeType, GraphRef, SceneGraphBundle
from reasoner.ast import QueryAST


@dataclass(frozen=True)
class CompileResult:
    ast: QueryAST | None
    outcome: Literal["compiled", "parser_failure", "out_of_schema"]
    compiler_name: str               # "rules_v1" | "llm_v1"
    notes: str = ""


@dataclass(frozen=True)
class CompletenessProfile:
    """Externally calibrated coverage priors for a (backend, extractor,
    scene) triple. Lives in eval/; attached by a calibration run. Never
    produced by an extractor or builder.

    source semantics:
      - "oracle"   → by construction recall = 1.0; executor short-circuits
                     no-bindings to `empty`.
      - "measured" → recall priors from a labeled calibration dataset;
                     executor applies the empty_recall_threshold rule.
      - "unknown"  → no calibration; executor short-circuits no-bindings
                     to `unknown`.
    """
    source: Literal["oracle", "measured", "unknown"]
    entity_recall_by_class: dict[str, float]
    edge_recall_by_type: dict[EdgeType, float]
    calibration_dataset: str | None = None


@dataclass(frozen=True)
class ExecutionContext:
    completeness: CompletenessProfile
    empty_recall_threshold: float = 0.95
    # --- answer-level confidence (v2-calibration) -------------------------
    # answer_tau: the Router returns `unknown` instead of a bindings/empty
    # answer whose confidence is < answer_tau. 0.0 is the DEFAULT and is a
    # no-op gate: every confidence is >= 0.0 by construction, so the frozen
    # pre-calibration behavior is reproduced exactly. tau is a parameter of
    # the run, never a constant in the code.
    answer_tau: float = 0.0
    # rejections: the EdgeRejection records the graph build threw away, used
    # to ask "did anything come close?" for an empty answer. These are BUILD
    # evidence (BuildDiagnostics.rejection_samples), re-attached per run by
    # the caller; they are deliberately NOT part of the serialized context
    # (see reasoner/serde.py). Empty by default -> no near-miss evidence ->
    # frozen behavior.
    #
    # LIMITATION, load-bearing for any result computed off this field: each
    # extractor caps its sample at 64 rejections
    # (graph/relations/*.py::max_rejection_samples), taken in iteration
    # order, NOT by margin. On replica_room_0 that is 384 retained out of
    # 4638 actual rejections (8.3%). "max margin over scoped rejections" is
    # therefore a max over an arbitrary 8% subsample and systematically
    # understates how close the nearest miss really was.
    rejections: tuple[EdgeRejection, ...] = ()


@dataclass(frozen=True)
class CandidateScope:
    """What the executor searched over, recorded so a later stage can ask
    which rejected edges were candidates for THIS query.

    Populated by RulesExecutor on every non-error result. `edge_type` is the
    STORED relation the query actually consulted, which is not always
    constraint.type: a SUPPORTS query is answered from ON_SURFACE or
    ON_ENTITY_SURFACE edges, so those are the rejection families that bear
    on it.

    A rejection is in scope iff its type matches `edge_type` and at least one
    of its endpoints is in `anchor_uids | surface_uids`. Both uid sets empty
    means the query anchored on nothing present in the scene -- there were no
    candidates at all, which is a different (and stronger) kind of empty than
    "candidates existed and all failed".
    """
    edge_type: EdgeType | None = None
    anchor_uids: tuple[str, ...] = ()
    surface_uids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionResult:
    outcome: Literal["bindings", "empty", "unknown", "abstain", "execution_error"]
    bindings: list[dict[str, GraphRef]]
    evidence: list[Edge]
    coverage_floor: float
    notes: str = ""
    scope: CandidateScope | None = None


@dataclass(frozen=True)
class Answer:
    text: str
    answered_by: Literal["rules_compiler", "llm_compiler", "verbalizer_abstain"]
    outcome: Literal["bindings", "empty", "unknown", "abstain", "parser_failure"]
    cited_uids: list[str]
    cited_edges: list[str]
    # Answer-level confidence in [0, 1], or None when no confidence was
    # computed. Stamped by the Router (see reasoner/confidence.py); the
    # Verbalizer does not produce it, because tau has to be applied before
    # the answer text is chosen.
    confidence: float | None = None
    confidence_parts: dict[str, object] = field(default_factory=dict)


class QueryCompiler(Protocol):
    name: str
    version: str

    def compile(self, question: str, scene: SceneGraphBundle) -> CompileResult: ...


class ASTExecutor(Protocol):
    name: str
    version: str

    def execute(
        self,
        ast: QueryAST,
        graph: SceneGraphBundle,
        ctx: ExecutionContext,
    ) -> ExecutionResult: ...


class Verbalizer(Protocol):
    name: str
    version: str

    def verbalize(
        self,
        question: str,
        compile_result: CompileResult,
        exec_result: ExecutionResult | None,
        scene: SceneGraphBundle,
    ) -> Answer: ...
