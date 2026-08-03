"""Router — phase0_design.md §5.5.

Orchestrates: compiler → executor → verbalizer.

Phase 1: only the rules compiler is wired. parser_failure routes to
verbalizer abstention. The LLM compiler is a Phase 4 deliverable; the
constructor accepts an `llm_compiler` parameter so the Phase 4 wiring
will be a single line. Until then it is ignored.

Per §5.5 routing policy:
  compiled       → ASTExecutor → Verbalizer
  parser_failure → (Phase 4: try llm_compiler; Phase 1: abstain)
  out_of_schema  → Verbalizer abstains

`empty` and `unknown` are returned as-is. Neither triggers escalation.

CONFIDENCE GATE (v2-calibration)
--------------------------------
The Router scores every executed answer via reasoner/confidence.py and, when
that score is below `ctx.answer_tau`, rewrites the ExecutionResult to
`unknown` BEFORE verbalizing, so the user-facing text is the honest "I don't
have enough evidence" string rather than a claim the system does not stand
behind. The score is then stamped onto the Answer either way.

The gate lives here and not in the Verbalizer because the Verbalizer
Protocol has no ExecutionContext and therefore cannot see tau; putting it
here keeps that Protocol unchanged. It does not move the empty/unknown
decision out of the executor -- the executor still decides what the graph
ASSERTS, tau only decides whether the assertion is trustworthy enough to say.

Default `answer_tau` is 0.0 and every confidence is >= 0.0, so by default
this gate never fires and the routing behavior is bit-identical to the
pre-calibration Router.
"""
from __future__ import annotations

from dataclasses import replace

from graph.schema import SceneGraphBundle
from reasoner.base import (
    ASTExecutor, Answer, CompileResult, ExecutionContext, ExecutionResult,
    QueryCompiler, Verbalizer,
)
from reasoner.confidence import (
    Aggregation, ConfidenceReport, DEFAULT_AGGREGATION, score_answer,
)


# Outcomes a low confidence may downgrade. `unknown` is already the target,
# and abstain / execution_error are refusals the gate must not overwrite
# (turning an execution error into "not enough evidence" would hide a bug).
_GATEABLE = frozenset({"bindings", "empty"})


class Router:
    name: str = "router_v1"
    version: str = "0.1"

    def __init__(
        self,
        *,
        compiler: QueryCompiler,
        executor: ASTExecutor,
        verbalizer: Verbalizer,
        llm_compiler: QueryCompiler | None = None,
        confidence_aggregation: Aggregation = DEFAULT_AGGREGATION,
    ):
        self.compiler = compiler
        self.executor = executor
        self.verbalizer = verbalizer
        self.llm_compiler = llm_compiler
        self.confidence_aggregation = confidence_aggregation

    def _finish(
        self,
        question: str,
        cr: CompileResult,
        er: ExecutionResult | None,
        graph: SceneGraphBundle,
        ctx: ExecutionContext,
    ) -> Answer:
        """Score, apply tau, verbalize, stamp."""
        report: ConfidenceReport = score_answer(
            er, graph, ctx, aggregation=self.confidence_aggregation
        )
        gated = False
        if (
            er is not None
            and er.outcome in _GATEABLE
            and report.value < ctx.answer_tau
        ):
            gated = True
            er = replace(
                er,
                outcome="unknown",
                bindings=[],
                evidence=[],
                notes=(
                    f"answer confidence {report.value:.4f} < answer_tau "
                    f"{ctx.answer_tau:.4f} ({report.basis}); "
                    f"downgraded from {er.outcome!r}"
                ),
            )
        answer = self.verbalizer.verbalize(question, cr, er, graph)
        parts = report.to_dict()
        parts["gated_by_tau"] = gated
        parts["answer_tau"] = ctx.answer_tau
        return replace(answer, confidence=report.value, confidence_parts=parts)

    def answer(
        self,
        question: str,
        graph: SceneGraphBundle,
        ctx: ExecutionContext,
    ) -> Answer:
        cr = self.compiler.compile(question, graph)

        if cr.outcome == "compiled" and cr.ast is not None:
            er = self.executor.execute(cr.ast, graph, ctx)
            return self._finish(question, cr, er, graph, ctx)

        if cr.outcome == "parser_failure" and self.llm_compiler is not None:
            cr2 = self.llm_compiler.compile(question, graph)
            if cr2.outcome == "compiled" and cr2.ast is not None:
                er = self.executor.execute(cr2.ast, graph, ctx)
                return self._finish(question, cr2, er, graph, ctx)
            # LLM also failed → abstain via verbalizer using the LLM's CompileResult
            return self._finish(question, cr2, None, graph, ctx)

        # parser_failure (no LLM in Phase 1) or out_of_schema → abstain
        return self._finish(question, cr, None, graph, ctx)
