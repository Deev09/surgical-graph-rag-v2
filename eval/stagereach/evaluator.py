"""StageReach3D evaluator — gating, invariants, attribution.

Pure functions over schema.Trace / schema.StagePath. No I/O.

Survival & attribution semantics (freeze doc section 5, verbatim): report
reached, pass, fail, and unknown separately at every stage. Causal survival
is computed over declared gating dependencies only; non-gating audit stages
remain visible but do not reduce downstream reachability. Attribution
reports the first `fail` on a gating stage; `unknown` stages are reported
as unmeasured, never attributed.

The six frozen invariants each raise InvariantViolation (freeze doc
section 6); invariants 1 and 4-6 live in metrics.py because they guard
counting, the rest are enforced here on every constructed trace.
"""
from __future__ import annotations

from .schema import (
    EXPECTED_OUTCOMES,
    MEASURED_STATUSES,
    PATHS,
    RESULTS,
    StagePath,
    StageRecord,
    Trace,
)

# Statuses at a gating dependency that block the dependent stage. `unknown`
# and `pass` never block (unknown never zeroes downstream survival);
# not_applicable never appears on a gating dependency because bypassed
# stages are off-path.
_BLOCKING = ("fail", "abstain", "not_reached")

# Observation statuses an adapter may supply (gating outcomes are computed
# here, so adapters never claim not_reached or not_applicable themselves).
_OBSERVABLE = ("pass", "fail", "unknown", "abstain")


class InvariantViolation(Exception):
    """A frozen schema invariant was violated. `invariant` is 1..6."""

    def __init__(self, invariant: int, message: str):
        self.invariant = invariant
        super().__init__(f"invariant {invariant}: {message}")


def _path(path_id: str) -> StagePath:
    if path_id not in PATHS:
        raise ValueError(f"unknown path {path_id!r}")
    return PATHS[path_id]


def evaluate_trace(
    *,
    question_id: str,
    scene_id: str,
    arm: str,
    path_id: str,
    scope: str,
    expected_outcome: str,
    observations: dict[str, tuple[str, str]],
    result: str,
    raw_category: str = "",
) -> Trace:
    """Resolve per-stage observations against the path's gating DAG.

    `observations` maps every on-path stage to (status, source) where status
    is one of pass/fail/unknown/abstain. Gating is applied here: a stage
    whose gating dependency resolved to fail/abstain (or was itself
    unreached) becomes not_reached, regardless of its observation. Declared
    bypassed stages are emitted as not_applicable. The finished trace is
    validated before it is returned.
    """
    path = _path(path_id)
    if scope not in path.allowed_scopes:
        raise ValueError(f"{path_id}: scope {scope!r} not allowed "
                         f"(allowed: {path.allowed_scopes})")
    missing = [s for s in path.stage_names() if s not in observations]
    if missing:
        raise ValueError(f"{question_id}/{arm}: no observation for {missing}; "
                         "an unmeasured stage must be observed as 'unknown', "
                         "never omitted")
    extra = [s for s in observations if s not in path.stage_names()]
    if extra:
        raise ValueError(f"{question_id}/{arm}: observations for off-path "
                         f"stages {extra}")

    resolved: dict[str, str] = {}
    records: list[StageRecord] = []
    for decl in path.stages:
        status, source = observations[decl.stage]
        if status not in _OBSERVABLE:
            raise ValueError(f"{question_id}/{decl.stage}: adapters may only "
                             f"observe {_OBSERVABLE}, got {status!r}")
        blockers = [d for d in decl.gating_deps if resolved[d] in _BLOCKING]
        if blockers:
            status = "not_reached"
            source = (f"not reached: gating dependency "
                      f"{blockers[0]} is {resolved[blockers[0]]}")
        resolved[decl.stage] = status
        records.append(StageRecord(stage=decl.stage, status=status,
                                   source=source))
    for stage in path.bypassed:
        records.append(StageRecord(
            stage=stage, status="not_applicable",
            source=f"declared bypass on path {path_id}"))

    trace = Trace(
        question_id=question_id, scene_id=scene_id, arm=arm,
        path_id=path_id, scope=scope, expected_outcome=expected_outcome,
        stages=tuple(records), result=result, raw_category=raw_category,
    )
    validate_trace(trace)
    return trace


def validate_trace(trace: Trace) -> None:
    """Enforce invariants 2 and 3 on a constructed trace.

    Invariant 2: a deployable-scope trace may not pass-consume any
    oracle_fed answer_path stage (CLAUDE.md deployable-lane rule 3 as code).
    Invariant 3: no silent bypass — any reached status downstream of a fail
    on a depended-on stage is an error, and not_applicable is legal only on
    stages the path declares bypassed.
    """
    path = _path(trace.path_id)
    on_path = set(path.stage_names())
    expected_records = tuple(path.stage_names()) + tuple(path.bypassed)
    got = tuple(r.stage for r in trace.stages)
    if got != expected_records:
        raise InvariantViolation(
            3, f"{trace.question_id}/{trace.arm}: trace records {got} do not "
               f"match the path's stages+bypasses {expected_records}")

    status = {r.stage: r.status for r in trace.stages}
    for r in trace.stages:
        if r.stage not in on_path:
            if r.status != "not_applicable":
                raise InvariantViolation(
                    3, f"{trace.question_id}/{r.stage}: bypassed stage must be "
                       f"not_applicable, got {r.status}")
            continue
        if r.status == "not_applicable":
            raise InvariantViolation(
                3, f"{trace.question_id}/{r.stage}: not_applicable on a stage "
                   f"path {trace.path_id} does not declare bypassed")
        decl = path.decl(r.stage)
        blocked = [d for d in decl.gating_deps if status[d] in _BLOCKING]
        if blocked and r.status != "not_reached":
            raise InvariantViolation(
                3, f"{trace.question_id}/{r.stage}: status {r.status} is a "
                   f"silent bypass — gating dependency {blocked[0]} is "
                   f"{status[blocked[0]]}, so this stage must be not_reached")
        if not blocked and r.status == "not_reached":
            raise InvariantViolation(
                3, f"{trace.question_id}/{r.stage}: not_reached with no failed "
                   "gating dependency")
        if (trace.scope == "deployable" and decl.oracle_fed
                and decl.answer_path and r.status == "pass"):
            raise InvariantViolation(
                2, f"{trace.question_id}/{r.stage}: a deployable-scope trace "
                   "may not pass-consume an oracle-fed answer-path stage")


def attribute(trace: Trace) -> dict | None:
    """First fail on a gating stage, in path order; None if the trace is
    clean. The terminal answer_generation stage is attributable (fail or
    abstain) even though nothing gates on it. Non-gating audit stages are
    never attributed, and unknown is never attributed (it is unmeasured).
    """
    path = _path(trace.path_id)
    attributable = set(path.gating_stages()) | {"answer_generation"}
    for decl in path.stages:
        r = trace.record(decl.stage)
        if decl.stage not in attributable:
            continue
        if r.status == "fail":
            return {"stage": decl.stage, "status": "fail", "source": r.source}
        if decl.stage == "answer_generation" and r.status == "abstain":
            return {"stage": decl.stage, "status": "abstain", "source": r.source}
    return None


def unmeasured_stages(trace: Trace) -> tuple[str, ...]:
    """Stages reported as unmeasured (unknown), per freeze doc section 5."""
    return tuple(r.stage for r in trace.stages if r.status == "unknown")
