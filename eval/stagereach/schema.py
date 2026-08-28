"""StageReach3D frozen schema — vocabularies, per-arm paths, trace records.

This file is data plus validation. It implements sections 1-4 and 7 of
docs/stagereach_schema_freeze.md verbatim:

- an ordered stage universe (a path uses an ordered subset);
- six statuses, with `unknown` never counted as pass, never attributed, and
  never zeroing downstream survival (enforced in evaluator/metrics);
- seven scopes mirroring the results registry;
- per-arm StagePath declarations with per-stage gating dependencies.
  `serialization_consistency` does NOT depend on `relation_correctness` on
  any path, and `relation_correctness` is a NON-GATING audit stage on every
  ARKit path (it appears in no other stage's gating dependencies there).

Gating-dependency design note (why the oracle arms ladder as 10->7):
on `graph_identity_oracle` and `geometry_ceiling`, a human injects identity,
so the deployable intermediate gates (object_delivery, relation_applicability,
serialization_consistency) do not causally gate that arm's answer -- the arm
answers from the stored representation regardless. They are therefore
declared as audit stages there (measured and visible on every trace, gating
nothing), and the causal ladder of an oracle arm is scored -> answer. On the
two deployable graph paths the full chain gates, which is exactly the
"held but unreachable" decomposition the paper reports.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ------------------------------------------------------------- vocabularies
# Ordered universe of stages; every path's stage list is an ordered subset.
STAGE_UNIVERSE: tuple[str, ...] = (
    "key_eligibility",
    "object_delivery",
    "relation_applicability",
    "relation_correctness",
    "serialization_consistency",
    "identity_injection",
    "referent_grounding",
    "answer_generation",
)

STATUSES: tuple[str, ...] = (
    "pass", "fail", "unknown", "not_applicable", "abstain", "not_reached",
)

# Statuses that carry a measured verdict for a stage that was reached.
MEASURED_STATUSES: tuple[str, ...] = ("pass", "fail", "abstain")

# Mirrors the project results registry exactly.
SCOPES: tuple[str, ...] = (
    "deployable", "identity_oracle", "delivered",
    "oracle_free_component_eval", "proposal_ceiling",
    "definition_change", "bug_diagnostic",
)

EXPECTED_OUTCOMES: tuple[str, ...] = ("answer", "empty", "defer")
RESULTS: tuple[str, ...] = ("correct", "wrong", "abstain", "excluded")

TRACE_SCHEMA = "stagereach_trace"
TRACE_SCHEMA_VERSION = 1

# Stage attributes fixed by the freeze doc: key_eligibility is
# answer_path=False, oracle_fed=True on every path; identity_injection is
# answer_path=True, oracle_fed=True. relation_correctness is an audit vs
# independent ground truth (oracle_fed, not on the answer path).
_STAGE_ATTRS: dict[str, tuple[bool, bool]] = {
    # stage: (answer_path, oracle_fed)
    "key_eligibility": (False, True),
    "object_delivery": (True, False),
    "relation_applicability": (True, False),
    "relation_correctness": (False, True),
    "serialization_consistency": (True, False),
    "identity_injection": (True, True),
    "referent_grounding": (True, False),
    "answer_generation": (True, False),
}


# ------------------------------------------------------------- declarations
@dataclass(frozen=True)
class StageDecl:
    """One stage on one path, with its declared gating dependencies.

    `gating_deps` are the topological predecessors that gate this stage's
    reachability. A stage that appears in no other stage's gating_deps is a
    non-gating audit stage on that path: measured and visible, but it never
    reduces downstream reachability and is never attributed.
    """
    stage: str
    answer_path: bool
    oracle_fed: bool
    gating_deps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.stage not in STAGE_UNIVERSE:
            raise ValueError(f"unknown stage {self.stage!r}")
        for dep in self.gating_deps:
            if dep not in STAGE_UNIVERSE:
                raise ValueError(f"{self.stage}: unknown gating dep {dep!r}")


@dataclass(frozen=True)
class StagePath:
    """A per-arm DAG: an ordered subset of the stage universe plus declared
    bypasses and the scopes a trace on this path may carry."""
    path_id: str
    stages: tuple[StageDecl, ...]
    bypassed: tuple[str, ...] = ()
    allowed_scopes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        names = [d.stage for d in self.stages]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.path_id}: duplicate stage")
        order = [STAGE_UNIVERSE.index(n) for n in names]
        if order != sorted(order):
            raise ValueError(
                f"{self.path_id}: stages must be an ordered subset of the universe")
        seen: set[str] = set()
        for decl in self.stages:
            for dep in decl.gating_deps:
                if dep not in seen:
                    raise ValueError(
                        f"{self.path_id}: {decl.stage} gates on {dep}, which is "
                        "not an earlier stage on this path")
            seen.add(decl.stage)
        for b in self.bypassed:
            if b not in STAGE_UNIVERSE:
                raise ValueError(f"{self.path_id}: unknown bypassed stage {b!r}")
            if b in names:
                raise ValueError(f"{self.path_id}: {b} both on-path and bypassed")
        for s in self.allowed_scopes:
            if s not in SCOPES:
                raise ValueError(f"{self.path_id}: unknown scope {s!r}")
        # serialization_consistency must not depend on relation_correctness
        # (freeze doc section 4) -- enforced structurally on every path.
        for decl in self.stages:
            if (decl.stage == "serialization_consistency"
                    and "relation_correctness" in decl.gating_deps):
                raise ValueError(
                    f"{self.path_id}: serialization_consistency must not depend "
                    "on relation_correctness")

    def stage_names(self) -> tuple[str, ...]:
        return tuple(d.stage for d in self.stages)

    def decl(self, stage: str) -> StageDecl:
        for d in self.stages:
            if d.stage == stage:
                return d
        raise KeyError(f"{self.path_id}: no stage {stage!r}")

    def gating_stages(self) -> tuple[str, ...]:
        """Stages some later stage gates on (audit stages excluded)."""
        gated_on: set[str] = set()
        for d in self.stages:
            gated_on.update(d.gating_deps)
        return tuple(n for n in self.stage_names() if n in gated_on)

    def ladder_stages(self) -> tuple[str, ...]:
        """The causal survival ladder: the transitive gating closure of
        answer_generation, in path order, plus answer_generation itself."""
        closure: set[str] = set()
        frontier = ["answer_generation"]
        while frontier:
            deps = self.decl(frontier.pop()).gating_deps
            for dep in deps:
                if dep not in closure:
                    closure.add(dep)
                    frontier.append(dep)
        return tuple(n for n in self.stage_names()
                     if n in closure or n == "answer_generation")


def _decl(stage: str, deps: tuple[str, ...] = ()) -> StageDecl:
    answer_path, oracle_fed = _STAGE_ATTRS[stage]
    return StageDecl(stage=stage, answer_path=answer_path,
                     oracle_fed=oracle_fed, gating_deps=deps)


def _deployable_graph_path(path_id: str) -> StagePath:
    """The full deployable chain. relation_correctness is on-path but gates
    nothing (non-gating audit; unknown on ARKit -- no independent semantic
    relation annotation exists). serialization_consistency depends on the
    chain up to relation_applicability, NOT on relation_correctness."""
    k = "key_eligibility"
    return StagePath(
        path_id=path_id,
        stages=(
            _decl(k),
            _decl("object_delivery", (k,)),
            _decl("relation_applicability", (k, "object_delivery")),
            _decl("relation_correctness", (k, "object_delivery",
                                           "relation_applicability")),
            _decl("serialization_consistency", (k, "object_delivery",
                                                "relation_applicability")),
            _decl("referent_grounding", (k, "object_delivery",
                                         "relation_applicability",
                                         "serialization_consistency")),
            _decl("answer_generation", (k, "object_delivery",
                                        "relation_applicability",
                                        "serialization_consistency",
                                        "referent_grounding")),
        ),
        bypassed=(),
        allowed_scopes=("deployable", "delivered"),
    )


def _paths() -> dict[str, StagePath]:
    k = "key_eligibility"
    paths = [
        _deployable_graph_path("graph_deployable_delivered"),
        _deployable_graph_path("graph_deployable_grounded"),
        # Oracle arm: human identity injection; the deployable intermediate
        # stages are measured audit stages that gate nothing here (see module
        # docstring). Causal ladder: key_eligibility -> answer_generation.
        StagePath(
            path_id="graph_identity_oracle",
            stages=(
                _decl(k),
                _decl("object_delivery", (k,)),
                _decl("relation_applicability", (k,)),
                _decl("relation_correctness", (k,)),
                _decl("serialization_consistency", (k,)),
                _decl("identity_injection", (k,)),
                _decl("answer_generation", (k,)),
            ),
            bypassed=("referent_grounding",),
            allowed_scopes=("identity_oracle",),
        ),
        StagePath(
            path_id="geometry_ceiling",
            stages=(
                _decl(k),
                _decl("object_delivery", (k,)),
                _decl("relation_applicability", (k,)),
                _decl("identity_injection", (k,)),
                _decl("answer_generation", (k,)),
            ),
            bypassed=("relation_correctness", "serialization_consistency",
                      "referent_grounding"),
            allowed_scopes=("proposal_ceiling", "identity_oracle"),
        ),
        StagePath(
            path_id="direct_rgb",
            stages=(
                _decl(k),
                _decl("answer_generation", (k,)),
            ),
            bypassed=("object_delivery", "relation_applicability",
                      "relation_correctness", "serialization_consistency",
                      "referent_grounding"),
            allowed_scopes=("deployable",),
        ),
        # Fault-injection battery path: full ground truth exists for every
        # stage, so relation_correctness DOES gate the answer here (a wrong
        # computed relation is a real, attributable fault). Serialization
        # still does not depend on relation_correctness, which is what lets
        # the battery distinguish "relation wrong before serialization"
        # (first fail: relation_correctness) from "correct relation corrupted
        # at serialization" (first fail: serialization_consistency).
        StagePath(
            path_id="fixture_diagnostic",
            stages=(
                _decl(k),
                _decl("object_delivery", (k,)),
                _decl("relation_applicability", (k, "object_delivery")),
                _decl("relation_correctness", (k, "object_delivery",
                                               "relation_applicability")),
                _decl("serialization_consistency", (k, "object_delivery",
                                                    "relation_applicability")),
                _decl("referent_grounding", (k, "object_delivery",
                                             "relation_applicability",
                                             "serialization_consistency")),
                _decl("answer_generation", (k, "object_delivery",
                                            "relation_applicability",
                                            "relation_correctness",
                                            "serialization_consistency",
                                            "referent_grounding")),
            ),
            bypassed=("identity_injection",),
            allowed_scopes=("bug_diagnostic", "oracle_free_component_eval"),
        ),
    ]
    return {p.path_id: p for p in paths}


PATHS: dict[str, StagePath] = _paths()

# The paths frozen by docs/stagereach_schema_freeze.md section 4.
FROZEN_ARKIT_PATH_IDS: tuple[str, ...] = (
    "graph_deployable_delivered", "graph_deployable_grounded",
    "graph_identity_oracle", "geometry_ceiling", "direct_rgb",
)


# ------------------------------------------------------------------- traces
@dataclass(frozen=True)
class StageRecord:
    """One stage's resolved status on one trace, with provenance."""
    stage: str
    status: str
    source: str

    def __post_init__(self) -> None:
        if self.stage not in STAGE_UNIVERSE:
            raise ValueError(f"unknown stage {self.stage!r}")
        if self.status not in STATUSES:
            raise ValueError(f"{self.stage}: unknown status {self.status!r}")


@dataclass(frozen=True)
class Trace:
    """One question on one arm, resolved against that arm's path."""
    question_id: str
    scene_id: str
    arm: str
    path_id: str
    scope: str
    expected_outcome: str
    stages: tuple[StageRecord, ...]
    result: str
    raw_category: str = field(default="")

    def __post_init__(self) -> None:
        if self.path_id not in PATHS:
            raise ValueError(f"unknown path {self.path_id!r}")
        if self.scope not in SCOPES:
            raise ValueError(f"unknown scope {self.scope!r}")
        if self.expected_outcome not in EXPECTED_OUTCOMES:
            raise ValueError(f"unknown expected outcome {self.expected_outcome!r}")
        if self.result not in RESULTS:
            raise ValueError(f"unknown result {self.result!r}")

    @property
    def positive_expected(self) -> bool:
        return self.expected_outcome == "answer"

    def status(self, stage: str) -> str:
        for r in self.stages:
            if r.stage == stage:
                return r.status
        raise KeyError(f"{self.question_id}/{self.arm}: no record for {stage!r}")

    def record(self, stage: str) -> StageRecord:
        for r in self.stages:
            if r.stage == stage:
                return r
        raise KeyError(f"{self.question_id}/{self.arm}: no record for {stage!r}")


def trace_to_dict(trace: Trace) -> dict:
    """Serialize per freeze doc section 7 (stagereach_trace, version 1)."""
    return {
        "schema": TRACE_SCHEMA,
        "schema_version": TRACE_SCHEMA_VERSION,
        "question_id": trace.question_id,
        "scene_id": trace.scene_id,
        "arm": trace.arm,
        "path_id": trace.path_id,
        "scope": trace.scope,
        "expected_outcome": trace.expected_outcome,
        "stages": [{"stage": r.stage, "status": r.status, "source": r.source}
                   for r in trace.stages],
        "final_outcome": {"result": trace.result,
                          "positive_expected": trace.positive_expected},
        "raw_category": trace.raw_category,
    }
