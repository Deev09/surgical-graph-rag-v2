"""StageReach3D metrics — survival ladders, outcome matrices, refusals.

Pure functions over Trace tuples. No I/O, no floats-as-headlines.

Counting rules (freeze doc sections 5-6):
- reached / pass / fail / unknown / abstain are reported separately at
  every stage; `unknown` is NEVER counted as pass (invariant 1).
- metrics are keyed by (expected_outcome, result); positives are never
  pooled with true-empties and no single "accuracy" is exposed by this API
  (invariant 4 — `pooled_accuracy` exists only to refuse).
- precision/recall/true-negative metrics are refused for non-exhaustive
  keys (invariant 5).
- definition_change-scope results cannot be pooled with frozen-track
  results (invariant 6).
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

from .evaluator import InvariantViolation
from .schema import (
    EXPECTED_OUTCOMES,
    MEASURED_STATUSES,
    PATHS,
    RESULTS,
    StagePath,
    Trace,
)


# --------------------------------------------------------------- accounting
def counts_as_pass(status: str) -> bool:
    """The one place survival counting decides what a pass is. `unknown`
    is not a pass, anywhere (invariant 1)."""
    return status == "pass"


def stage_report(traces: Sequence[Trace], path: StagePath) -> list[dict]:
    """Per-stage accounting: reached, pass, fail, unknown, abstain reported
    separately, plus not_reached / not_applicable. Verified before return."""
    rows = []
    n = len(traces)
    for stage in path.stage_names() + path.bypassed:
        c = Counter(t.status(stage) for t in traces)
        row = {
            "stage": stage,
            "n": n,
            "reached": n - c["not_reached"] - c["not_applicable"],
            "pass": c["pass"],
            "fail": c["fail"],
            "unknown": c["unknown"],
            "abstain": c["abstain"],
            "not_reached": c["not_reached"],
            "not_applicable": c["not_applicable"],
        }
        verify_stage_accounting(row)
        rows.append(row)
    return rows


def verify_stage_accounting(row: dict) -> None:
    """Invariant 1 as an executable check: pass+fail+unknown+abstain must
    equal reached, so an unknown folded into pass (or counted twice) is a
    hard error, not a quiet inflation."""
    measured_plus_unknown = (row["pass"] + row["fail"] + row["unknown"]
                             + row["abstain"])
    if measured_plus_unknown != row["reached"]:
        raise InvariantViolation(
            1, f"stage {row['stage']}: pass+fail+unknown+abstain = "
               f"{measured_plus_unknown} != reached {row['reached']} — an "
               "unknown was counted as a measured outcome (or vice versa)")
    if row["reached"] + row["not_reached"] + row["not_applicable"] != row["n"]:
        raise InvariantViolation(
            1, f"stage {row['stage']}: reached+not_reached+not_applicable "
               f"!= n={row['n']}")
    if any(v < 0 for k, v in row.items() if k != "stage"):
        raise InvariantViolation(1, f"stage {row['stage']}: negative count")


# ------------------------------------------------------------------ ladders
def survival_ladder(traces: Sequence[Trace], path: StagePath) -> list[dict]:
    """Causal survival over declared gating dependencies only.

    Rungs are the gating closure of answer_generation (plus the terminal
    stage itself), in path order. A rung's survivors are the traces that
    reached the stage AND passed it — for answer_generation, a pass is a
    correct answer. A stage with no measured status on any trace (all
    unknown / not_reached / not_applicable) is unmeasured: it is skipped,
    never zeroed, never passed (invariant 1). Non-gating audit stages are
    visible in stage_report but are not rungs."""
    rungs = []
    for stage in path.ladder_stages():
        statuses = [t.status(stage) for t in traces]
        if not any(s in MEASURED_STATUSES for s in statuses):
            continue  # unmeasured everywhere: reported as unknown, not a rung
        rungs.append({
            "stage": stage,
            "survivors": sum(1 for s in statuses if counts_as_pass(s)),
            "reached": sum(1 for s in statuses
                           if s not in ("not_reached", "not_applicable")),
        })
    return rungs


def ladder_counts(traces: Sequence[Trace], path: StagePath) -> list[int]:
    return [r["survivors"] for r in survival_ladder(traces, path)]


# ---------------------------------------------------------- outcome mapping
# ARKit relation-challenge vocabulary (freeze doc section 8). Total.
ARKIT_OUTCOME_MAP: dict[str, tuple[str, str]] = {
    "correct": ("answer", "correct"),
    "wrong": ("answer", "wrong"),
    "unanswered": ("answer", "abstain"),
    "excluded_no_human_answer": ("answer", "excluded"),
}

# router_qa's 9-category vocabulary (eval/router_qa.py). Total.
ROUTER_CATEGORIES: tuple[str, ...] = (
    "true_answer", "false_answer", "miss", "correct_defer", "true_empty",
    "true_unknown", "true_parser_failure", "true_execution_error",
    "unexpected",
)


def normalize_arkit_outcome(outcome: str) -> tuple[str, str]:
    if outcome not in ARKIT_OUTCOME_MAP:
        raise ValueError(f"unknown ARKit outcome {outcome!r}")
    return ARKIT_OUTCOME_MAP[outcome]


def _target_expected(expected_outcome: str) -> str:
    """Collapse router_qa's six expected outcomes onto the frozen target
    vocabulary {answer, empty, defer}: the harness-diagnostic expectations
    (unknown / parser_failure / execution_error) are expected non-answers
    and map to defer. raw_category always preserves the source label."""
    if expected_outcome in ("answer", "empty", "defer"):
        return expected_outcome
    if expected_outcome in ("unknown", "parser_failure", "execution_error"):
        return "defer"
    raise ValueError(f"unknown expected_outcome {expected_outcome!r}")


def normalize_router_record(record: dict) -> tuple[str, str]:
    """Map one router_qa per-question record onto (expected, result).

    Total over the 9-category vocabulary, and — per the freeze doc — the
    split is decided on the per-question RECORD, not the category alone:
    a miss that explicitly deferred/abstained is (answer, abstain); a miss
    that answered incompletely or returned empty without deferring is
    (answer, wrong). false_answer splits on the question's expected outcome.
    """
    category = record["category"]
    if category not in ROUTER_CATEGORIES:
        raise ValueError(f"unknown router_qa category {category!r}")
    expected = _target_expected(record["expected_outcome"])
    if category == "true_answer":
        return ("answer", "correct")
    if category == "true_empty":
        return ("empty", "correct")
    if category == "correct_defer":
        return ("defer", "correct")
    if category in ("true_unknown", "true_parser_failure",
                    "true_execution_error"):
        return ("defer", "correct")
    if category == "miss":
        if record.get("deferred") or record.get("actual_outcome") == "defer":
            return ("answer", "abstain")
        return ("answer", "wrong")
    if category == "false_answer":
        return (expected, "wrong")
    # unexpected: the outcome shape did not match the expectation. A
    # fabricated answer is wrong; a mismatched non-answer is an abstention.
    if record.get("actual_outcome") == "bindings":
        return (expected, "wrong")
    return (expected, "abstain")


# ----------------------------------------------------------------- matrices
def assert_poolable(scopes: Iterable[str]) -> None:
    """Invariant 6: definition_change results pool with nothing frozen."""
    scope_set = set(scopes)
    if "definition_change" in scope_set and len(scope_set) > 1:
        raise InvariantViolation(
            6, "definition_change-scope results cannot be pooled with "
               f"frozen-track results (scopes present: {sorted(scope_set)})")


def outcome_matrix(traces: Sequence[Trace]) -> dict[tuple[str, str], int]:
    """Counts keyed by (expected_outcome, result). Never pooled across the
    expected axis; scope pooling is guarded (invariant 6)."""
    assert_poolable(t.scope for t in traces)
    matrix: Counter = Counter()
    for t in traces:
        matrix[(t.expected_outcome, t.result)] += 1
    return dict(matrix)


def matrix_to_json(matrix: dict[tuple[str, str], int]) -> dict[str, int]:
    return {f"{e}|{r}": n for (e, r), n in sorted(matrix.items())}


def raw_category_counts(traces: Sequence[Trace]) -> dict[str, int]:
    return dict(sorted(Counter(t.raw_category for t in traces).items()))


def pooled_accuracy(*_args, **_kwargs):
    """Invariant 4: there is no pooled accuracy. Positives are never pooled
    with true-empties, and no single accuracy number exists in this API.
    This function exists only to refuse, loudly, at the place someone would
    reach for it."""
    raise InvariantViolation(
        4, "no pooled accuracy: metrics are keyed by (expected_outcome, "
           "result); report matrix cells, not a single accuracy")


def require_exhaustive_key(relation: str, exhaustive: bool) -> None:
    """Invariant 5: precision/recall/true-negative metrics require an
    exhaustive key. For a non-exhaustive key (e.g. Replica NEAR_SURFACE)
    this refuses instead of quietly emitting a number."""
    if not exhaustive:
        raise InvariantViolation(
            5, f"relation {relation}: the key is not exhaustive, so "
               "precision/recall/true-negative metrics are refused")


def precision_recall(tp: int, fp: int, fn: int, *, relation: str,
                     exhaustive: bool) -> dict:
    """P/R over an exhaustive key; refuses otherwise (invariant 5)."""
    require_exhaustive_key(relation, exhaustive)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {"relation": relation, "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall}
