"""ARKit relation-challenge adapter — per-arm StageReach traces.

Reads ONLY the packed report
`eval/results/project_census_v1/arkit_relation_challenge_report.json` and
derives one trace per question per arm, independently of
tools/paper_statistics.py (that tool is read for field semantics only and
is never imported here).

Field semantics, from the report itself:
- key_eligibility: the ceiling arm's outcome != 'excluded_no_human_answer'
  (2 of 12 items are owner-marked ambiguous and enter no tally).
- object_delivery: whether a human could map every referenced object to a
  DELIVERED instance uid — `rows[qid].uids` on the ceiling arm. NOT the
  delivered arm's reason string, which reports a labelling failure and is
  silent about delivery.
- relation_applicability: the NEAR convention can express the item —
  fail iff qid is in attribution.buckets.ceiling_unanswerable_no_exhaustive_set.
- relation_correctness: UNKNOWN on every ARKit trace — no independent
  semantic relation annotation exists (stored-vs-geometry agreement is a
  consistency check, not semantic ground truth).
- serialization_consistency: geometry_vs_stored_graph.rows[qid].agree.
- referent_grounding: measured only on the grounded arm — fail iff its
  reason is exactly the bridge-abstained string; the delivered arm is
  silent about grounding, so it is unknown there.
- identity_injection (oracle arms): the human uid mapping bound the
  referenced objects (same uids field, consumed as an oracle input).
- answer_generation: the arm's own outcome.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..evaluator import evaluate_trace
from ..metrics import normalize_arkit_outcome
from ..schema import Trace

REPORT_RELPATH = ("eval/results/project_census_v1/"
                  "arkit_relation_challenge_report.json")

# The grounded arm says exactly this when the bridge failed to bind an
# anchor; any other reason means the bridge DID bind and the failure is
# downstream.
BRIDGE_ABSTAINED = "the grounding bridge abstained on this anchor"

NO_RELATION_ANNOTATION = "no independent semantic relation annotation"

# (arm name in the report, path_id, scope)
ARMS: tuple[tuple[str, str, str], ...] = (
    ("delivered_graph", "graph_deployable_delivered", "deployable"),
    ("grounded_delivered_graph", "graph_deployable_grounded", "deployable"),
    ("stored_graph_human_identity", "graph_identity_oracle",
     "identity_oracle"),
    ("geometry_relation_ceiling", "geometry_ceiling", "proposal_ceiling"),
    ("blinded_rgb_vlm", "direct_rgb", "deployable"),
)


def load_report(repo_root: Path) -> dict:
    return json.loads((Path(repo_root) / REPORT_RELPATH).read_text())


def _common_observations(qid: str, ceiling_row: dict, agree: dict,
                         no_exhaustive: frozenset[str]) -> dict:
    """Stages whose measurement is shared across arms (all derive from the
    report's cross-arm blocks, not from any arm's own answer)."""
    eligible = ceiling_row["outcome"] != "excluded_no_human_answer"
    delivered = bool(ceiling_row.get("uids"))
    expressible = qid not in no_exhaustive
    obs = {
        "key_eligibility": (
            "pass" if eligible else "fail",
            "arms.geometry_relation_ceiling.rows[qid].outcome vs "
            "'excluded_no_human_answer'"),
        "object_delivery": (
            "pass" if delivered else "fail",
            "arms.geometry_relation_ceiling.rows[qid].uids — human mapping "
            "of every referenced object to a delivered instance"),
        "relation_applicability": (
            "pass" if expressible else "fail",
            "attribution.buckets.ceiling_unanswerable_no_exhaustive_set"),
        "relation_correctness": ("unknown", NO_RELATION_ANNOTATION),
        "serialization_consistency": (
            "pass" if agree.get(qid) else "fail",
            "geometry_vs_stored_graph.rows[qid].agree"),
        "identity_injection": (
            "pass" if delivered else "fail",
            "human uid mapping consumed as an oracle identity input"),
    }
    return obs


def _answer_observation(row: dict, arm: str) -> tuple[str, str]:
    outcome = row["outcome"]
    status = {"correct": "pass", "wrong": "fail",
              "unanswered": "abstain",
              "excluded_no_human_answer": "abstain"}[outcome]
    return (status, f"arms.{arm}.rows[qid].outcome")


def derive_traces(report: dict) -> dict[str, list[Trace]]:
    """One trace per question per arm, sorted by question id."""
    arms = report["arms"]
    rows_by_arm = {name: {r["id"]: r for r in arm["rows"]}
                   for name, arm in arms.items()}
    agree = {r["id"]: r["agree"]
             for r in report["geometry_vs_stored_graph"]["rows"]}
    no_exhaustive = frozenset(
        report["attribution"]["buckets"]
        ["ceiling_unanswerable_no_exhaustive_set"])

    ceiling_rows = rows_by_arm["geometry_relation_ceiling"]
    out: dict[str, list[Trace]] = {}
    for arm, path_id, scope in ARMS:
        traces: list[Trace] = []
        for qid in sorted(ceiling_rows):
            ceiling_row = ceiling_rows[qid]
            row = rows_by_arm[arm][qid]
            common = _common_observations(qid, ceiling_row, agree,
                                          no_exhaustive)
            obs = {"answer_generation": _answer_observation(row, arm)}
            if path_id in ("graph_deployable_delivered",
                           "graph_deployable_grounded"):
                for s in ("key_eligibility", "object_delivery",
                          "relation_applicability", "relation_correctness",
                          "serialization_consistency"):
                    obs[s] = common[s]
                if path_id == "graph_deployable_grounded":
                    grounded = rows_by_arm["grounded_delivered_graph"][qid]
                    bound = grounded.get("reason") != BRIDGE_ABSTAINED
                    obs["referent_grounding"] = (
                        "pass" if bound else "fail",
                        "arms.grounded_delivered_graph.rows[qid].reason vs "
                        f"{BRIDGE_ABSTAINED!r}")
                else:
                    obs["referent_grounding"] = (
                        "unknown",
                        "the delivered arm reports a labelling failure and "
                        "is silent about referent grounding")
            elif path_id == "graph_identity_oracle":
                for s in ("key_eligibility", "object_delivery",
                          "relation_applicability", "relation_correctness",
                          "serialization_consistency",
                          "identity_injection"):
                    obs[s] = common[s]
            elif path_id == "geometry_ceiling":
                for s in ("key_eligibility", "object_delivery",
                          "relation_applicability", "identity_injection"):
                    obs[s] = common[s]
            elif path_id == "direct_rgb":
                obs["key_eligibility"] = common["key_eligibility"]

            _, result = normalize_arkit_outcome(row["outcome"])
            traces.append(evaluate_trace(
                question_id=qid,
                scene_id=ceiling_row["scene_id"],
                arm=arm, path_id=path_id, scope=scope,
                expected_outcome="answer",
                observations=obs,
                result=result,
                raw_category=row["outcome"],
            ))
        out[arm] = traces
    return out


# --------------------------------------------------- legacy-ledger compat
def legacy_reachability_block(traces_by_arm: dict[str, list[Trace]]) -> dict:
    """LEGACY-LEDGER COMPATIBILITY ONLY — never the primary output.

    Reproduces tools/paper_statistics.py's mixed reachability ladder, which
    stitches stages measured on DIFFERENT arms into one column: delivery /
    applicability / serialization from the shared blocks, anchor grounding
    from the grounded arm, and the final rung from the delivered arm. The
    per-arm ladders (metrics.survival_ladder) are the primary output; this
    exists so the committed paper_statistics.json numbers stay reproducible
    field-by-field from StageReach traces.
    """
    delivered = traces_by_arm["delivered_graph"]
    grounded = traces_by_arm["grounded_delivered_graph"]
    stored = traces_by_arm["stored_graph_human_identity"]
    rgb = traces_by_arm["blinded_rgb_vlm"]

    def n_pass(traces: list[Trace], stage: str) -> int:
        return sum(1 for t in traces if t.status(stage) == "pass")

    survivors = [
        ("human_answerable", n_pass(delivered, "key_eligibility")),
        ("objects_delivered", n_pass(delivered, "object_delivery")),
        ("relation_expressible", n_pass(delivered, "relation_applicability")),
        ("edge_serialized", n_pass(delivered, "serialization_consistency")),
        ("anchor_grounded", n_pass(grounded, "referent_grounding")),
        ("graph_correct", n_pass(delivered, "answer_generation")),
    ]
    prev = [survivors[0][1]] + [s for _, s in survivors[:-1]]
    return {
        "label": ("legacy-ledger compatibility: mixes stages measured on "
                  "different arms into one ladder, exactly as "
                  "paper_statistics.py did; the per-arm ladders are the "
                  "primary output"),
        "n_scored": n_pass(delivered, "key_eligibility"),
        "held_by_representation": n_pass(stored, "answer_generation"),
        "reached_by_deployable_grounding": n_pass(grounded,
                                                  "answer_generation"),
        "reached_by_delivered_graph": n_pass(delivered, "answer_generation"),
        "reached_by_direct_rgb": n_pass(rgb, "answer_generation"),
        "survivors_by_stage": [
            {"stage": stage, "surviving": s, "lost_here": p - s}
            for (stage, s), p in zip(survivors, prev)
        ],
    }
