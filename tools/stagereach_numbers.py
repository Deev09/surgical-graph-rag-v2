#!/usr/bin/env python3
"""Every number the paper cites, as LaTeX macros — generated, never typed.

Reads the two committed StageReach artifacts
(eval/results/stagereach/{arkit,replica}_stagereach_v1.json), the committed
eval/results/paper_statistics.json, and re-runs the deterministic fault
battery, then emits docs/3dv/sec/generated_numbers.tex containing one
\\newcommand per quantity. The paper (Agent C) \\inputs the file and never
edits it.

Macro names are self-describing and contain no digits (spelled-out words
where a numeral would appear). Values are derived, never hard-coded:
if a gate number drifts, this file drifts, and the paper's --check fails.

p-value formatting: exact decimal when it terminates within six
significant digits (0.015625, 0.25), else rounded to three significant
figures (0.00342, 0.000122).

ANONYMITY: the emitted file lives inside the submission's LaTeX tree and
is scanned by the anonymisation test — it must never contain repository
paths, script names, or commit-hash-like strings.

    tools/stagereach_numbers.py [--check]

`--check` recomputes and fails if the committed .tex would change.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from decimal import Decimal
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eval.stagereach import faults  # noqa: E402

IN_ARKIT = REPO / "eval" / "results" / "stagereach" / "arkit_stagereach_v1.json"
IN_REPLICA = REPO / "eval" / "results" / "stagereach" / "replica_stagereach_v1.json"
IN_STATS = REPO / "eval" / "results" / "paper_statistics.json"
OUT_TEX = REPO / "docs" / "3dv" / "sec" / "generated_numbers.tex"


def fmt_p(p: float) -> str:
    """Exact decimal when short (<= 6 significant digits), else 3 s.f."""
    d = Decimal(str(p)).normalize()
    if len(d.as_tuple().digits) <= 6 and d.as_tuple().exponent >= -12:
        return format(d, "f")
    return f"{p:.3g}"


def exact_binomial_sign(b: int, c: int) -> float:
    """Two-sided exact binomial on the discordant units (same test shape
    as the committed statistics tool's exact McNemar)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _rung(arm_block: dict, stage: str) -> int:
    for r in arm_block["ladder"]:
        if r["stage"] == stage:
            return r["survivors"]
    raise KeyError(f"no ladder rung for {stage}")


def scene_sign_test(paired_top_one: dict) -> tuple[int, int, float]:
    """Scene-level sign test over the paired label result: per scene, the
    sign of (fixed - regressed) at top-1; ties drop. Instance-level
    clustering is exactly why this coarser test is reported alongside."""
    net: Counter = Counter()
    for item in paired_top_one["fixed_by_rgb"]:
        net[item.split(":")[0]] += 1
    for item in paired_top_one["regressed_by_rgb"]:
        net[item.split(":")[0]] -= 1
    b = sum(1 for v in net.values() if v > 0)
    c = sum(1 for v in net.values() if v < 0)
    return b, c, exact_binomial_sign(b, c)


def build_macros() -> list[tuple[str, str, str]]:
    """(group, macro name, value) triples, in emission order."""
    arkit = json.loads(IN_ARKIT.read_text())
    replica = json.loads(IN_REPLICA.read_text())
    stats = json.loads(IN_STATS.read_text())
    battery = faults.run_battery(faults.build_fixture())

    arms = arkit["arms"]
    delivered = arms["delivered_graph"]
    grounded = arms["grounded_delivered_graph"]
    stored = arms["stored_graph_human_identity"]
    rgb = arms["blinded_rgb_vlm"]
    ceiling = arms["geometry_relation_ceiling"]
    legacy = arkit["legacy_compat"]

    top1, top3 = stats["paired_label_tests"]
    assert top1["field"] == "top1_correct" and top3["field"] == "top3_correct"
    identity = stats["paired_identity_test"]
    sb, sc, sp = scene_sign_test(top1)

    tot = replica["totals"]
    raw = tot["raw_category_counts"]
    norm = tot["normalized_matrix"]
    scenes = replica["scenes"]

    m: list[tuple[str, str, str]] = []

    def add(group: str, name: str, value) -> None:
        m.append((group, name, str(value)))

    g = "ARKit population"
    add(g, "SRArkitQuestionTotal", arkit["n_questions"])
    add(g, "SRArkitScored", arkit["n_scored"])
    add(g, "SRArkitExcluded", arkit["n_questions"] - arkit["n_scored"])

    g = "Per-arm ladders (causal survival)"
    add(g, "SRDeliveredScored", _rung(delivered, "key_eligibility"))
    add(g, "SRDeliveredObjectsDelivered", _rung(delivered, "object_delivery"))
    add(g, "SRDeliveredRelationExpressible",
        _rung(delivered, "relation_applicability"))
    add(g, "SRDeliveredEdgeSerialized",
        _rung(delivered, "serialization_consistency"))
    add(g, "SRDeliveredAnswerCorrect", _rung(delivered, "answer_generation"))
    add(g, "SRGroundedScored", _rung(grounded, "key_eligibility"))
    add(g, "SRGroundedObjectsDelivered", _rung(grounded, "object_delivery"))
    add(g, "SRGroundedRelationExpressible",
        _rung(grounded, "relation_applicability"))
    add(g, "SRGroundedEdgeSerialized",
        _rung(grounded, "serialization_consistency"))
    add(g, "SRGroundedAnchorsBound", _rung(grounded, "referent_grounding"))
    add(g, "SRGroundedAnswerCorrect", _rung(grounded, "answer_generation"))
    add(g, "SRStoredScored", _rung(stored, "key_eligibility"))
    add(g, "SRStoredAnswerCorrect", _rung(stored, "answer_generation"))
    add(g, "SRCeilingScored", _rung(ceiling, "key_eligibility"))
    add(g, "SRCeilingAnswerCorrect", _rung(ceiling, "answer_generation"))
    add(g, "SRDirectRgbScored", _rung(rgb, "key_eligibility"))
    add(g, "SRDirectRgbAnswerCorrect", _rung(rgb, "answer_generation"))

    g = "Held vs reached (legacy-ledger compatibility)"
    add(g, "SRScoredCount", legacy["n_scored"])
    add(g, "SRHeldCount", legacy["held_by_representation"])
    add(g, "SRReachedByGrounding", legacy["reached_by_deployable_grounding"])
    add(g, "SRReachedByDeliveredGraph", legacy["reached_by_delivered_graph"])
    add(g, "SRReachedByDirectRgb", legacy["reached_by_direct_rgb"])

    g = "Replica raw categories"
    add(g, "SRReplicaTotalQuestions", tot["n"])
    add(g, "SRReplicaSceneCount", len(scenes))
    add(g, "SRReplicaTrueAnswer", raw["true_answer"])
    add(g, "SRReplicaTrueEmpty", raw["true_empty"])
    add(g, "SRReplicaMiss", raw["miss"])
    add(g, "SRReplicaFalseAnswer", raw["false_answer"])

    g = "Replica normalized matrix"
    add(g, "SRReplicaAnswerCorrect", norm["answer|correct"])
    add(g, "SRReplicaAnswerWrong", norm["answer|wrong"])
    add(g, "SRReplicaAnswerAbstain", norm["answer|abstain"])
    add(g, "SRReplicaEmptyCorrect", norm["empty|correct"])
    add(g, "SRReplicaEmptyWrong", norm["empty|wrong"])

    g = "Replica per-scene populations"
    add(g, "SRReplicaOfficeN", scenes["replica_office_0"]["n"])
    add(g, "SRReplicaRoomZeroN", scenes["replica_room_0"]["n"])
    add(g, "SRReplicaRoomOneN", scenes["replica_room_1"]["n"])
    add(g, "SRReplicaRoomTwoN", scenes["replica_room_2"]["n"])

    g = "Fault-injection battery"
    add(g, "SRFaultLocalized",
        f"{battery['n_localized']}/{battery['n_total']}")
    add(g, "SRFaultTotal", battery["n_total"])
    add(g, "SRFaultClasses", battery["n_fault_classes"])
    add(g, "SRFaultRelationTypes", battery["n_relation_types"])
    add(g, "SRFaultCleanFailures", battery["clean_failures"])

    g = "Paired tests (supplement)"
    add(g, "SRLabelInstanceCount", top1["n_instances"])
    add(g, "SRLabelFixedCount", top1["rgb_only_correct"])
    add(g, "SRLabelRegressedCount", top1["splat_only_correct"])
    add(g, "SRLabelTopOnePValue", fmt_p(top1["p_value_exact_mcnemar"]))
    add(g, "SRLabelTopThreeFixedCount", top3["rgb_only_correct"])
    add(g, "SRLabelTopThreeRegressedCount", top3["splat_only_correct"])
    add(g, "SRLabelTopThreePValue", fmt_p(top3["p_value_exact_mcnemar"]))
    add(g, "SRIdentityStoredCorrect", identity["stored_correct"])
    add(g, "SRIdentityDeliveredCorrect", identity["delivered_correct"])
    add(g, "SRIdentityItemCount", identity["n_items"])
    add(g, "SRIdentityPValue", fmt_p(identity["p_value_exact_mcnemar"]))
    add(g, "SRSceneSignTestPositiveScenes", sb)
    add(g, "SRSceneSignTestNegativeScenes", sc)
    add(g, "SRSceneSignTestPValue", fmt_p(sp))

    # Composed prose forms used directly by the paper text. Derived from the
    # same values as the atoms above, never typed.
    g = "Composed prose forms"
    arrow = r"\,$\to$\,"

    def ladder(*counts: int) -> str:
        return arrow.join(str(c) for c in counts)

    add(g, "SRSignTestP", fmt_p(sp))
    add(g, "SRArmDeliveredLadder", ladder(
        _rung(delivered, "key_eligibility"),
        _rung(delivered, "object_delivery"),
        _rung(delivered, "relation_applicability"),
        _rung(delivered, "serialization_consistency"),
        _rung(delivered, "answer_generation")))
    add(g, "SRArmGroundedLadder", ladder(
        _rung(grounded, "key_eligibility"),
        _rung(grounded, "object_delivery"),
        _rung(grounded, "relation_applicability"),
        _rung(grounded, "serialization_consistency"),
        _rung(grounded, "referent_grounding"),
        _rung(grounded, "answer_generation")))
    add(g, "SRArmStoredLadder", ladder(
        _rung(stored, "key_eligibility"),
        _rung(stored, "answer_generation")))
    add(g, "SRArmDirectLadder", ladder(
        _rung(rgb, "key_eligibility"),
        _rung(rgb, "answer_generation")))
    add(g, "SRReplicaTotal", tot["n"])
    add(g, "SRReplicaSceneNs", "/".join(str(scenes[s]["n"]) for s in (
        "replica_office_0", "replica_room_0",
        "replica_room_1", "replica_room_2")))
    add(g, "SRReplicaPooled",
        f"{raw['true_answer'] + raw['true_empty']}/{tot['n']}")

    names = [n for _, n, _ in m]
    assert len(names) == len(set(names)), "duplicate macro name"
    for n in names:
        assert not any(ch.isdigit() for ch in n), f"digit in macro name {n}"
    return m


def render(macros: list[tuple[str, str, str]]) -> bytes:
    lines = [
        "% GENERATED numbers file. Do not edit by hand: it is regenerated",
        "% (and byte-checked) from the committed evaluation artifacts by",
        "% the StageReach numbers generator. Every quantity the paper",
        "% cites is a macro here; none is typed into prose.",
    ]
    group = None
    for g, name, value in macros:
        if g != group:
            lines.append("")
            lines.append(f"% {g}")
            group = g
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    return ("\n".join(lines) + "\n").encode()


def main(argv: list[str]) -> int:
    data = render(build_macros())
    if "--check" in argv:
        before = OUT_TEX.read_bytes() if OUT_TEX.is_file() else b""
        if before != data:
            print("committed generated_numbers.tex is stale; re-run "
                  "tools/stagereach_numbers.py")
            return 1
        print("generated numbers are current")
        return 0
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_bytes(data)
    print(f"wrote {OUT_TEX.relative_to(REPO)} "
          f"({len([1 for _ in build_macros()])} macros)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
