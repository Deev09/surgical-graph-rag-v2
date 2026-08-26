#!/usr/bin/env python3
"""Paired statistics and a question-level reachability ledger.

Reads ONLY the tracked evidence pack. Runs no model, touches no threshold, and
produces no new measurement -- every number here is a re-reading of outcomes
that were already scored and committed.

Two analyses:

1.  A paired discordance table and an exact McNemar test for splat vs rgb_tight
    at top-1 and top-3. The label arms were evaluated on the SAME matched
    instances, so the unpaired marginals (1/21 vs 12/21) understate the
    evidence: what matters is how many individual instances changed verdict and
    in which direction. The test is exact (two-sided binomial on the discordant
    pairs), not the chi-square approximation, because the discordant counts are
    far too small for it.

2.  A reachability ledger: one row per question, with a column for each stage
    the answer must survive. This is the bookkeeping behind "held but
    unreachable" made explicit, so the transition where questions are lost can
    be read off rather than argued.

Every ledger column is a field the packed report states. Where the report does
not state something, this tool leaves it out rather than inferring it -- see
`objects_delivered`, which is read from whether a human could map every
referenced object to a delivered instance, and NOT from the delivered arm's
reason string (which reports a labelling failure and is silent about delivery).

    tools/paper_statistics.py [--check]

`--check` recomputes and fails if the committed outputs would change.
"""
from __future__ import annotations

import csv
import json
import sys
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACK = REPO / "eval" / "results" / "project_census_v1"
OUT_JSON = REPO / "eval" / "results" / "paper_statistics.json"
OUT_LEDGER = REPO / "docs" / "paper_reachability_ledger.csv"

SCENES = ("41069021", "41069025", "41069042")
ARMS = ("splat", "rgb_tight")

# The grounded arm says exactly this when the bridge failed to bind an anchor;
# any other reason means the bridge DID bind and the failure is downstream.
BRIDGE_ABSTAINED = "the grounding bridge abstained on this anchor"


# ---------------------------------------------------------------- paired test
def per_instance(scene: str, arm: str) -> dict:
    path = PACK / f"arkit_label_ab_{scene}_{arm}_label_eval.json"
    doc = json.loads(path.read_text())
    rows = doc["metrics"]["per_match"]
    return {r["object_uid"]: r for r in rows}


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact binomial test on the discordant pairs.

    Under the null the b+c instances that changed verdict are equally likely to
    have changed in either direction, so the count is Binomial(b+c, 1/2).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_table(field: str) -> dict:
    b = c = both = neither = 0
    fixed: list[str] = []
    regressed: list[str] = []
    for scene in SCENES:
        splat, rgb = per_instance(scene, "splat"), per_instance(scene, "rgb_tight")
        if set(splat) != set(rgb):
            raise SystemExit(f"{scene}: the two arms do not cover the same instances; "
                             "they are not paired and this test does not apply")
        for uid in sorted(splat):
            s, r = bool(splat[uid][field]), bool(rgb[uid][field])
            if r and not s:
                b += 1
                fixed.append(f"{scene}:{uid}")
            elif s and not r:
                c += 1
                regressed.append(f"{scene}:{uid}")
            elif s and r:
                both += 1
            else:
                neither += 1
    n = b + c + both + neither
    return {
        "field": field,
        "n_instances": n,
        "n_scenes": len(SCENES),
        "splat_correct": both + c,
        "rgb_tight_correct": both + b,
        "both_correct": both,
        "rgb_only_correct": b,
        "splat_only_correct": c,
        "neither_correct": neither,
        "discordant_pairs": b + c,
        "p_value_exact_mcnemar": exact_mcnemar(b, c),
        "test": "two-sided exact binomial on discordant pairs (not the chi-square "
                "approximation; the discordant counts are too small for it)",
        "fixed_by_rgb": sorted(fixed),
        "regressed_by_rgb": sorted(regressed),
        "clustering_limitation":
            f"The {n} instances are clustered within only {len(SCENES)} scenes. The "
            "test treats instances as independent, which they are not: instances in "
            "one room share a capture, a lighting condition and a reconstruction. "
            "The p-value is therefore an instance-level statement and NOT a "
            "scene-level or dataset-level one, and no claim about handheld capture "
            "in general follows from it.",
    }


# ----------------------------------------------------------------- the ledger
def ledger() -> list[dict]:
    doc = json.loads((PACK / "arkit_relation_challenge_report.json").read_text())
    arms = doc["arms"]
    by_id = {name: {r["id"]: r for r in arm["rows"]} for name, arm in arms.items()}
    agree = {r["id"]: r["agree"] for r in doc["geometry_vs_stored_graph"]["rows"]}
    no_set = set(doc["attribution"]["buckets"]["ceiling_unanswerable_no_exhaustive_set"])
    thin = set(doc["thin_evidence_subtest"]["thin_question_ids"])

    out = []
    for qid, ceiling in by_id["geometry_relation_ceiling"].items():
        stored = by_id["stored_graph_human_identity"][qid]
        delivered = by_id["delivered_graph"][qid]
        grounded = by_id["grounded_delivered_graph"][qid]
        rgb = by_id["blinded_rgb_vlm"][qid]
        views = ceiling.get("evidence_views")
        out.append({
            "question": qid,
            "scene": ceiling["scene_id"],
            "form": ceiling["form"],
            # a human supplied an answer for this item
            "human_answerable": ceiling["outcome"] != "excluded_no_human_answer",
            # a human mapped every referenced object to a DELIVERED instance uid
            "objects_delivered": bool(ceiling.get("uids")),
            # the NEAR convention can express the question at all
            "relation_expressible": qid not in no_set,
            # the serialized edge reproduces the recomputed geometry
            "edge_serialized": agree.get(qid),
            # the oracle-free bridge bound the anchors (failure, if any, is later)
            "anchor_grounded": grounded.get("reason") != BRIDGE_ABSTAINED,
            "ceiling_correct": ceiling["outcome"] == "correct",
            "stored_correct": stored["outcome"] == "correct",
            "graph_correct": delivered["outcome"] == "correct",
            "grounded_correct": grounded["outcome"] == "correct",
            "rgb_correct": rgb["outcome"] == "correct",
            "graph_abstained": delivered["outcome"] == "unanswered",
            "rgb_abstained": rgb["outcome"] == "unanswered",
            "evidence_views": views,
            # NOT the transfer test's "cross-view". There, a generator produced
            # questions whose referents are non-co-visible by construction. Here
            # "2+" only records that the answer drew on more than one view, which a
            # co-visible question can also do. The two must not be pooled.
            "evidence_spans_multiple_views": views == "2+",
            "thin_evidence": qid in thin,
        })
    return sorted(out, key=lambda r: r["question"])


def transitions(rows: list[dict]) -> dict:
    """Where questions are lost between representation and deployable answer."""
    scored = [r for r in rows if r["human_answerable"]]
    n = len(scored)

    def count(pred) -> int:
        return sum(1 for r in scored if pred(r))

    stages = [
        ("human_answerable", n, "a human supplied an answer"),
        ("objects_delivered", count(lambda r: r["objects_delivered"]),
         "every referenced object exists in the delivered partition"),
        ("relation_expressible", count(lambda r: r["objects_delivered"]
                                       and r["relation_expressible"]),
         "and the NEAR convention can express the question"),
        ("edge_serialized", count(lambda r: r["objects_delivered"]
                                  and r["relation_expressible"]
                                  and r["edge_serialized"]),
         "and the serialized edge reproduces the geometry"),
        ("anchor_grounded", count(lambda r: r["objects_delivered"]
                                  and r["relation_expressible"]
                                  and r["edge_serialized"]
                                  and r["anchor_grounded"]),
         "and the oracle-free bridge bound the anchors"),
        ("graph_correct", count(lambda r: r["graph_correct"]),
         "the delivered graph answered correctly"),
    ]
    return {
        "n_scored": n,
        "held_by_representation": count(lambda r: r["stored_correct"]),
        "reached_by_deployable_grounding": count(lambda r: r["grounded_correct"]),
        "reached_by_delivered_graph": count(lambda r: r["graph_correct"]),
        "reached_by_direct_rgb": count(lambda r: r["rgb_correct"]),
        "survivors_by_stage": [
            {"stage": k, "surviving": v, "lost_here": prev - v, "meaning": m}
            for (k, v, m), prev in zip(stages, [n] + [s[1] for s in stages[:-1]])
        ],
        "multi_view_evidence": {
            "note": "evidence_views == '2+' means the answer drew on more than one "
                    "view. It does NOT mean the referents are non-co-visible, which is "
                    "what the transfer test's cross-view items were. Do not pool these "
                    "counts with the transfer test's 0/3.",
            "n_multi_view": count(lambda r: r["evidence_spans_multiple_views"]),
            "rgb_correct_multi_view": count(lambda r: r["evidence_spans_multiple_views"]
                                            and r["rgb_correct"]),
            "rgb_correct_single_view": count(lambda r: not r["evidence_spans_multiple_views"]
                                             and r["rgb_correct"]),
        },
        "selective_risk": {
            "note": "coverage = answered / scored; selective risk = wrong / answered",
            "rgb_answered": count(lambda r: not r["rgb_abstained"]),
            "rgb_wrong": count(lambda r: not r["rgb_abstained"] and not r["rgb_correct"]),
            "graph_answered": count(lambda r: not r["graph_abstained"]),
        },
    }


def paired_identity_test(rows: list[dict]) -> dict:
    """The identity substitution itself, tested the same paired way.

    The stored-relation arm and the delivered arm answer the SAME questions over
    the SAME serialized edges and differ only in where object identity comes
    from, so the comparison is paired by construction. This is the central
    result of the paper, and it deserves the same treatment as the label result.
    """
    scored = [r for r in rows if r["human_answerable"]]
    b = sum(1 for r in scored if r["stored_correct"] and not r["graph_correct"])
    c = sum(1 for r in scored if r["graph_correct"] and not r["stored_correct"])
    return {
        "comparison": "stored relations with human identity vs the delivered graph",
        "n_items": len(scored),
        "stored_correct": sum(1 for r in scored if r["stored_correct"]),
        "delivered_correct": sum(1 for r in scored if r["graph_correct"]),
        "human_identity_only_correct": b,
        "delivered_only_correct": c,
        "discordant_pairs": b + c,
        "p_value_exact_mcnemar": exact_mcnemar(b, c),
        "scope": "identity_oracle on one side -- this measures the SIZE OF A BOUND, "
                 "not a system improvement. It says the two arms differ, not that "
                 "anything deployable achieves the higher number.",
        "limitation":
            f"{len(scored)} items authored over two rooms, with one human reviewer who "
            "is not blind to the hypothesis. The pairing is exact (same questions, same "
            "stored edges) but the item set is small and the p-value inherits every "
            "limitation of that key.",
    }


def build() -> tuple[dict, list[dict]]:
    rows = ledger()
    report = {
        "schema": "paper_statistics_v1",
        "purpose": "Paired significance and question-level reachability, computed "
                   "from the tracked evidence pack. No model was run.",
        "source": "eval/results/project_census_v1/",
        "paired_label_tests": [paired_table("top1_correct"),
                               paired_table("top3_correct")],
        "reachability": transitions(rows),
        "paired_identity_test": paired_identity_test(rows),
    }
    return report, rows


def write(report: dict, rows: list[dict]) -> None:
    OUT_JSON.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    with OUT_LEDGER.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]),
                           quoting=csv.QUOTE_ALL, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str]) -> int:
    report, rows = build()
    if "--check" in argv:
        before = (OUT_JSON.read_bytes() if OUT_JSON.is_file() else b"",
                  OUT_LEDGER.read_bytes() if OUT_LEDGER.is_file() else b"")
        write(report, rows)
        after = (OUT_JSON.read_bytes(), OUT_LEDGER.read_bytes())
        if before != after:
            print("committed statistics are stale; re-run tools/paper_statistics.py")
            return 1
        print("statistics are current")
        return 0
    write(report, rows)
    for t in report["paired_label_tests"]:
        print(f"  {t['field']}: splat {t['splat_correct']}/{t['n_instances']} -> "
              f"rgb_tight {t['rgb_tight_correct']}/{t['n_instances']}; "
              f"fixed {t['rgb_only_correct']}, regressed {t['splat_only_correct']}, "
              f"exact McNemar p = {t['p_value_exact_mcnemar']:.6g}")
    i = report["paired_identity_test"]
    print(f"  identity substitution: {i['delivered_correct']}/{i['n_items']} -> "
          f"{i['stored_correct']}/{i['n_items']}; {i['human_identity_only_correct']} fixed, "
          f"{i['delivered_only_correct']} regressed, exact McNemar p = "
          f"{i['p_value_exact_mcnemar']:.6g}")
    r = report["reachability"]
    print(f"  reachability: {r['held_by_representation']} held / "
          f"{r['reached_by_deployable_grounding']} reached by grounding / "
          f"{r['reached_by_delivered_graph']} by the delivered graph, "
          f"n = {r['n_scored']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
