"""Annotation-key sensitivity analysis over the frozen ARKit relation challenge.

Re-scores every recorded arm of the relation challenge under three evaluation
keys, changing NOTHING about the arms' answers (they are read verbatim from the
committed challenge report):

  (a) original   -- the owner-adjudicated key
                    (eval/human_feedback/arkitscenes_relation_challenge_key_v1.json)
  (b) annotator2 -- the independent second annotator's key exactly as returned
                    (eval/human_feedback/arkitscenes_relation_challenge_annotator2_returned.json)
  (c) agreement  -- the original key restricted to the items on which the two
                    annotators agree (final_answer_match in
                    eval/results/stagereach/annotator_agreement_v1.json)

Scoring rule reuse: grading is delegated to `grade()` in
tools/arkitscenes_relation_challenge_score.py (which applies `_matches` per
question form). No scoring judgment is introduced here.

Ambiguity / abstention handling rule (deterministic, uniform across arms):
  An item with no key answer is excluded from that key's denominator via the
  frozen scorer's own `excluded_no_human_answer` branch. Under the original
  key that covers the two owner-excluded ambiguous items (n = 10 scored).
  Under the annotator-2 key, a returned "cannot determine" is an abstention:
  the item's key answer is None and the same branch excludes it (n = 8
  scored). The agreement subset contains only items where both annotators
  returned the same effective answer (n = 6 scored).

Annotator-2 answer typing (mechanical, no judgment; the sheet returns free
text while the scorer's key is typed):
  binary_near      "yes"/"no" (case-insensitive) -> True/False;
                   "cannot determine" -> None (abstention).
  comparative_near the returned string must contain exactly one of the
                   question's two reference names (case-insensitive
                   substring); the key answer is that canonical name;
                   "cannot determine" -> None.
  near_set         split on commas; each token maps to the candidate object
                   it equals case-insensitively, else to the unique candidate
                   whose word set contains the token's words (this is the
                   adjudication-recorded equivalence "white heater" -> "white
                   convector heater"); an unmappable token is kept verbatim
                   (annotator 2's over-inclusion of "window" is preserved,
                   not repaired).
Any input outside these rules raises instead of guessing.

Usage:
  .venv/bin/python tools/annotation_sensitivity.py

Writes eval/results/stagereach/annotation_sensitivity_v1.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.arkitscenes_relation_challenge_score import _matches, grade  # noqa: E402

REPORT = REPO / "eval/results/project_census_v1/arkit_relation_challenge_report.json"
QUESTIONS = REPO / "eval/questions/arkitscenes_relation_challenge_v1.json"
KEY_V1 = REPO / "eval/human_feedback/arkitscenes_relation_challenge_key_v1.json"
ANNOT2 = REPO / "eval/human_feedback/arkitscenes_relation_challenge_annotator2_returned.json"
AGREEMENT = REPO / "eval/results/stagereach/annotator_agreement_v1.json"
OUT = REPO / "eval/results/stagereach/annotation_sensitivity_v1.json"

ARMS = [
    "geometry_relation_ceiling",
    "stored_graph_human_identity",
    "delivered_graph",
    "grounded_delivered_graph",
    "blinded_rgb_vlm",
    "evidence_aware_hybrid",
]

ABSTAIN = "cannot determine"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


# --------------------------------------------------------------------------
# annotator-2 key construction (mechanical typing of the returned free text)
# --------------------------------------------------------------------------
def _type_binary(text: str) -> object:
    t = text.strip().lower()
    if t == ABSTAIN:
        return None
    if t == "yes":
        return True
    if t == "no":
        return False
    raise ValueError(f"untypeable binary answer: {text!r}")


def _type_comparative(text: str, question: dict) -> object:
    t = text.strip().lower()
    if t == ABSTAIN:
        return None
    refs = [question["reference_a"], question["reference_b"]]
    hits = [r for r in refs if r.lower() in t]
    if len(hits) != 1:
        raise ValueError(
            f"comparative answer {text!r} names {len(hits)} of {refs}")
    return hits[0]


def _type_set(text: str, question: dict) -> object:
    t = text.strip().lower()
    if t == ABSTAIN:
        return None
    candidates = question["candidate_objects"]
    out = []
    for token in (p.strip().lower() for p in text.split(",")):
        if not token:
            continue
        exact = [c for c in candidates if c.lower() == token]
        if exact:
            out.append(exact[0])
            continue
        words = set(token.split())
        subset = [c for c in candidates if words <= set(c.lower().split())]
        if len(subset) == 1:
            out.append(subset[0])
        else:
            out.append(token)  # kept verbatim; over-inclusions are preserved
    return out


def build_annotator2_key(returned: dict, questions: list[dict]) -> dict:
    typers = {"binary_near": _type_binary,
              "comparative_near": _type_comparative,
              "near_set": _type_set}
    by_id = {q["id"]: q for q in questions}
    truth = []
    for qid, entry in sorted(returned["answers"].items()):
        q = by_id[qid]
        typer = typers[q["form"]]
        answer = (typer(entry["answer"]) if q["form"] == "binary_near"
                  else typer(entry["answer"], q))
        truth.append({
            "id": qid,
            "ambiguous": bool(entry.get("ambiguous")),
            "answer": answer,
            "returned_verbatim": entry["answer"],
            "evidence_views": None,
            "notes": entry.get("rationale") or None,
        })
    return {"human_relation_truth": truth}


# --------------------------------------------------------------------------
# scoring under one key
# --------------------------------------------------------------------------
def score_arm(rows: list[dict], key: dict, questions: list[dict],
              mask: set[str] | None = None) -> dict:
    """Grade `rows` with the frozen scorer's grade(); optionally restrict to
    a question-id mask (agreement subset). Returns tally + per-item outcomes."""
    clean = [{"id": r["id"], "answer": r["answer"],
              "outcome_hint": r["outcome_hint"], "source": r["source"],
              "form": r["form"]} for r in rows]
    graded = grade(clean, key, questions)
    if mask is not None:
        graded = [g for g in graded if g["id"] in mask]
    tally = {"correct": 0, "wrong": 0, "unanswered": 0,
             "excluded_no_human_answer": 0}
    for g in graded:
        tally[g["outcome"]] += 1
    n_scored = len(graded) - tally["excluded_no_human_answer"]
    n_answered = tally["correct"] + tally["wrong"]
    return {
        "tally": tally,
        "n_scored": n_scored,
        "correct": tally["correct"],
        "coverage": round(n_answered / n_scored, 4) if n_scored else None,
        "items": {g["id"]: g["outcome"] for g in graded},
    }


def main() -> None:
    report = _load(REPORT)
    questions = _load(QUESTIONS)["questions"]
    key_v1 = _load(KEY_V1)
    returned = _load(ANNOT2)
    agreement = _load(AGREEMENT)

    key_a2 = build_annotator2_key(returned, questions)

    # agreement mask: items whose adjudicated answers match across annotators
    mask = {i["id"] for i in agreement["items"] if i["final_answer_match"]}

    # determinism guard: on every mask item the typed annotator-2 answer must
    # equal the original key answer under the frozen matcher, so scoring the
    # agreement subset under either key is identical by construction.
    forms = {q["id"]: q["form"] for q in questions}
    v1_truth = {a["id"]: a for a in key_v1["human_relation_truth"]}
    a2_truth = {a["id"]: a for a in key_a2["human_relation_truth"]}
    for qid in sorted(mask):
        assert _matches(forms[qid], a2_truth[qid]["answer"],
                        v1_truth[qid]["answer"]), \
            f"agreement item {qid} disagrees after typing"

    arms_out = {}
    for arm in ARMS:
        rows = report["arms"][arm]["rows"]
        arms_out[arm] = {
            "deployable": report["arms"][arm]["deployable"],
            "original_key": score_arm(rows, key_v1, questions),
            "annotator2_key": score_arm(rows, key_a2, questions),
            "agreement_subset": score_arm(rows, key_v1, questions, mask=mask),
        }

    out = {
        "schema": "arkitscenes_annotation_sensitivity",
        "schema_version": 1,
        "purpose": ("evaluation-key reproducibility evidence; re-scores frozen "
                    "arm answers under alternative keys; retunes nothing"),
        "handling_rule": (
            "an item with no key answer is excluded from that key's "
            "denominator via the frozen scorer's excluded_no_human_answer "
            "branch, uniformly across arms; under the annotator-2 key a "
            "returned 'cannot determine' is an abstention (key answer None) "
            "and takes the same branch"),
        "keys": {
            "original_key": {"source": str(KEY_V1.relative_to(REPO)),
                             "n_scored": 10},
            "annotator2_key": {"source": str(ANNOT2.relative_to(REPO)),
                               "n_scored": 8,
                               "typed_key": key_a2["human_relation_truth"]},
            "agreement_subset": {"source": str(AGREEMENT.relative_to(REPO)),
                                 "n_scored": len(mask),
                                 "definition": ("items with final_answer_match "
                                                "true in the adjudicated "
                                                "agreement record")},
        },
        "agreement_mask": {i["id"]: i["final_answer_match"]
                           for i in agreement["items"]},
        "arms": arms_out,
        "inputs": {
            "report": str(REPORT.relative_to(REPO)),
            "questions": str(QUESTIONS.relative_to(REPO)),
        },
    }

    # sanity: reproduce the committed headline tallies under the original key
    headline = {"stored_graph_human_identity": 7, "delivered_graph": 0,
                "grounded_delivered_graph": 2, "blinded_rgb_vlm": 7}
    for arm, want in headline.items():
        got = arms_out[arm]["original_key"]["correct"]
        assert got == want, f"{arm}: original-key correct {got} != {want}"

    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT}")
    for arm in ARMS:
        a = arms_out[arm]
        print(f"{arm:32s} "
              f"orig {a['original_key']['correct']}/{a['original_key']['n_scored']}  "
              f"annot2 {a['annotator2_key']['correct']}/{a['annotator2_key']['n_scored']}  "
              f"agree {a['agreement_subset']['correct']}/{a['agreement_subset']['n_scored']}")


if __name__ == "__main__":
    main()
