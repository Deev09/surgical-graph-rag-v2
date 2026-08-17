"""Four-layer scorer for the two-scene NEAR relation challenge.

  python3 tools/arkitscenes_relation_challenge_score.py \
      --questions eval/questions/arkitscenes_relation_challenge_v1.json \
      --key eval/human_feedback/arkitscenes_relation_challenge_key_v1.json \
      --scene-inputs runs/arkit_relation_challenge/scene_inputs.json \
      --out runs/arkit_relation_challenge/report.json

EVALUATION ONLY. Runs no segmenter, no labeler and no relation stage, writes
nothing back into any bundle, and changes no perception, threshold or key.

WHY FOUR LAYERS
---------------
The previous kill test could not say WHY the graph scored zero, because a
missing relation type, a wrong label and unusable geometry all look identical
from the outside. Four layers separate them:

  geometry_relation_ceiling  human-verified UID mappings + delivered geometry.
                             CEILING, NOT DEPLOYABLE: it consumes the human key
                             to resolve identity. Answers "could this
                             representation express the answer at all?"
  delivered_graph            delivered entities, LEARNED labels, delivered
                             edges. The deployable structured path. Never
                             touches the human key.
  blinded_rgb_vlm            an answer-free multi-view RGB packet answered in a
                             separate fresh context. Never touches the key.
  evidence_aware_hybrid      predeclared routing over the two above. Abstains
                             rather than inventing an unavailable graph fact.

Ceiling passes + delivered fails  -> naming/instance-delivery/extraction binds.
Ceiling fails                     -> the representation cannot express it.
Ceiling unanswerable              -> no resolvable delivered instance exists;
                                     that is an instance-DELIVERY finding, and
                                     is reported separately from a wrong answer.

ANSWERING AND GRADING ARE SEPARATE FUNCTIONS, ON PURPOSE
--------------------------------------------------------
Every `answer_*` function returns answers and never sees an expected value.
`grade()` is the only function that reads the key. A deployable layer therefore
*cannot* consult the key even by accident, and a test asserts the deployable
answer functions never bind the name `key`. This is the same separation the
support review sheet enforces between human truth and system evidence.

THE NEAR THRESHOLD IS READ FROM THE SEALED ARTIFACT, NEVER DECLARED HERE
------------------------------------------------------------------------
`near_threshold_m` comes from each scene's slice manifest, which records it as
a "provisional Replica-era constant; not a transfer claim". This tool does not
define, default or tune it. Comparative questions do not use it at all, which
is why they are the threshold-independent core of the question set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry.surface_distance import aabb_to_aabb_surface
from tools.arkitscenes_spatial_qa_score import is_anonymous, normalize

SCHEMA = "arkitscenes_relation_challenge_score_v1"
QUESTIONS_SCHEMA = "arkitscenes_relation_challenge_questions_v1"
KEY_SCHEMA = "arkitscenes_relation_challenge_key_v1"
RESPONSE_SCHEMA = "arkitscenes_relation_challenge_rgb_responses_v1"

FORMS = ("binary_near", "comparative_near", "near_set")
SCENE_DIRS = {
    "arkitscenes_41069025": "41069025",
    "arkitscenes_41069042": "41069042",
}
OUTCOMES = ("correct", "wrong", "unanswered")

# Layer identity. `deployable` is carried into the report so a ceiling number
# can never be quoted as system performance without the flag travelling with it.
LAYERS = (
    ("geometry_relation_ceiling", "ceiling", False),
    ("delivered_graph", "delivered", True),
    ("blinded_rgb_vlm", "delivered", True),
    ("evidence_aware_hybrid", "delivered", True),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


# --------------------------------------------------------------------------
# unanswered reasons -- distinct causes must stay distinguishable in the report
# --------------------------------------------------------------------------
NO_MAPPING = "no human-verified UID mapping for this object"
NO_LABEL = "no delivered instance carries an admitted label for this object"
NO_DISTANCE = "neither pair has a stored distance, so no ordering is derivable"
THIN_EVIDENCE = "fewer than two valid cited views"
NO_GRAPH_FACT = "the delivered graph materializes no fact for this question"
# A `near_set` item asks for an EXHAUSTIVE set. If a candidate cannot be
# resolved, the honest answer is not a shorter set -- it is no set. Returning
# the resolvable subset would emit a false negative dressed as an answer, which
# is precisely the failure the unanswered/wrong split exists to prevent.
NO_EXHAUSTIVE_SET = ("cannot enumerate an exhaustive set: one or more candidate "
                     "objects are unresolvable")


def _unanswered(question: dict, reason: str, source: str) -> dict:
    return {"id": question["id"], "scene_id": question["scene_id"],
            "form": question["form"], "answer": None,
            "outcome_hint": "unanswered", "reason": reason, "source": source}


def _answered(question: dict, answer: object, source: str, **extra) -> dict:
    row = {"id": question["id"], "scene_id": question["scene_id"],
           "form": question["form"], "answer": answer,
           "outcome_hint": "answered", "source": source}
    row.update(extra)
    return row


# --------------------------------------------------------------------------
# layer 1 -- geometry relation ceiling (CEILING; consumes human UID mappings)
# --------------------------------------------------------------------------
def answer_geometry_ceiling(questions: list[dict], scenes: dict,
                            uid_mappings: dict) -> list[dict]:
    """Delivered geometry + HUMAN-VERIFIED identity. Not a deployable result.

    Uses `aabb_to_aabb_surface`, the exact function the delivered NEAR
    extractor uses, and each scene's own recorded threshold. No new metric is
    introduced: the only thing this layer changes versus `delivered_graph` is
    where object identity comes from.
    """
    rows = []
    for question in questions:
        scene = scenes[question["scene_id"]]
        aabb = scene["aabb_by_uid"]
        threshold = scene["near_threshold_m"]
        mapped = uid_mappings.get(question["scene_id"], {})

        def uid_for(name: str) -> str | None:
            entry = mapped.get(name)
            return entry if isinstance(entry, str) and entry in aabb else None

        if question["form"] == "binary_near":
            a, b = uid_for(question["subject"]), uid_for(question["object"])
            if a is None or b is None:
                rows.append(_unanswered(question, NO_MAPPING, "geometry_ceiling"))
                continue
            distance = aabb_to_aabb_surface(aabb[a], aabb[b])
            rows.append(_answered(question, bool(distance < threshold),
                                  "geometry_ceiling", distance_m=round(distance, 4),
                                  uids=[a, b], threshold_m=threshold))

        elif question["form"] == "comparative_near":
            s = uid_for(question["subject"])
            a = uid_for(question["reference_a"])
            b = uid_for(question["reference_b"])
            if s is None or a is None or b is None:
                rows.append(_unanswered(question, NO_MAPPING, "geometry_ceiling"))
                continue
            da = aabb_to_aabb_surface(aabb[s], aabb[a])
            db = aabb_to_aabb_surface(aabb[s], aabb[b])
            # Threshold-free by construction: an ordering, not a membership test.
            rows.append(_answered(
                question, question["reference_a"] if da < db else question["reference_b"],
                "geometry_ceiling", distance_a_m=round(da, 4),
                distance_b_m=round(db, 4), uids=[s, a, b]))

        elif question["form"] == "near_set":
            s = uid_for(question["subject"])
            if s is None:
                rows.append(_unanswered(question, NO_MAPPING, "geometry_ceiling"))
                continue
            # The candidate roster is stated on the question, so this is a
            # closed set for every layer rather than an open-vocabulary task
            # for one of them.
            unresolved = sorted(n for n in question["candidate_objects"]
                                if uid_for(n) is None)
            if unresolved:
                row = _unanswered(question, NO_EXHAUSTIVE_SET, "geometry_ceiling")
                row["unresolved_candidates"] = unresolved
                rows.append(row)
                continue
            hits = [name for name in question["candidate_objects"]
                    if uid_for(name) != s
                    and aabb_to_aabb_surface(aabb[s], aabb[uid_for(name)]) < threshold]
            rows.append(_answered(question, sorted(hits), "geometry_ceiling",
                                  uids=[s], threshold_m=threshold))
        else:
            rows.append(_unanswered(question, f"unknown form {question['form']}",
                                    "geometry_ceiling"))
    return rows


# --------------------------------------------------------------------------
# layer 2 -- delivered graph (DEPLOYABLE; learned labels only, no human key)
# --------------------------------------------------------------------------
def _label_index(entities: list[dict]) -> dict[str, list[str]]:
    """Learned-label -> uids. Anonymous placeholders assert nothing."""
    index: dict[str, list[str]] = {}
    for entity in entities:
        label = entity.get("display_label")
        if is_anonymous(label):
            continue
        index.setdefault(normalize(label), []).append(entity["uid"])
    return index


def _resolve_delivered(name: str, index: dict[str, list[str]],
                       synonyms: dict) -> str | None:
    """A question object resolves only if exactly one instance claims it.

    Zero matches means the system asserts nothing; several means it cannot say
    which. Both abstain. Guessing among several would score a coin flip as
    system capability.
    """
    wanted = {normalize(n) for n in synonyms.get(name, [name])}
    hits = sorted({uid for w in wanted for uid in index.get(w, [])})
    return hits[0] if len(hits) == 1 else None


def answer_delivered_graph(questions: list[dict], scenes: dict,
                           synonyms: dict) -> list[dict]:
    """Delivered entities + LEARNED labels + delivered edges. No human key.

    Edge absence is informative, not missing data: the extractor evaluates
    every unordered pair and emits one iff surface distance < threshold, so a
    missing edge means "measured, and at least threshold apart".
    """
    rows = []
    for question in questions:
        scene = scenes[question["scene_id"]]
        index = scene["label_index"]
        distances = scene["edge_distance"]
        threshold = scene["near_threshold_m"]

        def uid_for(name: str) -> str | None:
            return _resolve_delivered(name, index, synonyms)

        def pair_distance(a: str, b: str) -> float | None:
            return distances.get(frozenset((a, b)))

        if question["form"] == "binary_near":
            a, b = uid_for(question["subject"]), uid_for(question["object"])
            if a is None or b is None:
                rows.append(_unanswered(question, NO_LABEL, "delivered_graph"))
                continue
            distance = pair_distance(a, b)
            rows.append(_answered(question, distance is not None,
                                  "delivered_graph",
                                  edge_present=distance is not None,
                                  distance_m=(round(distance, 4)
                                              if distance is not None else None),
                                  uids=[a, b], threshold_m=threshold))

        elif question["form"] == "comparative_near":
            s = uid_for(question["subject"])
            a = uid_for(question["reference_a"])
            b = uid_for(question["reference_b"])
            if s is None or a is None or b is None:
                rows.append(_unanswered(question, NO_LABEL, "delivered_graph"))
                continue
            da, db = pair_distance(s, a), pair_distance(s, b)
            if da is None and db is None:
                # Both at least threshold apart; the graph stores no value for
                # either, so it cannot order them. Abstain rather than guess.
                rows.append(_unanswered(question, NO_DISTANCE, "delivered_graph"))
                continue
            if da is not None and db is not None:
                choice = question["reference_a"] if da < db else question["reference_b"]
                basis = "both_edges_stored"
            else:
                # One stored (< threshold), one absent (>= threshold): the
                # ordering follows from the threshold semantics alone.
                choice = (question["reference_a"] if da is not None
                          else question["reference_b"])
                basis = "one_edge_stored_one_beyond_threshold"
            rows.append(_answered(question, choice, "delivered_graph",
                                  distance_a_m=(round(da, 4) if da is not None else None),
                                  distance_b_m=(round(db, 4) if db is not None else None),
                                  ordering_basis=basis, uids=[s, a, b]))

        elif question["form"] == "near_set":
            s = uid_for(question["subject"])
            if s is None:
                rows.append(_unanswered(question, NO_LABEL, "delivered_graph"))
                continue
            unresolved = sorted(n for n in question["candidate_objects"]
                                if uid_for(n) is None)
            if unresolved:
                row = _unanswered(question, NO_EXHAUSTIVE_SET, "delivered_graph")
                row["unresolved_candidates"] = unresolved
                rows.append(row)
                continue
            hits = [name for name in question["candidate_objects"]
                    if uid_for(name) != s
                    and pair_distance(s, uid_for(name)) is not None]
            rows.append(_answered(question, sorted(hits), "delivered_graph",
                                  uids=[s], threshold_m=threshold))
        else:
            rows.append(_unanswered(question, f"unknown form {question['form']}",
                                    "delivered_graph"))
    return rows


# --------------------------------------------------------------------------
# layer 3 -- blinded multi-view RGB (DEPLOYABLE; sidecar, no human key)
# --------------------------------------------------------------------------
def answer_blinded_rgb(questions: list[dict], packets: dict,
                       response: dict) -> list[dict]:
    """Validate and adopt a blinded response produced in a separate context."""
    if response.get("schema") != RESPONSE_SCHEMA:
        raise ValueError("wrong blinded-response schema")
    model = response.get("model")
    if not isinstance(model, dict) or any(
            not isinstance(model.get(k), str) or not model[k].strip()
            for k in ("provider", "name", "version")):
        raise ValueError("blinded response must record provider/name/version")
    for scene_id, packet in packets.items():
        pinned = response.get("packet_sha256", {}).get(scene_id)
        if pinned != packet["packet_sha256"]:
            raise ValueError(f"{scene_id}: response does not pin the prepared packet")

    by_id = {a["id"]: a for a in response.get("answers", [])}
    if len(by_id) != len(response.get("answers", [])):
        raise ValueError("duplicate blinded-response question id")
    if set(by_id) != {q["id"] for q in questions}:
        raise ValueError("blinded response must answer every question exactly once")

    rows = []
    for question in questions:
        answer = by_id[question["id"]]
        valid = {f["id"] for f in packets[question["scene_id"]]["frames"]}
        cited = answer.get("evidence_frame_ids", [])
        if not isinstance(cited, list) or any(f not in valid for f in cited):
            raise ValueError(f"{question['id']}: invalid evidence frame id")
        confidence = answer.get("confidence")
        if (not isinstance(confidence, (int, float)) or isinstance(confidence, bool)
                or not 0.0 <= float(confidence) <= 1.0):
            raise ValueError(f"{question['id']}: confidence must be in [0, 1]")
        outcome = answer.get("outcome")
        if outcome not in {"answer", "unknown"}:
            raise ValueError(f"{question['id']}: outcome must be answer or unknown")
        extra = {"confidence": float(confidence), "evidence_frame_ids": cited,
                 "evidence_sufficient": len(set(cited)) >= 2}
        if outcome == "unknown":
            row = _unanswered(question, "model returned unknown", "blinded_rgb_vlm")
            row.update(extra)
        else:
            row = _answered(question, answer.get("answer"), "blinded_rgb_vlm", **extra)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# layer 4 -- evidence-aware hybrid (DEPLOYABLE; predeclared routing)
# --------------------------------------------------------------------------
def answer_hybrid(questions: list[dict], graph_rows: list[dict],
                  rgb_rows: list[dict]) -> list[dict]:
    """Predeclared policy, fixed before any answer was seen.

    1. A relation question routes to the delivered graph whenever the graph
       materializes a fact for it.
    2. Otherwise it falls back to blinded RGB, and is accepted only when the
       response cites at least two distinct supplied views.
    3. Otherwise it abstains. The hybrid never invents a graph fact.
    """
    graph = {r["id"]: r for r in graph_rows}
    rgb = {r["id"]: r for r in rgb_rows}
    rows = []
    for question in questions:
        g = graph[question["id"]]
        if g["outcome_hint"] == "answered":
            row = dict(g)
            row.update(source="evidence_aware_hybrid", route="typed_relation",
                       answered_by="delivered_graph")
            rows.append(row)
            continue
        r = dict(rgb[question["id"]])
        r.update(source="evidence_aware_hybrid", route="visual_evidence",
                 answered_by="blinded_rgb_vlm",
                 graph_abstained_because=g.get("reason", NO_GRAPH_FACT))
        if r["outcome_hint"] == "answered" and not r.get("evidence_sufficient"):
            r.update(outcome_hint="unanswered", answer=None,
                     reason=THIN_EVIDENCE, insufficiency=THIN_EVIDENCE)
        rows.append(r)
    return rows


# --------------------------------------------------------------------------
# grading -- the ONLY place the human key is read
# --------------------------------------------------------------------------
def _matches(form: str, answer: object, expected: object) -> bool:
    if form == "binary_near":
        return isinstance(answer, bool) and isinstance(expected, bool) \
            and answer == expected
    if form == "comparative_near":
        return isinstance(answer, str) and isinstance(expected, str) \
            and answer.strip().lower() == expected.strip().lower()
    if form == "near_set":
        if not isinstance(answer, list) or not isinstance(expected, list):
            return False
        return sorted(str(x).strip().lower() for x in answer) == \
            sorted(str(x).strip().lower() for x in expected)
    return False


def grade(rows: list[dict], key: dict, questions: list[dict]) -> list[dict]:
    """Attach outcomes. Answer production never sees any of this."""
    truth = {a["id"]: a for a in key["human_relation_truth"]}
    forms = {q["id"]: q["form"] for q in questions}
    graded = []
    for row in rows:
        item = truth[row["id"]]
        out = dict(row)
        expected = item.get("answer")
        out["expected"] = expected
        out["human_marked_ambiguous"] = bool(item.get("ambiguous"))
        out["evidence_views"] = item.get("evidence_views")
        if item.get("ambiguous") or expected is None:
            # The owner declined to assert an answer. Nothing can be right or
            # wrong against a non-answer; excluded from the tally, kept visible.
            out["outcome"] = "excluded_no_human_answer"
        elif row["outcome_hint"] == "unanswered":
            out["outcome"] = "unanswered"
        else:
            out["outcome"] = ("correct" if _matches(forms[row["id"]],
                                                    row["answer"], expected)
                              else "wrong")
        graded.append(out)
    return graded


def summarize(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["outcome"] in OUTCOMES]
    counts = {o: sum(1 for r in scored if r["outcome"] == o) for o in OUTCOMES}
    answered = counts["correct"] + counts["wrong"]
    total = len(scored)
    return {
        "n_questions_scored": total,
        "n_excluded_no_human_answer": len(rows) - total,
        "tally": counts,
        "coverage": round(answered / total, 4) if total else None,
        "exact_accuracy_all": round(counts["correct"] / total, 4) if total else None,
        "accuracy_when_answered": (round(counts["correct"] / answered, 4)
                                   if answered else None),
        "false_confident_rate": (round(counts["wrong"] / answered, 4)
                                 if answered else None),
    }


def per_scene(rows: list[dict]) -> dict:
    scenes = sorted({r["scene_id"] for r in rows})
    return {s: summarize([r for r in rows if r["scene_id"] == s]) for s in scenes}


# --------------------------------------------------------------------------
# cross-layer readouts
# --------------------------------------------------------------------------
def graph_unique_wins(graph_rows: list[dict], rgb_rows: list[dict]) -> list[dict]:
    """Questions the delivered graph got right and blinded RGB did not.

    This is the quantity the continuation bar is written against. It counts
    only the DELIVERED graph; a ceiling win is not a graph win.
    """
    rgb = {r["id"]: r for r in rgb_rows}
    wins = []
    for row in graph_rows:
        other = rgb.get(row["id"])
        if row["outcome"] == "correct" and other is not None \
                and other["outcome"] != "correct":
            wins.append({"id": row["id"], "scene_id": row["scene_id"],
                         "form": row["form"], "graph_answer": row["answer"],
                         "rgb_outcome": other["outcome"],
                         "rgb_answer": other.get("answer")})
    return wins


def attribution(ceiling_rows: list[dict], graph_rows: list[dict]) -> dict:
    """Which stage binds, per the predeclared interpretation table."""
    ceiling = {r["id"]: r for r in ceiling_rows}
    buckets = {
        "ceiling_correct_delivered_wrong": [],
        "ceiling_correct_delivered_unanswered": [],
        "ceiling_wrong": [],
        "ceiling_unanswerable_no_uid_mapping": [],
        "both_correct": [],
    }
    for row in graph_rows:
        c = ceiling.get(row["id"])
        if c is None:
            continue
        if c["outcome"] == "unanswered" and c.get("reason") == NO_MAPPING:
            buckets["ceiling_unanswerable_no_uid_mapping"].append(row["id"])
        elif c["outcome"] == "wrong":
            buckets["ceiling_wrong"].append(row["id"])
        elif c["outcome"] == "correct" and row["outcome"] == "wrong":
            buckets["ceiling_correct_delivered_wrong"].append(row["id"])
        elif c["outcome"] == "correct" and row["outcome"] == "unanswered":
            buckets["ceiling_correct_delivered_unanswered"].append(row["id"])
        elif c["outcome"] == "correct" and row["outcome"] == "correct":
            buckets["both_correct"].append(row["id"])
    reading = {
        "ceiling_correct_delivered_wrong":
            "naming, instance delivery or relation extraction is binding",
        "ceiling_correct_delivered_unanswered":
            "the fact is expressible but the delivered system cannot address it",
        "ceiling_wrong":
            "the current geometry/representation cannot answer the relation",
        "ceiling_unanswerable_no_uid_mapping":
            "no resolvable delivered instance exists; instance DELIVERY binds, "
            "which is distinct from a geometry failure",
    }
    return {"buckets": buckets, "reading": reading}


def thin_evidence_slice(rgb_rows: list[dict], hybrid_rows: list[dict]) -> dict:
    """Secondary subtest: does the two-view gate pay for its coverage cost?

    The slice is whatever the owner independently recorded as 0- or 1-view
    evidence. It is never manufactured, and it is analysed only after the
    questions are fixed.
    """
    thin_ids = {r["id"] for r in rgb_rows if r.get("evidence_views") in {"0", "1"}}
    if not thin_ids:
        return {"status": "empty",
                "note": "the owner recorded two or more views for every question; "
                        "this run cannot test the sufficiency gate",
                "n_thin": 0}
    rgb = {r["id"]: r for r in rgb_rows if r["id"] in thin_ids}
    hybrid = {r["id"]: r for r in hybrid_rows if r["id"] in thin_ids}
    gate_fired = [i for i in thin_ids
                  if rgb[i]["outcome"] in {"correct", "wrong"}
                  and hybrid[i]["outcome"] == "unanswered"]
    return {
        "status": "measured",
        "n_thin": len(thin_ids),
        "thin_question_ids": sorted(thin_ids),
        "rgb_on_thin_slice": summarize(list(rgb.values())),
        "hybrid_on_thin_slice": summarize(list(hybrid.values())),
        "gate_fired_on": sorted(gate_fired),
        "wrong_answers_suppressed": sorted(
            i for i in gate_fired if rgb[i]["outcome"] == "wrong"),
        "correct_answers_suppressed": sorted(
            i for i in gate_fired if rgb[i]["outcome"] == "correct"),
        "note": "coverage cost and error reduction are reported side by side; "
                "suppressing a correct answer is a real cost, not a rounding term",
    }


def decision(wins: list[dict], scenes: list[str]) -> dict:
    """Predeclared continuation bar. Screening gate, not a publication claim."""
    per = {s: sum(1 for w in wins if w["scene_id"] == s) for s in scenes}
    both = sum(1 for c in per.values() if c >= 1)
    return {
        "n_graph_unique_wins": len(wins),
        "graph_unique_wins_per_scene": per,
        "bar": "at least two delivered-graph-unique correct answers, "
               "preferably at least one in each scene",
        "meets_bar": len(wins) >= 2,
        "reproduced_across_scenes": both >= 2,
        "proceed_to_larger_hybrid": len(wins) >= 2,
        "interpretation": "screening gate on two scenes; not generalization "
                          "and not a publication claim",
    }


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def load_scene_inputs(spec: dict) -> dict:
    """Load delivered artifacts per scene, plus the recorded NEAR threshold."""
    scenes = {}
    for scene_id, paths in spec.items():
        entities_path = Path(paths["entities"])
        graph_path = Path(paths["graph"])
        slice_path = Path(paths["slice_manifest"])
        entity_manifest = json.loads(entities_path.read_text())
        graph_manifest = json.loads(graph_path.read_text())
        slice_manifest = json.loads(slice_path.read_text())

        relation_config = slice_manifest["relation_configuration"]
        if relation_config["near_metric"] != "aabb_surface":
            raise ValueError(f"{scene_id}: unexpected near metric "
                             f"{relation_config['near_metric']!r}")

        entities, aabb_by_uid = [], {}
        for entity in entity_manifest["entities"]:
            uid = entity["identity"]["object_uid"]
            entities.append({"uid": uid,
                             "display_label": entity["identity"].get("display_label")})
            aabb_by_uid[uid] = entity["bbox_aabb"]

        edge_distance = {}
        for edge in graph_manifest["edges"]:
            if edge.get("type") != "NEAR":
                continue
            pair = frozenset((edge["source"]["uid"], edge["target"]["uid"]))
            edge_distance[pair] = edge["evidence"]["distance_m"]

        stored = sorted(edge_distance.values())
        scenes[scene_id] = {
            "aabb_by_uid": aabb_by_uid,
            "label_index": _label_index(entities),
            "edge_distance": edge_distance,
            "near_threshold_m": relation_config["near_threshold_m"],
            "provenance": {
                "entities": str(entities_path),
                "entities_sha256": sha256(entities_path),
                "entity_bundle_hash": entity_manifest.get("bundle_hash"),
                "graph": str(graph_path),
                "graph_sha256": sha256(graph_path),
                "graph_bundle_hash": graph_manifest.get("bundle_hash"),
                "slice_manifest": str(slice_path),
                "n_entities": len(entities),
                "n_near_edges": len(edge_distance),
                "near_threshold_m": relation_config["near_threshold_m"],
                "near_metric": relation_config["near_metric"],
                "threshold_status": relation_config.get("threshold_status"),
                "stored_distance_m": {
                    "min": round(stored[0], 4) if stored else None,
                    "median": round(statistics.median(stored), 4) if stored else None,
                    "max": round(stored[-1], 4) if stored else None,
                },
            },
        }
    return scenes


def write_scene_inputs(doc: dict, out: Path) -> Path:
    """The scorer's input map. Generated, never hand-written, so the paths and
    the slice manifest that carries the NEAR threshold cannot drift apart."""
    spec = {}
    for scene_id in doc["scenes"]:
        short = SCENE_DIRS[scene_id]
        spec[scene_id] = {
            "entities": str(REPO_ROOT / "runs" / f"arkit_label_image_ab_{short}"
                            / "rgb_tight" / "entities" / "manifest.json"),
            "graph": str(REPO_ROOT / "runs" / "arkit_vertical_slice" / "sealed_pair"
                         / short / "graph" / "manifest.json"),
            "slice_manifest": str(REPO_ROOT / "runs" / "arkit_vertical_slice"
                                  / "sealed_pair" / short / "manifest.json"),
        }
    for scene_id, paths in spec.items():
        for name, value in paths.items():
            if not Path(value).is_file():
                raise ValueError(f"{scene_id}: missing {name} at {value}")
    path = out / "scene_inputs.json"
    path.write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n")
    return path


def check_ready(questions_doc: dict, key: dict, questions_path: Path) -> None:
    """Refuse to score anything that is not owner-confirmed and hash-pinned."""
    if questions_doc.get("schema") != QUESTIONS_SCHEMA:
        raise ValueError("wrong question-manifest schema")
    if key.get("schema") != KEY_SCHEMA:
        raise ValueError("wrong key schema")
    if questions_doc.get("status") != "OWNER_CONFIRMED":
        raise ValueError(
            f"question manifest status is {questions_doc.get('status')!r}; "
            "real scoring requires OWNER_CONFIRMED")
    if key.get("status") != "OWNER_CONFIRMED":
        raise ValueError(
            f"key status is {key.get('status')!r}; real scoring requires "
            "OWNER_CONFIRMED. A draft key must not produce a score.")
    # Pin the QUESTIONS, not the file. Confirming the manifest edits the file,
    # so a whole-file pin would break exactly when it is first used -- the key
    # would be invalidated by the very act of accepting it.
    content = json_sha256(questions_doc["questions"])
    if key.get("questions_content_sha256") != content:
        raise ValueError(
            "key does not pin these questions "
            f"(expected {content[:16]}..., got "
            f"{str(key.get('questions_content_sha256'))[:16]}...)")
    answered = {a["id"] for a in key["human_relation_truth"]}
    asked = {q["id"] for q in questions_doc["questions"]}
    if answered != asked:
        raise ValueError("key must answer exactly the asked questions")


def run(questions_path: Path, key_path: Path, scene_inputs_path: Path,
        response_path: Path | None, packets_path: Path | None,
        out: Path) -> Path:
    questions_doc = json.loads(questions_path.read_text())
    key = json.loads(key_path.read_text())
    check_ready(questions_doc, key, questions_path)

    questions = questions_doc["questions"]
    scenes = load_scene_inputs(json.loads(scene_inputs_path.read_text()))
    synonyms = questions_doc.get("object_synonyms", {})
    # The merge step already dropped every unresolved mapping and kept the
    # reason under `unresolved_mappings`, so this consumes {scene: {object:
    # uid}} directly. Re-deriving it here would be a second place for the
    # none/missing and ambiguous outcomes to be silently reinterpreted.
    uid_mappings = key.get("uid_mappings", {})
    for scene_id, mapping in uid_mappings.items():
        if not isinstance(mapping, dict):
            raise ValueError(
                f"{scene_id}: uid_mappings must be a resolved object->uid map; "
                "run the review kit's `merge` command on the per-scene returns")

    ceiling = grade(answer_geometry_ceiling(questions, scenes, uid_mappings),
                    key, questions)
    delivered = grade(answer_delivered_graph(questions, scenes, synonyms),
                      key, questions)
    arms = {
        "geometry_relation_ceiling": {
            "layer_kind": "ceiling", "deployable": False,
            "warning": "consumes human-verified identity; never quote as "
                       "deployable system performance",
            "rows": ceiling, "summary": summarize(ceiling),
            "per_scene": per_scene(ceiling),
        },
        "delivered_graph": {
            "layer_kind": "delivered", "deployable": True,
            "rows": delivered, "summary": summarize(delivered),
            "per_scene": per_scene(delivered),
        },
    }

    rgb = hybrid = None
    if response_path is not None and packets_path is not None:
        packets = json.loads(packets_path.read_text())
        response = json.loads(response_path.read_text())
        rgb = grade(answer_blinded_rgb(questions, packets, response), key, questions)
        hybrid = grade(answer_hybrid(questions, delivered, rgb), key, questions)
        arms["blinded_rgb_vlm"] = {
            "layer_kind": "delivered", "deployable": True,
            "rows": rgb, "summary": summarize(rgb), "per_scene": per_scene(rgb),
            "model": response.get("model"),
        }
        arms["evidence_aware_hybrid"] = {
            "layer_kind": "delivered", "deployable": True,
            "rows": hybrid, "summary": summarize(hybrid),
            "per_scene": per_scene(hybrid),
        }

    scene_ids = sorted({q["scene_id"] for q in questions})
    wins = graph_unique_wins(delivered, rgb) if rgb is not None else []
    report = {
        "schema": SCHEMA,
        "stage": "scored" if rgb is not None else "partial_pending_blinded_rgb",
        "evaluation_only": True,
        "perception_changed": False,
        "scene_ids": scene_ids,
        "relation_under_test": questions_doc["relation_under_test"],
        "near_convention": questions_doc["near_convention"],
        "inputs": {
            "questions": str(questions_path),
            "questions_sha256": sha256(questions_path),
            "key": str(key_path), "key_sha256": sha256(key_path),
            "blinded_responses": str(response_path) if response_path else None,
            "scenes": {s: scenes[s]["provenance"] for s in scenes},
        },
        "arms": arms,
        "attribution": attribution(ceiling, delivered),
        "graph_unique_wins": wins,
        "thin_evidence_subtest": (thin_evidence_slice(rgb, hybrid)
                                  if rgb is not None else
                                  {"status": "pending_blinded_rgb"}),
        "decision": (decision(wins, scene_ids) if rgb is not None
                     else {"status": "pending_blinded_rgb", "proceed": None}),
        "limits": [
            "Two human-reviewed scenes are a screening test, not generalization.",
            "The geometry ceiling consumes human-verified identity and is NOT a "
            "deployable result; it bounds what the representation could express.",
            "Binary and set questions depend on the delivered NEAR threshold, "
            "recorded as a provisional constant; comparative questions do not.",
            "Questions the owner marked ambiguous are excluded from every tally "
            "rather than resolved by the system's own answer.",
            "No arm changes perception, labels, thresholds or the human key.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    return out


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--key", type=Path)
    ap.add_argument("--scene-inputs", type=Path)
    ap.add_argument("--blinded-responses", type=Path)
    ap.add_argument("--packets", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--emit-scene-inputs", action="store_true",
                    help="write scene_inputs.json beside --out and exit; "
                         "scores nothing")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.emit_scene_inputs:
        doc = json.loads(args.questions.read_text())
        args.out.parent.mkdir(parents=True, exist_ok=True)
        print(f"scene inputs -> {write_scene_inputs(doc, args.out.parent)}")
        print("nothing was scored")
        return 0
    if args.key is None:
        parser().error("--key is required unless --emit-scene-inputs is given")
    path = run(args.questions, args.key, args.scene_inputs,
               args.blinded_responses, args.packets, args.out)
    report = json.loads(path.read_text())
    for name, arm in report["arms"].items():
        s = arm["summary"]
        flag = "" if arm["deployable"] else "  [CEILING, NOT DEPLOYABLE]"
        print(f"{name:26s} correct={s['tally']['correct']} "
              f"wrong={s['tally']['wrong']} unanswered={s['tally']['unanswered']} "
              f"accuracy={s['exact_accuracy_all']}{flag}")
    print(f"graph-unique wins: {len(report['graph_unique_wins'])}")
    print(f"report -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
