#!/usr/bin/env python3
"""Synthetic tests for the four-layer relation-challenge scorer.

SYNTHETIC FIXTURES ONLY. No real scene, no real key, no real score. The point
of these tests is to pin the separations that make the four-layer reading
trustworthy before any real answer exists:

  - deployable layers cannot read the human key (AST-enforced, not promised);
  - a draft key cannot produce a score;
  - the key must pin the exact question manifest it answers;
  - a ceiling result is flagged non-deployable wherever it is reported;
  - "no resolvable UID" is a distinct outcome from "geometry got it wrong".
"""
from __future__ import annotations

import ast
import inspect
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry.surface_distance import aabb_to_aabb_surface
from tools import arkitscenes_relation_challenge_score as mod
from tools.arkitscenes_relation_challenge_score import (
    KEY_SCHEMA,
    NO_DISTANCE,
    NO_LABEL,
    NO_EXHAUSTIVE_SET,
    NO_MAPPING,
    QUESTIONS_SCHEMA,
    RESPONSE_SCHEMA,
    THIN_EVIDENCE,
    answer_blinded_rgb,
    answer_delivered_graph,
    answer_geometry_ceiling,
    answer_hybrid,
    attribution,
    answer_stored_graph_human_identity,
    check_ready,
    decision,
    grade,
    graph_unique_wins,
    layer_agreement,
    summarize,
    thin_evidence_slice,
)

SCENE = "scene_a"


def box(x0, x1):
    """Unit-ish box spanning [x0,x1] on x, fixed on y and z."""
    return [[x0, 0.0, 0.0], [x1, 1.0, 1.0]]


def questions() -> list[dict]:
    return [
        {"id": "q1", "scene_id": SCENE, "form": "binary_near",
         "subject": "sofa", "object": "table"},
        {"id": "q2", "scene_id": SCENE, "form": "comparative_near",
         "subject": "sofa", "reference_a": "table", "reference_b": "fridge"},
        {"id": "q3", "scene_id": SCENE, "form": "near_set",
         "subject": "sofa", "candidate_objects": ["table", "fridge"]},
    ]


def scenes() -> dict:
    # sofa [0,1]; table [1.5,2] -> gap 0.5 (near); fridge [5,6] -> gap 4.0 (far)
    aabb = {"obj_0": box(0.0, 1.0), "obj_1": box(1.5, 2.0), "obj_2": box(5.0, 6.0)}
    entities = [
        {"uid": "obj_0", "display_label": "sofa"},
        {"uid": "obj_1", "display_label": "table"},
        {"uid": "obj_2", "display_label": "fridge"},
    ]
    return {SCENE: {
        "aabb_by_uid": aabb,
        "label_index": mod._label_index(entities),
        # Only the near pair is stored, exactly as the extractor would: it
        # emits an edge iff surface distance < threshold.
        "edge_distance": {frozenset(("obj_0", "obj_1")): 0.5},
        "near_threshold_m": 1.0,
    }}


def mappings() -> dict:
    return {SCENE: {"sofa": "obj_0", "table": "obj_1", "fridge": "obj_2"}}


def key(ambiguous: bool = False, views: str = "2+") -> dict:
    return {
        "schema": KEY_SCHEMA,
        "status": "OWNER_CONFIRMED",
        "human_relation_truth": [
            {"id": "q1", "answer": True, "ambiguous": ambiguous, "evidence_views": views},
            {"id": "q2", "answer": "table", "ambiguous": False, "evidence_views": views},
            {"id": "q3", "answer": ["table"], "ambiguous": False, "evidence_views": views},
        ],
    }


# ---------------------------------------------------------------- separation
def test_deployable_layers_never_bind_the_human_key():
    """AST-enforced: a deployable answer function cannot consult the key.

    Promising it in a docstring is not enforcement. If someone later threads
    the key into the delivered path to 'fix' a resolution failure, this fails.
    """
    forbidden = {"key", "expected", "human_relation_truth", "uid_mappings",
                 "truth", "answer_key"}
    for fn in (answer_delivered_graph, answer_blinded_rgb, answer_hybrid,
               mod._stored_edge_rows):
        tree = ast.parse(inspect.getsource(fn))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        args = {a.arg for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) for a in n.args.args}
        leaked = (names | args) & forbidden
        assert not leaked, f"{fn.__name__} references human-key names: {leaked}"


def test_only_grade_reads_expected_answers():
    """Two functions may touch the truth block, and only one may read answers.

    `check_ready` reads it to verify the key answers exactly the asked
    questions -- ids only. `grade` is the only function permitted to look at
    an expected answer. Asserting a loose subset here would let a future edit
    read answers inside the validator without failing anything.
    """
    tree = ast.parse(inspect.getsource(mod))
    readers = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
               and "human_relation_truth" in ast.dump(n)]
    assert sorted(readers) == ["check_ready", "grade"], \
        f"unexpected reader of human_relation_truth: {readers}"

    validator = next(n for n in ast.walk(tree)
                     if isinstance(n, ast.FunctionDef) and n.name == "check_ready")
    literals = {n.value for n in ast.walk(validator)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert "answer" not in literals, \
        "check_ready must validate ids only; it must never read an expected answer"


# ------------------------------------------------------------------- ceiling
def test_ceiling_uses_the_extractors_own_distance_function():
    rows = answer_geometry_ceiling(questions(), scenes(), mappings())
    binary = next(r for r in rows if r["id"] == "q1")
    expected = aabb_to_aabb_surface(box(0.0, 1.0), box(1.5, 2.0))
    assert binary["distance_m"] == round(expected, 4)
    assert abs(expected - 0.5) < 1e-9
    assert binary["answer"] is True


def test_ceiling_abstains_without_a_human_uid_mapping():
    rows = answer_geometry_ceiling(questions(), scenes(), {SCENE: {"sofa": "obj_0"}})
    for row in rows:
        assert row["outcome_hint"] == "unanswered"
    assert {r["reason"] for r in rows if r["form"] != "near_set"} == {NO_MAPPING}


def test_set_questions_abstain_rather_than_return_a_short_set():
    """An exhaustive-set answer missing an unresolvable member is a false negative.

    Returning the resolvable subset would look like a confident 'nothing else
    is near it', which is a wrong answer wearing the costume of a right one.
    """
    partial = {SCENE: {"sofa": "obj_0", "table": "obj_1"}}   # fridge unresolved
    ceiling = answer_geometry_ceiling(questions(), scenes(), partial)
    row = next(r for r in ceiling if r["form"] == "near_set")
    assert row["outcome_hint"] == "unanswered"
    assert row["reason"] == NO_EXHAUSTIVE_SET
    assert row["unresolved_candidates"] == ["fridge"]

    scene = scenes()
    scene[SCENE]["label_index"] = mod._label_index([
        {"uid": "obj_0", "display_label": "sofa"},
        {"uid": "obj_1", "display_label": "table"}])
    delivered = answer_delivered_graph(questions(), scene, {})
    row = next(r for r in delivered if r["form"] == "near_set")
    assert row["outcome_hint"] == "unanswered"
    assert row["unresolved_candidates"] == ["fridge"]


def test_ceiling_comparative_needs_no_threshold():
    rows = answer_geometry_ceiling(questions(), scenes(), mappings())
    comparative = next(r for r in rows if r["id"] == "q2")
    assert comparative["answer"] == "table"
    assert "threshold_m" not in comparative


def test_ceiling_is_flagged_non_deployable():
    kinds = {name: kind for name, kind, _ in mod.LAYERS}
    deployable = {name: flag for name, _, flag in mod.LAYERS}
    assert kinds["geometry_relation_ceiling"] == "ceiling"
    assert deployable["geometry_relation_ceiling"] is False
    assert all(deployable[n] for n in ("delivered_graph", "blinded_rgb_vlm",
                                       "evidence_aware_hybrid"))


# ---------------------------------------------------------- delivered graph
def test_delivered_graph_reads_edge_presence_not_geometry():
    rows = answer_delivered_graph(questions(), scenes(), {})
    binary = next(r for r in rows if r["id"] == "q1")
    assert binary["answer"] is True and binary["edge_present"] is True
    near_set = next(r for r in rows if r["id"] == "q3")
    assert near_set["answer"] == ["table"]


def test_delivered_graph_orders_when_one_edge_is_beyond_threshold():
    """Absence is a measurement, not missing data: every pair is evaluated."""
    rows = answer_delivered_graph(questions(), scenes(), {})
    comparative = next(r for r in rows if r["id"] == "q2")
    assert comparative["answer"] == "table"
    assert comparative["ordering_basis"] == "one_edge_stored_one_beyond_threshold"


def test_delivered_graph_abstains_when_neither_pair_has_a_distance():
    scene = scenes()
    scene[SCENE]["edge_distance"] = {}
    rows = answer_delivered_graph(questions(), scene, {})
    comparative = next(r for r in rows if r["id"] == "q2")
    assert comparative["outcome_hint"] == "unanswered"
    assert comparative["reason"] == NO_DISTANCE


def test_delivered_graph_abstains_on_ambiguous_or_absent_labels():
    scene = scenes()
    duplicated = [
        {"uid": "obj_0", "display_label": "sofa"},
        {"uid": "obj_9", "display_label": "sofa"},
        {"uid": "obj_1", "display_label": "table"},
    ]
    scene[SCENE]["label_index"] = mod._label_index(duplicated)
    rows = answer_delivered_graph(questions(), scene, {})
    assert all(r["reason"] == NO_LABEL for r in rows)

    anonymous = [{"uid": "obj_0", "display_label": "segment_0"},
                 {"uid": "obj_1", "display_label": "table"}]
    scene[SCENE]["label_index"] = mod._label_index(anonymous)
    rows = answer_delivered_graph(questions(), scene, {})
    assert all(r["reason"] == NO_LABEL for r in rows)


def test_delivered_graph_honours_declared_synonyms():
    scene = scenes()
    scene[SCENE]["label_index"] = mod._label_index([
        {"uid": "obj_0", "display_label": "couch"},
        {"uid": "obj_1", "display_label": "table"},
        {"uid": "obj_2", "display_label": "fridge"}])
    rows = answer_delivered_graph(questions(), scene, {"sofa": ["sofa", "couch"]})
    assert next(r for r in rows if r["id"] == "q1")["answer"] is True


# ------------------------------------------------------------------- hybrid
def packets() -> dict:
    return {SCENE: {"packet_sha256": "pkt", "frames": [
        {"id": "f0"}, {"id": "f1"}, {"id": "f2"}]}}


def response(cited=("f0", "f1")) -> dict:
    return {
        "schema": RESPONSE_SCHEMA,
        "model": {"provider": "p", "name": "n", "version": "v"},
        "packet_sha256": {SCENE: "pkt"},
        "answers": [
            {"id": "q1", "outcome": "answer", "answer": False, "confidence": 0.5,
             "evidence_frame_ids": list(cited)},
            {"id": "q2", "outcome": "answer", "answer": "fridge", "confidence": 0.5,
             "evidence_frame_ids": list(cited)},
            {"id": "q3", "outcome": "unknown", "answer": None, "confidence": 0.1,
             "evidence_frame_ids": []},
        ],
    }


def test_hybrid_routes_to_graph_when_the_graph_materializes_a_fact():
    q = questions()
    graph = answer_delivered_graph(q, scenes(), {})
    rgb = answer_blinded_rgb(q, packets(), response())
    rows = answer_hybrid(q, graph, rgb)
    assert all(r["route"] == "typed_relation" for r in rows)
    assert next(r for r in rows if r["id"] == "q1")["answer"] is True


def test_hybrid_falls_back_to_rgb_and_gates_thin_evidence():
    q = questions()
    scene = scenes()
    scene[SCENE]["label_index"] = {}          # graph cannot resolve anything
    graph = answer_delivered_graph(q, scene, {})
    rgb = answer_blinded_rgb(q, packets(), response(cited=("f0",)))
    rows = answer_hybrid(q, graph, rgb)
    assert all(r["route"] == "visual_evidence" for r in rows)
    answered = [r for r in rows if r["id"] in {"q1", "q2"}]
    assert all(r["outcome_hint"] == "unanswered" for r in answered)
    assert all(r["reason"] == THIN_EVIDENCE for r in answered)


def test_hybrid_never_invents_a_graph_fact():
    q = questions()
    scene = scenes()
    scene[SCENE]["label_index"] = {}
    graph = answer_delivered_graph(q, scene, {})
    rgb = answer_blinded_rgb(q, packets(), response())
    rows = answer_hybrid(q, graph, rgb)
    for row in rows:
        assert row["answered_by"] != "delivered_graph"
        assert row["graph_abstained_because"] == NO_LABEL


def test_blinded_response_must_pin_each_scene_packet():
    bad = response()
    bad["packet_sha256"] = {SCENE: "other"}
    try:
        answer_blinded_rgb(questions(), packets(), bad)
    except ValueError as exc:
        assert "packet" in str(exc)
    else:
        raise AssertionError("unpinned blinded response must not be accepted")


def test_blinded_response_rejects_invalid_frame_citations():
    bad = response(cited=("f0", "not_a_frame"))
    try:
        answer_blinded_rgb(questions(), packets(), bad)
    except ValueError as exc:
        assert "frame" in str(exc)
    else:
        raise AssertionError("invalid frame id must not be accepted")


# ------------------------------------------------------------------ grading
def test_grade_excludes_questions_the_owner_marked_ambiguous():
    q = questions()
    rows = grade(answer_geometry_ceiling(q, scenes(), mappings()),
                 key(ambiguous=True), q)
    excluded = next(r for r in rows if r["id"] == "q1")
    assert excluded["outcome"] == "excluded_no_human_answer"
    assert summarize(rows)["n_excluded_no_human_answer"] == 1
    assert summarize(rows)["n_questions_scored"] == 2


def test_grade_matches_each_form_by_its_own_rule():
    q = questions()
    rows = grade(answer_geometry_ceiling(q, scenes(), mappings()), key(), q)
    assert {r["id"]: r["outcome"] for r in rows} == \
        {"q1": "correct", "q2": "correct", "q3": "correct"}


def test_set_comparison_is_order_insensitive_but_membership_strict():
    q = [{"id": "q3", "scene_id": SCENE, "form": "near_set",
          "subject": "sofa", "candidate_objects": ["table", "fridge"]}]
    k = {"schema": KEY_SCHEMA, "status": "OWNER_CONFIRMED",
         "human_relation_truth": [{"id": "q3", "answer": ["TABLE"],
                                   "ambiguous": False, "evidence_views": "2+"}]}
    rows = grade(answer_geometry_ceiling(q, scenes(), mappings()), k, q)
    assert rows[0]["outcome"] == "correct"
    k["human_relation_truth"][0]["answer"] = ["table", "fridge"]
    rows = grade(answer_geometry_ceiling(q, scenes(), mappings()), k, q)
    assert rows[0]["outcome"] == "wrong"


# ------------------------------------------------------------- cross-layer
def test_graph_unique_wins_counts_delivered_only_and_needs_rgb_to_miss():
    graph = [{"id": "q1", "scene_id": SCENE, "form": "binary_near",
              "outcome": "correct", "answer": True}]
    rgb_wrong = [{"id": "q1", "outcome": "wrong", "answer": False}]
    rgb_right = [{"id": "q1", "outcome": "correct", "answer": True}]
    assert len(graph_unique_wins(graph, rgb_wrong)) == 1
    assert graph_unique_wins(graph, rgb_right) == []


def test_attribution_separates_unresolvable_identity_from_wrong_geometry():
    ceiling = [
        {"id": "q1", "outcome": "unanswered", "reason": NO_MAPPING},
        {"id": "q2", "outcome": "wrong"},
        {"id": "q3", "outcome": "correct"},
    ]
    delivered = [
        {"id": "q1", "outcome": "wrong"},
        {"id": "q2", "outcome": "wrong"},
        {"id": "q3", "outcome": "wrong"},
    ]
    buckets = attribution(ceiling, delivered)["buckets"]
    assert buckets["ceiling_unanswerable_no_uid_mapping"] == ["q1"]
    assert buckets["ceiling_wrong"] == ["q2"]
    assert buckets["ceiling_correct_delivered_wrong"] == ["q3"]


def test_thin_evidence_slice_is_empty_when_the_owner_saw_two_views_everywhere():
    rows = [{"id": "q1", "scene_id": SCENE, "outcome": "correct",
             "evidence_views": "2+"}]
    assert thin_evidence_slice(rows, rows)["status"] == "empty"
    assert thin_evidence_slice(rows, rows)["n_thin"] == 0


def test_thin_evidence_slice_reports_both_sides_of_the_gate():
    rgb = [
        {"id": "q1", "scene_id": SCENE, "outcome": "wrong", "evidence_views": "1"},
        {"id": "q2", "scene_id": SCENE, "outcome": "correct", "evidence_views": "1"},
    ]
    hybrid = [
        {"id": "q1", "scene_id": SCENE, "outcome": "unanswered", "evidence_views": "1"},
        {"id": "q2", "scene_id": SCENE, "outcome": "unanswered", "evidence_views": "1"},
    ]
    out = thin_evidence_slice(rgb, hybrid)
    assert out["wrong_answers_suppressed"] == ["q1"]
    assert out["correct_answers_suppressed"] == ["q2"], \
        "suppressing a correct answer is a cost and must be reported"


def test_decision_bar_requires_two_wins():
    win = {"id": "q1", "scene_id": "a", "form": "binary_near"}
    assert decision([win], ["a", "b"])["meets_bar"] is False
    two_same = decision([win, dict(win, id="q2")], ["a", "b"])
    assert two_same["meets_bar"] is True
    assert two_same["reproduced_across_scenes"] is False
    both = decision([win, {"id": "q2", "scene_id": "b", "form": "binary_near"}],
                    ["a", "b"])
    assert both["reproduced_across_scenes"] is True


# --------------------------------------------------------------- readiness
def _write(tmp: Path, name: str, payload: dict) -> Path:
    path = tmp / name
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    return path


def test_a_draft_key_cannot_produce_a_score():
    tmp = Path(tempfile.mkdtemp())
    doc = {"schema": QUESTIONS_SCHEMA, "status": "DRAFT_AWAITING_OWNER",
           "questions": questions()}
    path = _write(tmp, "q.json", doc)
    k = key()
    k["questions_content_sha256"] = mod.json_sha256(doc["questions"])
    try:
        check_ready(doc, k, path)
    except ValueError as exc:
        assert "OWNER_CONFIRMED" in str(exc)
    else:
        raise AssertionError("a draft question manifest must not be scorable")

    doc["status"] = "OWNER_CONFIRMED"
    path = _write(tmp, "q2.json", doc)
    draft_key = key()
    draft_key["status"] = "DRAFT_AWAITING_OWNER"
    draft_key["questions_content_sha256"] = mod.json_sha256(doc["questions"])
    try:
        check_ready(doc, draft_key, path)
    except ValueError as exc:
        assert "OWNER_CONFIRMED" in str(exc)
    else:
        raise AssertionError("a draft key must not be scorable")


def test_key_must_pin_the_exact_question_manifest():
    tmp = Path(tempfile.mkdtemp())
    doc = {"schema": QUESTIONS_SCHEMA, "status": "OWNER_CONFIRMED",
           "questions": questions()}
    path = _write(tmp, "q.json", doc)
    k = key()
    k["questions_content_sha256"] = "0" * 64
    try:
        check_ready(doc, k, path)
    except ValueError as exc:
        assert "pin" in str(exc)
    else:
        raise AssertionError("an unpinned key must not be scorable")


def test_key_must_answer_exactly_the_asked_questions():
    tmp = Path(tempfile.mkdtemp())
    doc = {"schema": QUESTIONS_SCHEMA, "status": "OWNER_CONFIRMED",
           "questions": questions()}
    path = _write(tmp, "q.json", doc)
    k = key()
    k["questions_content_sha256"] = mod.json_sha256(doc["questions"])
    k["human_relation_truth"] = k["human_relation_truth"][:2]
    try:
        check_ready(doc, k, path)
    except ValueError as exc:
        assert "exactly" in str(exc)
    else:
        raise AssertionError("a partial key must not be scorable")


def test_confirming_the_manifest_does_not_break_the_pin():
    """The pin is over the questions, not the file.

    A whole-file pin breaks exactly when it is first used: accepting a key
    means flipping the manifest to OWNER_CONFIRMED, which edits the file and
    invalidates every key that pinned it. This caught that for real.
    """
    tmp = Path(tempfile.mkdtemp())
    doc = {"schema": QUESTIONS_SCHEMA, "status": "OWNER_CONFIRMED",
           "questions": questions()}
    path = _write(tmp, "q.json", doc)
    k = key()
    k["questions_content_sha256"] = mod.json_sha256(doc["questions"])
    check_ready(doc, k, path)                       # baseline: accepted

    # Editorial change only -- status, notes, a confirmation date.
    doc["confirmed"] = "2026-08-17"
    doc["pin_note"] = "questions unchanged"
    path = _write(tmp, "q_confirmed.json", doc)
    assert mod.sha256(path) != mod.sha256(_write(tmp, "q.json", {
        "schema": QUESTIONS_SCHEMA, "status": "OWNER_CONFIRMED",
        "questions": questions()})), "the file hash must actually have changed"
    check_ready(doc, k, path)                       # same key still accepted

    # Changing a QUESTION must still break the pin.
    doc["questions"][0]["object"] = "fridge"
    path = _write(tmp, "q_edited.json", doc)
    try:
        check_ready(doc, k, path)
    except ValueError as exc:
        assert "pin" in str(exc)
    else:
        raise AssertionError("editing a question must invalidate the key")


def test_answers_are_deterministic():
    q, s, m = questions(), scenes(), mappings()
    assert answer_geometry_ceiling(q, s, m) == answer_geometry_ceiling(q, s, m)
    assert answer_delivered_graph(q, s, {}) == answer_delivered_graph(q, s, {})



# ------------------------------------------------- identity-oracle stored graph
def test_stored_graph_layer_recomputes_no_geometry():
    """The whole point of this layer is that it reads only what was serialized.

    If it recomputed aabb_to_aabb_surface it would silently become a second
    copy of the geometry ceiling, and the agreement between them -- which is
    the evidence that relation extraction is cleared -- would be circular.
    """
    for fn in (answer_stored_graph_human_identity, mod._stored_edge_rows):
        tree = ast.parse(inspect.getsource(fn))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert "aabb_to_aabb_surface" not in names, \
            f"{fn.__name__} must not recompute geometry"
        assert "aabb_by_uid" not in names or fn is answer_stored_graph_human_identity, \
            "only the resolver may touch the uid table, and only to check membership"


def test_stored_graph_layer_uses_human_identity_and_stored_edges():
    rows = answer_stored_graph_human_identity(questions(), scenes(), mappings())
    binary = next(r for r in rows if r["id"] == "q1")
    assert binary["answer"] is True and binary["edge_present"] is True
    assert binary["stored_distance_m"] == 0.5      # the serialized value, verbatim
    assert "distance_m" not in binary              # not a recomputed one
    comparative = next(r for r in rows if r["id"] == "q2")
    assert comparative["ordering_basis"] == "one_edge_stored_one_beyond_threshold"


def test_stored_graph_layer_abstains_without_a_human_mapping():
    rows = answer_stored_graph_human_identity(questions(), scenes(),
                                              {SCENE: {"sofa": "obj_0"}})
    assert {r["reason"] for r in rows if r["form"] != "near_set"} == {NO_MAPPING}


def test_identity_oracle_is_not_deployable():
    kinds = {name: kind for name, kind, _ in mod.LAYERS}
    deployable = {name: flag for name, _, flag in mod.LAYERS}
    assert kinds["stored_graph_human_identity"] == "identity_oracle"
    assert deployable["stored_graph_human_identity"] is False


def test_layer_agreement_flags_a_divergence():
    ceiling = [{"id": "q1", "scene_id": SCENE, "form": "binary_near",
                "outcome": "correct", "answer": True}]
    same = [{"id": "q1", "scene_id": SCENE, "form": "binary_near",
             "outcome": "correct", "answer": True}]
    assert layer_agreement(ceiling, same)["n_disagree"] == 0
    differs = [{"id": "q1", "scene_id": SCENE, "form": "binary_near",
                "outcome": "unanswered", "answer": None, "reason": NO_DISTANCE}]
    out = layer_agreement(ceiling, differs)
    assert out["n_disagree"] == 1
    assert out["disagreements"][0]["stored_reason"] == NO_DISTANCE


# ------------------------------------------------------------- attribution
def test_attribution_buckets_partition_every_item():
    """A bucket set that does not add up reads as a measured zero.

    The first version dropped every ceiling abstention whose reason was not
    NO_MAPPING, so it reported zero unanswerable items while the ceiling
    summary showed two.
    """
    ceiling = [
        {"id": "q1", "outcome": "unanswered", "reason": NO_EXHAUSTIVE_SET},
        {"id": "q2", "outcome": "unanswered", "reason": NO_MAPPING},
        {"id": "q3", "outcome": "unanswered", "reason": NO_DISTANCE},
        {"id": "q4", "outcome": "excluded_no_human_answer"},
        {"id": "q5", "outcome": "correct"},
    ]
    delivered = [{"id": f"q{i}", "outcome": "unanswered"} for i in range(1, 6)]
    out = attribution(ceiling, delivered)
    b = out["buckets"]
    assert b["ceiling_unanswerable_no_exhaustive_set"] == ["q1"]
    assert b["ceiling_unanswerable_no_uid_mapping"] == ["q2"]
    assert b["ceiling_unanswerable_other"] == ["q3"]
    assert b["excluded_no_human_answer"] == ["q4"]
    assert out["n_bucketed"] == out["n_items"] == 5


def test_attribution_refuses_to_emit_a_partial_partition():
    ceiling = [{"id": "q1", "outcome": "correct"}]
    delivered = [{"id": "q1", "outcome": "correct"},
                 {"id": "q_orphan", "outcome": "wrong"}]
    try:
        attribution(ceiling, delivered)
    except ValueError as exc:
        assert "partition" in str(exc)
    else:
        raise AssertionError("an incomplete partition must raise, not under-report")



def test_outcome_flag_is_derived_when_the_response_restates_the_answer():
    """Repairs our own ambiguous spec, never the model's answers.

    The packet asked for `"outcome": "answer or unknown"`, which reads as "put
    the answer, or unknown". Responses took that reading. The flag is fully
    determined by whether `answer` is null, so it is derived and the returned
    value is preserved -- while genuine contradictions still raise.
    """
    r = response()
    r["answers"][0]["outcome"] = "false"        # restated the boolean answer
    r["answers"][1]["outcome"] = "fridge"       # restated the comparative
    rows = answer_blinded_rgb(questions(), packets(), r)
    binary = next(x for x in rows if x["id"] == "q1")
    assert binary["outcome_hint"] == "answered"
    assert binary["answer"] is False            # untouched
    assert binary["outcome_as_returned"] == "false"
    assert binary["outcome_normalized"] is True

    contradiction = response()
    contradiction["answers"][0]["outcome"] = "unknown"   # but answer is False
    try:
        answer_blinded_rgb(questions(), packets(), contradiction)
    except ValueError as exc:
        assert "non-null" in str(exc)
    else:
        raise AssertionError("unknown with a real answer must still raise")

    empty = response()
    empty["answers"][2]["outcome"] = "answer"    # but answer is None
    try:
        answer_blinded_rgb(questions(), packets(), empty)
    except ValueError as exc:
        assert "null answer" in str(exc)
    else:
        raise AssertionError("answer with a null value must still raise")


def test_unique_wins_is_directional():
    a = [{"id": "q1", "scene_id": SCENE, "form": "binary_near",
          "outcome": "correct", "answer": True}]
    b = [{"id": "q1", "scene_id": SCENE, "form": "binary_near",
          "outcome": "unanswered", "answer": None, "reason": NO_MAPPING}]
    assert len(mod.unique_wins(a, b)) == 1
    assert mod.unique_wins(b, a) == []
    assert mod.unique_wins(a, b)[0]["other_reason"] == NO_MAPPING


def test_blinded_responses_merge_across_scenes():
    import tempfile as _tf
    tmp = Path(_tf.mkdtemp())
    one, two = response(), response()
    two["packet_sha256"] = {"scene_b": "pkt_b"}
    two["answers"] = [dict(a, id=a["id"] + "_b") for a in two["answers"]]
    p1, p2 = tmp / "a.json", tmp / "b.json"
    p1.write_text(json.dumps(one)); p2.write_text(json.dumps(two))
    merged = mod.merge_blinded([p1, p2])
    assert merged["packet_sha256"] == {SCENE: "pkt", "scene_b": "pkt_b"}
    assert len(merged["answers"]) == 6

    two["model"] = {"provider": "other", "name": "n", "version": "v"}
    p2.write_text(json.dumps(two))
    try:
        mod.merge_blinded([p1, p2])
    except ValueError as exc:
        assert "disagree on the model" in str(exc)
    else:
        raise AssertionError("mixed models must not merge silently")


def main() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
