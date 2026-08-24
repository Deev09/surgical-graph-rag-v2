#!/usr/bin/env python3
"""Synthetic tests for the direct-RGB transfer test.

SYNTHETIC FIXTURES ONLY -- no scene, no frames, no model. These pin the parts
of the procedure that exist to remove the author's discretion: 2-of-3 anchor
agreement, an object-agnostic name-matching rule, mechanical ordering, a fixed
template allocation, and a review sheet that forces nothing.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import arkitscenes_rgb_transfer as mod
from tools.arkitscenes_rgb_transfer import (
    COUNTING_CONVENTION,
    GATE_ACCURACY,
    GATE_COVERAGE,
    score_run,
    has_unique_referent,
    KEY_SCHEMA,
    MIN_ANCHORS,
    MIN_PASSES,
    NEAR_CONVENTION,
    RESPONSE_SCHEMA,
    build_questions,
    merge_passes,
    names_match,
    non_covisible_pairs,
    normalize,
    prompt_text,
)

FRAMES = [f"frame_{i:02d}_0.{i}" for i in range(6)]


def obj(name, frames, count=1, conf="certain"):
    return {"name": name, "frame_ids": frames, "count": count,
            "count_confidence": conf}


def three_passes():
    a = [obj("sofa", FRAMES[:2]), obj("kitchen counter", FRAMES[1:3]),
         obj("lamp", FRAMES[2:4]), obj("bed", FRAMES[3:5]),
         obj("rug", FRAMES[4:6]), obj("kettle", FRAMES[5:]),
         obj("stool", FRAMES[2:5]), obj("ghost", FRAMES[:1])]
    b = [obj("couch-like sofa", FRAMES[:2]), obj("counter", FRAMES[1:3]),
         obj("lamps", FRAMES[2:4]), obj("bed", FRAMES[3:5]),
         obj("rug", FRAMES[4:6]), obj("kettle", FRAMES[5:]),
         obj("stool", FRAMES[2:5])]
    c = [obj("sofa", FRAMES[:2]), obj("counter", FRAMES[1:3]),
         obj("lamp", FRAMES[2:4]), obj("bed", FRAMES[3:5]),
         obj("rug", FRAMES[4:6]), obj("kettle", FRAMES[5:]),
         obj("stool", FRAMES[2:5])]
    return [{"objects": a}, {"objects": b}, {"objects": c}]


# ------------------------------------------------------------ matching rule
def test_name_matching_is_object_agnostic():
    """Shares a head noun and one contains the other -> same object."""
    assert names_match(normalize("kitchen counter"), normalize("Counter"))
    assert names_match(normalize("lamps"), normalize("lamp"))
    # same head, neither contains the other -> two different objects
    assert not names_match(normalize("coffee table"), normalize("dining table"))
    # different head noun -> different objects, however similar the words
    assert not names_match(normalize("bed"), normalize("bed frame"))


def test_no_per_object_synonym_table_exists_in_the_tool():
    """A synonym table written after seeing the passes is where bias enters.

    This asserts against the STRUCTURE rather than against words. Two earlier
    word-list versions failed on `names_match`'s own "kitchen counter" example
    and then on `table {` in the page CSS -- English words are everywhere, and
    a guard that keeps crying wolf gets deleted. What cannot exist is a literal
    mapping an object name to a list of alternative names.
    """
    import ast
    tree = ast.parse(Path(mod.__file__).read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict) or not node.keys:
            continue
        keys_are_words = all(isinstance(k, ast.Constant) and isinstance(k.value, str)
                             for k in node.keys)
        values_are_name_lists = all(
            isinstance(v, (ast.List, ast.Tuple, ast.Set)) and v.elts and all(
                isinstance(e, ast.Constant) and isinstance(e.value, str)
                for e in v.elts) for v in node.values)
        assert not (keys_are_words and values_are_name_lists), (
            f"literal at line {node.lineno} has the shape of a per-object "
            "synonym table")


# --------------------------------------------------------- 2-of-3 agreement
def test_single_pass_sightings_are_discarded():
    merged = merge_passes(three_passes(), FRAMES)
    names = [a["anchor"] for a in merged["admitted"]]
    assert "ghost" not in names
    assert [r["anchor"] for r in merged["rejected_single_pass"]] == ["ghost"]
    assert merged["min_passes_required"] == MIN_PASSES


def test_ordering_is_best_attested_first():
    """Amendment 1: agreement, then observation count, then first appearance.

    Plain first-appearance put a small print seen in three frames by two
    passes at rank 0, where it carried three of eight questions.
    """
    merged = merge_passes(three_passes(), FRAMES)
    keys = [(-a["n_passes"], -len(a["frame_ids"]), a["first_frame_rank"],
             a["anchor"]) for a in merged["admitted"]]
    assert keys == sorted(keys)
    # a 3/3 anchor must outrank a 2/3 one regardless of who appeared first
    passes = three_passes()
    passes[0]["objects"].insert(0, obj("early trinket", [FRAMES[0]]))
    passes[1]["objects"].insert(0, obj("early trinket", [FRAMES[0]]))
    merged = merge_passes(passes, FRAMES)
    assert merged["admitted"][0]["n_passes"] == 3, \
        "a briefly-seen 2/3 anchor must not take rank 0"


def test_frame_sets_union_across_agreeing_passes():
    passes = three_passes()
    passes[0]["objects"][3] = obj("bed", [FRAMES[0]])   # one pass saw it early
    merged = merge_passes(passes, FRAMES)
    bed = next(a for a in merged["admitted"] if a["anchor"] == "bed")
    assert FRAMES[0] in bed["frame_ids"] and FRAMES[3] in bed["frame_ids"], \
        "a pair is non-co-visible only if NO pass saw them together"


# ------------------------------------------------------ template allocation
def test_allocation_is_fixed_and_uses_the_ordered_anchors():
    merged = merge_passes(three_passes(), FRAMES)
    questions, _ = build_questions(merged["admitted"])
    forms = [q["form"] for q in questions]
    assert forms.count("comparative_near") == 4
    assert sum(1 for f in forms if f in {"cardinality", "presence"}) == 3
    top = [a["anchor"] for a in merged["admitted"][:3]]
    assert [q["subject"] for q in questions[:3]] == top


def test_a_hedged_count_degrades_to_a_presence_question():
    """Asking for a number the passes disagreed on would key an ambiguity."""
    passes = three_passes()
    passes[0]["objects"][0] = obj("sofa", FRAMES[:2], count=2, conf="unsure")
    questions, notes = build_questions(merge_passes(passes, FRAMES)["admitted"])
    # position is not assumed: amendment 1 reordered anchors, so find the item
    # about the hedged anchor rather than the first slot.
    # the merged anchor keeps the most specific surface form seen, which here
    # is pass 2's "couch-like sofa" rather than bare "sofa"
    hedged = [q for q in questions if "sofa" in str(q.get("subject", ""))
              and q["form"] in {"presence", "cardinality"}]
    assert hedged, "the hedged anchor should still occupy a presence/cardinality slot"
    assert hedged[0]["form"] == "presence"
    assert hedged[0]["answer_type"] == "boolean"
    assert any("presence question" in n for n in notes)


def test_cross_view_pairs_are_only_genuinely_non_covisible_ones():
    merged = merge_passes(three_passes(), FRAMES)
    anchors = merged["admitted"]
    for i, j in non_covisible_pairs(anchors):
        assert not (set(anchors[i]["frame_ids"]) & set(anchors[j]["frame_ids"]))
    questions, _ = build_questions(anchors)
    for q in questions:
        if q.get("cross_view"):
            assert q["why_cross_view"] == "no supplied frame contains both objects"


def test_no_substitute_pair_is_invented_when_none_exist():
    everywhere = [{"objects": [obj(n, FRAMES) for n in
                   ("a bed", "b bed", "c bed", "d bed", "e bed", "f bed")]}] * 3
    questions, notes = build_questions(merge_passes(everywhere, FRAMES)["admitted"])
    assert not any(q.get("cross_view") for q in questions)
    assert any("no non-co-visible" in n for n in notes)


def test_too_few_anchors_stops_the_test():
    thin = [{"objects": [obj("sofa", FRAMES[:2]), obj("bed", FRAMES[2:4])]}] * 3
    try:
        build_questions(merge_passes(thin, FRAMES)["admitted"])
    except ValueError as exc:
        assert str(MIN_ANCHORS) in str(exc)
    else:
        raise AssertionError("fewer than six anchors must stop the test")


def test_relational_slots_use_only_unique_referent_anchors():
    """Amendment 2: "the cushion" picks out nothing when there are four.

    Presence and cardinality slots are deliberately exempt -- several framed
    pictures is exactly what makes "how many" worth asking.
    """
    passes = three_passes()
    for block in passes:                      # make the first anchor plural
        block["objects"][0] = obj(block["objects"][0]["name"], FRAMES[:2], count=4)
    merged = merge_passes(passes, FRAMES)
    anchors = merged["admitted"]
    plural = {a["anchor"] for a in anchors if not has_unique_referent(a)}
    assert plural, "fixture must contain a multi-instance anchor"

    questions, _ = build_questions(anchors)
    for q in questions:
        if q["form"] not in {"comparative_near", "binary_near"}:
            continue
        used = {q.get(k) for k in ("subject", "object", "reference_a",
                                   "reference_b") if q.get(k)}
        assert not (used & plural), f"{q['id']} names a multi-instance anchor"


def test_unique_referent_needs_unanimous_agreement_on_exactly_one():
    assert has_unique_referent({"counts": [{"count": 1}, {"count": 1}, {"count": 1}]})
    assert not has_unique_referent({"counts": [{"count": 1}, {"count": 2}]})
    assert not has_unique_referent({"counts": [{"count": 2}, {"count": 2}]})
    assert not has_unique_referent({"counts": []})


def test_too_few_unique_referent_anchors_stops_the_test():
    plural = [{"objects": [obj(n, FRAMES, count=3) for n in
               ("a bed", "b bed", "c bed", "d bed", "e bed", "f bed")]}] * 3
    try:
        build_questions(merge_passes(plural, FRAMES)["admitted"])
    except ValueError as exc:
        assert "unique-referent" in str(exc)
    else:
        raise AssertionError("comparative slots need six unique-referent anchors")


# --------------------------------------------------------------- the prompt
def test_prompt_carries_both_conventions_and_permits_unknown():
    merged = merge_passes(three_passes(), FRAMES)
    questions, _ = build_questions(merged["admitted"])
    packet = {"scene_id": "47331972", "packet_sha256": "p" * 64,
              "questions": questions,
              "frames": [{"id": f} for f in FRAMES]}
    text = prompt_text(packet)
    flat = " ".join(text.split())
    assert " ".join(COUNTING_CONVENTION.split()) in flat
    assert "the captured space" in flat
    assert "in this room" not in flat, "amendment 1 removed the room scoping"
    assert " ".join(NEAR_CONVENTION.split()) in flat
    assert "unknown" in flat and RESPONSE_SCHEMA in text
    assert "not the answer itself" in flat, "the outcome-flag fix must carry over"
    assert not re.search(r"\bobj_\d+\b", text)


def test_prompt_exposes_no_expected_answer():
    packet = {"scene_id": "47331972", "packet_sha256": "p" * 64,
              "questions": build_questions(
                  merge_passes(three_passes(), FRAMES)["admitted"])[0],
              "frames": [{"id": f} for f in FRAMES]}
    text = prompt_text(packet).lower()
    for token in ("expected", "correct answer", "ground truth", "annotation"):
        assert token not in text


# ---------------------------------------------------------- the review sheet
def sheet():
    merged = merge_passes(three_passes(), FRAMES)
    questions, _ = build_questions(merged["admitted"])
    frames = [{"id": f, "file": f"frames/{f}.png"} for f in FRAMES[:1]]
    import unittest.mock as m
    with m.patch.object(mod, "jpeg_uri", lambda *a, **k: "data:image/jpeg;base64,AA"):
        return mod.build_sheet("47331972", questions, frames, Path("/tmp"),
                               merged, {"questions_content_sha256": "c" * 64})


def test_sheet_forces_no_answer_and_records_visibility():
    html = sheet()
    assert html.count('class="ambiguous"') == 10
    for value in ("2+", "1", "0"):
        assert f'value="{value}"' in html
    assert not re.search(r"<input[^>]*\bchecked\b", html)
    bodies = re.findall(r"<textarea[^>]*>(.*?)</textarea>", html, flags=re.S)
    assert bodies and all(not b.strip() for b in bodies)


def test_sheet_has_no_uid_mapping_panel_for_a_scene_with_no_entities():
    html = sheet()
    assert not re.search(r"\bobj_\d+\b", html)
    assert "no delivered instances" in html.lower()


def test_sheet_is_offline_and_exports_the_key_schema():
    html = sheet()
    assert "http://" not in html and "https://" not in html
    assert "<link" not in html.lower() and "@import" not in html.lower()
    assert KEY_SCHEMA in html and "c" * 64 in html


def test_sheet_explains_how_questions_were_chosen():
    html = " ".join(sheet().lower().split())   # the sheet prose is hard-wrapped
    assert "at least two of the three named it" in html
    assert "no question was chosen because a system could answer it" in html


def test_sheet_generation_is_deterministic():
    assert sheet() == sheet()


# ------------------------------------------------------------------ scoring
def _score_fixture(model_answers):
    q = build_questions(merge_passes(three_passes(), FRAMES)["admitted"])[0]
    doc = {"scene_id": "s", "questions": q, "questions_content_sha256": "c"}
    packet = {"packet_sha256": "p", "questions": q}
    key = {"questions_content_sha256": "c", "human_truth": [
        {"id": x["id"], "answer": _truth_for(x), "ambiguous": False,
         "evidence_views": "2+"} for x in q]}
    resp = {"packet_sha256": "p", "answers": [
        dict(model_answers(x), id=x["id"]) for x in q]}
    return score_run(doc, key, resp, packet)


def _truth_for(q):
    if q["answer_type"] == "boolean":
        return True
    if q["answer_type"] == "integer":
        return 1
    return q["reference_a"]


def test_an_abstention_is_not_counted_as_correct():
    """A perfect abstainer must fail the accuracy gate, not ace it."""
    r = _score_fixture(lambda q: {"outcome": "unknown", "answer": None,
                                  "confidence": 0.0, "evidence_frame_ids": []})
    assert r["tally"]["correct"] == 0 and r["tally"]["wrong"] == 0
    assert r["exact_accuracy"] == 0.0 and r["answer_coverage"] == 0.0
    assert r["accuracy_when_answered"] is None
    assert not r["all_gates_pass"]
    assert "NO TRANSFER CLAIM" in r["claim"]


def test_all_correct_passes_both_gates_and_licenses_one_sentence():
    r = _score_fixture(lambda q: {"outcome": "answer", "answer": _truth_for(q),
                                  "confidence": 0.9, "evidence_frame_ids": []})
    assert r["exact_accuracy"] == 1.0 and r["answer_coverage"] == 1.0
    assert r["all_gates_pass"]
    assert r["claim"].startswith("Under a fixed procedure")
    assert "transferred to one previously untouched" in r["claim"]


def test_gates_are_the_predeclared_numbers():
    assert (GATE_ACCURACY, GATE_COVERAGE) == (0.60, 0.80)
    r = _score_fixture(lambda q: {"outcome": "answer", "answer": _truth_for(q),
                                  "confidence": 0.9, "evidence_frame_ids": []})
    assert r["gates"]["exact_accuracy"]["required"] == 0.60
    assert r["gates"]["answer_coverage"]["required"] == 0.80


def test_score_refuses_a_key_or_response_that_pins_something_else():
    q = build_questions(merge_passes(three_passes(), FRAMES)["admitted"])[0]
    doc = {"scene_id": "s", "questions": q, "questions_content_sha256": "c"}
    packet = {"packet_sha256": "p", "questions": q}
    key = {"questions_content_sha256": "OTHER", "human_truth": []}
    try:
        score_run(doc, key, {"packet_sha256": "p", "answers": []}, packet)
    except ValueError as exc:
        assert "pin" in str(exc)
    else:
        raise AssertionError("an unpinned key must not be scorable")


def main() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
