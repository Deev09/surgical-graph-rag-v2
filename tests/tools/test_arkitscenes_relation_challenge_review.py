#!/usr/bin/env python3
"""Synthetic tests for the relation-challenge review kit.

SYNTHETIC FIXTURES ONLY. These pin the properties that make the returned key
independent of the system being measured:

  - the module never reads a predicted label or a graph edge (AST-enforced);
  - no label is ever paired with a uid on the owner's page;
  - "ambiguous" and "unknown" are always available, so no answer is forced;
  - evidence visibility is recorded per question, making the thin-evidence
    slice natural rather than manufactured;
  - the blinded packet carries no answer, no uid and no expected value;
  - generation is deterministic.
"""
from __future__ import annotations

import ast
import inspect
import json
import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import arkitscenes_relation_challenge_review as mod
from tools.arkitscenes_relation_challenge_review import (
    KEY_SCHEMA,
    QUESTIONS_SCHEMA,
    RESPONSE_SCHEMA,
    anchors_for,
    build_index,
    build_scene_page,
    prompt_text,
    validate_returned,
)

SCENE = "arkitscenes_41069025"


def doc() -> dict:
    return {
        "schema": QUESTIONS_SCHEMA,
        "status": "DRAFT_AWAITING_OWNER",
        "relation_under_test": "NEAR",
        "purpose": "test purpose",
        "near_convention": {
            "statement": "within about one metre between nearest surfaces",
            "not_applied_to": "comparative items use no threshold",
            "declared_confound": "the delivered threshold is also 1.0 m",
        },
        "selection_method": {"not_consulted": "no graph edge was consulted",
                             "anchor_rule": "confident anchors only"},
        "excluded_question_classes": [{"class": "direction", "why": "not expressible"}],
        "object_synonyms": {},
        "scenes": {SCENE: {"description": "a room", "n_questions": 3,
                           "capture_note": "wide frames exist"}},
        "questions": [
            {"id": "q_bin", "scene_id": SCENE, "form": "binary_near",
             "question": "Is the sofa near the rug?", "subject": "sofa",
             "object": "rug", "cross_view": False},
            {"id": "q_cmp", "scene_id": SCENE, "form": "comparative_near",
             "question": "Is the rug closer to the sofa or the counter?",
             "subject": "rug", "reference_a": "sofa", "reference_b": "counter",
             "cross_view": True},
            {"id": "q_set", "scene_id": SCENE, "form": "near_set",
             "question": "Which are near the sofa?", "subject": "sofa",
             "candidate_objects": ["rug", "counter"], "cross_view": False},
        ],
        "owner_review_required": ["confirm mappings"],
        "interpretation_limit": "screening only",
    }


def regions() -> list[dict]:
    return [{"uid": f"obj_{i}", "n_vertices": 100 - i,
             "width_m": 1.0, "depth_m": 1.0, "height_m": 1.0,
             "footprint_m2": 1.0, "underside_above_floor_m": 0.0,
             "ctx": ["data:image/png;base64,AAAA", "data:image/png;base64,BBBB"],
             "iso": "data:image/png;base64,CCCC"} for i in range(3)]


def frames() -> list[dict]:
    return [{"id": f"frame_{i:02d}_1.0", "file": f"f{i}.png",
             "uri": "data:image/jpeg;base64,DDDD"} for i in range(3)]


def page() -> str:
    return build_scene_page(SCENE, doc(), regions(), frames(),
                            {"questions_sha256": "a" * 64})


# ------------------------------------------------------- source-level bans
def test_module_never_reads_a_predicted_label_or_a_graph_edge():
    """AST-enforced. A docstring promise is not a guarantee.

    Asking the owner 'is obj_12 near obj_8?' *because the extractor linked
    them* would launder a model guess into ground truth, so the names that
    would make that possible must not appear in this module at all.
    """
    source = Path(mod.__file__).read_text()
    tree = ast.parse(source)
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    literals = {n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    forbidden = {"display_label", "semantic_hypotheses", "learned_label",
                 "edges", "edge_id", "distance_m", "expected"}
    # Docstrings legitimately discuss what is excluded; strip them first.
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            text = ast.get_docstring(node)
            if text:
                docstrings.add(text)
    literals -= docstrings
    used = (names | literals) & forbidden
    assert not used, f"review kit references system-prediction fields: {used}"


def test_module_never_opens_a_graph_manifest():
    source = Path(mod.__file__).read_text()
    body = "\n".join(line for line in source.splitlines()
                     if not line.strip().startswith("#"))
    assert "sealed_pair" not in body, "the review kit must not reach for the graph"
    assert "graph/manifest" not in body


# --------------------------------------------------------------- the page
def test_no_semantic_word_is_ever_paired_with_a_uid():
    html = page()
    cards = html.split("4 · Delivered regions")[1]
    for word in ("sofa", "rug", "counter", "table", "chair", "bed"):
        assert not re.search(rf"\b{word}\b", cards, re.I), \
            f"region cards must carry no semantic word, found {word!r}"


def test_every_question_offers_ambiguous_and_an_evidence_count():
    html = page()
    for qid in ("q_bin", "q_cmp", "q_set"):
        assert f'data-qid="{qid}"' in html
        assert f'name="ev_{qid}"' in html, f"{qid} must record evidence visibility"
    assert html.count('class="ambiguous"') == 3, \
        "every question must be refusable; none may be forced"
    for value in ("2+", "1", "0"):
        assert f'value="{value}"' in html


def test_region_cards_lead_with_photos_and_degrade_honestly():
    """Real capture photos are the primary evidence, renders are secondary.

    Identifying an instance from texture-free point splats is the exact input
    pathology that produced this scene's label errors -- a sofa read as
    "projector". Asking a human to do the same thing repeats it one level up.
    When no usable photo exists the card must say so rather than quietly
    showing only renders.
    """
    with_photo = [dict(regions()[0], photos=["data:image/jpeg;base64,EEEE"],
                       best_visible_fraction=0.82)]
    html = mod.region_cards(with_photo)
    assert 'alt="capture photo 1"' in html
    assert "unoccluded" in html
    assert html.index("capture photo 1") < html.index("context A"), \
        "photos must come before the renders"

    without = [dict(regions()[0], photos=[])]
    html = mod.region_cards(without)
    assert "No usable capture photo" in html
    assert "mark it ambiguous" in html


def test_mapping_offers_all_four_outcomes():
    html = page()
    for token in ("none_missing", "ambiguous", "overmerged"):
        assert token in html, f"mapping must offer {token}"
    assert '<option value="obj_0">' in html


def test_page_exports_the_key_schema_and_pins_the_manifest():
    html = page()
    assert KEY_SCHEMA in html
    assert "a" * 64 in html, "the export must pin the question manifest hash"
    assert "human_relation_truth" in html and "uid_mappings" in html


def test_page_makes_no_external_request():
    html = page()
    assert "http://" not in html and "https://" not in html
    assert "<link" not in html.lower() and "@import" not in html.lower()


def test_page_carries_no_expected_answer_and_pre_answers_nothing():
    """The risk is a pre-filled value, not the word 'expected' in a disclaimer.

    The page's own note says no expected result appears on it; a naive
    substring ban would fail on that sentence while missing the thing that
    actually matters -- a control that arrives already answered, which would
    anchor the owner instead of asking them.
    """
    html = page()
    prose = re.sub(r'<p class="note">.*?</p>', "", html, flags=re.S).lower()
    for token in ("ground truth", "answer_key", "expected answer",
                  "expected result", "correct answer"):
        assert token not in prose, f"the owner sheet must not hint at {token!r}"

    assert not re.search(r"<input[^>]*\bchecked\b", html), \
        "no answer control may arrive pre-selected"
    assert not re.search(r"<option[^>]*\bselected\b", html), \
        "no mapping may arrive pre-chosen"
    bodies = re.findall(r"<textarea[^>]*>(.*?)</textarea>", html, flags=re.S)
    assert bodies and all(not b.strip() for b in bodies), \
        "notes fields must start empty"


def test_anchors_are_collected_from_every_question_field():
    assert anchors_for(SCENE, doc()) == ["sofa", "rug", "counter"]


def test_generation_is_deterministic():
    assert page() == page()
    d, p = doc(), {"questions_sha256": "a" * 64}
    assert build_index(d, {SCENE: "s.html"}, p) == build_index(d, {SCENE: "s.html"}, p)


# ------------------------------------------------------------ the packet
def packet() -> dict:
    return {
        "schema": "arkitscenes_relation_challenge_packet_v1",
        "scene_id": SCENE,
        "packet_sha256": "p" * 64,
        "questions": [q for q in doc()["questions"]],
        "frames": [{"id": "frame_00_1.0"}, {"id": "frame_01_2.0"}],
    }


def test_prompt_states_the_convention_and_permits_unknown():
    text = prompt_text(packet())
    flat = " ".join(text.split())          # the prompt is hard-wrapped
    assert "within about one metre" in flat
    assert "unknown" in flat
    assert "never appear together in a single frame" in flat, \
        "the prompt must warn that some pairs are not co-visible"
    assert RESPONSE_SCHEMA in text


def test_prompt_carries_no_uid_and_no_expected_answer():
    text = prompt_text(packet())
    assert not re.search(r"\bobj_\d+\b", text)
    for token in ("expected", "correct answer", "ground truth"):
        assert token not in text.lower()


def test_prompt_constrains_each_form_to_its_answer_space():
    text = prompt_text(packet())
    assert "true or false" in text
    assert '"sofa" or "counter"' in text
    assert "JSON array" in text


# --------------------------------------------------------------- validate
def _questions_file(payload: dict) -> Path:
    path = Path(tempfile.mkdtemp()) / "q.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    return path


def good_return(path: Path) -> dict:
    return {
        "schema": KEY_SCHEMA, "scene_id": SCENE, "status": "OWNER_CONFIRMED",
        "questions_sha256": mod.sha256(path),
        "uid_mappings": [{"object": "sofa", "uid": "obj_0"},
                         {"object": "rug", "none_missing": True},
                         {"object": "counter", "ambiguous": True}],
        "human_relation_truth": [
            {"id": "q_bin", "answer": True, "ambiguous": False, "evidence_views": "2+"},
            {"id": "q_cmp", "answer": "sofa", "ambiguous": False, "evidence_views": "1"},
            {"id": "q_set", "answer": ["rug"], "ambiguous": False, "evidence_views": "0"},
        ],
    }


def test_a_well_formed_return_validates():
    path = _questions_file(doc())
    assert validate_returned(doc(), path, good_return(path)) == []


def test_validate_rejects_an_unpinned_or_draft_return():
    path = _questions_file(doc())
    bad = good_return(path)
    bad["questions_sha256"] = "0" * 64
    assert any("pin" in p for p in validate_returned(doc(), path, bad))
    bad = good_return(path)
    bad["status"] = "DRAFT_AWAITING_OWNER"
    assert any("OWNER_CONFIRMED" in p for p in validate_returned(doc(), path, bad))


def test_validate_enforces_each_form_answer_space():
    path = _questions_file(doc())
    bad = good_return(path)
    bad["human_relation_truth"][0]["answer"] = "yes"
    assert any("true or false" in p for p in validate_returned(doc(), path, bad))

    bad = good_return(path)
    bad["human_relation_truth"][1]["answer"] = "fridge"
    assert any("comparative" in p for p in validate_returned(doc(), path, bad))

    bad = good_return(path)
    bad["human_relation_truth"][2]["answer"] = ["rug", "lamp"]
    assert any("non-candidates" in p for p in validate_returned(doc(), path, bad))


def test_validate_requires_evidence_visibility_and_full_coverage():
    path = _questions_file(doc())
    bad = good_return(path)
    bad["human_relation_truth"][0]["evidence_views"] = "many"
    assert any("evidence_views" in p for p in validate_returned(doc(), path, bad))

    bad = good_return(path)
    bad["human_relation_truth"] = bad["human_relation_truth"][:2]
    assert any("expected answers" in p for p in validate_returned(doc(), path, bad))

    bad = good_return(path)
    bad["uid_mappings"] = bad["uid_mappings"][:1]
    assert any("uid_mappings missing" in p for p in validate_returned(doc(), path, bad))


def test_validate_requires_ambiguous_items_to_carry_a_null_answer():
    path = _questions_file(doc())
    bad = good_return(path)
    bad["human_relation_truth"][0]["ambiguous"] = True
    assert any("null answer" in p for p in validate_returned(doc(), path, bad))

    ok = good_return(path)
    ok["human_relation_truth"][0].update(ambiguous=True, answer=None)
    assert validate_returned(doc(), path, ok) == []


def test_an_ambiguous_item_may_omit_the_visibility_judgement():
    """Excluded items are in no tally and in no thin-evidence slice.

    Demanding a view count for a question the owner has just said they cannot
    answer would force them to invent one. A non-ambiguous item still requires
    it, and a bad value is still rejected.
    """
    path = _questions_file(doc())
    ok = good_return(path)
    ok["human_relation_truth"][0].update(ambiguous=True, answer=None,
                                         evidence_views=None)
    assert validate_returned(doc(), path, ok) == []

    bad = good_return(path)
    bad["human_relation_truth"][0].update(ambiguous=True, answer=None,
                                          evidence_views="lots")
    assert any("or null" in p for p in validate_returned(doc(), path, bad))

    still_required = good_return(path)
    still_required["human_relation_truth"][1]["evidence_views"] = None
    assert any("must be one of" in p
               for p in validate_returned(doc(), path, still_required))


def test_validate_reports_every_problem_in_one_pass():
    """One round trip per review, not one per mistake."""
    path = _questions_file(doc())
    bad = good_return(path)
    bad["status"] = "DRAFT"
    bad["questions_sha256"] = "0" * 64
    bad["human_relation_truth"][0]["answer"] = "yes"
    assert len(validate_returned(doc(), path, bad)) >= 3


def main() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
