#!/usr/bin/env python3
"""Synthetic tests for the precomputed direct-RGB demo page.

SYNTHETIC FIXTURES ONLY. These pin the properties that let the page be shown
to someone without misleading them: it must not compute a metric in the
browser, must not reach the network, must not offer open-ended QA it cannot
do, and must never present a ceiling number as system performance.
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import arkitscenes_rgb_demo as mod
from tools.arkitscenes_rgb_demo import LAYERS, build

SCENE = "arkitscenes_41069025"


def questions_doc() -> dict:
    return {"questions": [
        {"id": "q1", "scene_id": SCENE, "form": "binary_near",
         "question": "Is the sofa near the rug?", "cross_view": False},
        {"id": "q2", "scene_id": SCENE, "form": "comparative_near",
         "question": "Closer to sofa or counter?", "cross_view": True},
    ]}


def row(qid, outcome, answer, **extra):
    base = {"id": qid, "scene_id": SCENE, "form": "binary_near",
            "outcome": outcome, "answer": answer, "expected": True}
    base.update(extra)
    return base


def report() -> dict:
    arms = {}
    for name, _, _, _ in LAYERS:
        rows = [row("q1", "correct", True), row("q2", "unanswered", None)]
        if name == "blinded_rgb_vlm":
            rows = [row("q1", "correct", True, confidence=0.9,
                        evidence_frame_ids=["f0", "f1"], evidence_views="2+"),
                    row("q2", "wrong", False, confidence=0.4,
                        evidence_frame_ids=[], evidence_views="0")]
        arms[name] = {"rows": rows, "summary": {
            "tally": {"correct": 7, "wrong": 2, "unanswered": 1},
            "n_questions_scored": 10, "coverage": 0.9}}
    return {"arms": arms}


def frames() -> dict:
    return {f"{SCENE}/f0": "data:image/jpeg;base64,AAAA",
            f"{SCENE}/f1": "data:image/jpeg;base64,BBBB"}


def page() -> str:
    return build(report(), questions_doc(), frames(),
                 {"q1": ["data:image/jpeg;base64,CCCC"], "q2": None},
                 {"inputs": {"report": "a" * 64}})


# --------------------------------------------------------- honesty of framing
def test_page_declares_itself_a_recorded_replay_not_live_qa():
    html = page()
    assert "recorded evaluation replay" in html.lower()
    assert "arbitrary typed questions are not supported" in html.lower()
    assert "live vision api" in html.lower()


def test_questions_are_a_fixed_list_with_no_free_text_input():
    html = page()
    assert "<select" in html
    assert html.count("<option") == 2
    assert not re.search(r'<input[^>]*type=["\']?text', html, re.I)
    assert "<textarea" not in html


def test_every_ceiling_layer_is_marked_not_deployable():
    html = page()
    for name, title, _, deployable in LAYERS:
        if not deployable:
            assert title in html
    # the two identity-oracle layers must carry the words, not just a flag
    assert html.count("not deployable") >= 2
    assert "human-verified" in html.lower()
    assert "never system performance" in html.lower()


def test_headline_states_the_same_score_on_different_questions():
    html = page().lower()
    assert "on different questions" in html
    assert "0 of 10" in html and "2 of 10" in html


# ------------------------------------------------------- no computation in JS
def test_browser_script_computes_nothing():
    """The page may hide and show; it may not calculate.

    Any arithmetic in the browser is a second, unreviewed implementation of a
    number that already exists in the committed report, and the two can
    disagree without anything failing.
    """
    script = re.search(r"<script>(.*?)</script>", page(), re.S).group(1)
    for token in ("Math.", "reduce", ".map(", "filter", "parseFloat",
                  "parseInt", "Number(", "toFixed", "+", "*", "%"):
        assert token not in script, f"script performs computation: {token!r}"
    assert "innerHTML" not in script, "the script must not author content"
    assert "hidden" in script


def test_page_makes_no_network_request():
    html = page()
    assert "http://" not in html and "https://" not in html
    for token in ("fetch(", "XMLHttpRequest", "WebSocket", "<link", "@import",
                  "api_key", "apiKey", "Authorization"):
        assert token not in html, f"page reaches outward: {token!r}"


# ------------------------------------------------------------- content wiring
def test_answer_state_confidence_frames_views_and_scene_are_all_shown():
    html = page()
    assert "Model confidence" in html and "0.9" in html
    assert "Owner-recorded evidence views" in html
    assert "2+" in html
    assert SCENE in html
    assert "f0" in html and "f1" in html


def test_every_layer_outcome_appears_for_each_question():
    html = page()
    for qid in ("q1", "q2"):
        panel = html.split(f'id="panel-{qid}"')[1].split("</article>")[0]
        for _, title, _, _ in LAYERS:
            assert title in panel, f"{qid} panel is missing layer {title}"


def test_a_question_with_no_cited_frames_says_so():
    html = page()
    panel = html.split('id="panel-q2"')[1].split("</article>")[0]
    assert "No frames cited" in panel


def test_3d_layer_is_labelled_evidence_not_the_answer_engine():
    html = page()
    assert "not the answer engine" in html.lower()
    panel = html.split('id="panel-q2"')[1].split("</article>")[0]
    assert "geometry ceiling abstained" in panel


def test_generation_is_deterministic():
    assert page() == page()


def test_input_hashes_are_disclosed():
    assert "a" * 64 in page()


def main() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
