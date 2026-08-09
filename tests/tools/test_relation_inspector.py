"""Guards for the relation inspector's offline and fidelity properties.

The inspector's value is that it shows the stored evidence verbatim. Two
ways that quietly breaks: the page starts fetching something (so it stops
being a self-contained artifact you can open anywhere), or the payload
starts dropping evidence keys (so the panel shows a tidier story than the
extractor actually recorded).

Dataset-dependent tests skip cleanly when Replica is not present.
"""
from __future__ import annotations

import json
import re
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import relation_inspector as RI

SCENE = "replica_room_2"


def _dataset_ready() -> bool:
    try:
        return RI._room_dir(SCENE).is_dir()
    except Exception:
        return False


def _build():
    from demo.question_battery import _runs
    from demo.replica_habitat_import import import_habitat_room
    from graph.builder import build_graph
    arts = import_habitat_room(RI._room_dir(SCENE), SCENE)
    return build_graph(arts, _runs(), density_policy="phase2_telemetry_only")


def test_template_makes_no_external_requests() -> None:
    """No CDN, no fonts, no analytics, no <img src>. Checked on the template
    itself so it holds even without the dataset."""
    t = RI._TEMPLATE
    for bad in ("http://", "https://", "//cdn", "<script src", "<link ",
                "@import", "fetch(", "XMLHttpRequest", "WebSocket"):
        if bad in t:
            raise AssertionError(
                f"template references {bad!r} — the page must be fully "
                f"self-contained and openable offline")


def test_template_has_no_stray_font_size_in_user_units() -> None:
    """The viewBox is in metres, so a px font-size in CSS renders metres
    tall. This bit once already; the size must be set inline from the scene
    extent."""
    css = RI._TEMPLATE[RI._TEMPLATE.index("<style>"):RI._TEMPLATE.index("</style>")]
    m = re.search(r"text\.lbl\{[^}]*\}", css)
    if not m:
        raise AssertionError("text.lbl rule vanished")
    if re.search(r"font\s*:\s*\d|font-size\s*:", m.group(0)):
        raise AssertionError(
            f"text.lbl sets a font size in CSS: {m.group(0)!r} — it must be "
            "an inline attribute derived from the scene extent")
    if 'font-size="${fs}"' not in RI._TEMPLATE:
        raise AssertionError("labels no longer take their size from `fs`")


def test_payload_preserves_every_evidence_key() -> None:
    if not _dataset_ready():
        print("  SKIP (Replica not present)")
        return
    bundle, diag = _build()
    data = RI.payload(SCENE, bundle, diag, [])
    by_id = {e["id"]: e for e in data["edges"]}
    for e in bundle.edges:
        shown = set(by_id[e.edge_id]["ev"])
        stored = set(e.evidence)
        if shown != stored:
            raise AssertionError(
                f"edge {e.edge_id} ({e.type}) evidence differs: "
                f"missing={stored - shown} added={shown - stored}")
    if len(data["rejections"]) != len(diag.rejection_samples):
        raise AssertionError("rejection samples were dropped")


def test_render_is_deterministic_and_embeds_the_payload() -> None:
    if not _dataset_ready():
        print("  SKIP (Replica not present)")
        return
    bundle, diag = _build()
    data = RI.payload(SCENE, bundle, diag, [])
    a, b = RI.render(data), RI.render(data)
    if a != b:
        raise AssertionError("render is not deterministic")
    if "__DATA__" in a or "__TITLE__" in a:
        raise AssertionError("template placeholder was not substituted")
    if "</script>" not in a:
        raise AssertionError("script block missing")


def test_payload_is_json_serialisable_without_loss() -> None:
    if not _dataset_ready():
        print("  SKIP (Replica not present)")
        return
    bundle, diag = _build()
    data = RI.payload(SCENE, bundle, diag, [])
    round_tripped = json.loads(json.dumps(data, sort_keys=True))
    if round_tripped != json.loads(json.dumps(data, sort_keys=True)):
        raise AssertionError("payload does not round-trip")
    if not data["nodes"] or not data["edges"]:
        raise AssertionError("payload is empty")


def test_closing_script_tag_in_data_is_escaped() -> None:
    """A label containing '</script>' would otherwise terminate the block
    and break the page. Cheap to guard, impossible to notice later."""
    data = {"scene_id": "x", "labels": {"a": "</script><b>"}, "nodes": [],
            "edges": [], "rejections": [], "rejections_total": {},
            "rejections_sampled": 0, "questions": [], "threshold_keys": [],
            "bulky_keys": [], "edges_by_type": {}}
    out = RI.render(data)
    if "</script><b>" in out:
        raise AssertionError("a '</script>' inside the payload was not escaped")
    if "<\\/script>" not in out:
        raise AssertionError("expected the escaped form in the payload")


TESTS = [
    test_template_makes_no_external_requests,
    test_template_has_no_stray_font_size_in_user_units,
    test_payload_preserves_every_evidence_key,
    test_render_is_deterministic_and_embeds_the_payload,
    test_payload_is_json_serialisable_without_loss,
    test_closing_script_tag_in_data_is_escaped,
]


def main() -> int:
    failed = 0
    for test in TESTS:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
            print()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
