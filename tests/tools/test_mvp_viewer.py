"""Tests for tools/mvp_viewer.py (MVP-v1 3D viewer generator).

Run: python tests/tools/test_mvp_viewer.py

Synthetic coverage: payload quantization round-trip, id arrays, embedded
JSON validity, zero external requests, determinism. The real two-scene
build is exercised only when the dataset + frozen bundles + MVP-v0
reports exist.
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.segmenter.test_c1_pipeline import PERFECT, _scene


def _fake_mvp_report():
    return {
        "key_questions": {"Q01": {
            "question": "what is on the table?",
            "expected_outcome": "empty", "exhaustive": True,
            "expected_must_contain": [], "expected_must_not_contain": [],
            "candidate_labels": {}}},
        "variants": {"A": {
            "micro_precision": None, "micro_recall": None,
            "semantic_citation": None, "n_graph_edges": 0,
            "questions": [{"question_id": "Q01", "question": "what is on the table?",
                           "expected_outcome": "empty", "actual_outcome": "empty",
                           "verbalized": "nothing", "cited": [], "missed": [],
                           "exhaustive": True, "precision": None, "recall": None}]}},
        "c1_status": "synthetic",
        "provenance": {"git_commit": "test", "key": {"fixture_id": "syn"},
                       "isolation_statement": "synthetic"},
    }


def test_payload_roundtrip_and_ids():
    import base64
    from tools.mvp_viewer import build_scene_payload
    with tempfile.TemporaryDirectory() as td:
        room, bundle = _scene(Path(td), PERFECT)
        p = build_scene_payload(room, bundle, "synthetic_scene",
                                _fake_mvp_report())
        n = p["n"]
        if n != 16:
            raise AssertionError(f"two cubes = 16 vertices: {n}")
        q = np.frombuffer(base64.b64decode(p["b64"]["pos"]), dtype=np.uint16)
        lo = np.array(p["bbox"][0])
        hi = np.array(p["bbox"][1])
        xyz = lo + q.reshape(n, 3) / 65535.0 * (hi - lo)
        span = float((hi - lo).max())
        if span <= 0 or np.any(xyz < lo - 1e-6) or np.any(xyz > hi + 1e-6):
            raise AssertionError("dequantized positions out of bbox")
        # quantization error must be under 1mm at synthetic scale (6m span)
        if span / 65535.0 > 0.001:
            raise AssertionError(f"quantization step too coarse: {span/65535}")
        oracle = np.frombuffer(base64.b64decode(p["b64"]["oracle"]),
                               dtype=np.int16)
        pred = np.frombuffer(base64.b64decode(p["b64"]["pred"]), dtype=np.int16)
        if set(oracle.tolist()) != {1, 2} or set(pred.tolist()) != {10, 20}:
            raise AssertionError(f"id arrays wrong: {set(oracle.tolist())} "
                                 f"{set(pred.tolist())}")
        if p["objects"]["1"]["label"] != "table" or p["objects"]["1"]["pred"] != 10:
            raise AssertionError(f"object meta wrong: {p['objects']}")


def test_html_valid_json_offline_deterministic():
    from tools.mvp_viewer import build_scene_payload, build_viewer_html
    with tempfile.TemporaryDirectory() as td:
        room, bundle = _scene(Path(td), PERFECT)
        p1 = build_scene_payload(room, bundle, "synthetic_scene",
                                 _fake_mvp_report())
        p2 = build_scene_payload(room, bundle, "synthetic_scene",
                                 _fake_mvp_report())
        h1 = build_viewer_html([p1])
        h2 = build_viewer_html([p2])
        if h1 != h2:
            raise AssertionError("viewer generation must be deterministic")
        m = re.search(r'<script id="data" type="application/json">(.*?)</script>',
                      h1, re.S)
        if not m:
            raise AssertionError("embedded data block missing")
        data = json.loads(m.group(1).replace("<\\/", "</"))
        if "synthetic_scene" not in data["scenes"]:
            raise AssertionError("scene missing from embedded data")
        if not data["disclosures"]:
            raise AssertionError("disclosures must be embedded")
        for pat in ("src=\"http", "src='http", "href=\"http", "href='http",
                    "fetch(", "XMLHttpRequest", "import("):
            if pat in h1:
                raise AssertionError(f"external/request pattern found: {pat}")


def test_unknown_oracle_ids_normalized_to_minus_one():
    """Regression (office_0 acceptance blocker): the semantic mesh can
    carry face ids with NO entry in info_semantic.json — the viewer
    payload must normalize them to -1, never show a phantom obj_N."""
    import base64
    from tools.mvp_viewer import build_scene_payload
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # three-cube mesh (oids 1,2,3) but metadata for the third cube is
        # REMOVED — its mesh id 3 has no oracle object
        pred = np.array([10] * 8 + [20] * 8 + [-1] * 8)
        room, bundle = _scene(tmp, pred, with_wall_cube=True)
        info_path = room / "habitat" / "info_semantic.json"
        info = json.loads(info_path.read_text())
        info["objects"] = [o for o in info["objects"] if o["id"] != 3]
        info_path.write_text(json.dumps(info))
        p = build_scene_payload(room, bundle, "synthetic_scene",
                                _fake_mvp_report())
        oracle = np.frombuffer(base64.b64decode(p["b64"]["oracle"]),
                               dtype=np.int16)
        if set(oracle[:16].tolist()) != {1, 2}:
            raise AssertionError(f"known ids must survive: {set(oracle[:16].tolist())}")
        if set(oracle[16:24].tolist()) != {-1}:
            raise AssertionError(f"metadata-less mesh id must normalize to -1: "
                                 f"{set(oracle[16:24].tolist())}")
        if "3" in p["objects"]:
            raise AssertionError("phantom object must not appear in meta")


def test_real_build_when_inputs_present():
    data = Path.home() / "Desktop/datasets/replica/room_2"
    bundle = REPO_ROOT / "runs" / "phase8_c1" / "bundles_ms02" / "room_2"
    mvp = REPO_ROOT / "runs" / "mvp_v0" / "replica_room_2_mvp.json"
    if not (data.exists() and bundle.exists() and mvp.exists()):
        print("  (skipped: dataset/bundle/mvp report not present)")
        return
    from tools.mvp_viewer import main as viewer_main
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "viewer.html"
        rc = viewer_main(["--out", str(out), "--scenes", "replica_room_2"])
        if rc != 0 or not out.exists():
            raise AssertionError(f"real build failed: rc={rc}")
        if out.stat().st_size < 5_000_000:
            raise AssertionError("real viewer suspiciously small")


TESTS = [
    test_payload_roundtrip_and_ids,
    test_html_valid_json_offline_deterministic,
    test_unknown_oracle_ids_normalized_to_minus_one,
    test_real_build_when_inputs_present,
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
