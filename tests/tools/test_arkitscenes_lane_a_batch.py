"""Atomic orchestration tests for the sealed oracle-free Lane A pair."""
from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import arkitscenes_lane_a_batch as B
from tools.arkitscenes_mask3d_transfer import ScenePin


PINS = (
    ScenePin("1", "arkitscenes_1", "a" * 64, 3),
    ScenePin("2", "arkitscenes_2", "b" * 64, 4),
)


def _ready(bundle_root: Path) -> dict:
    rows = []
    for n, pin in enumerate(PINS):
        rows.append({"scene_key": pin.scene_key,
                     "output_sha256": f"output-{n}"})
    payload = {"schema": "synthetic_gpu_pair", "scenes": rows}
    bundle_root.mkdir(parents=True, exist_ok=True)
    (bundle_root / "SEALED_PAIR_READY.json").write_text(
        json.dumps(payload, sort_keys=True) + "\n")
    return payload


def _fake_runner(calls: list[list[str]], *, fail_scene: str | None = None):
    def run(argv: list[str]) -> int:
        calls.append(list(argv))
        opts = {argv[i]: argv[i + 1] for i in range(0, len(argv) - 1)
                if argv[i].startswith("--")
                and not argv[i + 1].startswith("--")}
        video_id = Path(opts["--scene-dir"]).name
        if video_id == fail_scene:
            return 7
        scene_id = f"arkitscenes_{video_id}"
        out = Path(opts["--out"])
        out.mkdir(parents=True)
        outputs = {}
        for key, relative in (
                ("entity_manifest", "entities/manifest.json"),
                ("graph_manifest", "graph/manifest.json"),
                ("graph_diagnostics", "graph_diagnostics.json"),
                ("inspector", "inspector.html")):
            artifact = out / relative
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(key)
            outputs[key] = str(artifact)
        with_patches = "--with-support-patches" in argv
        if with_patches:
            artifact = out / "support_patches.json"
            artifact.write_text("{}\n")
            outputs["support_patch_evidence"] = str(artifact)
        else:
            outputs["support_patch_evidence"] = None
        seg_hash = "output-0" if video_id == "1" else "output-1"
        manifest = {
            "schema": "arkit_vertical_slice_v1",
            "scene_id": scene_id,
            "oracle_free": True,
            "input": {
                "representation_hash": f"representation-{video_id}",
                "segmentation_output_sha256": seg_hash,
            },
            "outputs": outputs,
            "counts": {"entities": int(video_id), "edges": 0,
                       "near_edges": 0},
            "available_capabilities": {
                "learned_semantic_hypotheses": (
                    "--with-learned-labels" in argv),
                "entity_horizontal_patch_evidence": with_patches,
            },
        }
        (out / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return 0
    return run


def test_gpu_pair_guard_runs_before_any_output() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        calls = []

        def blocked(_root: Path):
            raise FileNotFoundError("pair incomplete")

        try:
            B.build_lane_a_pair(
                root / "data", root / "bundles", root / "lane-a",
                runner=_fake_runner(calls), pair_guard=blocked, pins=PINS)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("incomplete GPU pair was accepted")
        if calls or (root / "lane-a").exists():
            raise AssertionError("output began before GPU pair readiness")


def test_second_scene_failure_publishes_nothing() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ready = _ready(root / "bundles")
        calls = []
        try:
            B.build_lane_a_pair(
                root / "data", root / "bundles", root / "lane-a",
                runner=_fake_runner(calls, fail_scene="2"),
                pair_guard=lambda _root: ready, pins=PINS)
        except RuntimeError:
            pass
        else:
            raise AssertionError("failed second scene was accepted")
        if len(calls) != 2:
            raise AssertionError(f"unexpected run count: {len(calls)}")
        if (root / "lane-a").exists():
            raise AssertionError("partial Lane A pair was published")


def test_complete_pair_finalizes_paths_flags_and_unlock_manifest() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ready = _ready(root / "bundles")
        calls = []
        out = root / "lane-a"
        result = B.build_lane_a_pair(
            root / "data", root / "bundles", out,
            with_learned_labels=True, with_support_patches=True,
            question="what is near object?",
            runner=_fake_runner(calls), pair_guard=lambda _root: ready,
            pins=PINS)
        if result["oracle_evaluation_unlocked"] is not True:
            raise AssertionError("complete pair did not publish unlock")
        if len(calls) != 2 or any(
                "--with-learned-labels" not in call
                or "--with-support-patches" not in call
                for call in calls):
            raise AssertionError(f"flags did not reach both runs: {calls}")
        for row in result["scenes"]:
            manifest = Path(row["manifest"])
            if not manifest.is_file() or str(out.resolve()) not in str(manifest):
                raise AssertionError(f"scene was not finalized under output: {row}")
            payload = json.loads(manifest.read_text())
            for value in payload["outputs"].values():
                if value is not None and not str(value).startswith(str(out.resolve())):
                    raise AssertionError(f"staging path leaked: {value}")
        if B.require_lane_a_pair_ready(out) != result:
            raise AssertionError("fresh Lane A pair did not verify")


def test_changed_scene_manifest_relocks_pair() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ready = _ready(root / "bundles")
        out = root / "lane-a"
        result = B.build_lane_a_pair(
            root / "data", root / "bundles", out,
            runner=_fake_runner([]), pair_guard=lambda _root: ready,
            pins=PINS)
        Path(result["scenes"][0]["manifest"]).write_text("{}\n")
        try:
            B.require_lane_a_pair_ready(out)
        except ValueError:
            pass
        else:
            raise AssertionError("changed scene manifest kept pair unlocked")


def test_gpu_pair_change_during_build_publishes_nothing() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ready = _ready(root / "bundles")
        changed = json.loads(json.dumps(ready))
        changed["scenes"][0]["output_sha256"] = "replacement"
        calls = 0

        def changing_guard(_root: Path) -> dict:
            nonlocal calls
            calls += 1
            return ready if calls == 1 else changed

        out = root / "lane-a"
        try:
            B.build_lane_a_pair(
                root / "data", root / "bundles", out,
                runner=_fake_runner([]), pair_guard=changing_guard, pins=PINS)
        except ValueError:
            pass
        else:
            raise AssertionError("mixed GPU pair was published")
        if out.exists():
            raise AssertionError("GPU source drift left a partial Lane A pair")


TESTS = [
    test_gpu_pair_guard_runs_before_any_output,
    test_second_scene_failure_publishes_nothing,
    test_complete_pair_finalizes_paths_flags_and_unlock_manifest,
    test_changed_scene_manifest_relocks_pair,
    test_gpu_pair_change_during_build_publishes_nothing,
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
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
