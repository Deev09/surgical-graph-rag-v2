"""Oracle-free guards for the sealed ARKitScenes Mask3D handoff."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segmenter.base import SegmentationOutput, save_segmentation_output
from tools import arkitscenes_mask3d_transfer as T


def _mesh(path: Path, n_vertices: int) -> T.ScenePin:
    body = "".join("0 0 0\n" for _ in range(n_vertices))
    content = ("ply\nformat ascii 1.0\n"
               f"element vertex {n_vertices}\n"
               "property float x\nproperty float y\nproperty float z\n"
               "element face 0\nproperty list uchar int vertex_indices\n"
               "end_header\n" + body).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    video_id = path.parent.name
    return T.ScenePin(video_id, f"arkitscenes_{video_id}",
                      hashlib.sha256(content).hexdigest(), n_vertices)


def _bundle(root: Path, pin: T.ScenePin, assignment: list[int]) -> None:
    bundle = root / f"bundle_{pin.scene_key}"
    cfg = {**T.EXPECTED_CONFIG,
           "checkpoint_sha256": T.EXPECTED_CHECKPOINT_SHA256}
    seg = SegmentationOutput(
        input_mesh_sha256=pin.mesh_sha256,
        n_vertices=pin.n_vertices,
        segmenter_name=T.EXPECTED_SEGMENTER,
        segmenter_version=T.EXPECTED_VERSION,
        config_params_json=json.dumps(cfg, sort_keys=True),
        vertex_instance_ids=np.asarray(assignment, dtype=np.int64),
    ).finalize()
    save_segmentation_output(seg, bundle)
    mask = np.ones((1, pin.n_vertices), dtype=np.uint8)
    np.savez_compressed(
        bundle / "raw_masks.npz",
        masks_packed=np.packbits(mask, axis=1),
        n_vertices=np.int64(pin.n_vertices),
        scores=np.asarray([0.9]),
    )


def _with_synthetic_pins(fn) -> None:
    original = T.SEALED_PINS
    try:
        fn()
    finally:
        T.SEALED_PINS = original


def test_real_pins_and_notebook_agree() -> None:
    expected = {
        "41069025": (1_064_216,
                      "361ce587a7af33c1247db5eb6b1a56f6188a94202281a49f880812fada7b8770"),
        "41069042": (422_763,
                      "fe2dc97c20d8566a9caded784388f635a5da997c5a6e713864c7f1f85c0ef661"),
    }
    actual = {p.video_id: (p.n_vertices, p.mesh_sha256)
              for p in T.SEALED_PINS}
    if actual != expected:
        raise AssertionError(f"sealed pins drifted: {actual}")
    source = (REPO_ROOT / "notebooks" / "c1_mask3d_colab.ipynb").read_text()
    for video_id, (n_vertices, sha) in expected.items():
        for required in (f"arkitscenes_{video_id}", str(n_vertices), sha):
            if required not in source:
                raise AssertionError(f"notebook does not pin {required}")
    if "bundle_{SCENE}" not in source or "{SCENE}_c1_bundle.tar.gz" not in source:
        raise AssertionError("notebook outputs are no longer scene-distinct")
    if "MIN_SCORE = 0.2 if SCENE in SEALED_TRANSFER_SCENES else 0.4" not in source:
        raise AssertionError(
            "sealed runs do not directly deliver the declared ms02 setting")


def test_prepare_is_an_atomic_two_scene_handoff() -> None:
    def run() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / "data"
            p1 = _mesh(data / "1" / "1_3dod_mesh_canonical.ply", 3)
            p2 = _mesh(data / "2" / "2_3dod_mesh_canonical.ply", 4)
            T.SEALED_PINS = (p1, p2)
            out = root / "upload"
            manifest = T.prepare_inputs(data, out)
            if len(manifest["scenes"]) != 2:
                raise AssertionError(f"not a pair manifest: {manifest}")
            for pin in (p1, p2):
                copied = out / pin.scene_key / "mesh.ply"
                if T.verify_mesh(copied, pin)["mesh_sha256"] != pin.mesh_sha256:
                    raise AssertionError(f"bad staged copy for {pin.scene_key}")
            if not (out / T.INPUT_MANIFEST).is_file():
                raise AssertionError("input manifest missing")
    _with_synthetic_pins(run)


def test_partial_pair_never_unlocks_evaluation() -> None:
    def run() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p1 = T.ScenePin("1", "arkitscenes_1", "a" * 64, 3)
            p2 = T.ScenePin("2", "arkitscenes_2", "b" * 64, 4)
            T.SEALED_PINS = (p1, p2)
            _bundle(root, p1, [-1, 0, 0])
            try:
                T.check_pair(root)
            except FileNotFoundError:
                pass
            else:
                raise AssertionError("one sealed bundle incorrectly unlocked pair")
            if (root / T.READY_FILENAME).exists():
                raise AssertionError("partial run left a readiness manifest")
            try:
                T.require_pair_ready(root)
            except FileNotFoundError:
                pass
            else:
                raise AssertionError("missing readiness manifest was accepted")
    _with_synthetic_pins(run)


def test_complete_pair_is_hash_guarded_and_distinct() -> None:
    def run() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p1 = T.ScenePin("1", "arkitscenes_1", "a" * 64, 3)
            p2 = T.ScenePin("2", "arkitscenes_2", "b" * 64, 4)
            T.SEALED_PINS = (p1, p2)
            _bundle(root, p1, [-1, 0, 0])
            _bundle(root, p2, [-1, 0, 0, 1])
            manifest = T.check_pair(root)
            if len(manifest["scenes"]) != 2:
                raise AssertionError(f"wrong ready manifest: {manifest}")
            if len({r["output_sha256"] for r in manifest["scenes"]}) != 2:
                raise AssertionError("pair outputs are not distinct")
            if T.require_pair_ready(root) != manifest:
                raise AssertionError("fresh pair readiness did not verify")

            # Even a still-valid raw-mask file must invalidate a stale pair
            # manifest when its evidence bytes change.
            bundle = root / f"bundle_{p2.scene_key}"
            mask = np.zeros((1, p2.n_vertices), dtype=np.uint8)
            np.savez_compressed(
                bundle / "raw_masks.npz",
                masks_packed=np.packbits(mask, axis=1),
                n_vertices=np.int64(p2.n_vertices),
                scores=np.asarray([0.8]),
            )
            try:
                T.require_pair_ready(root)
            except ValueError:
                pass
            else:
                raise AssertionError("changed evidence passed stale readiness")
    _with_synthetic_pins(run)


def test_sealed_eval_cli_requires_pair_before_scene_input() -> None:
    from tools import arkitscenes_mask3d_eval as evaluator
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        code = evaluator.main([
            "--scene", "41069025",
            "--data-root", str(root / "no-dataset"),
            "--bundle-root", str(root / "no-bundles"),
            "--delivered-only",
        ])
    if code != 1:
        raise AssertionError(f"sealed evaluation bypassed pair gate: {code}")


TESTS = [
    test_real_pins_and_notebook_agree,
    test_prepare_is_an_atomic_two_scene_handoff,
    test_partial_pair_never_unlocks_evaluation,
    test_complete_pair_is_hash_guarded_and_distinct,
    test_sealed_eval_cli_requires_pair_before_scene_input,
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
    sys.exit(main())
