"""ARKitScenes view generation: upload isolation and contract parity.

The load-bearing test is `tar_for_upload` withholding `ids.npz`. Those
buffers map pixels to vertex indices; shipping them to the GPU stage would
let mask generation see scene identity, which is exactly the isolation the
C1-P1 protocol exists to preserve. That property is tested against a
synthetic directory so it runs with or without the dataset on disk.

The contract-parity checks are dataset-guarded and read whatever
`tools/arkitscenes_render.py --all` last produced; they do not re-render.
"""
from __future__ import annotations

import json
import sys
import tarfile
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from tools.arkitscenes_render import IDENTITY_FRAME, tar_for_upload

VIEWS_ROOT = REPO_ROOT / "runs" / "arkitscenes_p1"


def _rendered() -> list[Path]:
    if not VIEWS_ROOT.is_dir():
        return []
    return sorted(d for d in VIEWS_ROOT.glob("views_arkitscenes_*")
                  if d.is_dir() and (d / "manifest.json").is_file())


def test_upload_tar_withholds_id_buffers() -> None:
    """Synthetic, always runs. An id buffer reaching the GPU stage would
    break the protocol's isolation guarantee."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "views_synthetic"
        d.mkdir()
        for i in range(40):
            (d / f"view_{i:02d}.png").write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(8))
        (d / "manifest.json").write_text("{}")
        np.savez_compressed(d / "ids.npz", ids_00=np.zeros((4, 4), np.int32))

        tar = tar_for_upload(d)
        with tarfile.open(tar) as tf:
            names = tf.getnames()
        if any(n.endswith(".npz") for n in names):
            raise AssertionError(f"id buffers leaked into upload tar: {names}")
        if sum(n.endswith(".png") for n in names) != 40:
            raise AssertionError(f"expected 40 PNGs in tar, got {names}")
        if not any(n.endswith("manifest.json") for n in names):
            raise AssertionError("manifest missing from upload tar")
        if not all(n.startswith("views_synthetic/") for n in names):
            raise AssertionError(
                f"tar root must be views_<scene>/ for the notebook glob: {names}")


def test_upload_tar_refuses_an_incomplete_view_dir() -> None:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "views_synthetic"
        d.mkdir()
        (d / "view_00.png").write_bytes(b"x")
        try:
            tar_for_upload(d)
        except FileNotFoundError:
            return
        raise AssertionError("packed a view dir with no manifest")


def test_render_frame_is_the_identity() -> None:
    """The adapter emits canonical coordinates, so the renderer must not
    transform again. A non-identity here would double-rotate the scene."""
    R = np.asarray(IDENTITY_FRAME["world_from_raw_rotation"])
    t = np.asarray(IDENTITY_FRAME["world_from_raw_translation"])
    if not np.array_equal(R, np.eye(3)) or not np.array_equal(t, np.zeros(3)):
        raise AssertionError(f"render frame is not the identity: {R!r} {t!r}")


def test_manifest_matches_the_replica_view_contract() -> None:
    """Same schema and constants as tools/c1p1_render.py, so the Colab
    notebook and tools/c1p1_fuse.py consume both datasets unchanged."""
    dirs = _rendered()
    if not dirs:
        print("  SKIP (no rendered ARKitScenes views on disk)")
        return
    from segmenter.view_render import (
        EYE_HEIGHT_M, NEAR_M, ORIGIN_FRAC, PITCH_DEG, SIZE, VFOV_DEG,
    )
    expected = {"size": SIZE, "vfov_deg": VFOV_DEG, "near_m": NEAR_M,
                "pitch_deg": PITCH_DEG, "eye_height_m": EYE_HEIGHT_M,
                "origin_frac": ORIGIN_FRAC, "n_views": 40,
                "splat_px": 3, "background": "black"}
    for d in dirs:
        man = json.loads((d / "manifest.json").read_text())
        if man["schema"] != "c1p1_view_manifest_v1":
            raise AssertionError(f"{d.name}: schema {man['schema']!r}")
        if man["contract"] != expected:
            raise AssertionError(
                f"{d.name}: view contract drifted from the Replica path\n"
                f"  got      {man['contract']}\n  expected {expected}")
        if len(man["views"]) != 40 or len(list(d.glob("view_*.png"))) != 40:
            raise AssertionError(f"{d.name}: not exactly 40 views")
        if man["source"]["frame_kind"] != "scene_canonical":
            raise AssertionError(f"{d.name}: frame_kind not scene_canonical")
        if man["source"]["reads_annotations"] is not False:
            raise AssertionError(f"{d.name}: manifest claims annotations read")


def test_id_buffers_cover_the_mesh() -> None:
    """Evidence-coverage sanity: a scene whose views collectively miss most
    of the mesh cannot support multiview fusion, and would waste the GPU
    budget. This is a floor, NOT the protocol's G5 gate -- G5 is per-entity
    and oracle-dependent, so it belongs in the evaluator."""
    dirs = _rendered()
    if not dirs:
        print("  SKIP (no rendered ARKitScenes views on disk)")
        return
    for d in dirs:
        man = json.loads((d / "manifest.json").read_text())
        n_v = man["n_vertices"]
        ids = np.load(d / "ids.npz")
        seen = np.zeros(n_v, dtype=bool)
        for k in ids.files:
            a = ids[k]
            seen[np.unique(a[a >= 0])] = True
        cov = float(seen.mean())
        if cov < 0.60:
            raise AssertionError(
                f"{man['scene_id']}: only {cov:.1%} of vertices appear in any "
                "view; multiview fusion has too little evidence")


TESTS = [
    test_upload_tar_withholds_id_buffers,
    test_upload_tar_refuses_an_incomplete_view_dir,
    test_render_frame_is_the_identity,
    test_manifest_matches_the_replica_view_contract,
    test_id_buffers_cover_the_mesh,
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
