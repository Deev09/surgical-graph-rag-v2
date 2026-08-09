"""Prepare and integrity-gate the two sealed ARKitScenes Mask3D runs.

This module is deliberately oracle-free.  It reads canonical meshes and
Mask3D output bundles only; it never opens scene annotations or computes a
metric.  The workflow is all-or-none:

  python3 tools/arkitscenes_mask3d_transfer.py prepare --out <upload-dir>
  # run notebooks/c1_mask3d_colab.ipynb once for each staged scene
  python3 tools/arkitscenes_mask3d_transfer.py check --bundle-root <dir>

`check` writes SEALED_PAIR_READY.json only after both distinct bundles pass
their input, output, checkpoint, raw-mask, and configuration checks.  The
ARKitScenes Mask3D evaluator requires that manifest before it will score
either sealed scene.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segmenter.base import load_segmentation_output, sha256_file


@dataclass(frozen=True)
class ScenePin:
    video_id: str
    scene_key: str
    mesh_sha256: str
    n_vertices: int


SEALED_PINS = (
    ScenePin(
        video_id="41069025",
        scene_key="arkitscenes_41069025",
        mesh_sha256=(
            "361ce587a7af33c1247db5eb6b1a56f6188a94202281a49f880812fada7b8770"),
        n_vertices=1_064_216,
    ),
    ScenePin(
        video_id="41069042",
        scene_key="arkitscenes_41069042",
        mesh_sha256=(
            "fe2dc97c20d8566a9caded784388f635a5da997c5a6e713864c7f1f85c0ef661"),
        n_vertices=422_763,
    ),
)
SEALED_VIDEO_IDS = frozenset(p.video_id for p in SEALED_PINS)
READY_FILENAME = "SEALED_PAIR_READY.json"
INPUT_MANIFEST = "sealed_inputs_manifest.json"

EXPECTED_SEGMENTER = "openmask3d_class_agnostic_mask3d"
EXPECTED_VERSION = "3bc3fc52693b"
EXPECTED_CHECKPOINT_SHA256 = (
    "da4b68cb52c7f204e6ba9f226ffc1d48693de238ed1d4749d7207fde6a12c4a2")
EXPECTED_CONFIG = {
    "num_queries": 150,
    "use_dbscan": "true",
    "dbscan_eps": 0.95,
    "min_score": 0.2,
    "min_vertices": 20,
}

DEFAULT_DATA_ROOT = Path.home() / "Desktop/datasets/arkitscenes/Validation"
DEFAULT_BUNDLE_ROOT = REPO_ROOT / "runs" / "arkitscenes_mask3d_transfer"


def ply_vertex_count(path: Path) -> int:
    """Read only the ASCII PLY header and return its declared vertex count."""
    count = None
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a PLY file")
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path}: PLY header has no end_header")
            try:
                text = line.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise ValueError(f"{path}: non-ASCII PLY header") from exc
            if text.startswith("element vertex "):
                if count is not None:
                    raise ValueError(f"{path}: duplicate vertex element")
                count = int(text.split()[-1])
            if text == "end_header":
                break
    if count is None:
        raise ValueError(f"{path}: PLY header has no vertex element")
    return count


def canonical_mesh_path(data_root: Path, pin: ScenePin) -> Path:
    return (data_root / pin.video_id
            / f"{pin.video_id}_3dod_mesh_canonical.ply")


def verify_mesh(path: Path, pin: ScenePin) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing canonical mesh: {path}")
    actual_sha = sha256_file(path)
    if actual_sha != pin.mesh_sha256:
        raise ValueError(
            f"{pin.scene_key}: mesh sha256 {actual_sha}, expected "
            f"{pin.mesh_sha256}")
    actual_n = ply_vertex_count(path)
    if actual_n != pin.n_vertices:
        raise ValueError(
            f"{pin.scene_key}: {actual_n} vertices, expected {pin.n_vertices}")
    return {
        "video_id": pin.video_id,
        "scene_key": pin.scene_key,
        "mesh_sha256": actual_sha,
        "n_vertices": actual_n,
    }


def prepare_inputs(data_root: Path, out_dir: Path) -> dict:
    """Atomically stage both pinned meshes in the notebook's Drive layout."""
    verified = [(pin, verify_mesh(canonical_mesh_path(data_root, pin), pin))
                for pin in SEALED_PINS]
    if out_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing transfer handoff: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix=f".{out_dir.name}.", dir=out_dir.parent) as tmp:
        staged = Path(tmp) / out_dir.name
        staged.mkdir()
        rows = []
        for pin, row in verified:
            target_dir = staged / pin.scene_key
            target_dir.mkdir()
            target = target_dir / "mesh.ply"
            shutil.copy2(canonical_mesh_path(data_root, pin), target)
            if sha256_file(target) != pin.mesh_sha256:
                raise ValueError(f"copy verification failed for {pin.scene_key}")
            rows.append({**row, "upload_path": f"{pin.scene_key}/mesh.ply"})
        manifest = {
            "schema": "arkitscenes_mask3d_sealed_inputs_v1",
            "all_or_none": True,
            "scenes": rows,
        }
        (staged / INPUT_MANIFEST).write_text(
            json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        staged.rename(out_dir)
    return manifest


def _verify_raw_masks(path: Path, expected_n: int) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing raw-mask evidence: {path}")
    with np.load(path) as packed:
        required = {"masks_packed", "n_vertices", "scores"}
        if not required.issubset(packed.files):
            raise ValueError(f"{path}: expected arrays {sorted(required)}")
        n_vertices = int(packed["n_vertices"])
        masks_shape = packed["masks_packed"].shape
        scores_shape = packed["scores"].shape
    if n_vertices != expected_n:
        raise ValueError(f"{path}: raw masks name {n_vertices} vertices, "
                         f"expected {expected_n}")
    expected_bytes = (expected_n + 7) // 8
    if (len(masks_shape) != 2 or masks_shape[1] != expected_bytes
            or scores_shape != (masks_shape[0],)):
        raise ValueError(f"{path}: inconsistent masks/scores shapes "
                         f"{masks_shape}/{scores_shape}")
    return {"n_raw_masks": int(masks_shape[0]),
            "raw_masks_sha256": sha256_file(path)}


def verify_bundle(bundle_root: Path, pin: ScenePin) -> dict:
    bundle_dir = bundle_root / f"bundle_{pin.scene_key}"
    if not bundle_dir.is_dir():
        raise FileNotFoundError(
            f"missing sealed bundle {bundle_dir}; both scenes must exist "
            "before either may be evaluated")
    seg = load_segmentation_output(bundle_dir)
    if seg.input_mesh_sha256 != pin.mesh_sha256:
        raise ValueError(f"{pin.scene_key}: bundle mesh hash does not match pin")
    if seg.n_vertices != pin.n_vertices:
        raise ValueError(f"{pin.scene_key}: bundle vertex count "
                         f"{seg.n_vertices}, expected {pin.n_vertices}")
    if (seg.segmenter_name != EXPECTED_SEGMENTER
            or seg.segmenter_version != EXPECTED_VERSION):
        raise ValueError(f"{pin.scene_key}: unexpected segmenter "
                         f"{seg.segmenter_name}@{seg.segmenter_version}")
    try:
        cfg = json.loads(seg.config_params_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{pin.scene_key}: invalid config_params_json") from exc
    if cfg.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"{pin.scene_key}: checkpoint sha does not match pin")
    for key, expected in EXPECTED_CONFIG.items():
        if cfg.get(key) != expected:
            raise ValueError(f"{pin.scene_key}: config {key}={cfg.get(key)!r}, "
                             f"expected {expected!r}")
    raw = _verify_raw_masks(bundle_dir / "raw_masks.npz", pin.n_vertices)
    return {
        "video_id": pin.video_id,
        "scene_key": pin.scene_key,
        "bundle_dir": bundle_dir.name,
        "input_mesh_sha256": seg.input_mesh_sha256,
        "n_vertices": seg.n_vertices,
        "output_sha256": seg.output_sha256,
        "segmenter_name": seg.segmenter_name,
        "segmenter_version": seg.segmenter_version,
        "checkpoint_sha256": cfg["checkpoint_sha256"],
        **{key: cfg[key] for key in EXPECTED_CONFIG},
        **raw,
    }


def check_pair(bundle_root: Path) -> dict:
    """Validate both bundles first, then atomically publish readiness."""
    rows = [verify_bundle(bundle_root, pin) for pin in SEALED_PINS]
    if len({row["output_sha256"] for row in rows}) != len(rows):
        raise ValueError("sealed scenes produced identical output hashes; "
                         "possible duplicate/wrong-scene bundle")
    manifest = {
        "schema": "arkitscenes_mask3d_sealed_pair_ready_v1",
        "evaluation_unlocked": True,
        "scenes": rows,
    }
    bundle_root.mkdir(parents=True, exist_ok=True)
    target = bundle_root / READY_FILENAME
    partial = bundle_root / f".{READY_FILENAME}.partial"
    partial.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    os.replace(partial, target)
    return manifest


def require_pair_ready(bundle_root: Path) -> dict:
    """Refuse sealed evaluation unless the current pair matches its manifest."""
    path = bundle_root / READY_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is absent; run this tool's `check` command only after "
            "both sealed bundles exist")
    manifest = json.loads(path.read_text())
    if (manifest.get("schema")
            != "arkitscenes_mask3d_sealed_pair_ready_v1"):
        raise ValueError(f"{path}: wrong readiness schema")
    current = [verify_bundle(bundle_root, pin) for pin in SEALED_PINS]
    if manifest.get("scenes") != current:
        raise ValueError(
            f"{path}: bundles changed after readiness was recorded; rerun check")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="stage both pinned meshes")
    prep.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    prep.add_argument("--out", type=Path, required=True)
    check = sub.add_parser("check", help="validate both downloaded bundles")
    check.add_argument("--bundle-root", type=Path,
                       default=DEFAULT_BUNDLE_ROOT)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            report = prepare_inputs(args.data_root, args.out)
            print(f"READY TO UPLOAD: {args.out}")
            for row in report["scenes"]:
                print(f"  MyDrive/c1/{row['upload_path']}  "
                      f"{row['n_vertices']} vertices  "
                      f"sha256 {row['mesh_sha256'][:16]}…")
            print("Run the notebook once per scene_key; do not evaluate yet.")
        else:
            report = check_pair(args.bundle_root)
            print(f"PAIR READY: {len(report['scenes'])}/2 bundles verified")
            print(f"  {args.bundle_root / READY_FILENAME}")
            print("Sealed evaluation is now unlocked for both scenes.")
    except (FileNotFoundError, FileExistsError, KeyError, ValueError) as exc:
        print(f"HARD FAIL: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
