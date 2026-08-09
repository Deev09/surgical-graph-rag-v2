"""Atomically build both sealed ARKitScenes oracle-free Lane A slices.

The driver is intentionally thin: after the GPU pair readiness guard passes,
it invokes ``tools.arkit_vertical_slice.main`` once per sealed scene with the
same CLI flags.  Both runs live under a temporary sibling directory.  Only
after both scene manifests validate are their paths finalized, a pair
manifest written, and the complete directory atomically published.

No annotation or evaluation module is imported here.  A later evaluation
orchestrator must call :func:`require_lane_a_pair_ready` before opening an
oracle; a single completed scene never creates the unlock manifest.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segmenter.base import sha256_file
from tools.arkit_vertical_slice import main as vertical_slice_main
from tools.arkitscenes_mask3d_transfer import (
    READY_FILENAME as GPU_PAIR_MANIFEST,
    SEALED_PINS,
    ScenePin,
    require_pair_ready,
)


PAIR_MANIFEST = "LANE_A_PAIR_READY.json"
PAIR_SCHEMA = "arkitscenes_sealed_lane_a_pair_v1"


def _inside(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Lane A output escapes scene directory: {path}") from exc


def _finalize_scene_manifest(
        staged_scene: Path, final_scene: Path, pin: ScenePin,
        gpu_row: dict, *, with_learned_labels: bool,
        with_support_patches: bool) -> dict:
    path = staged_scene / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"{pin.scene_key}: Lane A manifest missing")
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "arkit_vertical_slice_v1":
        raise ValueError(f"{pin.scene_key}: wrong Lane A manifest schema")
    if manifest.get("oracle_free") is not True:
        raise ValueError(f"{pin.scene_key}: Lane A output is not oracle-free")
    if manifest.get("scene_id") != pin.scene_key:
        raise ValueError(
            f"{pin.scene_key}: Lane A scene_id={manifest.get('scene_id')!r}")
    if (manifest.get("input", {}).get("segmentation_output_sha256")
            != gpu_row["output_sha256"]):
        raise ValueError(
            f"{pin.scene_key}: Lane A used a different segmentation bundle")

    available = manifest.get("available_capabilities", {})
    expected_flags = {
        "learned_semantic_hypotheses": with_learned_labels,
        "entity_horizontal_patch_evidence": with_support_patches,
    }
    for key, expected in expected_flags.items():
        if available.get(key) is not expected:
            raise ValueError(
                f"{pin.scene_key}: capability {key}={available.get(key)!r}, "
                f"expected {expected!r}")

    # build_slice records the paths it was given.  The all-or-none publish
    # renames the staging root, so finalize those paths to the destination
    # before hashing the scene manifest.
    for key, value in manifest.get("outputs", {}).items():
        if value is None:
            continue
        staged_output = Path(value)
        if not staged_output.is_absolute():
            staged_output = Path.cwd() / staged_output
        relative = _inside(staged_output, staged_scene)
        if not (staged_scene / relative).is_file():
            raise FileNotFoundError(
                f"{pin.scene_key}: declared output {key} is missing")
        manifest["outputs"][key] = str(final_scene.resolve() / relative)

    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "video_id": pin.video_id,
        "scene_id": pin.scene_key,
        "output_dir": str(final_scene.resolve()),
        "manifest": str((final_scene / "manifest.json").resolve()),
        "manifest_sha256": sha256_file(path),
        "representation_hash": manifest["input"]["representation_hash"],
        "segmentation_output_sha256": manifest["input"][
            "segmentation_output_sha256"],
        "counts": manifest["counts"],
    }


def build_lane_a_pair(
        data_root: Path, bundle_root: Path, out_dir: Path, *,
        with_learned_labels: bool = False,
        with_support_patches: bool = False,
        question: str | None = None,
        runner: Callable[[list[str]], int] = vertical_slice_main,
        pair_guard: Callable[[Path], dict] = require_pair_ready,
        pins: Iterable[ScenePin] = SEALED_PINS) -> dict:
    """Build and atomically publish the complete sealed Lane A pair."""
    pins = tuple(pins)
    if len(pins) != 2 or len({p.scene_key for p in pins}) != 2:
        raise ValueError("Lane A sealed execution requires two distinct scenes")

    # Must happen before an output/staging directory is created.
    gpu_manifest = pair_guard(bundle_root)
    gpu_rows = {row["scene_key"]: row for row in gpu_manifest["scenes"]}
    if set(gpu_rows) != {p.scene_key for p in pins}:
        raise ValueError("GPU readiness manifest does not name the sealed pair")

    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite Lane A pair: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    final_root = out_dir.resolve()

    with tempfile.TemporaryDirectory(
            prefix=f".{out_dir.name}.", dir=out_dir.parent) as temp:
        staged_root = Path(temp) / out_dir.name
        staged_root.mkdir()
        scene_rows = []
        for pin in pins:
            staged_scene = staged_root / pin.video_id
            argv = [
                "--scene-dir", str(data_root / pin.video_id),
                "--segmentation-dir",
                str(bundle_root / f"bundle_{pin.scene_key}"),
                "--out", str(staged_scene),
            ]
            if question is not None:
                argv.extend(["--question", question])
            if with_learned_labels:
                argv.append("--with-learned-labels")
            if with_support_patches:
                argv.append("--with-support-patches")
            code = runner(argv)
            if code != 0:
                raise RuntimeError(
                    f"{pin.scene_key}: vertical slice exited {code}")
            scene_rows.append(_finalize_scene_manifest(
                staged_scene, final_root / pin.video_id, pin,
                gpu_rows[pin.scene_key],
                with_learned_labels=with_learned_labels,
                with_support_patches=with_support_patches,
            ))

        # Detect bundle replacement while the two potentially long Lane A
        # runs were in flight.  A mixed pair is never published.
        if pair_guard(bundle_root) != gpu_manifest:
            raise ValueError("GPU pair changed during Lane A execution")
        manifest = {
            "schema": PAIR_SCHEMA,
            "lane": "A",
            "oracle_free": True,
            "all_or_none": True,
            "oracle_evaluation_unlocked": True,
            "source_gpu_pair_manifest_sha256": sha256_file(
                bundle_root / GPU_PAIR_MANIFEST),
            "options": {
                "with_learned_labels": with_learned_labels,
                "with_support_patches": with_support_patches,
                "question": question,
            },
            "scenes": scene_rows,
        }
        pair_path = staged_root / PAIR_MANIFEST
        pair_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        staged_root.rename(out_dir)
    return manifest


def require_lane_a_pair_ready(out_dir: Path) -> dict:
    """Validate both finalized Lane A manifests before later evaluation."""
    path = Path(out_dir) / PAIR_MANIFEST
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is absent; both Lane A scenes must finalize first")
    pair = json.loads(path.read_text())
    if (pair.get("schema") != PAIR_SCHEMA
            or pair.get("lane") != "A"
            or pair.get("oracle_free") is not True
            or pair.get("oracle_evaluation_unlocked") is not True):
        raise ValueError(f"{path}: invalid Lane A pair manifest")
    rows = pair.get("scenes", [])
    if len(rows) != 2 or len({row.get("scene_id") for row in rows}) != 2:
        raise ValueError(f"{path}: incomplete or duplicate scene pair")
    for row in rows:
        manifest = Path(row["manifest"])
        if not manifest.is_file():
            raise FileNotFoundError(f"Lane A scene manifest missing: {manifest}")
        if sha256_file(manifest) != row["manifest_sha256"]:
            raise ValueError(f"Lane A scene manifest changed: {manifest}")
        payload = json.loads(manifest.read_text())
        if (payload.get("oracle_free") is not True
                or payload.get("scene_id") != row["scene_id"]):
            raise ValueError(f"invalid Lane A scene manifest: {manifest}")
    return pair


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--question")
    parser.add_argument("--with-learned-labels", action="store_true")
    parser.add_argument("--with-support-patches", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = build_lane_a_pair(
            args.data_root, args.bundle_root, args.out,
            with_learned_labels=args.with_learned_labels,
            with_support_patches=args.with_support_patches,
            question=args.question,
        )
    except (FileNotFoundError, FileExistsError, KeyError, RuntimeError,
            ValueError) as exc:
        print(f"HARD FAIL: {exc}")
        return 1
    print(f"LANE A PAIR READY: {len(result['scenes'])}/2 scenes")
    print(f"  {args.out / PAIR_MANIFEST}")
    print("Oracle evaluation may now be orchestrated for the pair, never singly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
