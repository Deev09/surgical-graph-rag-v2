"""ARKitScenes view generation for the C1-P1 SAM stage (local, CPU).

  python3 tools/arkitscenes_render.py --scene 41069021
  python3 tools/arkitscenes_render.py --all --tar

Produces the SAME artifact contract as `tools/c1p1_render.py`
(`views_<scene>/view_XX.png` + `ids.npz` + `manifest.json`, schema
`c1p1_view_manifest_v1`) so `notebooks/c1p1_sam2_colab.ipynb` and
`tools/c1p1_fuse.py` consume ARKitScenes output unchanged. `render_scene`
is imported from `tools.c1p1_render` rather than reimplemented -- the two
datasets must not drift to different view contracts, or their proposal
banks stop being comparable.

TWO ISOLATION PROPERTIES, both inherited from the P1 protocol:

  * Geometry only. The renderer sees the canonical mesh and nothing else.
    ARKitScenes annotations are never read here, and cannot be: the source
    is `adapters.arkitscenes`, which has no JSON parser (see its docstring
    and `tests/adapters/test_arkitscenes.py`).
  * `--tar` packs RGB PNGs + manifest and DELIBERATELY EXCLUDES `ids.npz`.
    The id buffers map pixels to vertex indices; shipping them to the GPU
    stage would let mask generation see scene identity. They stay local and
    are rejoined at fusion time.

The frame transform applied here is the identity, because the adapter has
already produced canonical coordinates and `GeometryHandle.uri` points at
that mesh. Canonicalization lives in the adapter, not in the renderer --
see `docs/frame_decision.md`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.arkitscenes import (
    ARKitScenesAdapter, build_arkitscenes_capture_bundle, scene_id_for,
)
from adapters.base import ReconstructionConfig
from tools.c1p1_render import render_scene, sha256_bytes

DEFAULT_DATA_ROOT = Path.home() / "Desktop/datasets/arkitscenes/Validation"
IDENTITY_FRAME = {
    "world_from_raw_rotation": np.eye(3).tolist(),
    "world_from_raw_translation": [0.0, 0.0, 0.0],
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def render_one(scene_dir: Path, out_root: Path) -> Path:
    """Adapter -> canonical mesh -> 40 frozen views. Returns the views dir."""
    scene_id = scene_id_for(scene_dir)
    out = out_root / f"views_{scene_id}"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing views: {out}")

    bundle = ARKitScenesAdapter().reconstruct(
        build_arkitscenes_capture_bundle(scene_dir),
        ReconstructionConfig(name="arkitscenes_mesh", version="0.1"),
    )
    if bundle.frame.kind != "scene_canonical":
        raise ValueError(
            f"{scene_id}: expected a scene_canonical bundle, got "
            f"{bundle.frame.kind!r}; the identity frame below would be wrong")

    mesh_path = Path(bundle.geometry_handle.uri)
    mpath = render_scene(mesh_path, IDENTITY_FRAME, out,
                         _sha256_file(mesh_path), scene_id)

    # record where the geometry came from, alongside the shared manifest
    man = json.loads(mpath.read_text())
    man["source"] = {
        "dataset": "arkitscenes-3dod-raw",
        "video_id": scene_dir.name,
        "adapter": bundle.notes["adapter"],
        "adapter_version": bundle.notes["adapter_version"],
        "representation_hash": bundle.representation_hash,
        "frame_kind": bundle.frame.kind,
        "reads_annotations": bundle.notes["reads_annotations"],
        "source_up_axis_capture_frame":
            bundle.notes["source_up_axis_capture_frame"],
        "scale": bundle.notes["scale"],
        "render_frame_is_identity": True,
    }
    mpath.write_text(json.dumps(man, indent=1, sort_keys=True) + "\n",
                     encoding="utf-8")
    return out


def tar_for_upload(views_dir: Path) -> Path:
    """PNGs + manifest only. ids.npz is withheld from the GPU stage."""
    tar_path = views_dir.with_suffix(".tar.gz")
    members = sorted(views_dir.glob("view_*.png")) + [views_dir / "manifest.json"]
    missing = [p.name for p in members if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{views_dir}: missing {missing}")
    with tarfile.open(tar_path, "w:gz") as tf:
        for p in members:
            tf.add(p, arcname=f"{views_dir.name}/{p.name}")
    with tarfile.open(tar_path) as tf:
        names = tf.getnames()
    leaked = [n for n in names if n.endswith(".npz")]
    if leaked:
        raise AssertionError(f"id buffers leaked into the upload tar: {leaked}")
    return tar_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", help="ARKitScenes video_id, e.g. 41069021")
    ap.add_argument("--all", action="store_true", help="every scene on disk")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-root", type=Path,
                    default=REPO_ROOT / "runs" / "arkitscenes_p1")
    ap.add_argument("--tar", action="store_true",
                    help="also pack PNGs+manifest for Drive upload")
    args = ap.parse_args(argv)

    if args.all:
        scenes = sorted(d for d in args.data_root.iterdir()
                        if d.is_dir()
                        and (d / f"{d.name}_3dod_mesh.ply").is_file())
    elif args.scene:
        scenes = [args.data_root / args.scene]
    else:
        ap.error("pass --scene <video_id> or --all")

    if not scenes:
        print(f"no ARKitScenes scenes under {args.data_root}")
        return 1

    t0 = time.perf_counter()
    for scene_dir in scenes:
        out = render_one(scene_dir, args.out_root)
        man = json.loads((out / "manifest.json").read_text())
        vis = [v["n_visible_vertices"] for v in man["views"]]
        line = (f"{man['scene_id']}: {man['contract']['n_views']} views in "
                f"{man['render_seconds']}s  vertices={man['n_vertices']}  "
                f"visible/view min={min(vis)} med={sorted(vis)[len(vis)//2]} "
                f"max={max(vis)}")
        if args.tar:
            tar = tar_for_upload(out)
            line += f"\n  upload -> {tar} ({tar.stat().st_size/1048576:.1f} MB, ids.npz withheld)"
        print(line)
    print(f"\ntotal {time.perf_counter() - t0:.1f}s -> {args.out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
