"""C1-P1 view generation (local, CPU, isolation-critical).

  python3 tools/c1p1_render.py --scene replica_room_2

Renders the 40 frozen views (RGB PNG + int32 id buffers) from the raw
mesh only. Reads: the raw-input lock, the frozen frame sidecar, and
`mesh.ply`. NEVER reads semantic meshes, metadata, keys, or answers —
the Stage-0 isolation test enforces this on the core path.

Output: runs/phase8_c1p1/views_<scene>/ with view_XX.png, ids.npz, and
manifest.json (cameras, per-file sha256, mesh sha, contract constants).
The RGB PNGs are the exact SAM inputs for the Colab inference notebook.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry.mesh_surfaces import load_raw_triangle_mesh, transform_mesh
from segmenter.view_render import (
    EYE_HEIGHT_M, NEAR_M, ORIGIN_FRAC, PITCH_DEG, SIZE, SPLAT_OFFSETS,
    VFOV_DEG, render_all,
)
from tools.c3_surface_run import _load_generation_inputs


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def render_scene(mesh_path: Path, frame: dict, out_dir: Path,
                 mesh_sha: str, scene_id: str, *,
                 rgb_offsets=SPLAT_OFFSETS, id_offsets=None) -> Path:
    """Kernels default to the frozen 3x3 contract; see
    docs/arkitscenes_render_density_protocol.md."""
    mesh = transform_mesh(load_raw_triangle_mesh(mesh_path),
                          np.asarray(frame["world_from_raw_rotation"]),
                          np.asarray(frame["world_from_raw_translation"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    id_arrays: dict[str, np.ndarray] = {}
    rows = []
    for i, cam, img, ids in render_all(mesh.xyz, mesh.rgb,
                                       rgb_offsets=rgb_offsets,
                                       id_offsets=id_offsets):
        png = out_dir / f"view_{i:02d}.png"
        Image.fromarray(img).save(png, optimize=False)
        id_arrays[f"ids_{i:02d}"] = ids
        rows.append({
            "view": i,
            "origin": [round(float(x), 6) for x in cam.origin],
            "yaw_deg": cam.yaw_deg, "pitch_deg": cam.pitch_deg,
            "rgb_sha256": sha256_bytes(png.read_bytes()),
            "id_sha256": sha256_bytes(ids.tobytes()),
            "n_visible_vertices": int(len(np.unique(ids[ids >= 0]))),
        })
    ids_path = out_dir / "ids.npz"
    np.savez_compressed(ids_path, **id_arrays)
    manifest = {
        "schema": "c1p1_view_manifest_v1",
        "protocol": "docs/c1_p1_multiview_proposals_protocol.md",
        "scene_id": scene_id,
        "input_mesh_sha256": mesh_sha,
        "n_vertices": int(len(mesh.xyz)),
        "n_source_quads": int(mesh.n_source_quads),
        "contract": {"size": SIZE, "vfov_deg": VFOV_DEG, "near_m": NEAR_M,
                     "pitch_deg": PITCH_DEG, "eye_height_m": EYE_HEIGHT_M,
                     "origin_frac": ORIGIN_FRAC, "n_views": len(rows),
                     "splat_px": 3, "background": "black"},
        "views": rows,
        "ids_npz_sha256": sha256_bytes(ids_path.read_bytes()),
        "render_seconds": round(time.perf_counter() - t0, 1),
    }
    mpath = out_dir / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n",
                     encoding="utf-8")
    return mpath


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out-root", type=Path,
                    default=REPO_ROOT / "runs" / "phase8_c1p1")
    args = ap.parse_args(argv)
    mesh_path, _, frame = _load_generation_inputs(args.scene)
    lock = json.loads((REPO_ROOT / "tools" / "replica_scenes.lock.json")
                      .read_text())
    rows = {r["relpath"]: r for r in lock["files"]}
    short = args.scene.replace("replica_", "")
    mesh_sha = rows[f"{short}/mesh.ply"]["sha256"]
    out = args.out_root / f"views_{args.scene}"
    if out.exists():
        print(f"refusing to overwrite existing views: {out}")
        return 1
    mpath = render_scene(mesh_path, frame, out, mesh_sha, args.scene)
    m = json.loads(mpath.read_text())
    print(f"rendered {m['contract']['n_views']} views in "
          f"{m['render_seconds']}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
