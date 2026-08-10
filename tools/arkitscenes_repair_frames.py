"""Select repair frames and pack a hash-pinned portable bundle for SAM 2.1.

  python3 tools/arkitscenes_repair_frames.py --scene 41069021 --tar

Stage 1-2 of the SAM repair arm (`docs/repair_arm_design_note.md`). Local and
CPU-only: it chooses which real capture frames the GPU stage will segment,
copies those exact PNGs into a portable directory, and hash-pins every one of
them plus the selection itself.

WHY A SEPARATE PORTABLE ARTIFACT
--------------------------------
SAM inference runs off this machine (the pinned configuration is CUDA bf16 on
an A100; see `notebooks/repair_sam2_colab.ipynb`). The mask sidecar that comes
back has to be joinable to exactly the frames that produced it, months later,
without trusting a filename. So:

  * `frames.json` records, per slot, the source PNG name, its sha256, the
    timestamp, the pose and the intrinsics. Poses and intrinsics stay LOCAL.
  * `upload_manifest.json` carries only what the GPU stage needs -- slot,
    filename, sha256, size -- and is what ships inside the tar.
  * `selection_sha256` hashes the ordered (slot, source name, sha256) triples.
    The mask sidecar records the same value; the lifting stage refuses a
    sidecar whose selection hash does not match.

ISOLATION. The tar contains RGB PNGs and the upload manifest. It cannot
contain poses, id buffers, the mesh, Mask3D output or annotations, and a
post-write check asserts that. Poses are withheld not because they are secret
but because the GPU stage has no use for them, and every extra field is
another way for a downstream run to diverge from the pin.

Nothing here reads an annotation, and the selection is deterministic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.arkitscenes import scene_id_for
from extractors.arkitscenes_rgb_crops import load_frames
from segmenter.base import sha256_file
from segmenter.rgb_multiview_repair import FRAME_STRIDE
from segmenter.sam_multiview_repair import N_FRAMES, config_record, select_frames
from tools.arkitscenes_eval import DEFAULT_DATA_ROOT, load_canonical_geometry
from tools.arkitscenes_repair_eval import DEFAULT_OUT_ROOT

SCHEMA = "arkitscenes_repair_frames_v1"


def selection_digest(rows: list[dict]) -> str:
    """Hash of the ordered selection: slot, source filename, image sha256."""
    h = hashlib.sha256()
    for row in rows:
        h.update(f"{row['slot']}\x00{row['source_png']}\x00"
                 f"{row['sha256']}\x01".encode())
    return h.hexdigest()


def export_frames(scene_dir: Path, out_dir: Path, *, n_frames: int = N_FRAMES,
                  frame_stride: int = FRAME_STRIDE) -> dict:
    mesh, rotation, bundle = load_canonical_geometry(scene_dir)
    xyz_world = mesh.xyz @ rotation
    frames = load_frames(scene_dir, stride=frame_stride)
    if not frames:
        raise ValueError(f"no pose-matched frames under {scene_dir}")
    selected, selection_diagnostics = select_frames(
        frames, xyz_world, n_frames=n_frames)
    if len(selected) < 16:
        raise ValueError(
            f"only {len(selected)} frames survived selection; the brief "
            "requires at least 16")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    rows = []
    for slot, frame_index in enumerate(selected):
        frame = frames[frame_index]
        destination = out_dir / f"frame_{slot:02d}.png"
        shutil.copyfile(frame.png, destination)
        rows.append({
            "slot": slot,
            "frame_index": int(frame_index),
            "source_png": frame.png.name,
            "sha256": sha256_file(destination),
            "bytes": destination.stat().st_size,
            "timestamp": round(float(frame.timestamp), 6),
            "width": int(frame.width), "height": int(frame.height),
            "intrinsics": {"fx": frame.fx, "fy": frame.fy,
                           "cx": frame.cx, "cy": frame.cy},
            "R_wc": frame.R_wc.tolist(),
            "t_wc": frame.t_wc.tolist(),
        })

    digest = selection_digest(rows)
    local = {
        "schema": SCHEMA,
        "scene_id": scene_id_for(scene_dir),
        "video_id": scene_dir.name,
        "representation_hash": bundle.representation_hash,
        "canonical_mesh_sha256": sha256_file(
            scene_dir / f"{scene_dir.name}_3dod_mesh_canonical.ply"),
        "frame_stride": frame_stride,
        "selection_sha256": digest,
        "selection_diagnostics": selection_diagnostics,
        "config": config_record(),
        "annotations_read": False,
        "frames": rows,
    }
    (out_dir / "frames.json").write_text(
        json.dumps(local, indent=1, sort_keys=True) + "\n")

    # Upload side: no poses, no intrinsics, no mesh identity.
    upload = {
        "schema": SCHEMA + "_upload",
        "scene_id": local["scene_id"],
        "selection_sha256": digest,
        "n_frames": len(rows),
        "frames": [{"slot": r["slot"], "png": f"frame_{r['slot']:02d}.png",
                    "sha256": r["sha256"], "bytes": r["bytes"]}
                   for r in rows],
    }
    (out_dir / "upload_manifest.json").write_text(
        json.dumps(upload, indent=1, sort_keys=True) + "\n")
    return local


def tar_for_upload(out_dir: Path) -> Path:
    """PNGs + upload manifest only. Everything else is withheld."""
    tar_path = out_dir.parent / (out_dir.name + ".tar.gz")
    members = sorted(out_dir.glob("frame_*.png")) + [out_dir / "upload_manifest.json"]
    missing = [p.name for p in members if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{out_dir}: missing {missing}")
    with tarfile.open(tar_path, "w:gz") as tf:
        for p in members:
            tf.add(p, arcname=f"{out_dir.name}/{p.name}")
    with tarfile.open(tar_path) as tf:
        names = tf.getnames()
    leaked = [n for n in names
              if n.endswith((".npz", ".npy", ".ply"))
              or Path(n).name == "frames.json"]
    if leaked:
        raise AssertionError(f"withheld files leaked into the upload tar: {leaked}")
    return tar_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", default="41069021")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--n-frames", type=int, default=N_FRAMES)
    ap.add_argument("--frame-stride", type=int, default=FRAME_STRIDE)
    ap.add_argument("--tar", action="store_true",
                    help="also pack PNGs + upload manifest for the GPU stage")
    args = ap.parse_args(argv)

    if not 16 <= args.n_frames <= 32:
        ap.error("--n-frames must be in the 16-32 band the brief sets")

    scene_dir = args.data_root / args.scene
    scene_id = scene_id_for(scene_dir)
    out_dir = args.out_root / scene_id / f"repair_frames_{scene_id}"
    t0 = time.perf_counter()
    manifest = export_frames(scene_dir, out_dir, n_frames=args.n_frames,
                             frame_stride=args.frame_stride)
    diagnostics = manifest["selection_diagnostics"]

    print(f"=== {scene_id}")
    print(f"    candidate frames    : {diagnostics['n_candidate_frames']} "
          f"({diagnostics['n_valid_frames']} visibility-valid)")
    print(f"    selected            : {diagnostics['n_selected']} at "
          f">={diagnostics['min_angular_separation_deg']:.0f} deg separation")
    print(f"    mesh seen           : {diagnostics['mesh_seen_any_view']:.1%} "
          f"in >=1 view, {diagnostics['mesh_seen_two_views']:.1%} in >=2, "
          f"{diagnostics['mesh_seen_three_views']:.1%} in >=3")
    print(f"    selection sha256    : {manifest['selection_sha256'][:16]}…")
    print(f"    frames -> {out_dir}")
    if args.tar:
        tar_path = tar_for_upload(out_dir)
        print(f"    upload -> {tar_path} "
              f"({tar_path.stat().st_size / 1e6:.1f} MB, "
              f"sha256 {sha256_file(tar_path)[:16]}…)")
    print(f"    {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
