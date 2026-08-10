"""Loopback self-check: can the SAM arm's geometry recover objects it was handed?

  python3 tools/arkitscenes_repair_loopback.py --scene 41069021

Runs stages 3-5 of `segmenter/sam_multiview_repair.py` on SYNTHETIC masks
rendered from known 3D instances, so the answer is known before the run. It
uses no GPU, no SAM, and no annotation.

WHY THIS EXISTS
---------------
SAM inference is off-machine and costs a GPU session per scene. Before
spending one it is worth knowing whether the local half can recover an object
it is *given* a perfect 2D mask of. If lifting, association or fusion is
broken, this fails here for free instead of being mistaken for "SAM did not
find the object" after the run.

It is a NECESSARY-condition check, not a validation of the arm. Perfect
recovery here says the geometry is sound; it says nothing about whether SAM's
real masks cluster, because real masks are noisy, partial, overlapping at
several scales, and frequently absent.

HOW THE SYNTHETIC MASKS ARE MADE
--------------------------------
Each probe object is a frozen Mask3D proposal -- a real 3D vertex set, chosen
without reference to annotations. For each selected frame, the object's
visible vertices are projected and the pixels they land on become a 2D mask.
That mask is then handed back through the ordinary lifting path.

The round trip is deliberately lossy in the realistic direction: a pixel
carries every vertex on the front surface, so a mask picks up whatever else
shares those pixels -- exactly the boundary bleed real masks suffer. What it
does NOT simulate is SAM's actual failure modes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.arkitscenes import scene_id_for
from extractors.arkitscenes_rgb_crops import load_frames
from segmenter.base import sha256_file
from segmenter.rgb_multiview_repair import FRAME_STRIDE, visible_vertices
from segmenter.sam_multiview_repair import (
    MIN_MASK_VERTICES, associate, fuse_cluster, lift_masks,
)
from tools.arkitscenes_eval import DEFAULT_DATA_ROOT, load_canonical_geometry
from tools.arkitscenes_mask3d_eval import DEFAULT_BUNDLE_ROOT
from tools.arkitscenes_repair_eval import DEFAULT_OUT_ROOT, load_baseline_bank

# A probe must be visible enough in a frame for a synthetic mask to be worth
# making; below this the "perfect mask" is a handful of pixels.
MIN_PROBE_VISIBLE = 0.20
# Recovery is called successful at this IoU against the probe. Not 1.0: the
# projection round trip legitimately gains neighbouring surface at object
# boundaries and loses vertices no selected frame ever saw.
RECOVERY_IOU = 0.50


def synthetic_masks(xyz_world: np.ndarray, frame, probes: list[np.ndarray]
                    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """One perfect 2D mask per probe visible in this frame."""
    vertices, pixel = visible_vertices(xyz_world, frame)
    if len(vertices) == 0:
        return np.zeros((0, frame.height, frame.width), bool), \
            np.zeros((0, 2), np.float32), []
    visible_set = np.zeros(len(xyz_world), dtype=bool)
    visible_set[vertices] = True
    masks, owners = [], []
    for index, probe in enumerate(probes):
        seen = visible_set[probe]
        if seen.mean() < MIN_PROBE_VISIBLE:
            continue
        member = np.zeros(len(xyz_world), dtype=bool)
        member[probe] = True
        hit = member[vertices]
        if hit.sum() < MIN_MASK_VERTICES:
            continue
        canvas = np.zeros(frame.height * frame.width, dtype=bool)
        canvas[pixel[hit]] = True
        masks.append(canvas.reshape(frame.height, frame.width))
        owners.append(index)
    if not masks:
        return np.zeros((0, frame.height, frame.width), bool), \
            np.zeros((0, 2), np.float32), []
    return (np.stack(masks),
            np.ones((len(masks), 2), dtype=np.float32), owners)


def run(scene_dir: Path, bundle_dir: Path, out_dir: Path,
        n_probes: int) -> dict:
    frames_json = out_dir / f"repair_frames_{scene_id_for(scene_dir)}" / "frames.json"
    if not frames_json.is_file():
        raise FileNotFoundError(
            f"{frames_json} — run tools/arkitscenes_repair_frames.py first")
    manifest = json.loads(frames_json.read_text())

    mesh, rotation, _ = load_canonical_geometry(scene_dir)
    xyz_world = mesh.xyz @ rotation
    n_vertices = len(xyz_world)
    mesh_sha = sha256_file(scene_dir / f"{scene_dir.name}_3dod_mesh_canonical.ply")
    baseline_props, _ = load_baseline_bank(bundle_dir, n_vertices, mesh_sha)

    # Probes: the largest frozen Mask3D proposals that are not already
    # room-sized. Largest, because a synthetic mask of a 200-vertex fragment
    # tests nothing about association.
    ranked = sorted(baseline_props, key=lambda p: -len(p.vertices))
    probes = [p.vertices for p in ranked
              if len(p.vertices) <= 0.15 * n_vertices][:n_probes]

    frames = load_frames(scene_dir, stride=manifest.get("frame_stride",
                                                        FRAME_STRIDE))
    selected = [row["frame_index"] for row in manifest["frames"]]

    visibility, lifted, owner_of = {}, [], {}
    for slot, frame_index in enumerate(selected):
        frame = frames[frame_index]
        vertices, _ = visible_vertices(xyz_world, frame)
        visibility[frame_index] = vertices
        masks, scores, owners = synthetic_masks(xyz_world, frame, probes)
        view_masks = lift_masks(xyz_world, frame, frame_index, masks, scores)
        # lift_masks drops masks below the size floor, so re-derive ownership
        # from the mask index it preserved rather than assuming alignment.
        for lifted_mask in view_masks:
            owner_of[(frame_index, lifted_mask.mask_index)] = \
                owners[lifted_mask.mask_index]
        lifted.extend(view_masks)

    groups = associate(lifted, frames, n_vertices)
    clusters = []
    for group in groups:
        fused = fuse_cluster([lifted[i] for i in group], frames, visibility,
                             n_vertices)
        if fused is not None:
            clusters.append(fused)

    rows = []
    for index, probe in enumerate(probes):
        views = sum(1 for m in lifted
                    if owner_of.get((m.view, m.mask_index)) == index)
        best_iou, best_size = 0.0, 0
        for cluster in clusters:
            inter = np.intersect1d(cluster.vertices, probe,
                                   assume_unique=True).size
            if not inter:
                continue
            iou = inter / (len(cluster.vertices) + len(probe) - inter)
            if iou > best_iou:
                best_iou, best_size = iou, len(cluster.vertices)
        rows.append({
            "probe": index,
            "n_vertices": int(len(probe)),
            "n_synthetic_masks": views,
            "best_cluster_iou": round(best_iou, 4),
            "best_cluster_vertices": best_size,
            "recovered": best_iou >= RECOVERY_IOU,
        })

    eligible = [r for r in rows if r["n_synthetic_masks"] >= 2]
    recovered = [r for r in eligible if r["recovered"]]
    return {
        "check": "sam_repair_loopback",
        "scene_id": scene_id_for(scene_dir),
        "recovery_iou": RECOVERY_IOU,
        "n_frames": len(selected),
        "n_probes": len(probes),
        "n_probes_in_two_or_more_views": len(eligible),
        "n_recovered": len(recovered),
        "recovery_rate_of_eligible": (round(len(recovered) / len(eligible), 4)
                                      if eligible else None),
        "median_iou_of_eligible": (
            round(float(np.median([r["best_cluster_iou"] for r in eligible])), 4)
            if eligible else None),
        "n_lifted_masks": len(lifted),
        "n_clusters": len(clusters),
        "per_probe": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", default="41069021")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--n-probes", type=int, default=12)
    args = ap.parse_args(argv)

    scene_dir = args.data_root / args.scene
    scene_id = scene_id_for(scene_dir)
    out_dir = args.out_root / scene_id
    t0 = time.perf_counter()
    report = run(scene_dir, args.bundle_root / f"bundle_{scene_id}", out_dir,
                 args.n_probes)
    report["runtime_seconds"] = round(time.perf_counter() - t0, 1)
    path = out_dir / "repair_loopback.json"
    path.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")

    print(f"=== {scene_id} loopback   {report['n_frames']} frames, "
          f"{report['n_probes']} probes")
    print(f"    synthetic masks     : {report['n_lifted_masks']} lifted -> "
          f"{report['n_clusters']} supported clusters")
    print(f"    probes in >=2 views : {report['n_probes_in_two_or_more_views']}"
          f"/{report['n_probes']}")
    print(f"    recovered @IoU{RECOVERY_IOU:.2f}: {report['n_recovered']}"
          f"/{report['n_probes_in_two_or_more_views']}   "
          f"median IoU {report['median_iou_of_eligible']}")
    for row in report["per_probe"]:
        flag = "ok " if row["recovered"] else ("-- " if row["n_synthetic_masks"] < 2
                                               else "MISS")
        print(f"      {flag} probe {row['probe']:2d}  {row['n_vertices']:7d}v  "
              f"{row['n_synthetic_masks']:2d} masks  IoU {row['best_cluster_iou']:.3f}")
    print(f"    report -> {path}   ({report['runtime_seconds']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
