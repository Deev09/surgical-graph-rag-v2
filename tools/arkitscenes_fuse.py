"""Fuse ARKitScenes SAM masks into a C1-P1 proposal bank.

  python3 tools/arkitscenes_fuse.py --scene 41069021

`fuse_scene` is IMPORTED from tools/c1p1_fuse.py, not reimplemented -- it is
already dataset-agnostic (views_dir, masks_npz, faces, n_vertices). Only
`main()` in that module is Replica-bound, via the scene lock. This wrapper
supplies ARKitScenes geometry from the adapter and writes the same
`c1p1_proposal_bank_v1` artifact to runs/arkitscenes_p1/, so the bank a
Replica scene produces and the bank an ARKitScenes scene produces are the
same object and downstream cannot tell them apart.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.arkitscenes import scene_id_for
from tools.arkitscenes_eval import DEFAULT_DATA_ROOT, DEFAULT_VIEWS_ROOT
from tools.arkitscenes_eval import load_canonical_geometry
from segmenter.proposal_fusion import EVIDENCE_DENOMINATORS
from tools.arkitscenes_render import views_dir_for
from tools.c1p1_fuse import fuse_scene


def bank_paths(views_root: Path, scene_id: str,
               evidence_denominator: str,
               variant: str = "") -> tuple[Path, Path]:
    """`covisible` keeps the original names so existing artifacts and
    downstream defaults are untouched; other modes get a suffix so both
    banks coexist for a same-run comparison."""
    tag = "" if evidence_denominator == "covisible" else f".{evidence_denominator}"
    tag += variant
    return (views_root / f"bank_{scene_id}{tag}.npz",
            views_root / f"{scene_id}_bank{tag}.json")


def fuse_one(scene_dir: Path, views_root: Path,
             evidence_denominator: str = "covisible",
             rgb_splat: str = "3x3", id_splat: str = "3x3") -> int:
    scene_id = scene_id_for(scene_dir)
    views_dir = views_dir_for(views_root, scene_id, rgb_splat, id_splat)
    # masks are keyed to the RGB image only, so an arm that leaves RGB
    # untouched reuses the existing sidecar byte-for-byte
    masks_npz = views_root / f"c1p1_masks_{scene_id}.npz"
    variant = ("" if (rgb_splat, id_splat) == ("3x3", "3x3")
               else f".rgb{rgb_splat}_id{id_splat}")
    bank_npz, bank_json = bank_paths(views_root, scene_id,
                                     evidence_denominator, variant)

    for p, how in ((views_dir / "manifest.json", "tools/arkitscenes_render.py"),
                   (masks_npz, "notebooks/c1p1_sam2_colab.ipynb")):
        if not p.exists():
            print(f"{scene_id}: missing {p} — produce it with {how}")
            return 1
    if bank_npz.exists() or bank_json.exists():
        print(f"{scene_id}: refusing to overwrite finalized bank: {bank_npz}")
        return 1

    mesh, _R, bundle = load_canonical_geometry(scene_dir)
    man = json.loads((views_dir / "manifest.json").read_text())
    if man["source"]["representation_hash"] != bundle.representation_hash:
        print(f"{scene_id}: views were rendered from "
              f"{man['source']['representation_hash']}, adapter now produces "
              f"{bundle.representation_hash} — re-render before fusing")
        return 1

    t0 = time.perf_counter()
    bank, stats = fuse_scene(views_dir, masks_npz, mesh.faces, len(mesh.xyz),
                             evidence_denominator=evidence_denominator)
    verts = (np.concatenate([p["vertices"] for p in bank])
             if bank else np.empty(0, dtype=np.int64))
    offsets = np.zeros(len(bank) + 1, dtype=np.int64)
    for i, p in enumerate(bank):
        offsets[i + 1] = offsets[i] + len(p["vertices"])
    np.savez_compressed(bank_npz, vertices=verts, offsets=offsets,
                        cuts=np.array([p["cut"] for p in bank]),
                        n_vertices=np.int64(len(mesh.xyz)))
    payload = {
        "schema": "c1p1_proposal_bank_v1",
        "protocol": "docs/c1_p1_multiview_proposals_protocol.md",
        "scene_id": scene_id,
        "dataset": "arkitscenes-3dod-raw",
        "video_id": scene_dir.name,
        "evidence_denominator": evidence_denominator,
        "rgb_splat": rgb_splat,
        "id_splat": id_splat,
        "representation_hash": bundle.representation_hash,
        "masks_sidecar_sha256": hashlib.sha256(masks_npz.read_bytes()).hexdigest(),
        "views_manifest_sha256": hashlib.sha256(
            (views_dir / "manifest.json").read_bytes()).hexdigest(),
        **stats,
        "serialized_bytes": bank_npz.stat().st_size,
        "fuse_seconds": round(time.perf_counter() - t0, 1),
        "bank_npz_sha256": hashlib.sha256(bank_npz.read_bytes()).hexdigest(),
        "proposal_digests_sha256": hashlib.sha256(
            "".join(p["digest"] for p in bank).encode()).hexdigest(),
    }
    bank_json.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n",
                         encoding="utf-8")
    extra = {k: v for k, v in stats.items() if isinstance(v, (int, float))}
    print(f"{scene_id}: {len(bank)} proposals, "
          f"{payload['serialized_bytes']/1e6:.1f} MB, "
          f"{payload['fuse_seconds']}s -> {bank_npz}")
    print("   " + "  ".join(f"{k}={v}" for k, v in sorted(extra.items())))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", help="ARKitScenes video_id, e.g. 41069021")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--views-root", type=Path, default=DEFAULT_VIEWS_ROOT)
    ap.add_argument("--rgb-splat", default="3x3")
    ap.add_argument("--id-splat", default="3x3")
    ap.add_argument("--evidence-denominator", default="covisible",
                    choices=list(EVIDENCE_DENOMINATORS),
                    help="fusion evidence rule; see "
                         "docs/arkitscenes_fusion_evidence_protocol.md")
    args = ap.parse_args(argv)

    if args.all:
        scenes = sorted(d for d in args.data_root.iterdir()
                        if d.is_dir()
                        and (d / f"{d.name}_3dod_mesh.ply").is_file())
    elif args.scene:
        scenes = [args.data_root / args.scene]
    else:
        ap.error("pass --scene <video_id> or --all")
    return max(fuse_one(d, args.views_root, args.evidence_denominator,
                        args.rgb_splat, args.id_splat)
               for d in scenes)


if __name__ == "__main__":
    sys.exit(main())
