"""C1-P1 fusion: SAM 2D masks + id buffers -> hash-stamped proposal bank.

  python3 tools/c1p1_fuse.py --scene replica_room_2 \
      --masks runs/phase8_c1p1/c1p1_masks_replica_room_2.npz

Inputs: the local view directory (ids.npz + manifest) and the Colab mask
sidecar (per-view packbits masks + scores + env). All fusion logic is
the frozen segmenter/proposal_fusion.py. Oracle-free: nothing semantic
is read. Output: runs/phase8_c1p1/bank_<scene>.npz + <scene>_bank.json
(hash-stamped; finalized BEFORE any oracle evaluation).
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

from geometry.mesh_surfaces import load_raw_triangle_mesh
from segmenter.proposal_fusion import (
    CONFIDENCE_CUTS, build_bank, edge_confidence, lift_mask, mesh_edges,
)
from segmenter.view_render import SIZE
from tools.c3_surface_run import _load_generation_inputs


def fuse_scene(views_dir: Path, masks_npz: Path, faces: np.ndarray,
               n_vertices: int, *,
               evidence_denominator: str = "covisible",
               ) -> tuple[list[dict], dict]:
    """`evidence_denominator` is forwarded to
    `proposal_fusion.edge_confidence`; the default is the frozen
    behaviour. See docs/arkitscenes_fusion_evidence_protocol.md."""
    manifest = json.loads((views_dir / "manifest.json").read_text())
    ids = np.load(views_dir / "ids.npz")
    masks = np.load(masks_npz, allow_pickle=False)
    views = []
    n_masks_total = n_lifted = 0
    for row in manifest["views"]:
        i = row["view"]
        idbuf = ids[f"ids_{i:02d}"]
        visible = np.unique(idbuf[idbuf >= 0])
        key = f"masks_{i:02d}"
        lifted = []
        if key in masks:
            packed = masks[key]
            n_masks_total += len(packed)
            for m in range(len(packed)):
                mask2d = np.unpackbits(packed[m])[:SIZE * SIZE]
                verts = lift_mask(mask2d.reshape(SIZE, SIZE), idbuf)
                if len(verts):
                    lifted.append(verts)
                    n_lifted += 1
        views.append({"visible": visible, "masks": lifted})
    edges = mesh_edges(faces)
    co_vis, conf = edge_confidence(
        edges, n_vertices, views,
        evidence_denominator=evidence_denominator)
    bank = build_bank(edges, co_vis, conf, n_vertices)
    stats = {
        "n_2d_masks": int(n_masks_total),
        "n_lifted_masks": int(n_lifted),
        "n_mesh_edges": int(len(edges)),
        "n_covisible_edges": int(np.count_nonzero(co_vis)),
        "confidence_cuts": list(CONFIDENCE_CUTS),
        "evidence_denominator": evidence_denominator,
        "n_proposals": len(bank),
    }
    return bank, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--masks", required=True, type=Path)
    ap.add_argument("--out-root", type=Path,
                    default=REPO_ROOT / "runs" / "phase8_c1p1")
    args = ap.parse_args(argv)

    bank_npz = args.out_root / f"bank_{args.scene}.npz"
    bank_json = args.out_root / f"{args.scene}_bank.json"
    if bank_npz.exists() or bank_json.exists():
        print(f"refusing to overwrite finalized bank: {bank_npz}")
        return 1
    views_dir = args.out_root / f"views_{args.scene}"
    mesh_path, _, _ = _load_generation_inputs(args.scene)
    mesh = load_raw_triangle_mesh(mesh_path)

    t0 = time.perf_counter()
    bank, stats = fuse_scene(views_dir, args.masks, mesh.faces, len(mesh.xyz))
    verts_concat = (np.concatenate([p["vertices"] for p in bank])
                    if bank else np.empty(0, dtype=np.int64))
    offsets = np.zeros(len(bank) + 1, dtype=np.int64)
    for i, p in enumerate(bank):
        offsets[i + 1] = offsets[i] + len(p["vertices"])
    np.savez_compressed(bank_npz, vertices=verts_concat, offsets=offsets,
                        cuts=np.array([p["cut"] for p in bank]),
                        n_vertices=np.int64(len(mesh.xyz)))
    payload = {
        "schema": "c1p1_proposal_bank_v1",
        "protocol": "docs/c1_p1_multiview_proposals_protocol.md",
        "scene_id": args.scene,
        "masks_sidecar_sha256": hashlib.sha256(
            args.masks.read_bytes()).hexdigest(),
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
    print(f"bank: {len(bank)} proposals, "
          f"{payload['serialized_bytes']/1e6:.1f} MB, "
          f"{payload['fuse_seconds']}s -> {bank_npz}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
