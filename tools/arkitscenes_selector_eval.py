"""Score an ARKitScenes P1 bank with the oracle-free selector.

  python3 tools/arkitscenes_selector_eval.py --all

Requires the Colab SAM stage and `tools/c1p1_fuse.py` to have run:
`runs/arkitscenes_p1/c1p1_masks_<scene>.npz` and `bank_<scene>.npz`.

Mirrors `tools/p1_selector_eval.py` -- same scorer, same ablation table,
same ranking helpers, all IMPORTED rather than reimplemented, so the two
datasets cannot drift to different scoring and their AR@k stay comparable.
What is dataset-specific here is only I/O: where the geometry, id buffers,
masks and oracle come from.

ORACLE BOUNDARY, same discipline as the Replica tool: everything above the
boundary comment could run on a scene with no ground truth. The oracle side
delegates to `tools.arkitscenes_eval`, which is the one module allowed to
read annotations.

`build_views` deliberately reuses `proposal_fusion.lift_mask` -- the same
function that built the bank -- so the scorer sees exactly the co-membership
evidence the generator saw, and an agreement score cannot be inflated by
lifting masks a second, different way.
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

from segmenter import selector_free
from segmenter.proposal_fusion import lift_mask
from tools.arkitscenes_eval import (
    DEFAULT_DATA_ROOT, DEFAULT_VIEWS_ROOT, iou_matrix,
    load_canonical_geometry, load_oracle_entities,
)
from tools.p1_selector_eval import (
    ABLATIONS, DEFAULT_VARIANT, IOU_THRESHOLDS, KS, curve, rank_order,
)
from adapters.arkitscenes import scene_id_for
from segmenter.proposal_fusion import EVIDENCE_DENOMINATORS
from tools.arkitscenes_fuse import (
    FROZEN_STABILITY, bank_paths, stability_tag,
)

OUT_ROOT = REPO_ROOT / "runs" / "arkitscenes_selector"


# --------------------------------------------------------------------
# oracle-free half — nothing here reads an annotation
# --------------------------------------------------------------------
def build_views(views_dir: Path, masks_path: Path) -> list[dict]:
    """The 40 views in `proposal_fusion.edge_confidence`'s contract."""
    ids = np.load(views_dir / "ids.npz")
    packed = np.load(masks_path)
    views = []
    for v in range(len(ids.files)):
        buf = ids[f"ids_{v:02d}"]
        raw = packed[f"masks_{v:02d}"]
        scores = packed[f"scores_{v:02d}"]
        masks, quality = [], []
        for m in range(raw.shape[0]):
            img = np.unpackbits(raw[m])[:buf.size].reshape(buf.shape).astype(bool)
            lifted = lift_mask(img, buf)
            if lifted.size:
                masks.append(lifted.astype(np.int64))
                quality.append(scores[m])
        views.append({
            "visible": np.unique(buf[buf >= 0]).astype(np.int64),
            "masks": masks,
            "mask_quality": (np.asarray(quality, float) if quality
                             else np.zeros((0, 2))),
        })
    return views


def load_bank(bank_path: Path) -> list[np.ndarray]:
    z = np.load(bank_path)
    verts, off = z["vertices"], z["offsets"]
    return [verts[off[i]:off[i + 1]] for i in range(len(off) - 1)]


def rank_scene(scene_dir: Path, views_root: Path,
               evidence_denominator: str = "covisible",
               stability: float = FROZEN_STABILITY) -> dict:
    """Finished, oracle-free ranking. No annotation is open yet."""
    t0 = time.perf_counter()
    scene_id = scene_id_for(scene_dir)
    views_dir = views_root / f"views_{scene_id}"
    stab = stability_tag(stability)
    masks_path = views_root / f"c1p1_masks_{scene_id}{stab}.npz"
    bank_path, _ = bank_paths(views_root, scene_id,
                              evidence_denominator, stab)
    for p, how in ((views_dir / "ids.npz", "tools/arkitscenes_render.py"),
                   (masks_path, "notebooks/c1p1_sam2_colab.ipynb"),
                   (bank_path, "tools/c1p1_fuse.py")):
        if not p.exists():
            raise FileNotFoundError(f"missing {p} — produce it with {how}")

    mesh, _R, bundle = load_canonical_geometry(scene_dir)
    bank = load_bank(bank_path)
    views = build_views(views_dir, masks_path)
    n = len(mesh.xyz)
    sig = selector_free.proposal_signals(bank, n, mesh.xyz, views)
    return {
        "scene_id": scene_id,
        "video_id": scene_dir.name,
        "evidence_denominator": evidence_denominator,
        "stability_score_thresh": stability,
        "representation_hash": bundle.representation_hash,
        "n_vertices": n,
        "n_proposals": len(bank),
        "n_views": len(views),
        "n_lifted_masks": int(sum(len(v["masks"]) for v in views)),
        "proposals": bank,
        "signals": sig,
        "scores": {name: selector_free.score_proposals(sig, comps)
                   for name, comps in ABLATIONS.items()},
        "rank_seconds": round(time.perf_counter() - t0, 1),
    }


# --------------------------------------------------------------------
# ORACLE BOUNDARY — nothing below may influence the ranking
# --------------------------------------------------------------------
def evaluate(ranked: dict, scene_dir: Path) -> dict:
    mesh, R, _ = load_canonical_geometry(scene_dir)
    entities = load_oracle_entities(scene_dir, mesh.xyz, R)
    ious = iou_matrix(ranked["proposals"], entities)
    ceiling = {f"{t:.2f}": int((ious.max(axis=0) >= t).sum())
               for t in IOU_THRESHOLDS}
    order = rank_order(ranked["scores"][DEFAULT_VARIANT])
    out = {k: ranked[k] for k in
           ("scene_id", "video_id", "evidence_denominator",
            "stability_score_thresh",
            "representation_hash", "n_vertices",
            "n_proposals", "n_views", "n_lifted_masks", "rank_seconds")}
    out["n_entities"] = len(entities)
    out["oracle_ceiling"] = ceiling
    out["ar"] = {f"{t:.2f}": curve(order, ious, t) for t in IOU_THRESHOLDS}
    out["recovery"] = {
        t: {k: (round(v / ceiling[t], 3) if ceiling[t] else None)
            for k, v in c.items()}
        for t, c in out["ar"].items()}
    out["zero_overlap_share"] = {
        ("all" if k is None else str(k)): round(float(
            (ious[order if k is None else order[:k]].max(axis=1) < 0.10).mean()), 3)
        for k in KS}
    out["ablation"] = {
        name: {f"{t:.2f}": curve(rank_order(ranked["scores"][name]), ious, t)
               for t in IOU_THRESHOLDS}
        for name in ABLATIONS}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--views-root", type=Path, default=DEFAULT_VIEWS_ROOT)
    ap.add_argument("--stability", type=float, default=FROZEN_STABILITY)
    ap.add_argument("--evidence-denominator", default="covisible",
                    choices=list(EVIDENCE_DENOMINATORS))
    args = ap.parse_args(argv)

    if args.all:
        scenes = sorted(d for d in args.data_root.iterdir()
                        if d.is_dir()
                        and (d / f"{d.name}_3dod_mesh.ply").is_file())
    elif args.scene:
        scenes = [args.data_root / args.scene]
    else:
        ap.error("pass --scene <video_id> or --all")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ks = [("all" if k is None else str(k)) for k in KS]
    for scene_dir in scenes:
        try:
            ranked = rank_scene(scene_dir, args.views_root,
                                args.evidence_denominator,
                                args.stability)
        except FileNotFoundError as e:
            print(f"=== {scene_id_for(scene_dir)}\n    SKIP — {e}")
            continue
        rep = evaluate(ranked, scene_dir)
        ceil50 = rep["oracle_ceiling"]["0.50"]
        print(f"=== {rep['scene_id']}  ({rep['rank_seconds']}s rank)")
        print(f"    proposals={rep['n_proposals']}  entities={rep['n_entities']}"
              f"  lifted_masks={rep['n_lifted_masks']}")
        print(f"    ceiling@IoU0.50     : {ceil50}/{rep['n_entities']}")
        print("    k                   : " + " ".join(f"{k:>5s}" for k in ks))
        print("    AR@k  IoU0.50       : "
              + " ".join(f"{rep['ar']['0.50'][k]:5d}" for k in ks))
        print("    recovery            : "
              + " ".join(
                  f"{rep['recovery']['0.50'][k]:5.2f}"
                  if rep['recovery']['0.50'][k] is not None else "   --"
                  for k in ks))
        print("    no-entity share     : "
              + " ".join(f"{rep['zero_overlap_share'][k]:5.2f}" for k in ks))
        tag = (("" if args.evidence_denominator == "covisible"
                else f".{args.evidence_denominator}")
               + stability_tag(args.stability))
        path = OUT_ROOT / f"{rep['scene_id']}_selector_eval{tag}.json"
        path.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
        print(f"    report              -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
