"""M1 Stage-1-anchor: Replica room_2 under the same SAM threshold.

  python3 tools/m1_replica_anchor.py --stability 0.85

Protocol: `docs/arkitscenes_mask_coverage_protocol.md`, hazard H-C.

M1 breaks the frozen SAM pin, which ends the property that ARKitScenes and
Replica banks are products of one parameterisation. The anchor arm buys that
comparability back by running the SAME threshold on Replica room_2, so the
transfer gap can still be read like-for-like afterwards. It is REPORTED, NOT
GATED -- a decision input, not a success criterion.

Why this is a separate tool rather than a flag on `tools/p1_selector_eval.py`:
that module is the frozen Replica evaluation path and hardcodes
`bank_<scene>.npz` / `c1p1_masks_<scene>.npz`. Teaching it a variant suffix
would put the committed baseline reports one wrong flag away from being
regenerated -- the exact coupling that silently rewrote a frozen artifact
during the path-relativisation pass. Everything here is IMPORTED from it, so
the two cannot drift to different IoU, ceiling, or AR@k definitions.

ORACLE BOUNDARY: delegated wholesale to `p1_selector_eval.load_oracle`. This
module reads no annotation itself.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segmenter import selector_free
from segmenter.proposal_fusion import lift_mask
from tools.c1p1_fuse import FROZEN_STABILITY, stability_tag
from tools.p1_selector_eval import (
    ABLATIONS, DEFAULT_VARIANT, IOU_THRESHOLDS, KS, P1_ROOT, curve,
    iou_matrix, load_geometry, load_oracle, rank_order,
)

SCENE = "replica_room_2"
OUT = REPO_ROOT / "runs" / "arkitscenes_selector"

# Replica's committed 3x3 baseline, from runs/phase8_c1p1/. Quoted in the
# protocol; restated here so the comparison is self-contained.
BASELINE = {"ceiling_050": 25, "ceiling_025": 39, "n_proposals": 534,
            "occ_coverage": 0.465, "median_mask_px": 13804}


def load_bank_variant(stab: str) -> list[np.ndarray]:
    z = np.load(P1_ROOT / f"bank_{SCENE}{stab}.npz")
    verts, off = z["vertices"], z["offsets"]
    return [verts[off[i]:off[i + 1]] for i in range(len(off) - 1)]


def build_views_variant(stab: str) -> tuple[list[dict], dict]:
    """`p1_selector_eval.build_views` with the sidecar suffix threaded in.

    Same lift, same quality pair, same view contract -- only the filename
    differs. Also returns the render-side coverage statistics the protocol's
    M6 direction check is stated in, measured on the same pass.
    """
    ids = np.load(P1_ROOT / f"views_{SCENE}" / "ids.npz")
    packed = np.load(P1_ROOT / f"c1p1_masks_{SCENE}{stab}.npz")
    env = json.loads(str(packed["env"]))
    views, occ, areas = [], [], []
    for v in range(len(ids.files)):
        buf = ids[f"ids_{v:02d}"]
        raw, scores = packed[f"masks_{v:02d}"], packed[f"scores_{v:02d}"]
        masks, quality = [], []
        anym = np.zeros(buf.shape, bool)
        for m in range(raw.shape[0]):
            img = np.unpackbits(raw[m])[:buf.size].reshape(buf.shape).astype(bool)
            areas.append(int(img.sum()))
            anym |= img
            lifted = lift_mask(img, buf)
            if lifted.size:
                masks.append(lifted.astype(np.int64))
                quality.append(scores[m])
        occupied = buf >= 0
        occ.append((anym & occupied).sum() / max(occupied.sum(), 1))
        views.append({
            "visible": np.unique(buf[buf >= 0]).astype(np.int64),
            "masks": masks,
            "mask_quality": (np.asarray(quality, float) if quality
                             else np.zeros((0, 2))),
        })
    stats = {"n_2d_masks": len(areas),
             "median_mask_px": int(np.median(areas)),
             "occ_coverage": round(float(np.mean(occ)), 4),
             "recorded_stability": env.get("stability_score_thresh"),
             "recorded_scene": env.get("scene")}
    return views, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stability", type=float, default=0.85)
    args = ap.parse_args(argv)
    stab = stability_tag(args.stability)
    if not stab:
        print("--stability is the frozen pin; the anchor arm exists to "
              "compare a NON-pinned threshold against it")
        return 1

    views, mstats = build_views_variant(stab)
    if abs(float(mstats["recorded_stability"] or FROZEN_STABILITY)
           - args.stability) > 1e-9:
        print(f"sidecar records {mstats['recorded_stability']}, "
              f"asked for {args.stability} — wrong parameterisation")
        return 1

    bank = load_bank_variant(stab)
    xyz = load_geometry(SCENE)
    sig = selector_free.proposal_signals(bank, len(xyz), xyz, views)
    scores = {n: selector_free.score_proposals(sig, c)
              for n, c in ABLATIONS.items()}

    # ---- ORACLE BOUNDARY: ranking above is final ----
    oracle, oids, _classes = load_oracle(SCENE)
    ious = iou_matrix(bank, oracle, oids)
    ceiling = {f"{t:.2f}": int((ious.max(axis=0) >= t).sum())
               for t in IOU_THRESHOLDS}
    order = rank_order(scores[DEFAULT_VARIANT])
    ar = {f"{t:.2f}": curve(order, ious, t) for t in IOU_THRESHOLDS}

    sizes = np.diff(np.load(P1_ROOT / f"bank_{SCENE}{stab}.npz")["offsets"])
    rep = {"scene_id": SCENE, "role": "M1 Stage-1 anchor (reported, not gated)",
           "stability_score_thresh": args.stability,
           "n_entities": len(oids), "n_proposals": len(bank),
           "median_proposal_vertices": int(np.median(sizes)),
           "oracle_ceiling": ceiling, "ar": ar,
           "mask_stats": mstats, "baseline_3x3": BASELINE}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{SCENE}_m1_anchor{stab}.json"
    path.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")

    n = len(oids)
    print(f"=== Replica room_2 anchor @ stability={args.stability}  "
          f"({n} entities)")
    print(f"{'quantity':26s} {'baseline 0.95':>14s} {'anchor 0.85':>13s}")
    rows = [("ceiling @IoU0.50", f"{BASELINE['ceiling_050']}/{n}",
             f"{ceiling['0.50']}/{n}"),
            ("ceiling @IoU0.25", f"{BASELINE['ceiling_025']}/{n}",
             f"{ceiling['0.25']}/{n}"),
            ("AR@k=100 @IoU0.50", "—", str(ar["0.50"]["100"])),
            ("proposals", str(BASELINE["n_proposals"]), str(len(bank))),
            ("median mask px", str(BASELINE["median_mask_px"]),
             str(mstats["median_mask_px"])),
            ("occ. mask coverage", f"{BASELINE['occ_coverage']:.1%}",
             f"{mstats['occ_coverage']:.1%}")]
    for label, a, b in rows:
        print(f"{label:26s} {a:>14s} {b:>13s}")
    print(f"report -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
