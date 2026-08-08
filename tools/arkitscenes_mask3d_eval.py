"""ARKitScenes × Mask3D: compare P1, Mask3D and pooled banks on one scan.

  python3 tools/arkitscenes_mask3d_eval.py --scene 41069021

Contract: `docs/arkitscenes_mask3d_contract.md`. One dev scene, one
configuration, no sweep, fixed gates decided before the run.

This is the first proposal source in the project that never touches a
rendered image: Mask3D consumes the canonical mesh directly. That is the
whole point -- four refuted knobs (F1, R1 arm A, R1 arm C, M1) all sat
downstream of what SAM's masks mean on a stippled real-scan render, and this
tests a mechanism that does not inherit that.

Everything comparative is IMPORTED rather than reimplemented -- the oracle
and IoU from `tools.arkitscenes_eval`, the view contract and bank loader
from `tools.arkitscenes_selector_eval`, the scorer and rank order from
`tools.p1_selector_eval` -- so the three banks cannot be scored by three
subtly different rules, and the ARKitScenes numbers stay comparable to the
Replica ones quoted in the contract.

ORACLE BOUNDARY is inherited: `tools.arkitscenes_eval` is the only module
permitted to open the ARKitScenes annotation file, and nothing above the
boundary comment below reaches for one. The filename is deliberately not
spelled here -- the repo-level sweep in `tests/tools/test_arkitscenes_eval.py`
scans every module for it, and this one has no business needing an exemption.
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

from adapters.arkitscenes import scene_id_for
from segmenter import selector_free
from tools.arkitscenes_eval import (
    DEFAULT_DATA_ROOT, DEFAULT_VIEWS_ROOT, iou_matrix,
    load_canonical_geometry, load_oracle_entities,
)
from tools.arkitscenes_selector_eval import build_views, load_bank
from tools.c1_resolve_sweep import load_raw_masks
from tools.p1_selector_eval import ABLATIONS, DEFAULT_VARIANT, rank_order

DEFAULT_BUNDLE_ROOT = REPO_ROOT / "runs" / "arkitscenes_mask3d"
# The report is written BESIDE the bundle it came from, never to a fixed
# root: a dry run pointed at a scratch --bundle-root must not be able to
# drop a report into the real results directory, where its provenance would
# have to be read out of the JSON to notice it was synthetic.

# The contract's reporting grid -- deliberately NOT p1_selector_eval.KS /
# IOU_THRESHOLDS, which are the Replica grid. Stated here so a later edit to
# the Replica tool cannot silently redefine what this contract reported.
IOUS = (0.10, 0.25, 0.50)
KS = (25, 50, 100, 200, None)

# frozen resolve configuration, from the contract
MIN_SCORE = 0.2
MIN_VERTICES = 20

# share of mesh vertices above which a proposal is "giant" (diagnostic only)
GIANT_FRAC = 0.15

# Gates, fixed before the run. Mask3D alone at IoU 0.50, out of 18 entities.
GATE_STRONG = 6
GATE_PASS = 4
GATE_STOP = 3


def load_mask3d_banks(bundle_dir: Path, n_vertices: int,
                      mesh_sha256: str) -> tuple[list, list, dict]:
    """(raw, ms02, provenance). Hard-gated on the mesh the bundle ran against.

    A bundle produced from a different mesh -- or a different vertex count --
    would index into someone else's geometry and silently produce a bank that
    pools with nothing. Asserted rather than assumed, as everywhere else here.
    """
    meta = json.loads((bundle_dir / "meta.json").read_text())
    if int(meta["n_vertices"]) != n_vertices:
        raise ValueError(
            f"bundle ran on {meta['n_vertices']} vertices, the canonical mesh "
            f"has {n_vertices} — different geometry, banks cannot pool")
    if meta["input_mesh_sha256"] != mesh_sha256:
        raise ValueError(
            f"bundle ran on mesh {meta['input_mesh_sha256'][:16]}…, this "
            f"scene's canonical mesh is {mesh_sha256[:16]}… — wrong input")

    masks, scores = load_raw_masks(bundle_dir)
    if masks.shape[1] != n_vertices:
        raise ValueError(f"raw masks are {masks.shape[1]} wide, expected "
                         f"{n_vertices}")
    sizes = masks.sum(axis=1)
    raw = [np.flatnonzero(masks[k]).astype(np.int64) for k in range(len(masks))]
    keep = [k for k in range(len(masks))
            if scores[k] >= MIN_SCORE and sizes[k] >= MIN_VERTICES]
    ms02 = [np.flatnonzero(masks[k]).astype(np.int64) for k in keep]
    prov = {
        "segmenter_name": meta.get("segmenter_name"),
        "segmenter_version": meta.get("segmenter_version"),
        "hardware": meta.get("hardware"),
        "runtime_seconds": meta.get("runtime_seconds"),
        "input_mesh_sha256": meta["input_mesh_sha256"],
        "n_raw_masks": int(len(masks)),
        "n_after_min_score": int(len(ms02)),
        "min_score": MIN_SCORE, "min_vertices": MIN_VERTICES,
        "raw_score_range": [float(scores.min()), float(scores.max())],
        "ms02_scores": [float(scores[k]) for k in keep],
    }
    return raw, ms02, prov


def bank_report(name: str, proposals: list[np.ndarray], ious: np.ndarray,
                order: np.ndarray, n_vertices: int) -> dict:
    """Ceiling, AR@k, and the two diagnostics. Ceiling is ranking-free, which
    is why the gates are stated on it and not on AR@k."""
    ceiling = {f"{t:.2f}": int((ious.max(axis=0) >= t).sum()) for t in IOUS}
    ar = {f"{t:.2f}": {("all" if k is None else str(k)):
                       int((ious[order if k is None else order[:k]]
                            .max(axis=0) >= t).sum())
                       for k in KS}
          for t in IOUS}
    sizes = np.array([len(p) for p in proposals]) if proposals else np.zeros(0)
    return {
        "bank": name,
        "n_proposals": len(proposals),
        "median_proposal_vertices": int(np.median(sizes)) if len(sizes) else 0,
        "oracle_ceiling": ceiling,
        "ar": ar,
        "giant_mask_rate": (round(float((sizes > GIANT_FRAC * n_vertices).mean()), 4)
                            if len(sizes) else 0.0),
        "zero_overlap_rate": (round(float((ious.max(axis=1) < 0.10).mean()), 4)
                              if len(proposals) else 0.0),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", default="41069021")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--views-root", type=Path, default=DEFAULT_VIEWS_ROOT)
    ap.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    args = ap.parse_args(argv)

    scene_dir = args.data_root / args.scene
    scene_id = scene_id_for(scene_dir)
    bundle_dir = args.bundle_root / f"bundle_{scene_id}"
    if not (bundle_dir / "meta.json").is_file():
        print(f"missing {bundle_dir}/meta.json — extract the Colab bundle "
              f"there first (notebooks/c1_mask3d_colab.ipynb)")
        return 1

    from segmenter.base import sha256_file
    mesh_ply = scene_dir / f"{args.scene}_3dod_mesh_canonical.ply"
    mesh, R, bundle = load_canonical_geometry(scene_dir)
    n = len(mesh.xyz)

    raw, ms02, prov = load_mask3d_banks(bundle_dir, n, sha256_file(mesh_ply))
    p1 = load_bank(args.views_root / f"bank_{scene_id}.npz")
    views = build_views(args.views_root / f"views_{scene_id}",
                        args.views_root / f"c1p1_masks_{scene_id}.npz")

    # One scorer, one ranking rule, every bank -- so AR@k differences are the
    # banks and not the ranking. The agreement signal is multiview evidence
    # from the P1 views; for the Mask3D-only banks that is the sole view
    # evidence available, which is noted in the report rather than hidden.
    banks = [("p1", p1), ("mask3d_ms02", ms02), ("mask3d_raw", raw),
             ("pooled_p1_ms02", p1 + ms02), ("pooled_p1_raw", p1 + raw)]
    orders = {}
    for name, b in banks:
        sig = selector_free.proposal_signals(b, n, mesh.xyz, views)
        orders[name] = rank_order(
            selector_free.score_proposals(sig, ABLATIONS[DEFAULT_VARIANT]))

    # ---- ORACLE BOUNDARY: every ranking above is final ----
    entities = load_oracle_entities(scene_dir, mesh.xyz, R)
    reports = [bank_report(name, b, iou_matrix(b, entities), orders[name], n)
               for name, b in banks]

    m3d = next(r for r in reports if r["bank"] == "mask3d_ms02")
    got = m3d["oracle_ceiling"]["0.50"]
    verdict = ("STRONG PASS" if got >= GATE_STRONG else
               "PASS" if got >= GATE_PASS else
               "STOP" if got <= GATE_STOP else "INDETERMINATE")

    out = {
        "contract": "docs/arkitscenes_mask3d_contract.md",
        "scene_id": scene_id, "video_id": args.scene,
        "representation_hash": bundle.representation_hash,
        "n_vertices": n, "n_entities": len(entities),
        "n_views_for_agreement": len(views),
        "gated_bank": "mask3d_ms02",
        "gates": {"strong": GATE_STRONG, "pass": GATE_PASS, "stop": GATE_STOP},
        "gated_ceiling_iou050": got,
        "verdict": verdict,
        "mask3d_provenance": prov,
        "banks": reports,
    }
    args.bundle_root.mkdir(parents=True, exist_ok=True)
    path = args.bundle_root / f"{scene_id}_mask3d_eval.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")

    ne = len(entities)
    print(f"=== {scene_id}   {n} vertices, {ne} entities")
    print(f"    Mask3D: {prov['n_raw_masks']} raw masks -> "
          f"{prov['n_after_min_score']} at MIN_SCORE={MIN_SCORE}"
          f"/min_vertices={MIN_VERTICES}"
          f"   ({prov.get('hardware')}, {prov.get('runtime_seconds')}s)")
    print()
    hdr = f"{'bank':17s} {'K':>5s}  " + "  ".join(f"@{t:.2f}" for t in IOUS)
    print(hdr + "   " + " ".join(f"AR{k:>4}" for k in (25, 50, 100, 200))
          + "  giant  zero")
    for r in reports:
        c = "  ".join(f"{r['oracle_ceiling'][f'{t:.2f}']:5d}" for t in IOUS)
        a = " ".join(f"{r['ar']['0.50'][str(k)]:6d}" for k in (25, 50, 100, 200))
        print(f"{r['bank']:17s} {r['n_proposals']:5d}  {c}   {a}"
              f"  {r['giant_mask_rate']:5.1%} {r['zero_overlap_rate']:5.1%}")
    print()
    print(f"    gate: Mask3D(ms02) @IoU0.50 = {got}/{ne}   "
          f"(stop <={GATE_STOP}, pass >={GATE_PASS}, strong >={GATE_STRONG})")
    print(f"    VERDICT: {verdict}")
    print(f"    report -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
