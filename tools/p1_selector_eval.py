"""Evaluate the oracle-free proposal selector against the frozen C1-P1 banks.

  python3 tools/p1_selector_eval.py --scene replica_room_2
  python3 tools/p1_selector_eval.py --all

THIS FILE is where oracle IoU may be read — and only to SCORE the
ranking, never to produce it. The order of operations is enforced by the
code layout below: `rank_scene()` builds the views, calls
`segmenter.selector_free`, and returns a finished ranking BEFORE
`load_oracle()` is called at all. `segmenter/selector_free.py` itself
cannot read a file (see its docstring and
`tests/segmenter/test_selector_free.py`).

Reports, per scene:
  * AR@k — entity recall at IoU 0.25 and 0.50 for k in
    {10, 25, 50, 100, 200, all}, over the P1 bank alone and over the
    pooled P1 + Mask3D-raw bank;
  * oracle recovery — AR@k divided by what an oracle-guided selection of
    the same bank achieves (33/53 pooled on the dev scene);
  * an ablation over the score components and the raw signals.

Development scene: replica_room_2 (the only scene the constants in
selector_free.py were chosen on). replica_room_1 and replica_office_0
are reported as transfer and were not tuned on.

Outputs land in runs/selector_v0/.
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

from geometry.mesh_surfaces import load_raw_triangle_mesh, transform_mesh
from segmenter import selector_free
from segmenter.proposal_fusion import lift_mask
from tools.c3_surface_run import SCENE_TO_SHORT, _load_generation_inputs

DEV_SCENE = "replica_room_2"
TRANSFER_SCENES = ("replica_room_1", "replica_office_0")
KS = (10, 25, 50, 100, 200, None)          # None = the whole bank
IOU_THRESHOLDS = (0.25, 0.50)
P1_ROOT = REPO_ROOT / "runs" / "phase8_c1p1"
OUT_ROOT = REPO_ROOT / "runs" / "selector_v0"

ABLATIONS = {
    # v1 default: connectivity dropped. Headline AR@k / recovery /
    # zero_overlap are ranked by THIS variant.
    "v1_default": selector_free.DEFAULT_COMPONENTS,
    # v0, frozen before transfer scenes were read. Kept so the ablation table
    # in docs/selector_v0_results.md stays reproducible, and so the v1-vs-v0
    # comparison is available in one run.
    "full": selector_free.COMPONENTS_V0,
    "no_size": ("agreement", "connectivity", "redundancy"),
    "no_connectivity": ("agreement", "size", "redundancy"),
    "no_redundancy": ("agreement", "connectivity", "size"),
    "agreement_only": ("agreement",),
    "gates_only_no_agreement": ("connectivity", "size", "redundancy"),
}
# Which ABLATIONS entry the headline numbers are ranked by.
DEFAULT_VARIANT = "v1_default"
RAW_SIGNAL_RANKINGS = ("agreement", "support_frac", "sam_quality",
                       "connectivity", "size_prior", "n_vertices")


# --------------------------------------------------------------------
# oracle-free half: everything above the ORACLE BOUNDARY comment
# --------------------------------------------------------------------
def load_bank(scene: str) -> list[np.ndarray]:
    z = np.load(P1_ROOT / f"bank_{scene}.npz")
    verts, off = z["vertices"], z["offsets"]
    return [verts[off[i]:off[i + 1]] for i in range(len(off) - 1)]


def load_geometry(scene: str) -> np.ndarray:
    """Raw-mesh vertex positions in the frozen gravity/yaw frame."""
    mesh_path, _, frame = _load_generation_inputs(scene)
    mesh = transform_mesh(load_raw_triangle_mesh(mesh_path),
                          np.asarray(frame["world_from_raw_rotation"]),
                          np.asarray(frame["world_from_raw_translation"]))
    return mesh.xyz


def build_views(scene: str) -> list[dict]:
    """The 40 frozen views in `proposal_fusion.edge_confidence`'s contract.

    Masks are lifted with `proposal_fusion.lift_mask` — the same function
    that built the bank — so the scorer sees exactly the co-membership
    evidence the generator saw, plus SAM's own quality pair.
    """
    ids = np.load(P1_ROOT / f"views_{scene}" / "ids.npz")
    packed = np.load(P1_ROOT / f"c1p1_masks_{scene}.npz")
    views = []
    for v in range(len(ids.files)):
        buf = ids[f"ids_{v:02d}"]
        raw, scores = packed[f"masks_{v:02d}"], packed[f"scores_{v:02d}"]
        masks, quality = [], []
        for m in range(raw.shape[0]):
            img = np.unpackbits(raw[m]).reshape(buf.shape).astype(bool)
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


def load_mask3d(scene: str) -> list[np.ndarray]:
    """Frozen Mask3D raw masks as vertex-id arrays (same vertex indexing)."""
    from tools.c1_resolve_sweep import load_raw_masks
    bundle = (REPO_ROOT / "runs" / "phase8_c1" / "bundles_ms02"
              / SCENE_TO_SHORT[scene])
    masks, _ = load_raw_masks(bundle)
    return [np.flatnonzero(masks[k]).astype(np.int64)
            for k in range(len(masks))]


def rank_scene(scene: str) -> dict:
    """Produce the finished, oracle-free ranking. No oracle is open yet."""
    t0 = time.perf_counter()
    p1 = load_bank(scene)
    m3d = load_mask3d(scene)
    xyz = load_geometry(scene)
    views = build_views(scene)
    n = len(xyz)
    out = {"scene_id": scene, "n_vertices": n, "n_p1": len(p1),
           "n_mask3d": len(m3d), "banks": {}}
    for bank_name, proposals in (("p1", p1), ("pooled", p1 + m3d)):
        sig = selector_free.proposal_signals(proposals, n, xyz, views)
        out["banks"][bank_name] = {
            "proposals": proposals,
            "signals": sig,
            "scores": {name: selector_free.score_proposals(sig, comps)
                       for name, comps in ABLATIONS.items()},
        }
    out["rank_seconds"] = round(time.perf_counter() - t0, 1)
    return out


# --------------------------------------------------------------------
# ORACLE BOUNDARY — nothing below this line may influence the ranking
# --------------------------------------------------------------------
def load_oracle(scene: str) -> tuple[np.ndarray, list[int], dict[int, str]]:
    from demo.replica_mesh_import import _parse_semantic_ply
    from segmenter.base import load_segmentation_output
    from tools.c1_exact_eval import oracle_vertex_membership
    short = SCENE_TO_SHORT[scene]
    lock = json.loads((REPO_ROOT / "tools" / "replica_scenes.lock.json")
                      .read_text())
    room = Path(lock["data_root_relative_to_repo"]) / short
    seg = load_segmentation_output(REPO_ROOT / "runs" / "phase8_c1"
                                   / "bundles_ms02" / short)
    _, vidx, oid = _parse_semantic_ply(room / "habitat" / "mesh_semantic.ply")
    oracle = oracle_vertex_membership(vidx, oid, seg.n_vertices)
    # The entity set is the frozen C1-P1 evaluator's (structural classes
    # already dropped there); reusing it keeps this table comparable.
    frozen = json.loads((P1_ROOT / f"{scene}_eval.json").read_text())
    oids = [int(e["oid"]) for e in frozen["per_entity"]]
    classes = {int(e["oid"]): e["class"] for e in frozen["per_entity"]}
    return oracle, oids, classes


def iou_matrix(proposals: list[np.ndarray], oracle: np.ndarray,
               oids: list[int]) -> np.ndarray:
    """[n_proposals, n_entities] vertex IoU against every oracle entity."""
    sizes = {o: int((oracle == o).sum()) for o in oids}
    col = {o: i for i, o in enumerate(oids)}
    out = np.zeros((len(proposals), len(oids)))
    for k, p in enumerate(proposals):
        ids, counts = np.unique(oracle[p], return_counts=True)
        for o, c in zip(ids.tolist(), counts.tolist()):
            if o in col:
                out[k, col[o]] = c / (len(p) + sizes[o] - c)
    return out


def ar_at_k(order: np.ndarray, ious: np.ndarray, k: int | None,
            thr: float) -> int:
    sel = order if k is None else order[:k]
    if len(sel) == 0:
        return 0
    return int((ious[sel].max(axis=0) >= thr).sum())


def curve(order: np.ndarray, ious: np.ndarray, thr: float) -> dict[str, int]:
    return {("all" if k is None else str(k)): ar_at_k(order, ious, k, thr)
            for k in KS}


def rank_order(values: np.ndarray) -> np.ndarray:
    """Descending, ties broken by original index — deterministic."""
    return np.argsort(-np.asarray(values, float), kind="stable")


def evaluate_scene(ranked: dict) -> dict:
    scene = ranked["scene_id"]
    oracle, oids, classes = load_oracle(scene)
    rng = np.random.default_rng(0)
    report = {"scene_id": scene, "role": "dev" if scene == DEV_SCENE
              else "transfer", "n_entities": len(oids),
              "n_p1_proposals": ranked["n_p1"],
              "n_mask3d_proposals": ranked["n_mask3d"],
              "rank_seconds": ranked["rank_seconds"], "banks": {}}
    for bank_name, bank in ranked["banks"].items():
        proposals = bank["proposals"]
        ious = iou_matrix(proposals, oracle, oids)
        ceiling = {f"{t:.2f}": int((ious.max(axis=0) >= t).sum())
                   for t in IOU_THRESHOLDS}
        entry = {"n_proposals": len(proposals),
                 "oracle_ceiling": ceiling, "ar": {}, "recovery": {},
                 "ablation": {}, "raw_signal": {}}
        default_order = rank_order(bank["scores"][DEFAULT_VARIANT])
        for t in IOU_THRESHOLDS:
            key = f"{t:.2f}"
            c = curve(default_order, ious, t)
            entry["ar"][key] = c
            entry["recovery"][key] = {
                kk: (round(vv / ceiling[key], 3) if ceiling[key] else None)
                for kk, vv in c.items()}
        # dominant failure mode: proposals the selector ranks highly that
        # overlap NO scored entity at all (structural wall/floor/ceiling
        # patches reconstruct as perfectly stable single 2D masks).
        entry["zero_overlap_share"] = {
            ("all" if k is None else str(k)): round(float(
                (ious[default_order if k is None else default_order[:k]].max(axis=1)
                 < 0.10).mean()), 3) for k in KS}
        for name in ABLATIONS:
            o = rank_order(bank["scores"][name])
            entry["ablation"][name] = {f"{t:.2f}": curve(o, ious, t)
                                       for t in IOU_THRESHOLDS}
        sig = bank["signals"]
        for name in RAW_SIGNAL_RANKINGS:
            o = rank_order(getattr(sig, name))
            entry["raw_signal"][name] = {f"{t:.2f}": curve(o, ious, t)
                                         for t in IOU_THRESHOLDS}
        o = rng.permutation(len(proposals))
        entry["raw_signal"]["random_seed0"] = {
            f"{t:.2f}": curve(o, ious, t) for t in IOU_THRESHOLDS}
        # top-25 identity, for eyeballing what the selector actually picks
        entry["top25"] = []
        for r in default_order[:25].tolist():
            j = int(ious[r].argmax())
            entry["top25"].append({
                "proposal": r, "score": round(float(
                    bank["scores"]["full"][r]), 4),
                "best_entity": classes[oids[j]],
                "best_iou": round(float(ious[r, j]), 3),
                "agreement": round(float(sig.agreement[r]), 3),
                "connectivity": round(float(sig.connectivity[r]), 3),
                "size_prior": round(float(sig.size_prior[r]), 3),
                "n_nested_better": int(sig.n_nested_better[r])})
        report["banks"][bank_name] = entry
    return report


def print_report(rep: dict) -> None:
    ks = [("all" if k is None else str(k)) for k in KS]
    head = "  ".join(f"{k:>4}" for k in ks)
    print(f"\n=== {rep['scene_id']} ({rep['role']}) — "
          f"{rep['n_entities']} oracle entities, "
          f"{rep['n_p1_proposals']} P1 + {rep['n_mask3d_proposals']} Mask3D "
          f"proposals, ranked in {rep['rank_seconds']}s ===")
    for bank, e in rep["banks"].items():
        print(f"\n[{bank}] {e['n_proposals']} proposals; oracle-selection "
              f"ceiling @0.25 {e['oracle_ceiling']['0.25']}, "
              f"@0.50 {e['oracle_ceiling']['0.50']} of {rep['n_entities']}")
        print(f"  {'k':<26}{head}")
        for t in ("0.25", "0.50"):
            row = "  ".join(f"{e['ar'][t][k]:>4}" for k in ks)
            print(f"  {'AR@k IoU ' + t:<26}{row}")
            row = "  ".join(f"{e['recovery'][t][k]:>4.2f}" for k in ks)
            print(f"  {'  recovery of oracle sel.':<26}{row}")
        row = "  ".join(f"{e['zero_overlap_share'][k]:>4.2f}" for k in ks)
        print(f"  {'share ranked w/ no entity':<26}{row}")
        print(f"  -- ablation (AR@k @IoU0.50) --")
        for name, d in e["ablation"].items():
            row = "  ".join(f"{d['0.50'][k]:>4}" for k in ks)
            print(f"  {name:<26}{row}")
        print(f"  -- single raw signal (AR@k @IoU0.50) --")
        for name, d in e["raw_signal"].items():
            row = "  ".join(f"{d['0.50'][k]:>4}" for k in ks)
            print(f"  {name:<26}{row}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", action="append", default=None)
    ap.add_argument("--all", action="store_true",
                    help="dev scene then both transfer scenes")
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = ap.parse_args(argv)
    scenes = list(args.scene or [])
    if args.all or not scenes:
        scenes = [DEV_SCENE, *TRANSFER_SCENES]
    args.out_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for scene in scenes:
        rep = evaluate_scene(rank_scene(scene))
        out = args.out_root / f"{scene}_selector_eval.json"
        out.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n",
                       encoding="utf-8")
        print_report(rep)
        print(f"  report -> {out}")
        reports.append(rep)
    summary = {
        "schema": "p1_selector_eval_v0",
        "selector": "segmenter/selector_free.py",
        "dev_scene": DEV_SCENE, "transfer_scenes": list(TRANSFER_SCENES),
        "ks": [("all" if k is None else k) for k in KS],
        "iou_thresholds": list(IOU_THRESHOLDS),
        "scenes": {r["scene_id"]: {
            "role": r["role"], "n_entities": r["n_entities"],
            "banks": {b: {"oracle_ceiling": e["oracle_ceiling"],
                          "ar": e["ar"], "recovery": e["recovery"]}
                      for b, e in r["banks"].items()}} for r in reports},
    }
    sp = args.out_root / "summary.json"
    sp.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n",
                  encoding="utf-8")
    print(f"\nsummary -> {sp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
