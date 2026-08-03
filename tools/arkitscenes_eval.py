"""ARKitScenes oracle evaluator — the ONLY module that reads annotations.

  python3 tools/arkitscenes_eval.py --all                 # G5 pre-flight
  python3 tools/arkitscenes_eval.py --all --bank          # + AR@k once fused

Everything above the ORACLE BOUNDARY is annotation-free and could run on a
scene with no ground truth at all. Everything below reads
`<video_id>_3dod_annotation.json`. `adapters/arkitscenes.py` cannot even
parse JSON (see its docstring); the boundary lives here, on purpose, in the
one place where knowing the answer is the point.

TWO CORRECTNESS DECISIONS, both settled by measurement rather than by
reading someone's convention doc:

1. `obbAligned`, not `obb`. The raw `obb` block is in an arbitrary scan
   space -- its centroids span +/-334 with axis lengths of 166 on
   41069021, against a mesh that spans ~6.6m. `obbAligned` is in metres in
   the mesh's own frame (centroids [-2.97,-3.28,-0.51]..[-0.05,1.89,1.20]
   inside mesh bounds [-4.49,-3.69,-1.29]..[2.15,3.81,1.86]).

2. `normalizedAxes` reshaped (3,3) has the box axes as ROWS. Both readings
   are orthonormal, so neither errors -- the wrong one just silently
   misaligns every entity. Decided empirically on 41069021: rows contain
   81934 mesh vertices across the first 8 objects, columns 67396. The
   decisive case is the thin `tv_monitor` (axis lengths
   [0.043, 0.893, 0.513]): rows 4752 vertices, columns 975. Pinned by
   `test_obb_axis_convention_is_rows` -- if this is ever "corrected" to
   columns, that test fails.

The annotation boxes live in the CAPTURE frame; the adapter rotated the
mesh into `scene_canonical`. The same rotation is therefore applied to the
boxes here, taken from the adapter itself rather than re-derived, and the
result is range-checked against the canonical mesh bounds. Skipping that
rotation would be the `frame="world"` bug again, on a new dataset.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.arkitscenes import (
    ARKitScenesAdapter, build_arkitscenes_capture_bundle, read_mesh,
    scene_id_for,
)
from adapters.base import ReconstructionConfig

DEFAULT_DATA_ROOT = Path.home() / "Desktop/datasets/arkitscenes/Validation"
DEFAULT_VIEWS_ROOT = REPO_ROOT / "runs" / "arkitscenes_p1"
KS = (10, 25, 50, 100, 200, None)
IOU_THRESHOLDS = (0.25, 0.50)

# C1-P1 protocol gate G5, verbatim: "at least 80% of oracle entities have
# >=10% of their vertices present in the id buffers across the 40 views".
G5_MIN_ENTITY_FRAC = 0.10
G5_MIN_ENTITY_SHARE = 0.80

# Frame-agreement ceiling: share of annotated entities allowed to contain no
# mesh vertices before the box/geometry frames are declared mismatched.
# Empirical, from three scenes -- see load_oracle_entities for the table.
MAX_EMPTY_ENTITY_SHARE = 0.15


# --------------------------------------------------------------------
# oracle-free half — nothing here reads an annotation
# --------------------------------------------------------------------
def load_canonical_geometry(scene_dir: Path):
    """Re-run the adapter: it is deterministic and is the single source of
    the canonical mesh AND of the rotation the boxes need."""
    bundle = ARKitScenesAdapter().reconstruct(
        build_arkitscenes_capture_bundle(scene_dir),
        ReconstructionConfig(name="arkitscenes_mesh", version="0.1"),
    )
    mesh = read_mesh(Path(bundle.geometry_handle.uri))
    R = np.asarray(bundle.geometry_handle.notes["rotation_row_major"],
                   dtype=np.float64)
    return mesh, R, bundle


def load_id_buffers(views_dir: Path) -> list[np.ndarray]:
    z = np.load(views_dir / "ids.npz")
    return [z[k] for k in sorted(z.files)]


def load_bank(views_dir: Path, scene_id: str) -> list[np.ndarray] | None:
    """Fused P1 proposals, if the Colab + fuse steps have run."""
    p = views_dir.parent / f"bank_{scene_id}.npz"
    if not p.is_file():
        return None
    z = np.load(p)
    verts, off = z["vertices"], z["offsets"]
    return [verts[off[i]:off[i + 1]] for i in range(len(off) - 1)]


# --------------------------------------------------------------------
# ORACLE BOUNDARY — nothing below may influence extraction or ranking
# --------------------------------------------------------------------
@dataclass(frozen=True)
class OracleEntity:
    uid: str
    label: str
    centroid: np.ndarray      # canonical frame
    axes_lengths: np.ndarray
    axes: np.ndarray          # [3,3], ROWS are the box axes
    vertices: np.ndarray      # int64 vertex indices inside the box


def obb_contains(xyz: np.ndarray, centroid, lengths, axes) -> np.ndarray:
    """Rows of `axes` are the box axes. See module docstring for why."""
    d = (xyz - centroid) @ np.asarray(axes).T
    return np.all(np.abs(d) <= (np.asarray(lengths) / 2.0)[None, :], axis=1)


def load_oracle_entities(scene_dir: Path, xyz: np.ndarray,
                         R: np.ndarray) -> list[OracleEntity]:
    """Annotation boxes -> canonical-frame entities with vertex membership.

    Frame-agreement is checked on VERTEX MEMBERSHIP, not on centroid range.
    A centroid check is nearly useless here: a room is roughly symmetric, so
    a 90-degree error leaves every centroid comfortably inside the bounding
    box and the check passes while every entity is wrong. Membership has
    real signal -- a misaligned box slices through empty space.

    Measured on the three Validation scenes, share of entities containing
    ZERO mesh vertices:

        rotation            41069021   41069025   41069042
        correct                 0.0%       0.0%       0.0%
        90 deg about x         55.6%      30.0%      16.7%
        90 deg about z         44.4%      40.0%      66.7%
        180 deg about z        72.2%      50.0%      50.0%

    Hence MAX_EMPTY_ENTITY_SHARE below. It is an EMPIRICAL threshold set
    from three scenes, and the margin to the weakest wrong case (16.7%) is
    not large -- re-measure when more scenes are added rather than assuming
    it still separates.
    """
    vid = scene_dir.name
    ann = json.loads((scene_dir / f"{vid}_3dod_annotation.json").read_text())
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    span = float(np.linalg.norm(hi - lo))
    out: list[OracleEntity] = []
    for obj in ann.get("data", []):
        seg = obj.get("segments", {})
        oa = seg.get("obbAligned")
        if not oa:
            continue
        c = R @ np.asarray(oa["centroid"], dtype=np.float64)
        L = np.asarray(oa["axesLengths"], dtype=np.float64)
        A = np.asarray(oa["normalizedAxes"], dtype=np.float64).reshape(3, 3) @ R.T
        # cheap first pass: catches gross errors (unit mismatch, wrong space)
        if np.any(c < lo - 0.25 * span) or np.any(c > hi + 0.25 * span):
            raise ValueError(
                f"{vid}: annotation centroid {np.round(c, 2)} lies outside the "
                f"canonical mesh bounds {np.round(lo, 2)}..{np.round(hi, 2)}; "
                "the box frame does not match the geometry frame")
        inside = np.flatnonzero(obb_contains(xyz, c, L, A)).astype(np.int64)
        out.append(OracleEntity(
            uid=str(obj.get("uid", obj.get("objectId", len(out)))),
            label=str(obj.get("label", "?")),
            centroid=c, axes_lengths=L, axes=A, vertices=inside,
        ))

    if out:
        empty = [e for e in out if len(e.vertices) == 0]
        share = len(empty) / len(out)
        if share > MAX_EMPTY_ENTITY_SHARE:
            raise ValueError(
                f"{vid}: {len(empty)}/{len(out)} ({share:.1%}) annotated "
                "entities contain ZERO mesh vertices, above the "
                f"{MAX_EMPTY_ENTITY_SHARE:.0%} ceiling. The boxes and the "
                "geometry are not in the same frame -- check the rotation "
                f"passed in. Empty labels: {[e.label for e in empty][:6]}")
    return out


def evidence_coverage(entities: list[OracleEntity],
                      id_buffers: list[np.ndarray],
                      n_vertices: int) -> dict:
    """C1-P1 gate G5 — is there enough multiview evidence to be worth a GPU
    run? Per entity, the fraction of its vertices appearing in ANY view."""
    seen = np.zeros(n_vertices, dtype=bool)
    for ids in id_buffers:
        seen[np.unique(ids[ids >= 0])] = True
    per_entity = []
    for e in entities:
        frac = float(seen[e.vertices].mean()) if len(e.vertices) else 0.0
        per_entity.append({"uid": e.uid, "label": e.label,
                           "n_vertices": int(len(e.vertices)),
                           "seen_frac": round(frac, 4)})
    covered = [p for p in per_entity if p["seen_frac"] >= G5_MIN_ENTITY_FRAC]
    share = len(covered) / len(per_entity) if per_entity else 0.0
    empty = [p for p in per_entity if p["n_vertices"] == 0]
    return {
        "n_entities": len(per_entity),
        "n_entities_with_no_vertices": len(empty),
        "entities_at_or_above_min_frac": len(covered),
        "share": round(share, 4),
        "gate_g5_pass": bool(share >= G5_MIN_ENTITY_SHARE),
        "mesh_vertex_coverage": round(float(seen.mean()), 4),
        "per_entity": sorted(per_entity, key=lambda p: p["seen_frac"]),
    }


def iou_matrix(proposals: list[np.ndarray],
               entities: list[OracleEntity]) -> np.ndarray:
    """[n_proposals, n_entities] vertex-set IoU."""
    out = np.zeros((len(proposals), len(entities)), dtype=np.float64)
    ent_sets = [np.asarray(e.vertices) for e in entities]
    for i, p in enumerate(proposals):
        ps = np.asarray(p)
        for j, es in enumerate(ent_sets):
            if ps.size == 0 or es.size == 0:
                continue
            inter = np.intersect1d(ps, es, assume_unique=False).size
            if inter:
                out[i, j] = inter / (ps.size + es.size - inter)
    return out


def ar_at_k(order: np.ndarray, ious: np.ndarray, thresh: float) -> dict:
    res = {}
    for k in KS:
        sel = order if k is None else order[:k]
        key = "all" if k is None else str(k)
        res[key] = int((ious[sel].max(axis=0) >= thresh).sum()) if len(sel) else 0
    return res


def evaluate(scene_dir: Path, views_root: Path, want_bank: bool) -> dict:
    t0 = time.perf_counter()
    scene_id = scene_id_for(scene_dir)
    views_dir = views_root / f"views_{scene_id}"
    if not views_dir.is_dir():
        raise FileNotFoundError(
            f"no rendered views at {views_dir}; run tools/arkitscenes_render.py")

    mesh, R, bundle = load_canonical_geometry(scene_dir)
    man = json.loads((views_dir / "manifest.json").read_text())
    if man["source"]["representation_hash"] != bundle.representation_hash:
        raise ValueError(
            f"{scene_id}: views were rendered from representation "
            f"{man['source']['representation_hash']}, adapter now produces "
            f"{bundle.representation_hash}; re-render before evaluating")

    entities = load_oracle_entities(scene_dir, mesh.xyz, R)
    ids = load_id_buffers(views_dir)
    report = {
        "scene_id": scene_id,
        "video_id": scene_dir.name,
        "n_vertices": int(len(mesh.xyz)),
        "n_views": len(ids),
        "representation_hash": bundle.representation_hash,
        "labels": sorted({e.label for e in entities}),
        "evidence": evidence_coverage(entities, ids, len(mesh.xyz)),
    }
    if want_bank:
        bank = load_bank(views_dir, scene_id)
        if bank is None:
            report["bank"] = None
        else:
            ious = iou_matrix(bank, entities)
            order = np.arange(len(bank))     # unranked: bank order
            report["bank"] = {
                "n_proposals": len(bank),
                "oracle_ceiling": {
                    f"{t:.2f}": int((ious.max(axis=0) >= t).sum())
                    for t in IOU_THRESHOLDS},
                "ar": {f"{t:.2f}": ar_at_k(order, ious, t)
                       for t in IOU_THRESHOLDS},
            }
    report["eval_seconds"] = round(time.perf_counter() - t0, 1)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--views-root", type=Path, default=DEFAULT_VIEWS_ROOT)
    ap.add_argument("--bank", action="store_true",
                    help="also score a fused proposal bank if one exists")
    args = ap.parse_args(argv)

    if args.all:
        scenes = sorted(d for d in args.data_root.iterdir()
                        if d.is_dir()
                        and (d / f"{d.name}_3dod_mesh.ply").is_file())
    elif args.scene:
        scenes = [args.data_root / args.scene]
    else:
        ap.error("pass --scene <video_id> or --all")

    out_dir = args.views_root
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    for scene_dir in scenes:
        rep = evaluate(scene_dir, args.views_root, args.bank)
        ev = rep["evidence"]
        verdict = "PASS" if ev["gate_g5_pass"] else "FAIL"
        failed += (not ev["gate_g5_pass"])
        print(f"=== {rep['scene_id']}  ({rep['eval_seconds']}s)")
        print(f"    entities            : {ev['n_entities']} "
              f"({ev['n_entities_with_no_vertices']} with no mesh vertices)")
        print(f"    labels              : {', '.join(rep['labels'])}")
        print(f"    mesh vertex coverage: {ev['mesh_vertex_coverage']:.1%}")
        print(f"    G5 (>={G5_MIN_ENTITY_FRAC:.0%} of verts seen, "
              f"needs >={G5_MIN_ENTITY_SHARE:.0%} of entities): "
              f"{ev['entities_at_or_above_min_frac']}/{ev['n_entities']} "
              f"= {ev['share']:.1%}  {verdict}")
        worst = ev["per_entity"][:3]
        print("    weakest entities    : " + ", ".join(
            f"{p['label']}({p['seen_frac']:.0%},{p['n_vertices']}v)"
            for p in worst))
        if args.bank:
            b = rep.get("bank")
            print(f"    bank                : "
                  + ("not fused yet" if b is None else
                     f"{b['n_proposals']} proposals, ceiling@0.50="
                     f"{b['oracle_ceiling']['0.50']}/{ev['n_entities']}"))
        path = out_dir / f"{rep['scene_id']}_oracle_eval.json"
        path.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
        print(f"    report              -> {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
