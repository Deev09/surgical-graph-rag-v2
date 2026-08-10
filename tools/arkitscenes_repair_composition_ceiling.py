"""Oracle-guided composition ceiling: are SAM parts ASSEMBLABLE into entities?

  python3 tools/arkitscenes_repair_composition_ceiling.py --scene 41069021 \
      --masks runs/arkitscenes_repair/arkitscenes_41069021/repair_sam_masks_arkitscenes_41069021.npz

Zero GPU. Reuses the existing pinned sidecar; changes no SAM parameter.

WHAT THIS IS
------------
Checkpoint D measured that SAM repair proposals are correct object PARTS:
precision 0.82-1.00 against annotated entities, recall 0.08-0.31. That leaves
one question unanswered, and it decides the next mechanism. Are the parts of an
object *present and separable* in the bank -- so that some union of them would
reach IoU 0.50 -- or are they simply absent?

This measures the ceiling of that union, choosing the parts WITH THE ANSWER IN
HAND. For each annotated entity it greedily grows a union of proposals to
maximise IoU with that entity, and reports the best achievable at
<=1/2/4/8/16 parts.

WHAT THIS IS NOT
----------------
It is NOT a method and NOT a result that can be pooled, ranked or gated. The
union is selected by consulting the annotation, so its IoU is an upper bound no
oracle-free assembler can exceed and most will not approach. The report is
stamped `oracle_guided: true` and `deployable: false`, is written to its own
file, and never touches a `ProposalArtifact` or the gate sheet in
`eval/detection_repair.py`.

Its only job is to discriminate between two futures:

  * ceiling reaches 0.50 for previously missed entities  -> the parts exist and
    the next mechanism is ORACLE-FREE PART ASSEMBLY;
  * ceiling stays below 0.50                             -> no assembler can
    help, and the mask source or the view budget has to change.

THREE BANKS, deliberately increasing in permissiveness
------------------------------------------------------
  emitted (60)          what the arm actually proposed, after support, vote,
                        classification, dedupe and the emission cap
  supported (94)        every fused cluster, before classification and the cap
  pre_support (318)     every association group, as the raw UNION of its member
                        masks -- no vote, no support filter, singletons
                        included. The most permissive view of what SAM saw.

If the ceiling is low even on `pre_support`, no filtering decision in this arm
is responsible.

GREEDY, and how far that is from optimal
----------------------------------------
Optimal subset selection is combinatorial. Union-IoU is not submodular, so
greedy carries no general guarantee; the tool therefore also computes the
EXHAUSTIVE best pair and reports any case where greedy@2 falls short. Every
number here is a lower bound on the true ceiling, which only strengthens a
negative finding and weakens a positive one.
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
from segmenter.sam_multiview_repair import associate, fuse_cluster, lift_masks
from tools.arkitscenes_eval import DEFAULT_DATA_ROOT, load_canonical_geometry
from tools.arkitscenes_mask3d_eval import DEFAULT_BUNDLE_ROOT
from tools.arkitscenes_repair_eval import (
    DEFAULT_OUT_ROOT, load_baseline_bank, load_repair_bank,
)
from tools.arkitscenes_repair_propose_sam import load_sidecar

PART_BUDGETS = (1, 2, 4, 8, 16)
TARGET_IOU = 0.50


def rebuild_banks(scene_dir: Path, masks_path: Path, out_dir: Path):
    """Recompute the pre-support and supported cluster banks from the sidecar.

    Deterministic and annotation-free: the same code path the propose CLI ran,
    stopped at two earlier stages. Cheap enough (~7 s) not to be worth caching.
    """
    scene_id = scene_id_for(scene_dir)
    frames_json = out_dir / f"repair_frames_{scene_id}" / "frames.json"
    manifest = json.loads(frames_json.read_text())
    mesh, rotation, _ = load_canonical_geometry(scene_dir)
    xyz_world = mesh.xyz @ rotation
    n_vertices = len(xyz_world)

    sidecar_masks, env = load_sidecar(masks_path, manifest)
    frames = load_frames(scene_dir, stride=manifest.get("frame_stride",
                                                        FRAME_STRIDE))
    selected = [row["frame_index"] for row in manifest["frames"]]

    visibility, lifted = {}, []
    for slot, frame_index in enumerate(selected):
        frame = frames[frame_index]
        vertices, _ = visible_vertices(xyz_world, frame)
        visibility[frame_index] = vertices
        lifted.extend(lift_masks(xyz_world, frame, frame_index,
                                 sidecar_masks[slot]["masks"],
                                 sidecar_masks[slot]["scores"]))

    groups = associate(lifted, frames, n_vertices)
    pre_support = [
        np.unique(np.concatenate([lifted[i].vertices for i in group]))
        for group in groups
    ]
    supported = []
    for group in groups:
        fused = fuse_cluster([lifted[i] for i in group], frames, visibility,
                             n_vertices)
        if fused is not None:
            supported.append(fused.vertices)
    return mesh, rotation, n_vertices, pre_support, supported, env, len(lifted)


def greedy_union(candidates: list[np.ndarray], entity: np.ndarray,
                 n_vertices: int, budgets=PART_BUDGETS) -> dict:
    """Greedily grow a union of parts to maximise IoU with one entity.

    Only proposals that intersect the entity are considered: a part that
    touches nothing can never raise the numerator and can only inflate the
    denominator, so no optimal union contains one.
    """
    entity_mask = np.zeros(n_vertices, dtype=bool)
    entity_mask[entity] = True
    usable = [(i, p) for i, p in enumerate(candidates)
              if entity_mask[p].any()]
    results = {}
    union = np.zeros(n_vertices, dtype=bool)
    size = inter = 0
    chosen: list[int] = []
    best_by_k: dict[int, dict] = {}
    remaining = dict(usable)

    for step in range(max(budgets)):
        best = None
        for i, p in remaining.items():
            novel = p[~union[p]]
            if novel.size == 0:
                continue
            add_size = int(novel.size)
            add_inter = int(entity_mask[novel].sum())
            new_inter = inter + add_inter
            new_size = size + add_size
            denominator = new_size + len(entity) - new_inter
            iou = new_inter / denominator if denominator else 0.0
            if best is None or iou > best[0]:
                best = (iou, i, novel, add_size, add_inter)
        if best is None or (chosen and best[0] <= best_by_k[len(chosen)]["iou"]):
            break
        iou, index, novel, add_size, add_inter = best
        union[novel] = True
        size += add_size
        inter += add_inter
        chosen.append(index)
        remaining.pop(index)
        best_by_k[len(chosen)] = {
            "iou": round(iou, 4),
            "precision": round(inter / size, 4) if size else 0.0,
            "recall": round(inter / len(entity), 4) if len(entity) else 0.0,
            "n_parts": len(chosen),
            "union_vertices": size,
            "part_indices": list(chosen),
        }

    last = {"iou": 0.0, "precision": 0.0, "recall": 0.0, "n_parts": 0,
            "union_vertices": 0, "part_indices": []}
    for k in budgets:
        available = [j for j in best_by_k if j <= k]
        if available:
            # IoU is not monotone in k, so report the best union AT MOST k.
            pick = max(available, key=lambda j: best_by_k[j]["iou"])
            last = best_by_k[pick]
        results[str(k)] = dict(last)
    results["n_candidates_touching_entity"] = len(usable)
    return results


def exhaustive_pair(candidates: list[np.ndarray], entity: np.ndarray,
                    n_vertices: int) -> float:
    """Best IoU over all single proposals and all pairs. Greedy sanity check."""
    entity_mask = np.zeros(n_vertices, dtype=bool)
    entity_mask[entity] = True
    usable = [p for p in candidates if entity_mask[p].any()]
    best = 0.0
    stats = []
    for p in usable:
        inter = int(entity_mask[p].sum())
        stats.append((p, inter))
        denominator = len(p) + len(entity) - inter
        best = max(best, inter / denominator if denominator else 0.0)
    for a in range(len(stats)):
        pa, ia = stats[a]
        for b in range(a + 1, len(stats)):
            pb, ib = stats[b]
            union = np.union1d(pa, pb)
            inter = int(entity_mask[union].sum())
            denominator = len(union) + len(entity) - inter
            if denominator:
                best = max(best, inter / denominator)
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", default="41069021")
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--skip-exhaustive", action="store_true")
    ap.add_argument("--extra-bank", action="append", default=[],
                    metavar="NAME=PATH",
                    help="score an additional finalized bank, e.g. the "
                         "topology-cut components. Repeatable.")
    args = ap.parse_args(argv)

    scene_dir = args.data_root / args.scene
    scene_id = scene_id_for(scene_dir)
    out_dir = args.out_root / scene_id
    t0 = time.perf_counter()

    mesh, rotation, n_vertices, pre_support, supported, env, n_lifted = \
        rebuild_banks(scene_dir, args.masks, out_dir)
    mesh_sha = sha256_file(scene_dir / f"{args.scene}_3dod_mesh_canonical.ply")
    emitted_props, _ = load_repair_bank(out_dir / "repair_bank_sam.npz",
                                        n_vertices)
    emitted = [p.vertices for p in emitted_props]
    baseline_props, _ = load_baseline_bank(
        args.bundle_root / f"bundle_{scene_id}", n_vertices, mesh_sha)
    baseline = [p.vertices for p in baseline_props]

    banks = {"emitted": emitted, "supported": supported,
             "pre_support": pre_support}
    for spec in args.extra_bank:
        name, _, path = spec.partition("=")
        extra, _ = load_repair_bank(Path(path), n_vertices)
        banks[name] = [p.vertices for p in extra]
    print(f"=== {scene_id}   composition ceiling (ORACLE-GUIDED, diagnostic)")
    print(f"    lifted masks {n_lifted} | banks: " + ", ".join(
        f"{k}={len(v)}" for k, v in banks.items()))

    # ---- ORACLE BOUNDARY: every bank above is fixed and annotation-free ----
    from tools.arkitscenes_eval import load_oracle_entities
    entities = load_oracle_entities(scene_dir, mesh.xyz, rotation)

    def best_baseline_iou(entity) -> float:
        best = 0.0
        for p in baseline:
            inter = np.intersect1d(p, entity.vertices, assume_unique=True).size
            if inter:
                best = max(best, inter / (len(p) + len(entity.vertices) - inter))
        return best

    report = {
        "diagnostic": "oracle_guided_composition_ceiling",
        "oracle_guided": True,
        "deployable": False,
        "interpretation_limit": (
            "Parts are selected with the annotation in hand. These IoUs are an "
            "upper bound no oracle-free assembler can exceed. They are not a "
            "proposal bank, not a gate input, and not a result."),
        "scene_id": scene_id,
        "target_iou": TARGET_IOU,
        "part_budgets": list(PART_BUDGETS),
        "greedy": True,
        "sam_env": env,
        "bank_sizes": {k: len(v) for k, v in banks.items()},
        "n_entities": len(entities),
        "per_bank": {},
    }

    for name, bank in banks.items():
        rows, greedy_gap = [], 0
        for entity in entities:
            base_iou = best_baseline_iou(entity)
            ceiling = greedy_union(bank, entity.vertices, n_vertices)
            row = {
                "uid": entity.uid, "label": entity.label,
                "entity_vertices": int(len(entity.vertices)),
                "baseline_iou": round(base_iou, 4),
                "missed_by_baseline_at_050": base_iou < TARGET_IOU,
                "n_candidates": ceiling.pop("n_candidates_touching_entity"),
                "by_budget": ceiling,
            }
            if not args.skip_exhaustive:
                exact = exhaustive_pair(bank, entity.vertices, n_vertices)
                row["exhaustive_pair_iou"] = round(exact, 4)
                # Compare like with like. `ceiling` values are rounded to 4
                # decimals; comparing an unrounded float against them with a
                # 1e-6 tolerance counts pure rounding noise as a greedy
                # failure, which reported 9 spurious shortfalls on the first
                # run of this tool. The real worst case is +0.005.
                row["greedy_pair_shortfall"] = round(
                    max(0.0, exact - ceiling["2"]["iou"]), 4)
                if row["greedy_pair_shortfall"] >= 1e-3:
                    greedy_gap += 1
            rows.append(row)

        missed = [r for r in rows if r["missed_by_baseline_at_050"]]
        summary = {}
        for k in PART_BUDGETS:
            key = str(k)
            reaching = [r for r in missed
                        if r["by_budget"][key]["iou"] >= TARGET_IOU]
            summary[key] = {
                "entities_reaching_target": sum(
                    1 for r in rows if r["by_budget"][key]["iou"] >= TARGET_IOU),
                "missed_entities_reaching_target": len(reaching),
                # 1 part means a genuinely disconnected surface was separated;
                # more means the ORACLE stitched fragments, which no
                # annotation-free assembler could reproduce.
                "missed_reached_by_single_part": sum(
                    1 for r in reaching if r["by_budget"][key]["n_parts"] == 1),
                "missed_reached_by_oracle_assembly": sum(
                    1 for r in reaching if r["by_budget"][key]["n_parts"] > 1),
                "median_iou": round(float(np.median(
                    [r["by_budget"][key]["iou"] for r in rows])), 4),
                "median_precision": round(float(np.median(
                    [r["by_budget"][key]["precision"] for r in rows])), 4),
                "median_recall": round(float(np.median(
                    [r["by_budget"][key]["recall"] for r in rows])), 4),
            }

        # Junk growth, on the same definition eval/detection_repair.py uses.
        # Recomputed locally because this bank is a diagnostic and must never
        # be handed to the evaluator.
        best_per_proposal = []
        for proposal in bank:
            best = 0.0
            for entity in entities:
                inter = np.intersect1d(proposal, entity.vertices,
                                       assume_unique=True).size
                if inter:
                    best = max(best, inter / (len(proposal)
                                              + len(entity.vertices) - inter))
            best_per_proposal.append(best)
        best_per_proposal = np.asarray(best_per_proposal)

        report["per_bank"][name] = {
            "n_proposals": len(bank),
            "n_missed_by_baseline": len(missed),
            "greedy_below_exhaustive_pair_count": (
                None if args.skip_exhaustive else greedy_gap),
            "zero_overlap_rate": (round(float((best_per_proposal < 0.10).mean()), 4)
                                  if len(bank) else None),
            "median_proposal_vertices": int(np.median(
                [len(p) for p in bank])) if bank else 0,
            "summary_by_budget": summary,
            "per_entity": rows,
        }

        zero = report["per_bank"][name]["zero_overlap_rate"]
        print(f"\n    --- {name}: {len(bank)} proposals, "
              f"zero-overlap {zero:.1%}, "
              f"{len(missed)} entities missed by Mask3D ---")
        print(f"    {'parts':>6} {'reach .50':>10} {'of missed':>10} "
              f"{'1-part':>7} {'assembled':>10} "
              f"{'med IoU':>8} {'med prec':>9} {'med rec':>8}")
        for k in PART_BUDGETS:
            s = summary[str(k)]
            print(f"    {k:>6} {s['entities_reaching_target']:>10} "
                  f"{s['missed_entities_reaching_target']:>10} "
                  f"{s['missed_reached_by_single_part']:>7} "
                  f"{s['missed_reached_by_oracle_assembly']:>10} "
                  f"{s['median_iou']:>8.3f} {s['median_precision']:>9.3f} "
                  f"{s['median_recall']:>8.3f}")

    path = out_dir / "composition_ceiling.json"
    report["runtime_seconds"] = round(time.perf_counter() - t0, 1)
    path.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")

    best = max(report["per_bank"][b]["summary_by_budget"]["16"]
               ["missed_entities_reaching_target"] for b in banks)
    print()
    print(f"    previously missed entities reachable at IoU {TARGET_IOU:.2f} "
          f"with <=16 parts, best bank: {best}")
    print("    VERDICT: " + (
        "part assembly is worth building (>=2 reachable)" if best >= 2 else
        "part assembly cannot reach the target; the mask source or view "
        "budget must change"))
    print(f"    report -> {path}   ({report['runtime_seconds']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
