"""Detection-repair evaluation on ARKitScenes: Mask3D alone vs Mask3D + repair.

  python3 tools/arkitscenes_repair_eval.py --scene 41069021           # baseline only
  python3 tools/arkitscenes_repair_eval.py --scene 41069021 --repair runs/.../repair.npz

Design note: `docs/repair_arm_design_note.md`.

This is the arm's ONLY evaluator, and it exists before the repair algorithm on
purpose. Everything it measures is detection: whether a proposal bank contains
an annotated entity. It measures nothing about labels, relations or answers,
and `eval/detection_repair.py` stamps that limitation into every report it
writes.

THE ORDERING THIS FILE OWNS
---------------------------
`build_proposals` is annotation-free and returns FINALIZED, hash-stamped
artifacts. Only after it returns does execution cross the oracle boundary
below and load annotation boxes. `eval.detection_repair` re-verifies each
digest before scoring, so the ordering is checked rather than promised: a
proposal set regenerated or filtered after the boxes were opened cannot be
scored by this path at all.

BASELINE DEFINITION, fixed here
-------------------------------
The baseline is the `mask3d_ms02` bank from
`docs/arkitscenes_mask3d_contract.md`: the frozen GPU bundle's raw masks at
`min_score=0.2, min_vertices=20`, with each proposal carrying its own Mask3D
score as confidence. It is the same bank, the same constants and the same
loader as `tools/arkitscenes_mask3d_eval.py` -- imported, not restated -- so
the recorded contract numbers are reproducible through this evaluator instead
of merely consistent with it.

The delivered dense partition is a DIFFERENT artifact (a winner-takes-all
assignment, shipped at min_score=0.4) and is not the baseline here: repair
proposals are pooled with a proposal bank, so the comparison has to be against
a proposal bank.
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
from eval.detection_repair import (
    Proposal, ProposalArtifact, compare_banks, development_gates, score_bank,
)
from segmenter.base import sha256_file
from tools.arkitscenes_eval import DEFAULT_DATA_ROOT, load_canonical_geometry
from tools.arkitscenes_mask3d_eval import (
    DEFAULT_BUNDLE_ROOT, MIN_SCORE, MIN_VERTICES,
)
from tools.c1_resolve_sweep import load_raw_masks

DEFAULT_OUT_ROOT = REPO_ROOT / "runs" / "arkitscenes_repair"


def load_baseline_bank(bundle_dir: Path, n_vertices: int,
                       mesh_sha256: str) -> tuple[list[Proposal], dict]:
    """The frozen Mask3D ms02 proposal bank, with per-proposal scores.

    Hard-gated on the mesh the bundle ran against, for the reason
    `tools/arkitscenes_mask3d_eval.load_mask3d_banks` gives: a bundle from a
    different mesh indexes into someone else's geometry and pools with
    nothing.
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
        raise ValueError(
            f"raw masks are {masks.shape[1]} wide, expected {n_vertices}")
    sizes = masks.sum(axis=1)
    keep = [k for k in range(len(masks))
            if scores[k] >= MIN_SCORE and sizes[k] >= MIN_VERTICES]
    proposals = [
        Proposal(np.flatnonzero(masks[k]).astype(np.int64),
                 "mask3d", "baseline", float(scores[k]))
        for k in keep
    ]
    provenance = {
        "bank": "mask3d_ms02",
        "contract": "docs/arkitscenes_mask3d_contract.md",
        "segmenter_name": meta.get("segmenter_name"),
        "segmenter_version": meta.get("segmenter_version"),
        "hardware": meta.get("hardware"),
        "input_mesh_sha256": meta["input_mesh_sha256"],
        "n_raw_masks": int(len(masks)),
        "n_after_min_score": len(proposals),
        "min_score": MIN_SCORE,
        "min_vertices": MIN_VERTICES,
    }
    return proposals, provenance


def load_repair_bank(path: Path, n_vertices: int) -> tuple[list[Proposal], dict]:
    """Read a repair bank emitted by `tools/arkitscenes_repair_propose.py`.

    The sidecar manifest is required and its mesh identity is checked here:
    a repair bank generated against a different scene would otherwise index
    into this mesh and score as pure junk rather than as an error.
    """
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{path} has no manifest at {manifest_path}; repair banks must be "
            "finalized by tools/arkitscenes_repair_propose.py")
    manifest = json.loads(manifest_path.read_text())
    if int(manifest["n_vertices"]) != n_vertices:
        raise ValueError(
            f"repair bank was built on {manifest['n_vertices']} vertices, "
            f"this mesh has {n_vertices}")
    z = np.load(path, allow_pickle=False)
    verts, offsets = z["vertices"], z["offsets"]
    kinds = [str(k) for k in z["kind"]]
    confidences = z["confidence"].astype(float)
    proposals = [
        Proposal(verts[offsets[i]:offsets[i + 1]].astype(np.int64),
                 "repair", kinds[i], float(confidences[i]))
        for i in range(len(offsets) - 1)
    ]
    return proposals, manifest


def build_proposals(scene_dir: Path, bundle_dir: Path, out_dir: Path,
                    repair_path: Path | None):
    """Annotation-free half. Returns finalized artifacts, mesh and rotation."""
    mesh_ply = scene_dir / f"{scene_dir.name}_3dod_mesh_canonical.ply"
    mesh, rotation, bundle = load_canonical_geometry(scene_dir)
    n_vertices = len(mesh.xyz)
    mesh_sha = sha256_file(mesh_ply)

    baseline_props, baseline_prov = load_baseline_bank(
        bundle_dir, n_vertices, mesh_sha)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = ProposalArtifact.finalize(
        "mask3d_ms02", baseline_props, n_vertices,
        out_dir / "baseline_bank.npz",
        {**baseline_prov, "canonical_mesh_sha256": mesh_sha})

    repair = pooled = None
    if repair_path is not None:
        repair_props, repair_manifest = load_repair_bank(
            repair_path, n_vertices)
        built_against = repair_manifest.get("provenance", {}).get(
            "canonical_mesh_sha256")
        if built_against not in (None, mesh_sha):
            raise ValueError(
                f"repair bank was built against mesh {built_against[:16]}…, "
                f"this scene is {mesh_sha[:16]}…")
        repair = ProposalArtifact.finalize(
            "repair", repair_props, n_vertices,
            out_dir / "repair_bank.finalized.npz",
            {"source_artifact": str(repair_path),
             "source_sha256": sha256_file(repair_path),
             "generator_manifest": repair_manifest})
        pooled = baseline.pooled_with(
            repair, "pooled_mask3d_repair", out_dir / "pooled_bank.npz")
    return mesh, rotation, bundle, baseline, repair, pooled


def evaluate(scene_dir: Path, bundle_dir: Path, out_dir: Path,
             repair_path: Path | None,
             audited_cases_hit: list[str] | None = None) -> dict:
    mesh, rotation, bundle, baseline, repair, pooled = build_proposals(
        scene_dir, bundle_dir, out_dir, repair_path)

    # ---- ORACLE BOUNDARY: every proposal set above is finalized and hashed ----
    from tools.arkitscenes_eval import load_oracle_entities
    entities = load_oracle_entities(scene_dir, mesh.xyz, rotation)

    report = {
        "design_note": "docs/repair_arm_design_note.md",
        "evaluation": "detection_repair",
        "scene_id": scene_id_for(scene_dir),
        "video_id": scene_dir.name,
        "representation_hash": bundle.representation_hash,
        "n_vertices": int(len(mesh.xyz)),
        "n_entities": len(entities),
        "entity_labels": sorted(e.label for e in entities),
        "baseline_provenance": baseline.provenance,
        "baseline": score_bank(baseline, entities),
    }
    if pooled is None:
        report["repair"] = None
        report["comparison"] = None
        report["gates"] = None
        return report
    report["repair_provenance"] = repair.provenance
    report["repair_only"] = score_bank(repair, entities)
    report["comparison"] = compare_banks(baseline, pooled, entities)
    report["gates"] = development_gates(
        report["comparison"], audited_cases_hit=audited_cases_hit)
    return report


def _print_bank(title: str, bank: dict) -> None:
    print(f"    {title:22s} {bank['n_proposals']:5d} proposals   "
          f"@0.25={bank['n_recovered']['0.25']:2d}  "
          f"@0.50={bank['n_recovered']['0.50']:2d}   "
          f"giant={bank['giant_mask_rate']:5.1%}   "
          f"zero(top100)="
          f"{bank['zero_overlap']['confidence']['100']['zero_overlap_rate']:.1%}"
          f"  zero(all)="
          f"{bank['zero_overlap']['confidence']['all']['zero_overlap_rate']:.1%}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", default="41069021")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--repair", type=Path, default=None,
                    help="repair bank .npz from tools/arkitscenes_repair_propose.py")
    ap.add_argument("--audited-case", action="append", default=[],
                    help="audited human-feedback case id this run covers; "
                         "supplied by the operator, never inferred from IoU")
    args = ap.parse_args(argv)

    scene_dir = args.data_root / args.scene
    scene_id = scene_id_for(scene_dir)
    bundle_dir = args.bundle_root / f"bundle_{scene_id}"
    if not (bundle_dir / "meta.json").is_file():
        print(f"missing {bundle_dir}/meta.json — extract the Mask3D bundle there first")
        return 1

    out_dir = args.out_root / scene_id
    report = evaluate(scene_dir, bundle_dir, out_dir, args.repair,
                      args.audited_case or None)
    path = out_dir / "detection_repair_eval.json"
    path.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")

    print(f"=== {scene_id}   {report['n_vertices']} vertices, "
          f"{report['n_entities']} annotated entities")
    _print_bank("mask3d_ms02 (base)", report["baseline"])
    if report["comparison"] is None:
        print("    repair bank         : not supplied (baseline reproduction only)")
        print(f"    report -> {path}")
        return 0

    _print_bank("repair only", report["repair_only"])
    _print_bank("pooled", report["comparison"]["pooled"])
    print()
    for t in ("0.25", "0.50"):
        moved = report["comparison"]["entity_movement"][t]
        labels = ", ".join(moved["unique_recovered_labels"]) or "—"
        print(f"    IoU {t}: {moved['baseline_recovered']} -> "
              f"{moved['pooled_recovered']} recovered   "
              f"+{moved['n_unique_recovered']} unique ({labels})   "
              f"lost={moved['n_lost_baseline_matches']}")
    delta = report["comparison"]["zero_overlap_delta"]["confidence"]
    print(f"    zero-overlap delta  : top100={delta['100']:+.1%}  "
          f"all={delta['all']:+.1%}   "
          f"giant delta={report['comparison']['giant_mask_delta']:+.1%}")
    print()
    verdict = report["gates"]
    for name, gate in verdict["gates"].items():
        print(f"    [{'PASS' if gate['pass'] else 'FAIL'}] {name} = {gate['value']}")
    print(f"    VERDICT: {'ALL GATES PASS' if verdict['all_pass'] else 'FAILED: ' + ', '.join(verdict['failed'])}")
    print(f"    report -> {path}")
    return 0 if verdict["all_pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
