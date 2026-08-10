"""Generate multi-view RGB repair proposals for one ARKitScenes scan.

  python3 tools/arkitscenes_repair_propose.py --scene 41069021

Design note: `docs/repair_arm_design_note.md`. Mechanism:
`segmenter/rgb_multiview_repair.py`.

This is the ANNOTATION-FREE half of the arm, in its own executable on purpose.
It ends by writing a sha256-stamped proposal artifact; only then may
`tools/arkitscenes_repair_eval.py` open annotation boxes and score it. Keeping
generation in a separate process makes that ordering an operational fact
rather than a code-reading exercise -- this program cannot see an oracle
because it never loads one, and the artifact it writes is frozen before the
evaluator starts.

It also writes `repair_diagnostics.json`: which frames were chosen, how many
2D regions each contributed, how many components fusion produced, and exactly
why each rejected component was rejected. That file is the inspectable
artifact the design note promises in place of a long protocol document.
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
from eval.detection_repair import Proposal, ProposalArtifact
from geometry.mesh_surfaces import load_raw_triangle_mesh
from segmenter.base import sha256_file
from segmenter.rgb_multiview_repair import (
    FRAME_STRIDE, N_FRAMES, generate_repair_proposals,
)
from tools.arkitscenes_eval import DEFAULT_DATA_ROOT, load_canonical_geometry
from tools.arkitscenes_mask3d_eval import DEFAULT_BUNDLE_ROOT
from tools.arkitscenes_repair_eval import DEFAULT_OUT_ROOT, load_baseline_bank


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", default="41069021")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--n-frames", type=int, default=N_FRAMES)
    ap.add_argument("--frame-stride", type=int, default=FRAME_STRIDE)
    args = ap.parse_args(argv)

    scene_dir = args.data_root / args.scene
    scene_id = scene_id_for(scene_dir)
    bundle_dir = args.bundle_root / f"bundle_{scene_id}"
    if not (bundle_dir / "meta.json").is_file():
        print(f"missing {bundle_dir}/meta.json — extract the Mask3D bundle there first")
        return 1
    out_dir = args.out_root / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    mesh_ply = scene_dir / f"{args.scene}_3dod_mesh_canonical.ply"
    mesh, rotation, bundle = load_canonical_geometry(scene_dir)
    mesh_sha = sha256_file(mesh_ply)
    raw = load_raw_triangle_mesh(mesh_ply)
    if len(raw.xyz) != len(mesh.xyz):
        raise ValueError(
            f"triangle mesh has {len(raw.xyz)} vertices, canonical geometry "
            f"has {len(mesh.xyz)}")

    baseline_props, baseline_prov = load_baseline_bank(
        bundle_dir, len(mesh.xyz), mesh_sha)
    baseline = [p.vertices for p in baseline_props]
    print(f"=== {scene_id}   {len(mesh.xyz)} vertices, "
          f"{len(raw.faces)} faces, {len(baseline)} baseline proposals")

    proposals, diagnostics = generate_repair_proposals(
        scene_dir, mesh.xyz, rotation, raw.faces, baseline,
        n_frames=args.n_frames, frame_stride=args.frame_stride,
        progress=lambda m: print(f"    {m}", flush=True))

    artifact = ProposalArtifact.finalize(
        "repair_rgb_multiview",
        [Proposal(p.vertices, "repair", p.kind, p.confidence,
                  {"cut": p.cut, "parent_index": p.parent_index,
                   "containment": round(p.containment, 4),
                   "consensus_views": round(p.consensus_views, 3)})
         for p in proposals],
        len(mesh.xyz), out_dir / "repair_bank.npz",
        {"mechanism": "segmenter/rgb_multiview_repair.py",
         "design_note": "docs/repair_arm_design_note.md",
         "canonical_mesh_sha256": mesh_sha,
         "representation_hash": bundle.representation_hash,
         "baseline_bank": baseline_prov,
         "config": diagnostics["config"]})

    diagnostics["scene_id"] = scene_id
    diagnostics["video_id"] = args.scene
    diagnostics["canonical_mesh_sha256"] = mesh_sha
    diagnostics["proposal_sha256"] = artifact.sha256
    diagnostics["runtime_seconds"] = round(time.perf_counter() - t0, 1)
    diagnostics["proposals"] = [{
        "kind": p.kind,
        "n_vertices": int(len(p.vertices)),
        "confidence": round(p.confidence, 4),
        "consensus_views": round(p.consensus_views, 3),
        "cut": p.cut,
        "parent_index": p.parent_index,
        "containment": round(p.containment, 4),
    } for p in proposals]
    (out_dir / "repair_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=1, sort_keys=True) + "\n")

    sizes = np.array([len(p.vertices) for p in proposals]) if proposals else np.zeros(0)
    print()
    print(f"    components          : {diagnostics['n_components']}")
    print(f"    candidates          : {diagnostics['n_candidates']}")
    print(f"    emitted             : {len(proposals)} "
          f"({diagnostics['by_kind']['additional']} additional, "
          f"{diagnostics['by_kind']['split']} split)")
    if len(sizes):
        print(f"    proposal vertices   : median {int(np.median(sizes))}, "
              f"max {int(sizes.max())}")
    print(f"    rejected            : " + ", ".join(
        f"{k}={v}" for k, v in diagnostics["rejected"].items() if v))
    print(f"    proposal sha256     : {artifact.sha256[:16]}…")
    print(f"    runtime             : {diagnostics['runtime_seconds']}s")
    print(f"    bank  -> {out_dir / 'repair_bank.npz'}")
    print(f"    diags -> {out_dir / 'repair_diagnostics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
