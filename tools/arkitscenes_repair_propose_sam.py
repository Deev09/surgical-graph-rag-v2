"""SAM 2.1 repair proposals: pinned mask sidecar -> hash-stamped proposal bank.

  python3 tools/arkitscenes_repair_propose_sam.py --scene 41069021 \
      --masks runs/arkitscenes_repair/arkitscenes_41069021/repair_sam_masks_arkitscenes_41069021.npz

Design note: `docs/repair_arm_design_note.md`. Mechanism:
`segmenter/sam_multiview_repair.py`. Frames and their pins come from
`tools/arkitscenes_repair_frames.py`; masks come from
`notebooks/repair_sam2_colab.ipynb`.

ANNOTATION-FREE, in its own executable, for the same reason as the previous
arm: it ends by writing a sha256-stamped proposal artifact, and only then may
`tools/arkitscenes_repair_eval.py` open annotation boxes and score it.

THREE JOINS THAT ARE CHECKED, NOT ASSUMED
-----------------------------------------
A mask sidecar that came back from a GPU run months ago can silently belong to
a different selection, a different scene, or a different model pin. All three
are refused here:

  * `selection_sha256` in the sidecar must equal the one in `frames.json`.
    This is the join that matters -- masks are indexed by slot, and a slot
    that means a different photograph lifts onto the wrong geometry with no
    other symptom.
  * `checkpoint_sha256` and `sam2_commit` must match the frozen pin.
  * the canonical mesh hash in `frames.json` must match this scene's mesh.
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
from extractors.arkitscenes_rgb_crops import load_frames
from segmenter.base import sha256_file
from segmenter.rgb_multiview_repair import FRAME_STRIDE
from segmenter.sam_multiview_repair import generate_from_sidecar
from tools.arkitscenes_eval import DEFAULT_DATA_ROOT, load_canonical_geometry
from tools.arkitscenes_mask3d_eval import DEFAULT_BUNDLE_ROOT
from tools.arkitscenes_repair_eval import DEFAULT_OUT_ROOT, load_baseline_bank

PINNED_SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
PINNED_CHECKPOINT_SHA256 = (
    "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318")


def load_sidecar(path: Path, frames_manifest: dict) -> tuple[dict, dict]:
    """Unpack the SAM sidecar and verify it belongs to this frame selection."""
    z = np.load(path, allow_pickle=False)
    env = json.loads(str(z["env"]))
    if env.get("sam2_commit") != PINNED_SAM2_COMMIT:
        raise ValueError(
            f"sidecar ran sam2 @ {env.get('sam2_commit')}, pin is "
            f"{PINNED_SAM2_COMMIT}")
    if env.get("checkpoint_sha256") != PINNED_CHECKPOINT_SHA256:
        raise ValueError(
            f"sidecar checkpoint sha {env.get('checkpoint_sha256')} is not the "
            "pinned SAM 2.1 Hiera-L weight")
    if env.get("selection_sha256") != frames_manifest["selection_sha256"]:
        raise ValueError(
            f"sidecar was produced from selection "
            f"{str(env.get('selection_sha256'))[:16]}…, these frames are "
            f"{frames_manifest['selection_sha256'][:16]}… — the masks do not "
            "belong to this frame set")
    n_frames = len(frames_manifest["frames"])
    if int(env.get("n_frames", -1)) != n_frames:
        raise ValueError(
            f"sidecar covers {env.get('n_frames')} frames, the selection has "
            f"{n_frames}")

    masks: dict[int, dict] = {}
    for slot in range(n_frames):
        key = f"masks_{slot:02d}"
        if key not in z.files:
            raise ValueError(f"sidecar is missing {key}")
        row = frames_manifest["frames"][slot]
        height, width = int(row["height"]), int(row["width"])
        if f"shape_{slot:02d}" in z.files:
            got = z[f"shape_{slot:02d}"].tolist()
            if got != [height, width]:
                raise ValueError(
                    f"slot {slot}: sidecar masks are {got}, the frame is "
                    f"{[height, width]}")
        packed = z[key]
        count = height * width
        unpacked = (np.unpackbits(packed, axis=1, count=count).astype(bool)
                    .reshape(len(packed), height, width)
                    if len(packed) else np.zeros((0, height, width), dtype=bool))
        scores = z.get(f"scores_{slot:02d}")
        if scores is None:
            scores = np.ones((len(unpacked), 2), dtype=np.float32)
        masks[slot] = {"masks": unpacked, "scores": np.asarray(scores)}
    return masks, env


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", default="41069021")
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = ap.parse_args(argv)

    scene_dir = args.data_root / args.scene
    scene_id = scene_id_for(scene_dir)
    out_dir = args.out_root / scene_id
    frames_json = out_dir / f"repair_frames_{scene_id}" / "frames.json"
    if not frames_json.is_file():
        print(f"missing {frames_json} — run tools/arkitscenes_repair_frames.py first")
        return 1
    bundle_dir = args.bundle_root / f"bundle_{scene_id}"
    if not (bundle_dir / "meta.json").is_file():
        print(f"missing {bundle_dir}/meta.json — extract the Mask3D bundle there")
        return 1

    t0 = time.perf_counter()
    frames_manifest = json.loads(frames_json.read_text())
    mesh, rotation, bundle = load_canonical_geometry(scene_dir)
    mesh_sha = sha256_file(scene_dir / f"{args.scene}_3dod_mesh_canonical.ply")
    if frames_manifest["canonical_mesh_sha256"] != mesh_sha:
        raise ValueError(
            f"frames were selected against mesh "
            f"{frames_manifest['canonical_mesh_sha256'][:16]}…, this scene is "
            f"{mesh_sha[:16]}…")

    sidecar_masks, env = load_sidecar(args.masks, frames_manifest)
    baseline_props, baseline_prov = load_baseline_bank(
        bundle_dir, len(mesh.xyz), mesh_sha)
    baseline = [p.vertices for p in baseline_props]

    xyz_world = mesh.xyz @ rotation
    frames = load_frames(scene_dir, stride=frames_manifest.get(
        "frame_stride", FRAME_STRIDE))
    selected = [row["frame_index"] for row in frames_manifest["frames"]]
    for row in frames_manifest["frames"]:
        actual = frames[row["frame_index"]].png.name
        if actual != row["source_png"]:
            raise ValueError(
                f"slot {row['slot']}: frame index {row['frame_index']} is now "
                f"{actual}, the selection recorded {row['source_png']}")

    print(f"=== {scene_id}   {len(mesh.xyz)} vertices, "
          f"{len(baseline)} baseline proposals, {len(selected)} frames")
    print(f"    SAM: {env['device']}, {env['elapsed_seconds']}s, "
          f"selection {env['selection_sha256'][:16]}…")

    proposals, diagnostics = generate_from_sidecar(
        xyz_world, frames, selected, sidecar_masks, baseline,
        progress=lambda m: print(f"    {m}", flush=True))

    artifact = ProposalArtifact.finalize(
        "repair_sam_multiview",
        [Proposal(p.vertices, "repair", p.kind, p.confidence,
                  {"support_views": p.consensus_views,
                   "parent_index": p.parent_index,
                   "containment": round(p.containment, 4)})
         for p in proposals],
        len(mesh.xyz), out_dir / "repair_bank_sam.npz",
        {"mechanism": "segmenter/sam_multiview_repair.py",
         "design_note": "docs/repair_arm_design_note.md",
         "canonical_mesh_sha256": mesh_sha,
         "representation_hash": bundle.representation_hash,
         "selection_sha256": frames_manifest["selection_sha256"],
         "sam_env": env,
         "baseline_bank": baseline_prov,
         "config": diagnostics["config"]})

    diagnostics.update({
        "scene_id": scene_id, "video_id": args.scene,
        "canonical_mesh_sha256": mesh_sha,
        "selection_sha256": frames_manifest["selection_sha256"],
        "proposal_sha256": artifact.sha256,
        "sam_env": env,
        "runtime_seconds": round(time.perf_counter() - t0, 1),
        "proposals": [{
            "kind": p.kind, "n_vertices": int(len(p.vertices)),
            "confidence": round(p.confidence, 4),
            "support_views": int(p.consensus_views),
            "parent_index": p.parent_index,
            "containment": round(p.containment, 4),
        } for p in proposals],
    })
    (out_dir / "repair_sam_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=1, sort_keys=True) + "\n")

    sizes = (np.array([len(p.vertices) for p in proposals]) if proposals
             else np.zeros(0))
    print()
    print(f"    lifted masks        : {diagnostics['n_lifted_masks']}")
    print(f"    clusters            : {diagnostics['n_raw_clusters']} raw, "
          f"{diagnostics['n_clusters']} supported "
          f"({diagnostics['n_unsupported_clusters']} dropped)")
    print(f"    emitted             : {len(proposals)} "
          f"({diagnostics['by_kind']['additional']} additional, "
          f"{diagnostics['by_kind']['split']} split)")
    if len(sizes):
        print(f"    proposal vertices   : median {int(np.median(sizes))}, "
              f"max {int(sizes.max())}")
    rejected = ", ".join(f"{k}={v}" for k, v in diagnostics["rejected"].items() if v)
    print(f"    rejected            : {rejected or 'none'}")
    print(f"    proposal sha256     : {artifact.sha256[:16]}…")
    print(f"    runtime             : {diagnostics['runtime_seconds']}s")
    print(f"    bank  -> {out_dir / 'repair_bank_sam.npz'}")
    print(f"    diags -> {out_dir / 'repair_sam_diagnostics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
