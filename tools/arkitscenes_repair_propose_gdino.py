"""Detector-guided repair proposals: Grounding DINO + SAM 2.1 sidecar -> hashed bank.

  python3 tools/arkitscenes_repair_propose_gdino.py --scene 41069021 \
      --masks runs/arkitscenes_repair/arkitscenes_41069021/repair_gdino_masks_arkitscenes_41069021.npz

Design note: `docs/repair_arm_design_note.md`. Pin: `docs/grounding_dino_pin.json`.
Mechanism: `segmenter/detector_guided_repair.py`.

Annotation-free, in its own executable, ending in a sha256-stamped artifact;
`tools/arkitscenes_repair_eval.py` may open annotation boxes only afterwards.
`eval/detection_repair.py` is untouched.

FIVE JOINS CHECKED, NOT ASSUMED
-------------------------------
  * `selection_sha256` must match the frame bundle. Masks are indexed by slot;
    a slot meaning a different photograph lifts onto the wrong geometry with no
    other symptom.
  * SAM `sam2_commit` and `checkpoint_sha256` must match the frozen pin.
  * The detector `model_id`, and once frozen its `revision` and `sha256`, must
    match `docs/grounding_dino_pin.json`. While those are null the run is
    reported as UNPINNED and the observed values are printed for the freeze
    commit -- the procedure `docs/c1_p1_multiview_proposals_protocol.md` used
    for the SAM weights.
  * `box_threshold` / `text_threshold` must equal the published defaults in the
    pin. A sidecar produced at any other operating point is refused outright;
    that is what "no sweep" means mechanically.
  * The sidecar's vocabulary must hash to the repo's 41-class list. A silently
    reordered or edited vocabulary changes both the prompt and the labels
    association depends on.
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

from adapters.arkitscenes import scene_id_for
from eval.detection_repair import Proposal, ProposalArtifact
from extractors.arkitscenes_rgb_crops import load_frames
from segmenter.base import sha256_file
from segmenter.detector_guided_repair import (
    VOCABULARY, generate_from_sidecar, label_from_prompt_phrase,
)
from segmenter.rgb_multiview_repair import FRAME_STRIDE
from tools.arkitscenes_eval import DEFAULT_DATA_ROOT, load_canonical_geometry
from tools.arkitscenes_mask3d_eval import DEFAULT_BUNDLE_ROOT
from tools.arkitscenes_repair_eval import DEFAULT_OUT_ROOT, load_baseline_bank
from tools.arkitscenes_repair_propose_sam import (
    PINNED_CHECKPOINT_SHA256, PINNED_SAM2_COMMIT,
)

PIN_PATH = REPO_ROOT / "docs" / "grounding_dino_pin.json"


def vocabulary_sha256(vocabulary) -> str:
    return hashlib.sha256("\n".join(vocabulary).encode()).hexdigest()


def check_pins(env: dict, pin: dict) -> dict:
    """Verify every declared pin. Returns a provenance record."""
    if env.get("sam2_commit") != PINNED_SAM2_COMMIT:
        raise ValueError(f"sidecar ran sam2 @ {env.get('sam2_commit')}, pin is "
                         f"{PINNED_SAM2_COMMIT}")
    if env.get("checkpoint_sha256") != PINNED_CHECKPOINT_SHA256:
        raise ValueError("sidecar SAM checkpoint is not the pinned Hiera-L weight")

    detector = pin["detector"]
    if env.get("detector_model_id") != detector["model_id"]:
        raise ValueError(
            f"sidecar detector {env.get('detector_model_id')!r} is not the "
            f"pinned {detector['model_id']!r}")
    pinned = True
    for field, key in (("revision", "detector_revision"),
                       ("checkpoint_sha256", "detector_sha256")):
        expected = detector.get(field)
        observed = env.get(key)
        if expected is None:
            pinned = False
            continue
        if observed != expected:
            raise ValueError(
                f"sidecar detector {field} {observed!r} does not match the "
                f"pinned {expected!r}")

    thresholds = pin["thresholds"]
    for key in ("box_threshold", "text_threshold"):
        if float(env.get(key, -1)) != float(thresholds[key]):
            raise ValueError(
                f"sidecar {key}={env.get(key)} but the pin fixes "
                f"{thresholds[key]}; no sweep is permitted")

    expected_vocab = vocabulary_sha256(VOCABULARY)
    if env.get("vocabulary_sha256") != expected_vocab:
        raise ValueError(
            f"sidecar vocabulary hashes to {str(env.get('vocabulary_sha256'))[:16]}…, "
            f"the repo's 41-class list hashes to {expected_vocab[:16]}…")
    if list(env.get("vocabulary", [])) != list(VOCABULARY):
        raise ValueError("sidecar vocabulary differs from the repo's list")

    return {
        "detector_model_id": env["detector_model_id"],
        "detector_revision": env.get("detector_revision"),
        "detector_sha256": env.get("detector_sha256"),
        "box_threshold": env["box_threshold"],
        "text_threshold": env["text_threshold"],
        "vocabulary_sha256": expected_vocab,
        "detector_pin_frozen": pinned,
    }


def load_sidecar(path: Path, frames_manifest: dict) -> tuple[dict, dict, dict]:
    """Unpack the detector sidecar; canonicalize phrases to vocabulary labels."""
    z = np.load(path, allow_pickle=False)
    env = json.loads(str(z["env"]))
    if env.get("selection_sha256") != frames_manifest["selection_sha256"]:
        raise ValueError(
            f"sidecar was produced from selection "
            f"{str(env.get('selection_sha256'))[:16]}…, these frames are "
            f"{frames_manifest['selection_sha256'][:16]}…")
    n_frames = len(frames_manifest["frames"])
    if int(env.get("n_frames", -1)) != n_frames:
        raise ValueError(f"sidecar covers {env.get('n_frames')} frames, the "
                         f"selection has {n_frames}")

    unresolved: dict[str, int] = {}
    out: dict[int, dict] = {}
    for slot in range(n_frames):
        row = frames_manifest["frames"][slot]
        height, width = int(row["height"]), int(row["width"])
        if f"shape_{slot:02d}" in z.files:
            got = z[f"shape_{slot:02d}"].tolist()
            if got != [height, width]:
                raise ValueError(
                    f"slot {slot}: sidecar masks are {got}, frame is "
                    f"{[height, width]}")
        packed = z[f"masks_{slot:02d}"]
        count = height * width
        masks = (np.unpackbits(packed, axis=1, count=count).astype(bool)
                 .reshape(len(packed), height, width)
                 if len(packed) else np.zeros((0, height, width), dtype=bool))
        phrases = [str(p) for p in z[f"phrases_{slot:02d}"]]
        labels = []
        for phrase in phrases:
            label = label_from_prompt_phrase(phrase)
            if label is None:
                unresolved[phrase] = unresolved.get(phrase, 0) + 1
            labels.append(label)
        out[slot] = {
            "masks": masks,
            "sam_scores": np.stack([
                z[f"sam_scores_{slot:02d}"],
                np.ones(len(masks), dtype=np.float32)], axis=1)
            if len(masks) else np.zeros((0, 2), dtype=np.float32),
            "labels": labels,
            "detector_scores": z[f"det_scores_{slot:02d}"],
            "boxes": z[f"boxes_{slot:02d}"],
            "phrases": phrases,
        }
    return out, env, unresolved


def drop_unresolved(sidecar: dict) -> tuple[dict, int, int]:
    """Remove detections whose phrase did not resolve to exactly one class.

    A guessed label would corrupt label-guided association, which is the whole
    mechanism under test, so an ambiguous detection is dropped rather than
    coerced. The count is reported, not hidden.
    """
    kept_total = dropped_total = 0
    for slot, entry in sidecar.items():
        keep = [i for i, label in enumerate(entry["labels"]) if label is not None]
        dropped_total += len(entry["labels"]) - len(keep)
        kept_total += len(keep)
        if len(keep) == len(entry["labels"]):
            continue
        index = np.asarray(keep, dtype=np.int64)
        entry["masks"] = entry["masks"][index] if len(index) else \
            entry["masks"][:0]
        entry["sam_scores"] = entry["sam_scores"][index] if len(index) else \
            entry["sam_scores"][:0]
        entry["detector_scores"] = entry["detector_scores"][index] if len(index) \
            else entry["detector_scores"][:0]
        entry["boxes"] = entry["boxes"][index] if len(index) else \
            entry["boxes"][:0]
        entry["labels"] = [entry["labels"][i] for i in keep]
        entry["phrases"] = [entry["phrases"][i] for i in keep]
    return sidecar, kept_total, dropped_total


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
        print(f"missing {bundle_dir}/meta.json")
        return 1

    t0 = time.perf_counter()
    pin = json.loads(PIN_PATH.read_text())
    frames_manifest = json.loads(frames_json.read_text())
    mesh, rotation, bundle = load_canonical_geometry(scene_dir)
    mesh_sha = sha256_file(scene_dir / f"{args.scene}_3dod_mesh_canonical.ply")
    if frames_manifest["canonical_mesh_sha256"] != mesh_sha:
        raise ValueError("frames were selected against a different mesh")

    sidecar, env, unresolved = load_sidecar(args.masks, frames_manifest)
    provenance = check_pins(env, pin)
    sidecar, n_kept, n_dropped = drop_unresolved(sidecar)

    baseline_props, baseline_prov = load_baseline_bank(
        bundle_dir, len(mesh.xyz), mesh_sha)
    baseline = [p.vertices for p in baseline_props]
    xyz_world = mesh.xyz @ rotation
    frames = load_frames(scene_dir, stride=frames_manifest.get(
        "frame_stride", FRAME_STRIDE))
    selected = [row["frame_index"] for row in frames_manifest["frames"]]

    print(f"=== {scene_id}   {len(mesh.xyz)} vertices, {len(baseline)} baseline "
          f"proposals, {len(selected)} frames")
    print(f"    detector: {provenance['detector_model_id']} "
          f"@ box>={provenance['box_threshold']} text>={provenance['text_threshold']}"
          f"   [{'PINNED' if provenance['detector_pin_frozen'] else 'UNPINNED'}]")
    if not provenance["detector_pin_frozen"]:
        print(f"    !! docs/grounding_dino_pin.json has null revision/sha. "
              f"Observed revision={env.get('detector_revision')} "
              f"sha={str(env.get('detector_sha256'))[:16]}… — freeze these "
              "before claiming a result.")
    print(f"    detections: {n_kept + n_dropped} total, {n_dropped} dropped for "
          f"unresolvable phrases ({len(unresolved)} distinct)")

    proposals, diagnostics = generate_from_sidecar(
        xyz_world, frames, selected, sidecar, baseline,
        progress=lambda m: print(f"    {m}", flush=True))

    labels = diagnostics["emitted_labels"]
    artifact = ProposalArtifact.finalize(
        "repair_gdino_sam",
        [Proposal(p.vertices, "repair", p.kind, p.confidence,
                  {"label": label, "support_views": p.consensus_views,
                   "parent_index": p.parent_index,
                   "containment": round(p.containment, 4)})
         for p, label in zip(proposals, labels)],
        len(mesh.xyz), out_dir / "repair_bank_gdino.npz",
        {"mechanism": "segmenter/detector_guided_repair.py",
         "design_note": "docs/repair_arm_design_note.md",
         "detector_pin": provenance,
         "canonical_mesh_sha256": mesh_sha,
         "representation_hash": bundle.representation_hash,
         "selection_sha256": frames_manifest["selection_sha256"],
         "sam_env": {k: v for k, v in env.items() if k != "vocabulary"},
         "baseline_bank": baseline_prov,
         "config": diagnostics["config"]})

    diagnostics.update({
        "scene_id": scene_id, "video_id": args.scene,
        "canonical_mesh_sha256": mesh_sha,
        "proposal_sha256": artifact.sha256,
        "detector_pin": provenance,
        "unresolved_phrases": unresolved,
        "n_detections_kept": n_kept, "n_detections_dropped": n_dropped,
        "runtime_seconds": round(time.perf_counter() - t0, 1),
        "proposals": [{
            "kind": p.kind, "label": label,
            "n_vertices": int(len(p.vertices)),
            "confidence": round(p.confidence, 4),
            "support_views": int(p.consensus_views),
            "parent_index": p.parent_index,
            "containment": round(p.containment, 4),
        } for p, label in zip(proposals, labels)],
    })
    (out_dir / "repair_gdino_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=1, sort_keys=True) + "\n")

    sizes = (np.array([len(p.vertices) for p in proposals]) if proposals
             else np.zeros(0))
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
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
    print(f"    labels              : " + ", ".join(
        f"{k}×{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:10])
        or "none")
    print(f"    proposal sha256     : {artifact.sha256[:16]}…")
    print(f"    bank  -> {out_dir / 'repair_bank_gdino.npz'}")
    print(f"    diags -> {out_dir / 'repair_gdino_diagnostics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
