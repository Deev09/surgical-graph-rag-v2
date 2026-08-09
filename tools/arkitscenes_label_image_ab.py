"""Paired A/B/C: does the labeler improve when it sees real capture RGB?

  python3 tools/arkitscenes_label_image_ab.py

Screening experiment, dev scene only. The label stage currently classifies
each instance from three point-splat renders of that instance ALONE, which is
the same input pathology that refuted four C1-P1 protocols. This varies ONLY
the image origin: same CLIP weights, same vocabulary, same top_k, same 0.28
admission threshold, same evaluator, same delivered instances.

Arms
----
splat         three isolated point-splat views (the current behaviour)
rgb_tight     real capture RGB, context_pad=0.15, target NOT marked
rgb_context   real capture RGB, context_pad=0.60, everything outside the
              target's projected pixels dimmed

`rgb_tight` exists to catch the confound that makes a naive RGB win
worthless: a wide crop of a doorway reads as a kitchen, so a gain could be
CLIP recognising rooms rather than objects. Tight isolates "real pixels" from
"more context"; the marked context arm then tests whether context helps once
the subject is unambiguous.

Padding values are recorded, not frozen. If an arm fails for a diagnosed
reason, change it and record a new run -- this is a development comparison,
not a protocol with sign-off.

Interpretation limits, stated before the numbers exist:
  * the paired dev baseline is 0/7 top-1 and 0/7 top-3 on geometry-matched
    instances. n=7. This is screening evidence; 2/7 could be noise.
  * `curtain` and several visible categories are absent from the 41-class
    vocabulary, so some instances cannot be scored correctly by ANY arm.
    Reported separately as vocabulary-ineligible.
  * the 0.28 admission threshold was chosen against splat scores. It is kept
    for a controlled comparison and is NOT calibrated for RGB.

Arms are written to separate directories and never overwrite one another.
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

from adapters.arkitscenes import (
    ARKitScenesAdapter, build_arkitscenes_capture_bundle,
)
from extractors.arkitscenes_rgb_crops import RgbCropSource
from extractors.arkitscenes_segments import build_arkitscenes_segment_artifacts
from extractors.learned_labels import (
    GLOBAL_INDOOR_VOCABULARY_V1, LearnedLabelConfig, attach_learned_labels,
)
from extractors.serde import dump_entity_artifacts
from adapters.base import ReconstructionConfig

DEFAULT_SCENE = Path.home() / "Desktop/datasets/arkitscenes/Validation/41069021"
DEFAULT_SEG = REPO_ROOT / "runs/arkit_vertical_slice_ms02/bundle_arkitscenes_41069021"
DEFAULT_OUT = REPO_ROOT / "runs/arkit_label_image_ab"

ARMS = (
    ("splat", None, None),
    ("rgb_tight", 0.15, False),
    ("rgb_context", 0.60, True),
)


def build_arm(name, representation, anonymous, seg_dir, scene_dir,
              mesh_xyz, R, pad, mark, out_root):
    src = None
    coverage = None
    if pad is not None:
        crops = RgbCropSource(scene_dir, mesh_xyz, R, stride=6, n_views=3,
                              context_pad=pad, occlusion=True,
                              mark_target=bool(mark))
        src = crops.crops_for
        ids = np.load(Path(seg_dir) / "vertex_instance_ids.npy")
        per = {}
        for inst in sorted(int(i) for i in np.unique(ids) if i >= 0):
            per[inst] = crops.coverage(np.flatnonzero(ids == inst))
        coverage = {
            "n_instances": len(per),
            "n_with_three_views": sum(1 for c in per.values() if c["has_full_views"]),
            "n_with_any_view": sum(1 for c in per.values() if c["n_views_available"]),
            "median_best_visible_fraction": float(np.median(
                [c["best_visible_fraction"] for c in per.values()])),
            "per_instance": {str(k): v for k, v in per.items()},
        }

    labeled = attach_learned_labels(
        representation, anonymous, segmentation_dir=Path(seg_dir),
        config=LearnedLabelConfig(),
        image_source=src,
        image_source_name=(
            "instance_point_splat_3view" if pad is None
            else f"arkitscenes_rgb_crop_pad{pad}_mark{int(bool(mark))}"),
    )
    arm_dir = out_root / name
    dump_entity_artifacts(labeled, arm_dir / "entities")
    if coverage is not None:
        (arm_dir / "coverage.json").write_text(
            json.dumps(coverage, indent=1, sort_keys=True) + "\n")
    return labeled, coverage, arm_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE)
    ap.add_argument("--segmentation-dir", type=Path, default=DEFAULT_SEG)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--arms", nargs="*", default=[a[0] for a in ARMS])
    args = ap.parse_args(argv)

    from tools.arkitscenes_eval import load_canonical_geometry
    mesh, R, _ = load_canonical_geometry(args.scene_dir)

    representation = ARKitScenesAdapter().reconstruct(
        build_arkitscenes_capture_bundle(args.scene_dir),
        ReconstructionConfig(name="arkit_label_image_ab", version="0.1"),
    )
    anonymous = build_arkitscenes_segment_artifacts(
        representation, args.segmentation_dir, min_vertices=20)
    print(f"anonymous entities: {len(anonymous.entities)}")

    # Every delivered instance must be labeled in every arm: the evaluator
    # requires labels to cover the delivered partition exactly, and dropping
    # an instance from only the RGB arms would break pairing. Small targets
    # therefore get grown crops rather than no crop.
    args.out.mkdir(parents=True, exist_ok=True)
    results = {}

    for name, pad, mark in ARMS:
        if name not in args.arms:
            continue
        print(f"\n=== arm {name}  (pad={pad} mark={mark})")
        labeled, cov, arm_dir = build_arm(
            name, representation, anonymous, args.segmentation_dir,
            args.scene_dir, mesh.xyz, R, pad, mark, args.out)
        if cov:
            print(f"    coverage: {cov['n_with_three_views']}/"
                  f"{cov['n_instances']} instances with 3 verified views, "
                  f"median visible fraction {cov['median_best_visible_fraction']:.2f}")

        # ---- ORACLE BOUNDARY: predictions are finalized on disk above ----
        from tools.arkitscenes_learned_label_eval import evaluate_labels
        rep = evaluate_labels(args.scene_dir, arm_dir / "entities",
                              args.segmentation_dir)
        pm = rep["metrics"]["per_match"]
        t1 = sum(1 for r in pm if r["top1_correct"])
        t3 = sum(1 for r in pm if r["top3_correct"])
        elig = [r for r in pm
                if r["oracle_class_normalized"] in GLOBAL_INDOOR_VOCABULARY_V1]
        et1 = sum(1 for r in elig if r["top1_correct"])
        et3 = sum(1 for r in elig if r["top3_correct"])
        adm = rep["metrics"]["admission"]
        results[name] = {
            "bundle_hash": labeled.bundle_hash,
            "matched": len(pm), "top1": t1, "top3": t3,
            "vocab_eligible": len(elig), "elig_top1": et1, "elig_top3": et3,
            "admitted": adm["n_admitted"], "admission_precision": adm["precision"],
            "coverage": cov,
        }
        (arm_dir / "label_eval.json").write_text(
            json.dumps(rep, indent=1, sort_keys=True) + "\n")
        print(f"    top1 {t1}/{len(pm)}   top3 {t3}/{len(pm)}   "
              f"vocab-eligible {et1}/{len(elig)} | {et3}/{len(elig)}   "
              f"admitted {adm['n_admitted']} (prec {adm['precision']:.2f})")

    (args.out / "summary.json").write_text(
        json.dumps(results, indent=1, sort_keys=True) + "\n")
    print(f"\n{'arm':14s} {'top1':>8s} {'top3':>8s} {'elig top3':>11s} "
          f"{'admitted':>9s} {'3-view cov':>11s}")
    for name in [a[0] for a in ARMS]:
        r = results.get(name)
        if not r:
            continue
        cov = r["coverage"]
        cs = (f"{cov['n_with_three_views']}/{cov['n_instances']}" if cov else "n/a")
        print(f"{name:14s} {r['top1']:4d}/{r['matched']:<3d} "
              f"{r['top3']:4d}/{r['matched']:<3d} "
              f"{r['elig_top3']:5d}/{r['vocab_eligible']:<5d} "
              f"{r['admitted']:9d} {cs:>11s}")
    print(f"-> {args.out/'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
