"""Selector wiring for ARKitScenes: contract, plumbing, and eval maths.

The Colab SAM stage has not run, so there are no real masks. Rather than
wait and discover a shape mismatch after paying for GPU time, this test
synthesises a mask file from the REAL id buffers and drives the whole path:
build_views -> proposal_signals -> score_proposals -> AR@k.

The synthetic masks are plumbing fixtures, NOT a result. They live only
here, in a temp dir, and nothing they produce is written to runs/ or
reported as a measurement. `tools/arkitscenes_selector_eval.py` itself
refuses to run without real masks.

Dataset-guarded: self-skips when the rendered views are not on disk.
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from adapters.arkitscenes import scene_id_for
from tools import arkitscenes_selector_eval as ASE
from tools.arkitscenes_eval import load_canonical_geometry, load_oracle_entities

DATA_ROOT = Path.home() / "Desktop/datasets/arkitscenes/Validation"
VIEWS_ROOT = REPO_ROOT / "runs" / "arkitscenes_p1"
N_SYNTH_MASKS = 6


def _smallest_ready_scene() -> Path | None:
    """Smallest scene that has rendered views — keeps the test quick."""
    if not DATA_ROOT.is_dir():
        return None
    ready = []
    for d in sorted(DATA_ROOT.iterdir()):
        if not d.is_dir():
            continue
        views = VIEWS_ROOT / f"views_{scene_id_for(d)}"
        mesh = d / f"{d.name}_3dod_mesh.ply"
        if (views / "ids.npz").is_file() and mesh.is_file():
            ready.append((mesh.stat().st_size, d))
    return min(ready)[1] if ready else None


def _write_synthetic_masks(views_dir: Path, dst: Path) -> None:
    """Deterministic pseudo-masks in the Colab notebook's on-disk format:
    packbits over the id-buffer shape, plus a [n,2] quality array."""
    ids = np.load(views_dir / "ids.npz")
    out: dict[str, np.ndarray] = {}
    for v in range(len(ids.files)):
        buf = ids[f"ids_{v:02d}"]
        h, w = buf.shape
        packed, scores = [], []
        for m in range(N_SYNTH_MASKS):
            img = np.zeros((h, w), dtype=bool)
            y0 = (m * h) // N_SYNTH_MASKS
            y1 = ((m + 1) * h) // N_SYNTH_MASKS
            img[y0:y1, :] = True
            packed.append(np.packbits(img.ravel()))
            scores.append([0.90 + 0.01 * m, 0.95])
        out[f"masks_{v:02d}"] = np.stack(packed)
        out[f"scores_{v:02d}"] = np.asarray(scores, dtype=float)
    np.savez_compressed(dst, **out)


def _stage(scene_dir: Path, tmp: Path) -> Path:
    """A views-root with real id buffers, synthetic masks, oracle-derived bank."""
    scene_id = scene_id_for(scene_dir)
    (tmp / f"views_{scene_id}").symlink_to(VIEWS_ROOT / f"views_{scene_id}")
    _write_synthetic_masks(VIEWS_ROOT / f"views_{scene_id}",
                           tmp / f"c1p1_masks_{scene_id}.npz")

    # Bank = the oracle entities themselves plus junk. Perfect proposals are
    # present by construction, so the EVALUATION MATHS is checkable exactly.
    # This says nothing about selector quality and is not asserted to.
    mesh, R, _ = load_canonical_geometry(scene_dir)
    ents = load_oracle_entities(scene_dir, mesh.xyz, R)
    rng = np.random.default_rng(0)
    proposals = [e.vertices for e in ents]
    for _ in range(10):
        proposals.append(np.sort(rng.choice(len(mesh.xyz), 400, replace=False))
                         .astype(np.int64))
    offs = np.cumsum([0] + [len(p) for p in proposals]).astype(np.int64)
    np.savez_compressed(tmp / f"bank_{scene_id}.npz",
                        vertices=np.concatenate(proposals).astype(np.int64),
                        offsets=offs)
    return tmp


def test_ablation_table_matches_the_replica_tool() -> None:
    """Both datasets must be scored by the same variants, or their AR@k stop
    being comparable. Always runs."""
    from tools.p1_selector_eval import ABLATIONS as REPLICA_ABLATIONS
    if ASE.ABLATIONS is not REPLICA_ABLATIONS:
        raise AssertionError(
            "ARKitScenes selector eval no longer shares the Replica ablation "
            "table; the two datasets can now drift to different scoring")
    if ASE.DEFAULT_VARIANT not in ASE.ABLATIONS:
        raise AssertionError(f"unknown default variant {ASE.DEFAULT_VARIANT!r}")


def test_refuses_to_run_without_real_masks() -> None:
    scene_dir = _smallest_ready_scene()
    if scene_dir is None:
        print("  SKIP (no rendered ARKitScenes views on disk)")
        return
    with tempfile.TemporaryDirectory() as td:
        try:
            ASE.rank_scene(scene_dir, Path(td))
        except FileNotFoundError as e:
            if "c1p1_sam2_colab" not in str(e) and "ids.npz" not in str(e):
                raise AssertionError(f"unhelpful error: {e}")
            return
    raise AssertionError("ranked a scene with no masks present")


def test_build_views_honours_the_fusion_contract() -> None:
    scene_dir = _smallest_ready_scene()
    if scene_dir is None:
        print("  SKIP (no rendered ARKitScenes views on disk)")
        return
    scene_id = scene_id_for(scene_dir)
    views_dir = VIEWS_ROOT / f"views_{scene_id}"
    with tempfile.TemporaryDirectory() as td:
        masks = Path(td) / "m.npz"
        _write_synthetic_masks(views_dir, masks)
        views = ASE.build_views(views_dir, masks)

    if len(views) != 40:
        raise AssertionError(f"expected 40 views, got {len(views)}")
    n_masks = 0
    for i, v in enumerate(views):
        if set(v) != {"visible", "masks", "mask_quality"}:
            raise AssertionError(f"view {i}: keys {sorted(v)}")
        if v["visible"].dtype != np.int64:
            raise AssertionError(f"view {i}: visible dtype {v['visible'].dtype}")
        if v["mask_quality"].shape != (len(v["masks"]), 2):
            raise AssertionError(
                f"view {i}: quality {v['mask_quality'].shape} vs "
                f"{len(v['masks'])} masks")
        for m in v["masks"]:
            if m.dtype != np.int64 or m.ndim != 1:
                raise AssertionError(f"view {i}: mask dtype/shape wrong")
        n_masks += len(v["masks"])
    if n_masks == 0:
        raise AssertionError("no mask lifted from any view; lifting is broken")


def test_full_path_scores_and_the_eval_maths_is_exact() -> None:
    """End-to-end: rank -> evaluate. With the oracle entities present in the
    bank verbatim, the ceiling and AR@all are known exactly, so a wrong IoU
    or curve implementation cannot hide."""
    scene_dir = _smallest_ready_scene()
    if scene_dir is None:
        print("  SKIP (no rendered ARKitScenes views on disk)")
        return
    with tempfile.TemporaryDirectory() as td:
        root = _stage(scene_dir, Path(td))
        ranked = ASE.rank_scene(scene_dir, root)
        rep = ASE.evaluate(ranked, scene_dir)

    n_ent = rep["n_entities"]
    if rep["oracle_ceiling"]["0.50"] != n_ent:
        raise AssertionError(
            f"bank contains every entity verbatim, so ceiling@0.50 must be "
            f"{n_ent}, got {rep['oracle_ceiling']['0.50']}")
    if rep["ar"]["0.50"]["all"] != n_ent:
        raise AssertionError(
            f"AR@all must reach the ceiling: {rep['ar']['0.50']['all']} != {n_ent}")
    if rep["recovery"]["0.50"]["all"] != 1.0:
        raise AssertionError(f"recovery@all must be 1.0: {rep['recovery']}")

    scores = ranked["scores"][ASE.DEFAULT_VARIANT]
    if len(scores) != ranked["n_proposals"]:
        raise AssertionError("one score per proposal expected")
    if not np.all((scores >= 0.0) & (scores <= 1.0)):
        raise AssertionError(f"scores outside [0,1]: {scores.min()}..{scores.max()}")
    if set(rep["ablation"]) != set(ASE.ABLATIONS):
        raise AssertionError("ablation coverage incomplete")


TESTS = [
    test_ablation_table_matches_the_replica_tool,
    test_refuses_to_run_without_real_masks,
    test_build_views_honours_the_fusion_contract,
    test_full_path_scores_and_the_eval_maths_is_exact,
]


def main() -> int:
    failed = 0
    for test in TESTS:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
            print()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
