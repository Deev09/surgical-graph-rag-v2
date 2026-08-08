"""ARKitScenes oracle evaluator: axis convention, frame check, boundary.

Two of these tests exist because the failure they guard against is SILENT.
Both the row and column readings of `normalizedAxes` are orthonormal, and a
box rotated by the wrong matrix still sits plausibly inside the room -- so a
mistake in either place produces entities that look fine and are wrong, and
every AR@k built on them is quietly meaningless.

Dataset-guarded: self-skips when the ARKitScenes meshes are not on disk.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from adapters.arkitscenes import ANNOTATION_SUFFIX, read_mesh
from tools.arkitscenes_eval import (
    G5_MIN_ENTITY_FRAC, evidence_coverage, load_canonical_geometry,
    load_oracle_entities, obb_contains,
)

DATA_ROOT = Path.home() / "Desktop/datasets/arkitscenes/Validation"


def _scenes() -> list[Path]:
    if not DATA_ROOT.is_dir():
        return []
    return sorted(d for d in DATA_ROOT.iterdir()
                  if d.is_dir() and (d / f"{d.name}_3dod_mesh.ply").is_file())


def test_obb_axis_convention_is_rows() -> None:
    """`normalizedAxes` reshaped (3,3) has the box axes as ROWS.

    Decided by measurement, not by convention doc. The discriminating case
    is a thin panel: read the axes the wrong way and the short axis points
    somewhere else, so the box slices through empty space. If someone ever
    'corrects' obb_contains to columns, this fails.
    """
    scenes = _scenes()
    if not scenes:
        print("  SKIP (ARKitScenes data not on disk)")
        return
    scene = scenes[0]
    mesh = read_mesh(scene / f"{scene.name}_3dod_mesh_canonical.ply")
    ann = json.loads(
        (scene / f"{scene.name}{ANNOTATION_SUFFIX}").read_text())

    thin = None
    for obj in ann["data"]:
        oa = obj.get("segments", {}).get("obbAligned")
        if not oa:
            continue
        L = np.asarray(oa["axesLengths"])
        if thin is None or L.min() < np.asarray(
                thin["segments"]["obbAligned"]["axesLengths"]).min():
            thin = obj
    if thin is None:
        raise AssertionError("no annotated object with an obbAligned block")

    oa = thin["segments"]["obbAligned"]
    c = np.asarray(oa["centroid"])
    L = np.asarray(oa["axesLengths"])
    M = np.asarray(oa["normalizedAxes"]).reshape(3, 3)
    rows = int(obb_contains(mesh.xyz, c, L, M).sum())
    cols = int(obb_contains(mesh.xyz, c, L, M.T).sum())
    if rows <= cols:
        raise AssertionError(
            f"thinnest object {thin.get('label')!r} (lengths {np.round(L,3)}): "
            f"rows-as-axes contains {rows} vertices, columns {cols}. The "
            "empirical basis for the rows convention no longer holds -- "
            "re-measure before changing obb_contains.")


def test_annotation_boxes_land_inside_the_mesh() -> None:
    """The boxes are in capture axes; the adapter rotated the mesh. If the
    same rotation is not applied to the boxes, entities drift off the
    geometry -- the frame='world' bug, on a new dataset."""
    scenes = _scenes()
    if not scenes:
        print("  SKIP (ARKitScenes data not on disk)")
        return
    for scene in scenes:
        mesh, R, _ = load_canonical_geometry(scene)
        ents = load_oracle_entities(scene, mesh.xyz, R)
        if not ents:
            raise AssertionError(f"{scene.name}: no oracle entities parsed")
        empty = [e.label for e in ents if len(e.vertices) == 0]
        if empty:
            raise AssertionError(
                f"{scene.name}: {len(empty)} annotated entities contain zero "
                f"mesh vertices {empty[:5]}; boxes and geometry disagree")
        lo, hi = mesh.xyz.min(axis=0), mesh.xyz.max(axis=0)
        for e in ents:
            if np.any(e.centroid < lo) or np.any(e.centroid > hi):
                raise AssertionError(
                    f"{scene.name}: {e.label} centroid outside mesh bounds")


def test_frame_check_rejects_a_wrong_rotation() -> None:
    """The range check must have teeth: feed it a rotation that is not the
    adapter's and it has to refuse."""
    scenes = _scenes()
    if not scenes:
        print("  SKIP (ARKitScenes data not on disk)")
        return
    scene = scenes[0]
    mesh, _R, _ = load_canonical_geometry(scene)
    # 90 deg about x: legal rotation, wrong one
    bad = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    try:
        load_oracle_entities(scene, mesh.xyz, bad)
    except ValueError:
        return
    raise AssertionError(
        "load_oracle_entities accepted a wrong rotation; the frame check is "
        "decorative")


def test_g5_gate_arithmetic() -> None:
    """Gate maths on a synthetic scene, so the threshold is pinned even
    when the dataset is absent."""
    from tools.arkitscenes_eval import OracleEntity

    def ent(uid, verts):
        return OracleEntity(uid=uid, label="x", centroid=np.zeros(3),
                            axes_lengths=np.ones(3), axes=np.eye(3),
                            vertices=np.asarray(verts, dtype=np.int64))

    n = 100
    ids = [np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int32)]
    # e1: all 10 vertices seen. e2: 1 of 10 seen (10%, exactly at the floor).
    # e3: 0 of 10 seen.
    ents = [ent("a", range(10)), ent("b", [9] + list(range(50, 59))),
            ent("c", range(60, 70))]
    cov = evidence_coverage(ents, ids, n)
    if cov["n_entities"] != 3:
        raise AssertionError(cov)
    if abs(cov["per_entity"][-1]["seen_frac"] - 1.0) > 1e-9:
        raise AssertionError(f"fully-seen entity not at 1.0: {cov['per_entity']}")
    # exactly at the floor must count as covered (>=, not >)
    at_floor = [p for p in cov["per_entity"]
                if abs(p["seen_frac"] - G5_MIN_ENTITY_FRAC) < 1e-9]
    if not at_floor:
        raise AssertionError(f"expected an entity exactly at the floor: {cov}")
    if cov["entities_at_or_above_min_frac"] != 2:
        raise AssertionError(
            f"expected 2 covered entities, got "
            f"{cov['entities_at_or_above_min_frac']}: {cov['per_entity']}")
    if cov["gate_g5_pass"]:
        raise AssertionError("2/3 = 66.7% must not pass an 80% gate")


def test_only_the_evaluator_reads_annotations() -> None:
    """Repo-level invariant: annotations stay behind the evaluation
    boundary. Any NEW file that learns to open them should have to justify
    itself by editing this list."""
    allowed = {
        "tools/arkitscenes_eval.py",           # the evaluator itself
        "tools/arkitscenes_render.py",         # docstring mention only
        "adapters/arkitscenes.py",             # defines the suffix; never opens
        "tests/tools/test_arkitscenes_eval.py",
        "tests/adapters/test_arkitscenes.py",
        # Enforces the same boundary from the other side: AST-scans
        # tools/arkitscenes_mask3d_eval.py for a code-evaluated string
        # naming the annotation file, which is stricter than this substring
        # sweep. It needs the literal in order to look for it.
        "tests/tools/test_arkitscenes_mask3d_contract.py",
    }
    offenders = []
    for p in sorted(REPO_ROOT.glob("**/*.py")):
        rel = p.relative_to(REPO_ROOT).as_posix()
        if rel.startswith((".venv/", "runs/")) or rel in allowed:
            continue
        text = p.read_text(errors="ignore")
        if ANNOTATION_SUFFIX in text or "3dod_annotation" in text:
            offenders.append(rel)
    if offenders:
        raise AssertionError(
            "these files reference the ARKitScenes annotation file and are "
            f"not on the allowed list: {offenders}")


TESTS = [
    test_obb_axis_convention_is_rows,
    test_annotation_boxes_land_inside_the_mesh,
    test_frame_check_rejects_a_wrong_rotation,
    test_g5_gate_arithmetic,
    test_only_the_evaluator_reads_annotations,
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
