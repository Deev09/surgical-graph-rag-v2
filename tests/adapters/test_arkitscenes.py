"""ARKitScenes adapter: frame contract, and the annotation boundary.

The load-bearing test here is the annotation boundary. ARKitScenes ships
ground-truth oriented boxes next to the mesh; an adapter that quietly reads
them would rebuild the oracle entity path on a new dataset and make every
downstream number an oracle number without anyone noticing. Two independent
checks below, one static and one at runtime.

Dataset-guarded: self-skips when the ARKitScenes meshes are not on disk.
"""
from __future__ import annotations

import ast
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from adapters.arkitscenes import (
    ANNOTATION_SUFFIX, ARKitScenesAdapter, ARKitScenesMesh,
    build_arkitscenes_capture_bundle, read_mesh, write_mesh,
)
from adapters.base import ReconstructionConfig

DATA_ROOT = Path.home() / "Desktop/datasets/arkitscenes/Validation"
MODULE = REPO_ROOT / "adapters" / "arkitscenes.py"
CFG = ReconstructionConfig(name="arkitscenes_mesh", version="0.1")


def _scenes() -> list[Path]:
    if not DATA_ROOT.is_dir():
        return []
    return sorted(d for d in DATA_ROOT.iterdir()
                  if d.is_dir() and (d / f"{d.name}_3dod_mesh.ply").is_file())


# --- the annotation boundary ---------------------------------------------

def test_adapter_cannot_parse_json_at_all() -> None:
    """Static: the module never imports json. ARKitScenes annotations are a
    JSON file; a module with no JSON parser cannot read one, whatever a
    future edit does to its logic."""
    tree = ast.parse(MODULE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
        else:
            continue
        if "json" in names:
            raise AssertionError(
                "adapters/arkitscenes.py imports json -- it can now parse the "
                "annotation file. Annotations belong behind the evaluation "
                "boundary; see the module docstring.")


def test_reconstruct_opens_no_annotation_file() -> None:
    """Runtime: audit every file opened during a real reconstruct()."""
    scenes = _scenes()
    if not scenes:
        print("  SKIP (ARKitScenes data not on disk)")
        return
    opened: list[str] = []

    def hook(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes)):
            opened.append(str(args[0]))

    sys.addaudithook(hook)
    cap = build_arkitscenes_capture_bundle(scenes[0])
    ARKitScenesAdapter().reconstruct(cap, CFG)
    if not opened:
        raise AssertionError("audit hook captured nothing; it is not a control")
    bad = [p for p in opened if ANNOTATION_SUFFIX in p]
    if bad:
        raise AssertionError(f"reconstruct() opened annotation file(s): {bad}")


def test_capture_bundle_excludes_semantic_export() -> None:
    scenes = _scenes()
    if not scenes:
        print("  SKIP (ARKitScenes data not on disk)")
        return
    cap = build_arkitscenes_capture_bundle(scenes[0])
    if cap.semantic_export is not None:
        raise AssertionError(
            f"semantic_export must be None, got {cap.semantic_export!r}")
    if not cap.notes.get("annotation_present_but_unread"):
        raise AssertionError(
            "expected the annotation file to exist on disk and be recorded "
            "as present-but-unread; the boundary test is vacuous otherwise")


def test_reconstruct_refuses_a_semantic_export() -> None:
    scenes = _scenes()
    if not scenes:
        print("  SKIP (ARKitScenes data not on disk)")
        return
    cap = build_arkitscenes_capture_bundle(scenes[0])
    poisoned = type(cap)(
        **{**cap.__dict__,
           "semantic_export": scenes[0] / f"{scenes[0].name}{ANNOTATION_SUFFIX}"})
    try:
        ARKitScenesAdapter().reconstruct(poisoned, CFG)
    except ValueError:
        return
    raise AssertionError("reconstruct() accepted a semantic_export")


# --- the frame contract ---------------------------------------------------

def test_bundle_declares_scene_canonical_and_means_it() -> None:
    """frame.kind must describe the coordinates the geometry is actually in.
    Declaring scene_canonical while shipping unrotated coordinates is the
    exact bug docs/frame_decision.md fixed."""
    scenes = _scenes()
    if not scenes:
        print("  SKIP (ARKitScenes data not on disk)")
        return
    ad = ARKitScenesAdapter()
    for scene in scenes:
        b = ad.reconstruct(build_arkitscenes_capture_bundle(scene), CFG)
        if b.frame.kind != "scene_canonical":
            raise AssertionError(f"{scene.name}: frame.kind={b.frame.kind!r}")
        if b.notes["reads_annotations"] is not False:
            raise AssertionError(f"{scene.name}: reads_annotations not False")

        # the handle must point at the canonical mesh, not the source
        uri = Path(b.geometry_handle.uri)
        if not uri.is_file():
            raise AssertionError(f"{scene.name}: geometry handle {uri} missing")
        if uri == Path(b.geometry_handle.notes["source_mesh"]):
            raise AssertionError(
                f"{scene.name}: handle points at the UNROTATED source mesh")

        # and the geometry it points at must actually be +z up: re-estimate
        from geometry.frame import angle_between_deg, estimate_scene_frame
        m = read_mesh(uri)
        est = estimate_scene_frame(m.xyz, m.faces)
        off = angle_between_deg(np.asarray(est.up_axis), np.array([0.0, 0.0, 1.0]))
        off = min(off, 180.0 - off)
        if off > 0.05:
            raise AssertionError(
                f"{scene.name}: canonical mesh up is {off:.4f} deg off +z; "
                "frame.kind claims scene_canonical")


def test_ply_round_trip_preserves_geometry() -> None:
    scenes = _scenes()
    if not scenes:
        print("  SKIP (ARKitScenes data not on disk)")
        return
    src = scenes[0] / f"{scenes[0].name}_3dod_mesh.ply"
    m = read_mesh(src)
    tmp = Path("/tmp/_ark_roundtrip.ply")
    write_mesh(tmp, ARKitScenesMesh(xyz=m.xyz, rgb=m.rgb, faces=m.faces))
    back = read_mesh(tmp)
    tmp.unlink(missing_ok=True)
    if not np.allclose(m.xyz, back.xyz, atol=1e-3):
        raise AssertionError("round-trip moved vertices beyond float32 precision")
    if not np.array_equal(m.faces, back.faces):
        raise AssertionError("round-trip changed face indices")
    if not np.array_equal(m.rgb, back.rgb):
        raise AssertionError("round-trip changed vertex colour")


def test_renderer_consumes_the_bundle_geometry() -> None:
    """The reason vertex colour is read at all: the splat renderer needs it,
    and its id_buffer is what makes 2D->3D lifting portable. This is the
    check that the C1-P1 mechanism ports to a non-Replica capture."""
    scenes = _scenes()
    if not scenes:
        print("  SKIP (ARKitScenes data not on disk)")
        return
    from segmenter.proposal_fusion import lift_mask
    from segmenter.view_render import render_all

    b = ARKitScenesAdapter().reconstruct(
        build_arkitscenes_capture_bundle(scenes[0]), CFG)
    m = read_mesh(Path(b.geometry_handle.uri))
    for i, _cam, img, ids in render_all(m.xyz, m.rgb):
        if ids.shape != img.shape[:2]:
            raise AssertionError("id_buffer shape does not match the image")
        vis = ids[ids >= 0]
        if vis.size == 0:
            raise AssertionError(f"view {i}: id_buffer is entirely empty")
        if vis.max() >= len(m.xyz):
            raise AssertionError(
                f"view {i}: id_buffer holds an out-of-range vertex index")
        mask = np.zeros(ids.shape, dtype=bool)
        h, w = ids.shape
        mask[h // 3:2 * h // 3, w // 3:2 * w // 3] = True
        if len(lift_mask(mask, ids)) == 0:
            raise AssertionError(f"view {i}: lift_mask returned nothing")
        break   # one view is enough to prove the contract holds


TESTS = [
    test_adapter_cannot_parse_json_at_all,
    test_reconstruct_opens_no_annotation_file,
    test_capture_bundle_excludes_semantic_export,
    test_reconstruct_refuses_a_semantic_export,
    test_bundle_declares_scene_canonical_and_means_it,
    test_ply_round_trip_preserves_geometry,
    test_renderer_consumes_the_bundle_geometry,
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
