"""Guards for the real-RGB crop source.

The expensive failure here is silent: a wrong camera convention still
produces crops, they still look like photographs, and the label experiment
then measures nothing. So the convention is pinned by construction on a
synthetic camera, and separately corroborated on the real capture when it is
present.

Dataset-dependent tests skip cleanly.
"""
from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extractors import arkitscenes_rgb_crops as RC

DEV = Path.home() / "Desktop/datasets/arkitscenes/Validation/41069021"


def _dataset_ready() -> bool:
    return (DEV / "lowres_wide.traj").is_file() and (DEV / "lowres_wide").is_dir()


def test_angle_axis_matches_rodrigues_properties() -> None:
    for a in (np.zeros(3), np.array([0.3, -1.2, 0.7]), np.array([math.pi, 0, 0])):
        M = RC.angle_axis_to_matrix(a)
        if not np.allclose(M @ M.T, np.eye(3), atol=1e-9):
            raise AssertionError(f"not orthonormal for {a}")
        if not math.isclose(float(np.linalg.det(M)), 1.0, abs_tol=1e-9):
            raise AssertionError(f"det != 1 for {a} (reflection, not rotation)")
    # rotating about z by 90 deg sends +x to +y
    M = RC.angle_axis_to_matrix(np.array([0.0, 0.0, math.pi / 2]))
    if not np.allclose(M @ np.array([1.0, 0, 0]), np.array([0, 1, 0]), atol=1e-9):
        raise AssertionError("z-rotation handedness is wrong")


def test_projection_uses_opencv_camera_axes() -> None:
    """Pin the dataset convention: +x right, +y down, +z forward."""
    f = RC.Frame(0.0, Path("x.png"), np.eye(3), np.zeros(3),
                 200.0, 200.0, 128.0, 96.0, 256, 192)
    above = np.array([[0.0, -0.5, 2.0]])
    uv, inb = RC.project(above, f)
    if not inb[0]:
        raise AssertionError("a +z point in front of the camera was culled")
    if uv[0, 1] >= f.cy:
        raise AssertionError(
            f"point above the axis projected to v={uv[0,1]:.1f} >= cy={f.cy}; "
            "the OpenCV +y-down convention was lost")
    behind = np.array([[0.0, 0.0, -2.0]])
    _, inb2 = RC.project(behind, f)
    if inb2[0]:
        raise AssertionError("a point behind the camera was reported visible")


def test_projection_right_stays_right() -> None:
    f = RC.Frame(0.0, Path("x.png"), np.eye(3), np.zeros(3),
                 200.0, 200.0, 128.0, 96.0, 256, 192)
    uv, inb = RC.project(np.array([[0.5, 0.0, 2.0]]), f)
    if not inb[0] or uv[0, 0] <= f.cx:
        raise AssertionError("+x did not project to the right of centre")


def test_synchronized_depth_pins_camera_convention() -> None:
    """Three separated real frames reject the legacy axis flip.

    This is deliberately dataset-guarded: the repository does not distribute
    ARKitScenes. Sensor depth is independent of the mesh's baked RGB, so the
    regression cannot pass because two visually similar textures happened to
    be compared.
    """
    depth_dir = DEV / "lowres_depth"
    timestamps = (305.377, 380.363, 455.366)
    depth_paths = [depth_dir / f"41069021_{stamp:.3f}.png"
                   for stamp in timestamps]
    mesh_path = DEV / "41069021_3dod_mesh.ply"
    if not (_dataset_ready() and mesh_path.is_file()
            and all(path.is_file() for path in depth_paths)):
        print("  SKIP (ARKitScenes synchronized depth not present)")
        return

    from adapters.arkitscenes import read_mesh
    from tools.arkitscenes_camera_alignment import depth_alignment_metrics

    mesh = read_mesh(mesh_path)
    frames = RC.load_frames(DEV)
    for timestamp, depth_path in zip(timestamps, depth_paths):
        frame = min(frames, key=lambda item: abs(item.timestamp - timestamp))
        if abs(frame.timestamp - timestamp) > 0.001:
            raise AssertionError(
                f"no exact pose for {timestamp}; nearest={frame.timestamp}")
        depth = np.asarray(Image.open(depth_path), dtype=np.uint16)
        direct = depth_alignment_metrics(mesh.xyz, frame, depth)
        legacy = depth_alignment_metrics(
            mesh.xyz, frame, depth, legacy_axis_flip=True)
        if direct["median_abs_error_m"] > 0.06:
            raise AssertionError(
                f"{timestamp}: direct projection median depth error is "
                f"{direct['median_abs_error_m']:.3f} m")
        if direct["share_abs_error_le_0_10m"] < 0.60:
            raise AssertionError(
                f"{timestamp}: only "
                f"{direct['share_abs_error_le_0_10m']:.1%} of common pixels "
                "agree within 10 cm")
        if legacy["median_abs_error_m"] < 4.0 * direct["median_abs_error_m"]:
            raise AssertionError(
                f"{timestamp}: legacy flip was not decisively worse: "
                f"direct={direct['median_abs_error_m']:.3f} m, "
                f"legacy={legacy['median_abs_error_m']:.3f} m")


def test_camera_centres_lie_inside_the_scan() -> None:
    """Corroborates the pose convention on real data: a handheld capture is
    taken from INSIDE the room, at plausible heights. A transposed or
    un-inverted pose puts the camera metres outside the mesh."""
    if not _dataset_ready():
        print("  SKIP (ARKitScenes RGB not present)")
        return
    from adapters.arkitscenes import MESH_SUFFIX, read_mesh
    world = read_mesh(DEV / f"{DEV.name}{MESH_SUFFIX}").xyz
    lo, hi = world.min(axis=0), world.max(axis=0)
    centres = np.array([-Rwc.T @ t
                        for _ts, Rwc, t in RC.load_trajectory(DEV / "lowres_wide.traj")])
    inside = ((centres >= lo - 0.5) & (centres <= hi + 0.5)).all(axis=1)
    if inside.mean() < 0.95:
        raise AssertionError(
            f"only {inside.mean():.1%} of camera centres fall inside the mesh "
            f"bounds — the trajectory convention is wrong")


def test_frames_have_poses_and_intrinsics() -> None:
    if not _dataset_ready():
        print("  SKIP (ARKitScenes RGB not present)")
        return
    frames = RC.load_frames(DEV, stride=200)
    if not frames:
        raise AssertionError("no usable posed frames")
    for f in frames[:5]:
        if f.width <= 0 or f.height <= 0 or f.fx <= 0:
            raise AssertionError(f"bad intrinsics on {f.png.name}")
        if not f.png.is_file():
            raise AssertionError(f"missing image {f.png}")


def test_crops_are_context_padded_and_deterministic() -> None:
    if not _dataset_ready():
        print("  SKIP (ARKitScenes RGB not present)")
        return
    bundle = REPO_ROOT / "runs/arkitscenes_mask3d/bundle_arkitscenes_41069021"
    ids_path = bundle / "vertex_instance_ids.npy"
    if not ids_path.is_file():
        print("  SKIP (Mask3D bundle not extracted)")
        return
    from tools.arkitscenes_eval import load_canonical_geometry
    mesh, R, _ = load_canonical_geometry(DEV)
    ids = np.load(ids_path)
    inst = [i for i in np.unique(ids) if i >= 0]
    vi = np.flatnonzero(ids == inst[0])

    src = RC.RgbCropSource(DEV, mesh.xyz, R, stride=60, n_views=2)
    a = src.crops_for(vi)
    b = RC.RgbCropSource(DEV, mesh.xyz, R, stride=60, n_views=2).crops_for(vi)
    if [x.size for x in a] != [x.size for x in b]:
        raise AssertionError("crop selection is not deterministic")
    for im in a:
        if min(im.size) < RC.MIN_CROP_PX:
            raise AssertionError(f"crop {im.size} below MIN_CROP_PX")

    tight = RC.RgbCropSource(DEV, mesh.xyz, R, stride=60, n_views=2,
                             context_pad=0.0).crops_for(vi)
    if tight and a:
        if tight[0].size[0] > a[0].size[0] or tight[0].size[1] > a[0].size[1]:
            raise AssertionError(
                "context padding did not enlarge the crop; the context arm "
                "would be indistinguishable from the tight arm")


TESTS = [
    test_angle_axis_matches_rodrigues_properties,
    test_projection_uses_opencv_camera_axes,
    test_projection_right_stays_right,
    test_synchronized_depth_pins_camera_convention,
    test_camera_centres_lie_inside_the_scan,
    test_frames_have_poses_and_intrinsics,
    test_crops_are_context_padded_and_deterministic,
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
