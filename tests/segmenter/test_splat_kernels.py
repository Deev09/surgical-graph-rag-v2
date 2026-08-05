"""Stage-0 gate W4 for the render splat-density protocol.

Protocol: `docs/arkitscenes_render_density_protocol.md`.

The load-bearing property is that the DEFAULT path is bit-for-bit what it
was before the kernel became a parameter. Everything frozen in this repo —
the Replica banks, the 40-view manifests, every recorded sha256 — depends on
that, so it is asserted directly rather than assumed from the code shape.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from segmenter.view_render import (
    SIZE, SPLAT_KERNELS, SPLAT_OFFSETS, SPLAT_OFFSETS_3X3, SPLAT_OFFSETS_5X5,
    Camera, render_view,
)


def _single_vertex_scene(offset_m=(0.0, 0.0, 0.0)):
    """One vertex straight ahead of a camera at the origin looking +x."""
    xyz = np.array([[3.0 + offset_m[0], offset_m[1], offset_m[2]]],
                   dtype=np.float64)
    rgb = np.array([[200, 100, 50]], dtype=np.uint8)
    cam = Camera(origin=(0.0, 0.0, 0.0), yaw_deg=0.0, pitch_deg=0.0)
    return xyz, rgb, cam


def test_kernel_shapes_are_square_and_centred() -> None:
    if len(SPLAT_OFFSETS_3X3) != 9:
        raise AssertionError(f"3x3 kernel has {len(SPLAT_OFFSETS_3X3)} px")
    if len(SPLAT_OFFSETS_5X5) != 25:
        raise AssertionError(f"5x5 kernel has {len(SPLAT_OFFSETS_5X5)} px")
    for k in (SPLAT_OFFSETS_3X3, SPLAT_OFFSETS_5X5):
        if (0, 0) not in k:
            raise AssertionError("kernel is not centred on the vertex")
        if len(set(k)) != len(k):
            raise AssertionError("kernel has duplicate offsets")
    if SPLAT_KERNELS["3x3"] is not SPLAT_OFFSETS_3X3:
        raise AssertionError("kernel registry disagrees with the constants")


def test_w4_one_vertex_paints_exactly_k_pixels() -> None:
    """W4: 3x3 paints exactly 9 id pixels, 5x5 exactly 25."""
    xyz, rgb, cam = _single_vertex_scene()
    for name, kernel, expect in (("3x3", SPLAT_OFFSETS_3X3, 9),
                                 ("5x5", SPLAT_OFFSETS_5X5, 25)):
        _img, ids = render_view(xyz, rgb, cam, far_m=100.0, rgb_offsets=kernel)
        n = int((ids >= 0).sum())
        if n != expect:
            raise AssertionError(
                f"{name}: painted {n} id pixels, expected {expect}")
        if set(np.unique(ids[ids >= 0]).tolist()) != {0}:
            raise AssertionError(f"{name}: unexpected vertex ids in buffer")


def test_w4_clipping_at_the_canvas_edge_does_not_wrap() -> None:
    """A vertex near the edge must clamp, never wrap to the far side."""
    xyz = np.array([[3.0, 2.99, 0.0]], dtype=np.float64)   # far to one side
    rgb = np.array([[10, 20, 30]], dtype=np.uint8)
    cam = Camera(origin=(0.0, 0.0, 0.0), yaw_deg=0.0, pitch_deg=0.0)
    _img, ids = render_view(xyz, rgb, cam, far_m=100.0,
                            rgb_offsets=SPLAT_OFFSETS_5X5)
    painted = np.argwhere(ids >= 0)
    if painted.size == 0:
        raise AssertionError("fixture painted nothing; it is not a control")
    cols = painted[:, 1]
    if cols.max() - cols.min() > 4:
        raise AssertionError(
            f"splat spans {cols.min()}..{cols.max()} — it wrapped instead of "
            "clamping")


def test_default_render_is_bit_identical_to_the_frozen_path() -> None:
    """The whole protocol rests on this: passing the default kernel
    explicitly, and passing nothing, must produce identical bytes."""
    rng = np.random.default_rng(0)
    xyz = rng.uniform(-2.0, 2.0, size=(4000, 3))
    xyz[:, 0] += 4.0
    rgb = rng.integers(0, 256, size=(4000, 3), dtype=np.uint8)
    cam = Camera(origin=(0.0, 0.0, 0.0), yaw_deg=15.0)

    a_img, a_ids = render_view(xyz, rgb, cam, far_m=50.0)
    b_img, b_ids = render_view(xyz, rgb, cam, far_m=50.0,
                               rgb_offsets=SPLAT_OFFSETS)
    c_img, c_ids = render_view(xyz, rgb, cam, far_m=50.0,
                               rgb_offsets=SPLAT_OFFSETS,
                               id_offsets=SPLAT_OFFSETS)
    for label, (img, ids) in (("explicit rgb_offsets", (b_img, b_ids)),
                              ("explicit both", (c_img, c_ids))):
        if not np.array_equal(a_img, img) or not np.array_equal(a_ids, ids):
            raise AssertionError(f"{label}: default path is not bit-identical")


def test_split_kernels_dilate_ids_without_touching_rgb() -> None:
    """Arm C's mechanism: id buffer grows, RGB stays byte-identical to the
    3x3 render, so a segmenter sees exactly the same image."""
    rng = np.random.default_rng(1)
    xyz = rng.uniform(-2.0, 2.0, size=(3000, 3))
    xyz[:, 0] += 4.0
    rgb = rng.integers(0, 256, size=(3000, 3), dtype=np.uint8)
    cam = Camera(origin=(0.0, 0.0, 0.0), yaw_deg=0.0)

    base_img, base_ids = render_view(xyz, rgb, cam, far_m=50.0)
    armc_img, armc_ids = render_view(xyz, rgb, cam, far_m=50.0,
                                     rgb_offsets=SPLAT_OFFSETS_3X3,
                                     id_offsets=SPLAT_OFFSETS_5X5)
    if not np.array_equal(base_img, armc_img):
        raise AssertionError(
            "arm C changed the RGB image; it must not — that is the whole "
            "point of the arm")
    n_base = int((base_ids >= 0).sum())
    n_armc = int((armc_ids >= 0).sum())
    if n_armc <= n_base:
        raise AssertionError(
            f"id buffer did not dilate: {n_base} -> {n_armc}")
    # every pixel the 3x3 buffer claimed must still be claimed
    covered = base_ids >= 0
    if not np.all(armc_ids[covered] >= 0):
        raise AssertionError("dilated buffer lost pixels the 3x3 buffer had")


def test_dilated_ids_stay_in_range() -> None:
    rng = np.random.default_rng(2)
    xyz = rng.uniform(-2.0, 2.0, size=(500, 3))
    xyz[:, 0] += 4.0
    rgb = rng.integers(0, 256, size=(500, 3), dtype=np.uint8)
    cam = Camera(origin=(0.0, 0.0, 0.0), yaw_deg=0.0)
    _img, ids = render_view(xyz, rgb, cam, far_m=50.0,
                            rgb_offsets=SPLAT_OFFSETS_3X3,
                            id_offsets=SPLAT_OFFSETS_5X5)
    seen = ids[ids >= 0]
    if seen.size and (seen.min() < 0 or seen.max() >= len(xyz)):
        raise AssertionError("dilated id buffer holds an out-of-range index")
    if ids.shape != (SIZE, SIZE):
        raise AssertionError(f"id buffer shape {ids.shape}")


TESTS = [
    test_kernel_shapes_are_square_and_centred,
    test_w4_one_vertex_paints_exactly_k_pixels,
    test_w4_clipping_at_the_canvas_edge_does_not_wrap,
    test_default_render_is_bit_identical_to_the_frozen_path,
    test_split_kernels_dilate_ids_without_touching_rgb,
    test_dilated_ids_stay_in_range,
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
