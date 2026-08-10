"""Render ARKitScenes mesh/RGB alignment without annotation access.

This is a diagnostic, not an evaluator.  It projects the captured coloured
mesh into exact-timestamp RGB frames using the dataset's trajectory and
per-frame intrinsics, then writes source/render/overlay triptychs.  A wrong
camera convention can still produce plausible crops; these overlays make the
camera contract directly inspectable before any learned label is computed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.arkitscenes import MESH_SUFFIX, read_mesh
from extractors.arkitscenes_rgb_crops import Frame, load_frames


def camera_points(points_world: np.ndarray, frame: Frame) -> np.ndarray:
    """World points in ARKitScenes' documented OpenCV camera frame."""
    return (frame.R_wc @ np.asarray(points_world, dtype=np.float64).T).T + frame.t_wc


def project_opencv(
        points_world: np.ndarray, frame: Frame,
        *, legacy_axis_flip: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(uv, depth)``; invalid rows have depth <= 0.

    ``legacy_axis_flip`` exists only to reproduce the invalidated crop arm.
    The official loader applies no such flip.
    """
    cam = camera_points(points_world, frame)
    if legacy_axis_flip:
        cam = cam * np.array([1.0, -1.0, -1.0])
    depth = cam[:, 2]
    uv = np.full((len(cam), 2), np.nan, dtype=np.float64)
    front = depth > 1e-6
    uv[front, 0] = frame.fx * cam[front, 0] / depth[front] + frame.cx
    uv[front, 1] = frame.fy * cam[front, 1] / depth[front] + frame.cy
    return uv, depth


def render_vertices(
        xyz: np.ndarray, rgb: np.ndarray, frame: Frame,
        *, legacy_axis_flip: bool = False, radius_px: int = 1,
        stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-point RGB rendering and coverage mask at the frame size."""
    xyz = xyz[::stride]
    rgb = rgb[::stride]
    uv, depth = project_opencv(xyz, frame, legacy_axis_flip=legacy_axis_flip)
    u = np.zeros(len(uv), dtype=np.int64)
    v = np.zeros(len(uv), dtype=np.int64)
    finite = np.isfinite(uv).all(axis=1)
    u[finite] = np.rint(uv[finite, 0]).astype(np.int64)
    v[finite] = np.rint(uv[finite, 1]).astype(np.int64)
    valid = (
        (depth > 1e-6)
        & finite
        & (u >= -radius_px) & (u < frame.width + radius_px)
        & (v >= -radius_px) & (v < frame.height + radius_px)
    )
    # Painter's algorithm, far to near.  Near samples overwrite far samples.
    order = np.flatnonzero(valid)[np.argsort(depth[valid])[::-1]]
    canvas = np.zeros((frame.height, frame.width, 3), dtype=np.uint8)
    covered = np.zeros((frame.height, frame.width), dtype=bool)
    for dy in range(-radius_px, radius_px + 1):
        vv = v[order] + dy
        for dx in range(-radius_px, radius_px + 1):
            uu = u[order] + dx
            inside = (uu >= 0) & (uu < frame.width) & (vv >= 0) & (vv < frame.height)
            canvas[vv[inside], uu[inside]] = rgb[order[inside]]
            covered[vv[inside], uu[inside]] = True
    return canvas, covered


def render_depth(
        xyz: np.ndarray, frame: Frame, *, legacy_axis_flip: bool = False,
        radius_px: int = 1, stride: int = 1) -> np.ndarray:
    """Nearest mesh-vertex depth per pixel in metres."""
    uv, depth = project_opencv(
        xyz[::stride], frame, legacy_axis_flip=legacy_axis_flip)
    finite = np.isfinite(uv).all(axis=1)
    u = np.zeros(len(uv), dtype=np.int64)
    v = np.zeros(len(uv), dtype=np.int64)
    u[finite] = np.rint(uv[finite, 0]).astype(np.int64)
    v[finite] = np.rint(uv[finite, 1]).astype(np.int64)
    base = depth > 1e-6
    out = np.full((frame.height, frame.width), np.inf, dtype=np.float64)
    for dy in range(-radius_px, radius_px + 1):
        vv = v + dy
        for dx in range(-radius_px, radius_px + 1):
            uu = u + dx
            inside = (
                base & finite
                & (uu >= 0) & (uu < frame.width)
                & (vv >= 0) & (vv < frame.height)
            )
            np.minimum.at(out, (vv[inside], uu[inside]), depth[inside])
    return out


def depth_alignment_metrics(
        xyz: np.ndarray, frame: Frame, sensor_depth_mm: np.ndarray,
        *, legacy_axis_flip: bool = False) -> dict:
    """Independent mesh/pose check against synchronized sensor depth."""
    sensor = np.asarray(sensor_depth_mm, dtype=np.float64) / 1000.0
    if sensor.shape != (frame.height, frame.width):
        raise ValueError(
            f"depth shape {sensor.shape} does not match frame "
            f"{(frame.height, frame.width)}")
    mesh = render_depth(
        xyz, frame, legacy_axis_flip=legacy_axis_flip, radius_px=1)
    valid_sensor = sensor > 0
    common = valid_sensor & np.isfinite(mesh)
    if not np.any(common):
        raise ValueError(f"{frame.png}: no common sensor/mesh depth pixels")
    residual = mesh[common] - sensor[common]
    absolute = np.abs(residual)
    return {
        "timestamp": frame.timestamp,
        "n_sensor_valid": int(valid_sensor.sum()),
        "n_common": int(common.sum()),
        "common_share_of_sensor": float(common.sum() / valid_sensor.sum()),
        "median_signed_error_m": float(np.median(residual)),
        "median_abs_error_m": float(np.median(absolute)),
        "p90_abs_error_m": float(np.quantile(absolute, 0.9)),
        "share_abs_error_le_0_10m": float(np.mean(absolute <= 0.10)),
        "share_abs_error_le_0_20m": float(np.mean(absolute <= 0.20)),
    }


def alignment_triptych(
        xyz: np.ndarray, rgb: np.ndarray, frame: Frame,
        *, legacy_axis_flip: bool = False) -> Image.Image:
    source = np.asarray(Image.open(frame.png).convert("RGB"), dtype=np.uint8)
    if source.shape[:2] != (frame.height, frame.width):
        raise ValueError(
            f"{frame.png}: image shape {source.shape[:2]} does not match "
            f"intrinsics {(frame.height, frame.width)}")
    rendered, covered = render_vertices(
        xyz, rgb, frame, legacy_axis_flip=legacy_axis_flip)
    overlay = source.copy()
    overlay[covered] = np.rint(
        0.45 * source[covered].astype(np.float64)
        + 0.55 * rendered[covered].astype(np.float64)
    ).astype(np.uint8)
    return Image.fromarray(np.concatenate([source, rendered, overlay], axis=1))


def _nearest_frame(frames: list[Frame], timestamp: float) -> Frame:
    return min(frames, key=lambda frame: abs(frame.timestamp - timestamp))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--timestamp", type=float, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--legacy-axis-flip", action="store_true")
    parser.add_argument("--depth-dir", type=Path)
    args = parser.parse_args(argv)

    scene_dir = args.scene_dir
    video_id = scene_dir.name
    mesh = read_mesh(scene_dir / f"{video_id}{MESH_SUFFIX}")
    frames = load_frames(scene_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    metrics = []
    for requested in args.timestamp:
        frame = _nearest_frame(frames, requested)
        if abs(frame.timestamp - requested) > 0.001:
            raise ValueError(
                f"no exact matched frame near {requested}; nearest={frame.timestamp}")
        image = alignment_triptych(
            mesh.xyz, mesh.rgb, frame,
            legacy_axis_flip=args.legacy_axis_flip)
        suffix = "legacy_flip" if args.legacy_axis_flip else "opencv"
        image.save(args.out / f"{video_id}_{frame.timestamp:.3f}_{suffix}.png")
        if args.depth_dir is not None:
            depth_path = args.depth_dir / frame.png.name
            if not depth_path.is_file():
                raise FileNotFoundError(depth_path)
            depth = np.asarray(Image.open(depth_path), dtype=np.uint16)
            metrics.append(depth_alignment_metrics(
                mesh.xyz, frame, depth,
                legacy_axis_flip=args.legacy_axis_flip))
    if metrics:
        (args.out / "depth_alignment.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
