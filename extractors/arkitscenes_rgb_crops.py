"""Real capture-RGB crops for a delivered instance, as a labeler image source.

Oracle-free. Reads only the capture stream that shipped with the scan --
`lowres_wide/` frames, `lowres_wide_intrinsics/` and `lowres_wide.traj` --
plus the instance's own vertex indices. No annotation, no box, no label.

WHY THIS EXISTS
---------------
`extractors/learned_labels.py` classifies each instance from three
point-splat renders of that instance ALONE. That is the same input pathology
that refuted four C1-P1 protocols in a row: a stippled render of isolated
geometry is not what any 2D model was trained on. SAM failed on it, and the
measured label errors have exactly the shape it predicts -- a sofa read as
"projector", two cushions read as "rug", a plane read as "counter". Those are
what texture-free splat fragments look like, not what sofas look like.

This module supplies the alternative input: the actual photographs the
device took, cropped around where the instance really appears, WITH its
surroundings. Everything downstream -- model, vocabulary, top-k, admission
threshold, evaluator -- is unchanged, so image source is the only variable.

FRAME CONVENTION, established empirically (see the module tests)
---------------------------------------------------------------
Entities live in the `scene_canonical` frame; the trajectory lives in the
original ARKit world frame. `adapters/arkitscenes.py` builds the canonical
mesh as ``xyz_canon = xyz_orig @ R.T``, so the inverse is ``xyz_orig =
xyz_canon @ R``.

A trajectory row is ``ts ax ay az tx ty tz`` where ``(ax,ay,az)`` is the
angle-axis of the WORLD->CAMERA rotation and ``(tx,ty,tz)`` its translation;
the camera centre is ``-R.T @ t``. Verified rather than assumed: those
centres fall inside the mesh bounds at handheld heights (0.06-1.34 m), and a
z-buffered colour reprojection reproduces the photograph.

ARKit's camera looks down -z with +y up, while the `.pincam` intrinsics are
pinhole/OpenCV (+z forward, +y down), hence the ``[1,-1,-1]`` flip. Getting
that wrong yields crops that look plausible and mean nothing, which is why
it is asserted in tests instead of trusted.

KNOWN LIMITATION, stated rather than hidden: no depth buffer is used, so an
instance occluded by furniture can still score as visible. Depth
(`lowres_depth`) was not downloaded. Visibility here means "projects into
frame and in front of the camera", not "unoccluded".
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# Frames are 60 Hz and the trajectory ~10 Hz, so most frames have no exact
# pose. A frame is usable only if a pose exists within this many seconds.
MAX_POSE_DT_S = 0.05
# Points sampled per instance when scoring frames. Scoring only needs a
# reliable visible-fraction, and the full vertex set is wasteful.
SCORE_SAMPLE = 400
# Fraction of the instance's projected extent added on each side. The whole
# hypothesis is that CONTEXT is what the splat renders lack, so the crop is
# deliberately not tight.
CONTEXT_PAD = 0.6
# A crop smaller than this on either side is discarded: upsampling a 12 px
# patch to CLIP's input tests nothing.
MIN_CROP_PX = 24
# Minimum instance points landing in frame for a view to count.
MIN_VISIBLE_POINTS = 12


def angle_axis_to_matrix(a: np.ndarray) -> np.ndarray:
    """Rodrigues, matching ARKitScenes' cv2.Rodrigues use."""
    theta = float(np.linalg.norm(a))
    if theta < 1e-12:
        return np.eye(3)
    k = np.asarray(a, dtype=np.float64) / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


@dataclass(frozen=True)
class Frame:
    """One usable capture frame: image path, pose and intrinsics."""
    timestamp: float
    png: Path
    R_wc: np.ndarray          # world -> camera rotation (ARKit axes)
    t_wc: np.ndarray          # world -> camera translation
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def load_trajectory(traj_path: Path) -> list[tuple[float, np.ndarray, np.ndarray]]:
    out = []
    for line in traj_path.read_text().splitlines():
        tok = line.split()
        if len(tok) != 7:
            continue
        out.append((float(tok[0]),
                    angle_axis_to_matrix(np.array([float(tok[1]), float(tok[2]),
                                                   float(tok[3])])),
                    np.array([float(tok[4]), float(tok[5]), float(tok[6])])))
    out.sort(key=lambda r: r[0])
    return out


def load_frames(scene_dir: Path, stride: int = 1) -> list[Frame]:
    """Frames that have a pose within MAX_POSE_DT_S, every `stride`-th.

    `stride` subsamples the 60 Hz stream; consecutive frames are nearly
    identical viewpoints, so scoring all of them buys nothing.
    """
    traj = load_trajectory(scene_dir / "lowres_wide.traj")
    if not traj:
        raise ValueError(f"no trajectory rows in {scene_dir}")
    ts = np.array([r[0] for r in traj])
    frames: list[Frame] = []
    pngs = sorted((scene_dir / "lowres_wide").glob("*.png"))[::stride]
    for png in pngs:
        try:
            stamp = float(png.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        j = int(np.argmin(np.abs(ts - stamp)))
        if abs(ts[j] - stamp) > MAX_POSE_DT_S:
            continue
        pin = scene_dir / "lowres_wide_intrinsics" / f"{png.stem}.pincam"
        if not pin.is_file():
            continue
        w, h, fx, fy, cx, cy = (float(v) for v in pin.read_text().split())
        frames.append(Frame(stamp, png, traj[j][1], traj[j][2],
                            fx, fy, cx, cy, int(w), int(h)))
    return frames


def project(points_world: np.ndarray, f: Frame) -> tuple[np.ndarray, np.ndarray]:
    """(uv [n,2], valid [n]) for original-frame world points."""
    cam = (f.R_wc @ points_world.T).T + f.t_wc
    cam = cam * np.array([1.0, -1.0, -1.0])       # ARKit -> OpenCV axes
    z = cam[:, 2]
    ok = z > 1e-6
    uv = np.full((len(points_world), 2), -1.0)
    uv[ok, 0] = f.fx * cam[ok, 0] / z[ok] + f.cx
    uv[ok, 1] = f.fy * cam[ok, 1] / z[ok] + f.cy
    inb = ok & (uv[:, 0] >= 0) & (uv[:, 0] < f.width) \
             & (uv[:, 1] >= 0) & (uv[:, 1] < f.height)
    return uv, inb


class RgbCropSource:
    """Best real-RGB crops for an instance, ranked by in-frame point count.

    `xyz_canonical` and `R` come from `tools.arkitscenes_eval
    .load_canonical_geometry`; conversion back to the capture frame happens
    once, here, so no caller has to remember the direction.
    """

    def __init__(self, scene_dir: Path, xyz_canonical: np.ndarray,
                 R: np.ndarray, *, stride: int = 6,
                 n_views: int = 3, context_pad: float = CONTEXT_PAD,
                 rng_seed: int = 0):
        self.scene_dir = Path(scene_dir)
        self.xyz_world = np.asarray(xyz_canonical, dtype=np.float64) @ np.asarray(R)
        self.frames = load_frames(self.scene_dir, stride=stride)
        if not self.frames:
            raise ValueError(f"no usable posed frames under {scene_dir}")
        self.n_views = int(n_views)
        self.context_pad = float(context_pad)
        self._rng = np.random.default_rng(rng_seed)

    def _sample(self, vertex_idx: np.ndarray) -> np.ndarray:
        if len(vertex_idx) <= SCORE_SAMPLE:
            return vertex_idx
        step = max(1, len(vertex_idx) // SCORE_SAMPLE)
        return vertex_idx[::step][:SCORE_SAMPLE]

    def score_frames(self, vertex_idx: np.ndarray) -> list[tuple[int, int]]:
        """[(frame_index, n_visible)] sorted best-first. Deterministic."""
        pts = self.xyz_world[self._sample(np.asarray(vertex_idx))]
        scored = []
        for i, f in enumerate(self.frames):
            _, inb = project(pts, f)
            n = int(inb.sum())
            if n >= MIN_VISIBLE_POINTS:
                scored.append((i, n))
        scored.sort(key=lambda r: (-r[1], r[0]))     # ties -> earliest frame
        return scored

    def crops_for(self, vertex_idx: np.ndarray) -> list[Image.Image]:
        """Up to `n_views` context crops, best view first. May be empty."""
        vertex_idx = np.asarray(vertex_idx)
        pts_all = self.xyz_world[self._sample(vertex_idx)]
        out: list[Image.Image] = []
        for fi, _n in self.score_frames(vertex_idx):
            f = self.frames[fi]
            uv, inb = project(pts_all, f)
            if not inb.any():
                continue
            u, v = uv[inb, 0], uv[inb, 1]
            u0, u1, v0, v1 = u.min(), u.max(), v.min(), v.max()
            pw, ph = (u1 - u0) * self.context_pad, (v1 - v0) * self.context_pad
            box = (max(0, int(math.floor(u0 - pw))),
                   max(0, int(math.floor(v0 - ph))),
                   min(f.width, int(math.ceil(u1 + pw))),
                   min(f.height, int(math.ceil(v1 + ph))))
            if box[2] - box[0] < MIN_CROP_PX or box[3] - box[1] < MIN_CROP_PX:
                continue
            out.append(Image.open(f.png).convert("RGB").crop(box))
            if len(out) >= self.n_views:
                break
        return out
