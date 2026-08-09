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

OCCLUSION
---------
Views are filtered by a MESH z-buffer: the reconstructed mesh is rasterised
from the same pose, and an instance point counts as visible only if nothing
else in the mesh sits measurably in front of it. Without this, an instance
behind a wall still projects into frame and can win view selection, so a
"failed" crop might simply show the far side of a wall.

What this is NOT: sensor depth. The mesh IS what Mask3D segmented, so this
faithfully reports "visible" for a geometrically wrong instance -- an
overmerged plane passes cleanly while still being the wrong object. It is
also blind to anything absent from the reconstruction (thin surfaces,
objects the scan missed, temporary occluders). It answers "should this be
visible given the reconstruction", which is enough to screen views, and it
is not evidence that reconstructed depth equals recorded depth. If crops
still select hidden targets, `lowres_depth` becomes justified.
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
# Every Nth mesh vertex forms the occlusion z-buffer. The buffer only has to
# decide "is another surface in front of this point", which does not need
# full density, and the full 1M-vertex mesh per candidate frame would
# dominate runtime.
ZBUF_STRIDE = 4
# A point is occluded when the z-buffer at its pixel is closer than the point
# by more than this, in metres. Loose enough to tolerate the gaps a sparse
# point rasterisation leaves, tight enough to catch a wall in between.
DEPTH_TOLERANCE_M = 0.12
# Frames re-checked with the z-buffer, taken from the cheap in-frame ranking.
# Rasterising all ~1900 frames per instance would be wasteful when only the
# best few can win.
VERIFY_TOP_K = 30
# Fraction of an instance's in-frame points that must survive occlusion.
MIN_VISIBLE_FRACTION = 0.25
# How far outside the target mask is dimmed in the marked arm: 1.0 keeps the
# surroundings untouched, 0.0 blacks them out.
CONTEXT_DIM = 0.45


def angle_axis_to_matrix(a: np.ndarray) -> np.ndarray:
    """Rodrigues, matching ARKitScenes' cv2.Rodrigues use."""
    theta = float(np.linalg.norm(a))
    if theta < 1e-12:
        return np.eye(3)
    k = np.asarray(a, dtype=np.float64) / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def _grow_to_min(box, width: int, height: int):
    """Expand a box about its centre to at least MIN_CROP_PX, inside bounds."""
    u0, v0, u1, v1 = box
    if width < MIN_CROP_PX or height < MIN_CROP_PX:
        return None
    if u1 - u0 < MIN_CROP_PX:
        c = (u0 + u1) / 2.0
        u0 = int(max(0, min(width - MIN_CROP_PX, round(c - MIN_CROP_PX / 2))))
        u1 = u0 + MIN_CROP_PX
    if v1 - v0 < MIN_CROP_PX:
        c = (v0 + v1) / 2.0
        v0 = int(max(0, min(height - MIN_CROP_PX, round(c - MIN_CROP_PX / 2))))
        v1 = v0 + MIN_CROP_PX
    return (u0, v0, u1, v1)


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
                 occlusion: bool = True, mark_target: bool = False,
                 rng_seed: int = 0):
        self.scene_dir = Path(scene_dir)
        self.xyz_world = np.asarray(xyz_canonical, dtype=np.float64) @ np.asarray(R)
        self.frames = load_frames(self.scene_dir, stride=stride)
        if not self.frames:
            raise ValueError(f"no usable posed frames under {scene_dir}")
        self.n_views = int(n_views)
        self.context_pad = float(context_pad)
        self.occlusion = bool(occlusion)
        self.mark_target = bool(mark_target)
        self._rng = np.random.default_rng(rng_seed)
        # z-buffers are per FRAME, not per instance, and instances compete for
        # the same good viewpoints, so caching across instances pays for
        # itself immediately.
        self._zbuf: dict[int, np.ndarray] = {}
        self._zsrc = self.xyz_world[::ZBUF_STRIDE]

    def zbuffer(self, frame_index: int) -> np.ndarray:
        """Nearest mesh depth per pixel; +inf where the mesh does not cover."""
        cached = self._zbuf.get(frame_index)
        if cached is not None:
            return cached
        f = self.frames[frame_index]
        cam = (f.R_wc @ self._zsrc.T).T + f.t_wc
        cam = cam * np.array([1.0, -1.0, -1.0])
        z = cam[:, 2]
        ok = z > 1e-6
        u = (f.fx * cam[ok, 0] / z[ok] + f.cx).astype(np.int32)
        v = (f.fy * cam[ok, 1] / z[ok] + f.cy).astype(np.int32)
        d = z[ok]
        m = (u >= 0) & (u < f.width) & (v >= 0) & (v < f.height)
        buf = np.full((f.height, f.width), np.inf)
        # np.minimum.at is the correct reduction for colliding indices; a
        # plain assignment would keep whichever point happened to be last.
        np.minimum.at(buf, (v[m], u[m]), d[m])
        self._zbuf[frame_index] = buf
        return buf

    def visible_counts(self, points_world: np.ndarray,
                       frame_index: int) -> tuple[int, int]:
        """(n_in_frame, n_unoccluded) for one frame."""
        f = self.frames[frame_index]
        uv, inb = project(points_world, f)
        n_in = int(inb.sum())
        if n_in == 0:
            return 0, 0
        if not self.occlusion:
            return n_in, n_in
        cam = (f.R_wc @ points_world.T).T + f.t_wc
        depth = (cam * np.array([1.0, -1.0, -1.0]))[:, 2]
        buf = self.zbuffer(frame_index)
        u = uv[inb, 0].astype(np.int32)
        v = uv[inb, 1].astype(np.int32)
        front = buf[v, u]
        # visible when nothing sits measurably in front of the point
        return n_in, int((depth[inb] <= front + DEPTH_TOLERANCE_M).sum())

    def _sample(self, vertex_idx: np.ndarray) -> np.ndarray:
        if len(vertex_idx) <= SCORE_SAMPLE:
            return vertex_idx
        step = max(1, len(vertex_idx) // SCORE_SAMPLE)
        return vertex_idx[::step][:SCORE_SAMPLE]

    def score_frames(self, vertex_idx: np.ndarray) -> list[tuple[int, int]]:
        """[(frame_index, n_in_frame)] best-first, BEFORE occlusion.

        Cheap pre-ranking only: rasterising a z-buffer for every one of ~1900
        frames per instance would dominate runtime when only a handful can
        win. Occlusion is applied to the top of this list by `ranked_views`.
        """
        pts = self.xyz_world[self._sample(np.asarray(vertex_idx))]
        scored = []
        for i, f in enumerate(self.frames):
            _, inb = project(pts, f)
            n = int(inb.sum())
            if n >= MIN_VISIBLE_POINTS:
                scored.append((i, n))
        scored.sort(key=lambda r: (-r[1], r[0]))     # ties -> earliest frame
        return scored

    def ranked_views(self, vertex_idx: np.ndarray) -> list[tuple[int, int, float]]:
        """[(frame_index, n_unoccluded, visible_fraction)] best-first."""
        pts = self.xyz_world[self._sample(np.asarray(vertex_idx))]
        out = []
        for fi, _n in self.score_frames(vertex_idx)[:VERIFY_TOP_K]:
            n_in, n_vis = self.visible_counts(pts, fi)
            frac = (n_vis / n_in) if n_in else 0.0
            if n_vis >= MIN_VISIBLE_POINTS and frac >= MIN_VISIBLE_FRACTION:
                out.append((fi, n_vis, frac))
        out.sort(key=lambda r: (-r[1], r[0]))
        return out

    def _mark(self, img: Image.Image, box, uv_vis: np.ndarray) -> Image.Image:
        """Dim everything outside the target's projected pixels.

        Padding alone cannot say WHICH object is being asked about: a wide
        crop of a doorway reads as a kitchen. This keeps the surroundings
        legible while making the subject unambiguous.
        """
        arr = np.asarray(img, dtype=np.float32)
        keep = np.zeros(arr.shape[:2], dtype=bool)
        u = np.clip(uv_vis[:, 0].astype(int) - box[0], 0, arr.shape[1] - 1)
        v = np.clip(uv_vis[:, 1].astype(int) - box[1], 0, arr.shape[0] - 1)
        keep[v, u] = True
        # dilate so a sparse point set reads as a region, not confetti
        for _ in range(3):
            k = keep.copy()
            k[1:, :] |= keep[:-1, :]; k[:-1, :] |= keep[1:, :]
            k[:, 1:] |= keep[:, :-1]; k[:, :-1] |= keep[:, 1:]
            keep = k
        arr[~keep] *= CONTEXT_DIM
        return Image.fromarray(arr.clip(0, 255).astype(np.uint8))

    def crop_boxes(self, vertex_idx: np.ndarray) -> list[tuple[int, tuple, np.ndarray]]:
        """[(frame_index, box, uv_in_frame)] that survive EVERY filter.

        `coverage` and `crops_for` both go through this, so a view can never
        be counted as available and then rejected at crop time -- which is
        exactly what happened when they applied different filters.
        """
        vertex_idx = np.asarray(vertex_idx)
        # FULL vertex set here, not the scoring sample. The sample is fine for
        # ranking views, but using it for the crop box and the target mask
        # produced a scatter of ~400 speckles instead of a filled region --
        # the marked arm was then testing "dimmed photo with confetti on it",
        # which is not the experiment.
        pts_all = self.xyz_world[vertex_idx]
        out = []
        for fi, _n, _frac in self.ranked_views(vertex_idx):
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
            # A small target gets its box GROWN to the minimum rather than
            # discarded. Discarding silently removed whole instances from the
            # RGB arms, which would have made the comparison unpaired -- and
            # a distant object is exactly the case the experiment needs to
            # keep, not drop.
            box = _grow_to_min(box, f.width, f.height)
            if box is None:
                continue
            out.append((fi, box, uv[inb]))
            if len(out) >= self.n_views:
                break
        return out

    def crops_for(self, vertex_idx: np.ndarray) -> list[Image.Image]:
        """Up to `n_views` crops, best view first. May be empty."""
        images = []
        for fi, box, uv_vis in self.crop_boxes(vertex_idx):
            img = Image.open(self.frames[fi].png).convert("RGB").crop(box)
            if self.mark_target:
                img = self._mark(img, box, uv_vis)
            images.append(img)
        return images

    def coverage(self, vertex_idx: np.ndarray) -> dict:
        """Per-instance CROP availability -- what labeling will actually get."""
        views = self.ranked_views(vertex_idx)
        boxes = self.crop_boxes(vertex_idx)
        return {
            "n_views_available": len(views),
            "n_crops": len(boxes),
            "has_full_views": len(boxes) >= self.n_views,
            "has_any_crop": bool(boxes),
            "best_visible_points": int(views[0][1]) if views else 0,
            "best_visible_fraction": round(float(views[0][2]), 4) if views else 0.0,
            "smallest_crop_px": (min(min(b[2] - b[0], b[3] - b[1]) for _, b, _ in boxes)
                                 if boxes else 0),
        }
