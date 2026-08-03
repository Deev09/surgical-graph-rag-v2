"""Geometry-only up-axis, floor plane, and scene-scale estimation.

DIAGNOSTIC MODULE — additive, read-only with respect to the pipeline.
Nothing under extractors/, graph/, or reasoner/ imports this. It exists to
answer three questions from mesh geometry ALONE, with no dataset metadata,
no semantic labels, and no Replica-specific assumption:

    1. which unsigned axis is vertical (the gravity axis),
    2. which way along that axis is up, and where the floor plane sits,
    3. how big this scene is (room diagonal, floor extent, storey height).

Why this exists: every relation extractor in graph/relations/ hardcodes
`frame="world"` and compares against absolute-metre thresholds (0.3 m axis
dominance, 2.5 m sparse distance cap, 0.02 m contact band, ...). Those two
facts are only safe if (a) the world frame really is gravity-aligned and
(b) every scene really is the same size. This module measures both instead
of assuming them. It does NOT change any threshold and does not feed any
extractor; tools/frame_scale_audit.py reports what it measures.

Method (all steps are deterministic; no RNG, no RANSAC sampling):

  Axis.   Accumulate area-weighted triangle normals. Score each of K
          Fibonacci-sampled hemisphere directions d by the total mesh area
          whose unit normal lies within `cone_deg` of the ±d axis. The
          gravity axis is the peak: floor + ceiling + every tabletop are
          mutually parallel and jointly outweigh any single wall. Runner-up
          peaks are reported with their scores so the margin — i.e. how
          much this assumption is actually load-bearing — is visible rather
          than implied. The winner is then refined to sub-degree precision
          by the top eigenvector of the area-weighted normal scatter
          matrix over the faces inside the winning cone.

  Sign.   An axis has no sign; gravity does. Three independent physical
          cues vote, and all three are reported with their raw areas:
            (A) horizontal-area asymmetry — more up-facing horizontal area
                (floor + tabletops) than down-facing (ceiling + undersides).
                This textbook cue is WRONG on 4 of the 6 Replica scenes:
                furniture occludes the floor while the ceiling scans clean.
            (B) clutter asymmetry — non-horizontal area is denser in the
                band just above the floor than just below the ceiling.
                Right on 6/6.
            (C) interior horizontal mass — tabletops and seats sit in the
                lower half. Right on 6/6.
          Majority decides. Per-cue votes and agreement are in the
          diagnostics; a 2-1 split is visible, never hidden.

  Floor.  Bin the up-facing horizontal faces by their offset along the up
          axis (1 cm bins, area-weighted). The floor is the largest bin in
          the lowest `plane_search_frac` of the vertical extent; the
          ceiling is the first strong down-facing bin at least
          `min_storey_height_m` above the floor. Each is then least-squares
          refit over the triangle vertices within `plane_band_m` of the
          peak, which yields a plane normal independent of the axis
          estimate (their disagreement is itself a reported number). The
          full peak list is returned, so a multi-storey scene shows up as
          multiple peaks instead of a silently wrong floor.

  Scale.  A canonical frame is built as (e1, e2, up) where e1 is the
          dominant horizontal wall-normal direction (90-degree-symmetric
          circular mean of vertical-face azimuths, area-weighted). Extents
          are reported both raw (min/max) and robust (1st/99th percentile),
          because a single stray scanned triangle moves a raw AABB by
          metres. Downstream scale ratios use the robust extent.

Units are whatever the mesh is in; Replica is metres. Nothing here converts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any

import numpy as np

from common.types import Vec3
# Header parsing is shared with the C3.0-S estimator so there is one PLY
# header grammar in the repo. The face-body reader below is separate: the
# Habitat instance meshes carry a per-face `object_id` scalar alongside the
# vertex-index list, a record shape load_raw_triangle_mesh does not accept.
from geometry.mesh_surfaces import (
    _PLY_SCALAR_DTYPES,
    _elements_by_name,
    _read_ply_header,
    load_raw_triangle_mesh,
)


@dataclass(frozen=True)
class FrameEstimatorConfig:
    """Every knob is an angle, a percentile, or a fraction of the scene's
    own measured extent. The only absolute-length constants are the
    histogram bin (1 cm) and the plane-refit band (5 cm), which are
    sensor-resolution quantities, not room-scale quantities."""
    n_candidate_dirs: int = 512      # Fibonacci hemisphere samples
    cone_deg: float = 10.0           # "parallel to the axis" tolerance
    nms_deg: float = 20.0            # runner-up separation for reporting
    refine_iters: int = 3            # eigen refinements of the winning axis
    vertical_deg: float = 10.0       # "perpendicular to the axis" tolerance
    offset_bin_m: float = 0.01       # floor/ceiling offset histogram bin
    plane_band_m: float = 0.05       # refit band around a peak
    plane_search_frac: float = 0.25  # floor sought in lowest 25% of extent
    clutter_band_frac: float = 0.35  # sign cue B band, capped below
    clutter_band_max_m: float = 1.0
    interior_exclude_frac: float = 0.05   # sign cue C: exclude the two end planes
    interior_exclude_min_m: float = 0.20
    # The single human-scale absolute-length prior in this module: a storey
    # is taller than this. Used only to pick the FIRST ceiling above the
    # floor in a multi-level scene, so a two-storey capture does not report
    # a 5 m "room height". Value taken from the existing repo precedent,
    # MeshSurfaceConfig.floor_ceiling_separation_min_m. At 1.5 a soffit
    # 1.51 m above frl_apartment_0's floor is mistaken for its ceiling.
    min_storey_height_m: float = 1.8
    peak_min_area_frac: float = 0.05      # strong-peak cut, share of tallest bin
    peak_separation_m: float = 0.15
    ceiling_peak_min_area_frac: float = 0.25
    extent_percentile: float = 1.0   # robust extent = p1..p99
    n_reported_peaks: int = 6
    chunk_faces: int = 200_000

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


DEFAULT_CONFIG = FrameEstimatorConfig()


@dataclass(frozen=True)
class AxisCandidate:
    """One peak of the area-weighted normal histogram. `area_frac` is the
    share of total mesh area whose normal is parallel to this axis."""
    direction: Vec3
    area_m2: float
    area_frac: float
    angle_to_winner_deg: float


@dataclass(frozen=True)
class PlaneEstimate:
    """A refit horizontal plane. `offset_m` is the signed height along the
    estimated up axis; `normal` is the independently refit plane normal, so
    `tilt_to_up_deg` measures axis/plane self-consistency."""
    normal: Vec3
    offset_m: float
    d: float
    mesh_area_m2: float
    projected_area_m2: float
    rms_m: float
    n_faces: int
    tilt_to_up_deg: float


@dataclass(frozen=True)
class SceneScale:
    """All lengths in the canonical (e1, e2, up) frame."""
    extent_raw_m: Vec3
    extent_robust_m: Vec3
    room_diagonal_m: float           # 3D diagonal of the robust extent
    floor_diagonal_m: float          # horizontal diagonal of the robust extent
    floor_area_m2: float | None      # projected area of the refit floor
    storey_height_m: float | None    # ceiling offset - floor offset
    vertical_extent_m: float         # robust extent along up


@dataclass(frozen=True)
class FrameEstimate:
    up_axis: Vec3
    rotation: tuple[Vec3, Vec3, Vec3]   # rows e1, e2, up; world -> canonical
    yaw_deg: float
    axis_candidates: list[AxisCandidate]
    floor: PlaneEstimate | None
    ceiling: PlaneEstimate | None
    scale: SceneScale
    horizontal_peaks: list[tuple[float, float]]  # (offset_m, area_m2)
    diagnostics: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# mesh loading
# --------------------------------------------------------------------------

def _scalar_size(type_name: str) -> int:
    if type_name not in _PLY_SCALAR_DTYPES:
        raise ValueError(f"unsupported PLY scalar type {type_name!r}")
    return np.dtype(_PLY_SCALAR_DTYPES[type_name]).itemsize


def _read_binary_vertices(f, vertex: dict) -> np.ndarray:
    fields = []
    for p in vertex["properties"]:
        if p["kind"] != "scalar":
            raise ValueError("vertex element must be all scalar properties")
        fields.append((p["name"], _PLY_SCALAR_DTYPES[p["type"]]))
    vdtype = np.dtype(fields, align=False)
    n_v = int(vertex["count"])
    raw = f.read(n_v * vdtype.itemsize)
    if len(raw) != n_v * vdtype.itemsize:
        raise ValueError("binary PLY ended inside vertex data")
    verts = np.frombuffer(raw, dtype=vdtype, count=n_v)
    return np.column_stack([verts["x"], verts["y"], verts["z"]]).astype(np.float64)


def _read_binary_faces(f, face: dict) -> tuple[np.ndarray, int]:
    """Fixed-arity binary face reader that tolerates extra per-face scalar
    properties (Replica's Habitat instance meshes carry `object_id` after
    the vertex-index list). Quads are split on the v0-v2 diagonal, the rule
    already frozen for Replica quads elsewhere in this repo."""
    props = face["properties"]
    list_idx = [i for i, p in enumerate(props) if p["kind"] == "list"][0]
    lp = props[list_idx]
    before = sum(_scalar_size(p["type"]) for p in props[:list_idx])
    after = sum(_scalar_size(p["type"]) for p in props[list_idx + 1:])
    count_size = _scalar_size(lp["count_type"])
    item_dtype = np.dtype(_PLY_SCALAR_DTYPES[lp["item_type"]])
    n_f = int(face["count"])
    raw = f.read()
    if n_f == 0:
        return np.zeros((0, 3), dtype=np.int64), 0
    if len(raw) % n_f:
        raise ValueError("binary PLY face block is not a fixed-size record "
                         "(mixed face arity is out of scope for frame.py)")
    per_face = len(raw) // n_f
    payload = per_face - before - count_size - after
    if payload <= 0 or payload % item_dtype.itemsize:
        raise ValueError(f"cannot resolve face arity from {per_face} bytes/face")
    k = payload // item_dtype.itemsize
    if k not in (3, 4):
        raise ValueError(f"frame.py accepts face arity 3 or 4 only (got {k})")
    fields: list[tuple[str, str]] = []
    for j, p in enumerate(props[:list_idx]):
        fields.append((f"pre{j}", _PLY_SCALAR_DTYPES[p["type"]]))
    fields.append(("n", _PLY_SCALAR_DTYPES[lp["count_type"]]))
    for j in range(k):
        fields.append((f"v{j}", item_dtype.str))
    for j, p in enumerate(props[list_idx + 1:]):
        fields.append((f"post{j}", _PLY_SCALAR_DTYPES[p["type"]]))
    rec = np.frombuffer(raw, dtype=np.dtype(fields, align=False), count=n_f)
    if not np.all(rec["n"] == k):
        raise ValueError("binary PLY face arity is not uniform")
    idx = np.column_stack([rec[f"v{j}"] for j in range(k)]).astype(np.int64)
    if k == 3:
        return idx, 0
    tris = np.empty((2 * n_f, 3), dtype=np.int64)
    tris[0::2] = idx[:, (0, 1, 2)]
    tris[1::2] = idx[:, (0, 2, 3)]
    return tris, n_f


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return (xyz float64 [V,3], triangles int64 [F,3], provenance).

    Binary little-endian PLYs are read here (including the Habitat instance
    meshes with a trailing per-face `object_id`). Anything else is delegated
    to geometry.mesh_surfaces.load_raw_triangle_mesh."""
    path = Path(path)
    with open(path, "rb") as f:
        fmt, elements = _read_ply_header(f)
        vertex, face = _elements_by_name(elements)
        if fmt != "binary_little_endian":
            mesh = load_raw_triangle_mesh(path)
            return mesh.xyz, mesh.faces, {
                "loader": "geometry.mesh_surfaces.load_raw_triangle_mesh",
                "ply_format": fmt,
                "n_source_quads": int(mesh.n_source_quads),
            }
        xyz = _read_binary_vertices(f, vertex)
        faces, n_quads = _read_binary_faces(f, face)
    if not np.isfinite(xyz).all():
        raise ValueError("mesh contains non-finite vertices")
    if faces.size and (faces.min() < 0 or faces.max() >= len(xyz)):
        raise ValueError("mesh face index out of range")
    return xyz, faces, {
        "loader": "geometry.frame.load_mesh",
        "ply_format": fmt,
        "n_source_quads": int(n_quads),
    }


# --------------------------------------------------------------------------
# face geometry
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FaceGeometry:
    normal: np.ndarray    # float32 [F,3], unit
    area: np.ndarray      # float32 [F]
    centroid: np.ndarray  # float32 [F,3]
    n_degenerate: int


def face_geometry(xyz: np.ndarray, faces: np.ndarray,
                  cfg: FrameEstimatorConfig = DEFAULT_CONFIG) -> FaceGeometry:
    """Unit normal, area, and centroid per triangle. Degenerate triangles
    (zero area) get a zero normal and zero area, so they contribute nothing
    to any area-weighted statistic."""
    n_f = len(faces)
    normal = np.zeros((n_f, 3), dtype=np.float32)
    area = np.zeros(n_f, dtype=np.float32)
    centroid = np.zeros((n_f, 3), dtype=np.float32)
    n_degenerate = 0
    for lo in range(0, n_f, cfg.chunk_faces):
        hi = min(lo + cfg.chunk_faces, n_f)
        tri = xyz[faces[lo:hi]]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        norm = np.linalg.norm(cross, axis=1)
        good = norm > 0.0
        n_degenerate += int(np.count_nonzero(~good))
        a = 0.5 * norm
        area[lo:hi] = a.astype(np.float32)
        unit = np.zeros_like(cross)
        unit[good] = cross[good] / norm[good, None]
        normal[lo:hi] = unit.astype(np.float32)
        centroid[lo:hi] = (tri.mean(axis=1)).astype(np.float32)
    return FaceGeometry(normal=normal, area=area, centroid=centroid,
                        n_degenerate=n_degenerate)


# --------------------------------------------------------------------------
# axis estimation
# --------------------------------------------------------------------------

def hemisphere_directions(n: int) -> np.ndarray:
    """Area-uniform Fibonacci sampling of the upper unit hemisphere. Used as
    unsigned axes (scored via |n . d|), so a hemisphere covers all axes."""
    i = np.arange(n, dtype=np.float64) + 0.5
    z = i / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = math.pi * (1.0 + math.sqrt(5.0)) * i
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _axis_scores(fg: FaceGeometry, dirs: np.ndarray, cos_cone: float,
                 cfg: FrameEstimatorConfig) -> np.ndarray:
    """scores[k] = mesh area whose unit normal is within cone_deg of ±dirs[k].

    Chunked so the dense [chunk, K] projection stays around 64 MB whatever
    the mesh size; apartment_0 is 9.1 M triangles."""
    d32 = np.ascontiguousarray(dirs.T, dtype=np.float32)
    scores = np.zeros(len(dirs), dtype=np.float64)
    chunk = max(4096, 16_000_000 // max(1, len(dirs)))
    for lo in range(0, len(fg.area), chunk):
        hi = min(lo + chunk, len(fg.area))
        proj = np.abs(fg.normal[lo:hi] @ d32)
        mask = (proj >= np.float32(cos_cone)).astype(np.float32)
        scores += fg.area[lo:hi] @ mask
    return scores


def _refine_axis(fg: FaceGeometry, axis: np.ndarray, cos_cone: float,
                 iters: int) -> tuple[np.ndarray, int]:
    """Top eigenvector of the area-weighted normal scatter matrix restricted
    to faces inside the cone. n n^T is sign-invariant, so this refines the
    unsigned axis without any orientation assumption."""
    u = axis / np.linalg.norm(axis)
    n_sel = 0
    for _ in range(max(1, iters)):
        proj = np.abs(fg.normal @ u.astype(np.float32))
        sel = proj >= np.float32(cos_cone)
        n_sel = int(np.count_nonzero(sel))
        if n_sel < 3:
            break
        n = fg.normal[sel].astype(np.float64)
        w = fg.area[sel].astype(np.float64)
        m = (n * w[:, None]).T @ n
        eigvals, eigvecs = np.linalg.eigh(m)
        nxt = eigvecs[:, int(np.argmax(eigvals))]
        nxt = nxt / np.linalg.norm(nxt)
        if float(np.dot(nxt, u)) < 0:
            nxt = -nxt
        if float(np.linalg.norm(nxt - u)) < 1e-12:
            u = nxt
            break
        u = nxt
    return u, n_sel


def _axis_candidates(fg: FaceGeometry, cfg: FrameEstimatorConfig,
                     total_area: float) -> tuple[np.ndarray, list[AxisCandidate]]:
    dirs = hemisphere_directions(cfg.n_candidate_dirs)
    cos_cone = math.cos(math.radians(cfg.cone_deg))
    scores = _axis_scores(fg, dirs, cos_cone, cfg)
    order = np.argsort(-scores)
    cos_nms = math.cos(math.radians(cfg.nms_deg))
    picked: list[int] = []
    for k in order:
        if scores[k] <= 0.0:
            break
        if any(abs(float(np.dot(dirs[k], dirs[j]))) >= cos_nms for j in picked):
            continue
        picked.append(int(k))
        if len(picked) >= 3:
            break
    if not picked:
        raise ValueError("no oriented mesh area: cannot estimate a frame")
    winner, _ = _refine_axis(fg, dirs[picked[0]], cos_cone, cfg.refine_iters)
    refined_score = float(_axis_scores(fg, winner[None, :], cos_cone, cfg)[0])
    cands = [AxisCandidate(
        direction=tuple(float(v) for v in winner),
        area_m2=refined_score,
        area_frac=refined_score / total_area if total_area else 0.0,
        angle_to_winner_deg=0.0,
    )]
    for k in picked[1:]:
        dot = min(1.0, abs(float(np.dot(dirs[k], winner))))
        cands.append(AxisCandidate(
            direction=tuple(float(v) for v in dirs[k]),
            area_m2=float(scores[k]),
            area_frac=float(scores[k]) / total_area if total_area else 0.0,
            angle_to_winner_deg=math.degrees(math.acos(dot)),
        ))
    return winner, cands


# --------------------------------------------------------------------------
# sign, planes, yaw
# --------------------------------------------------------------------------

def _sign_cues(fg: FaceGeometry, axis: np.ndarray, cfg: FrameEstimatorConfig,
               ) -> tuple[float, dict[str, Any]]:
    """Return (+1/-1 for `axis` being up, cue diagnostics).

    Three independent cues vote; every cue's raw areas and margin are
    reported so a close call is visible rather than implied.

      A  horizontal-area asymmetry: total up-facing vs down-facing area.
         The textbook cue, and MEASURABLY WRONG on 4 of the 6 Replica
         scenes — furniture occludes the floor while the ceiling scans
         complete, so the down-facing total wins. Kept in the vote and
         reported because a cue that fails is a finding, not a bug.
      B  clutter asymmetry: non-horizontal area in the band inside each
         extreme. Furniture stands on the floor; the band under the
         ceiling is bare wall. Correct on 6/6 at every band width tried
         (0.15 / 0.30 / 0.50 / 1.00 m).
      C  interior horizontal mass: tabletops, seats and shelf tops sit in
         the lower half of the room, so the area-weighted mean height of
         the horizontal faces BETWEEN the two end planes is nearer the
         floor. Correct on 6/6. Sensitive to the exclusion band: excluding
         a full metre at each end cuts desk height out of the window and
         flips office_0, which is why the exclusion is a narrow
         end-plane band, not the cue-B clutter band.

    Majority of the three decides."""
    cos_cone = math.cos(math.radians(cfg.cone_deg))
    proj = fg.normal @ axis.astype(np.float32)
    up_area = float(fg.area[proj >= np.float32(cos_cone)].sum())
    down_area = float(fg.area[proj <= np.float32(-cos_cone)].sum())
    cue_a = 1.0 if up_area >= down_area else -1.0

    offsets = (fg.centroid @ axis.astype(np.float32)).astype(np.float64)
    weights = fg.area.astype(np.float64)
    lo = float(_weighted_percentile(offsets, weights, cfg.extent_percentile))
    hi = float(_weighted_percentile(offsets, weights, 100.0 - cfg.extent_percentile))
    span = max(hi - lo, 1e-9)
    band = min(cfg.clutter_band_max_m, cfg.clutter_band_frac * span)

    horizontal = np.abs(proj) >= np.float32(cos_cone)
    non_horizontal = ~horizontal
    low_band = non_horizontal & (offsets >= lo) & (offsets <= lo + band)
    high_band = non_horizontal & (offsets <= hi) & (offsets >= hi - band)
    low_area = float(fg.area[low_band].sum())
    high_area = float(fg.area[high_band].sum())
    cue_b = 1.0 if low_area >= high_area else -1.0

    exclude = max(cfg.interior_exclude_min_m, cfg.interior_exclude_frac * span)
    interior = horizontal & (offsets > lo + exclude) & (offsets < hi - exclude)
    interior_area = float(fg.area[interior].sum())
    if interior_area > 0:
        mean_off = float((offsets[interior] * weights[interior]).sum() / interior_area)
        interior_t = (mean_off - lo) / span
        cue_c = 1.0 if interior_t < 0.5 else -1.0
    else:
        interior_t = None
        cue_c = cue_b

    votes = [cue_a, cue_b, cue_c]
    sign = 1.0 if sum(votes) > 0 else -1.0

    def _ratio(x: float, y: float) -> float:
        return max(x, y) / min(x, y) if min(x, y) > 0 else float("inf")

    return sign, {
        "sign_cue_a_up_facing_area_m2": up_area,
        "sign_cue_a_down_facing_area_m2": down_area,
        "sign_cue_a_ratio": _ratio(up_area, down_area),
        "sign_cue_a_vote": cue_a,
        "sign_cue_b_low_band_clutter_area_m2": low_area,
        "sign_cue_b_high_band_clutter_area_m2": high_area,
        "sign_cue_b_ratio": _ratio(low_area, high_area),
        "sign_cue_b_band_m": band,
        "sign_cue_b_vote": cue_b,
        "sign_cue_c_interior_horizontal_area_m2": interior_area,
        "sign_cue_c_interior_mean_height_frac": interior_t,
        "sign_cue_c_exclude_band_m": exclude,
        "sign_cue_c_vote": cue_c,
        "sign_votes": votes,
        "sign_unanimous": bool(abs(sum(votes)) == 3),
        "sign_decided_by": "majority_of_three_cues",
    }


def _weighted_percentile(values: np.ndarray, weights: np.ndarray,
                         pct: float) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cum = np.cumsum(w)
    if cum[-1] <= 0:
        return float(v[0])
    target = (pct / 100.0) * cum[-1]
    return float(v[int(np.searchsorted(cum, target, side="left"))])


def _horizontal_peaks(offsets: np.ndarray, areas: np.ndarray,
                      bin_m: float) -> tuple[np.ndarray, np.ndarray]:
    if not len(offsets):
        return np.zeros(0), np.zeros(0)
    lo = float(offsets.min())
    idx = np.floor((offsets - lo) / bin_m).astype(np.int64)
    counts = np.bincount(idx, weights=areas)
    centers = lo + (np.arange(len(counts)) + 0.5) * bin_m
    return centers, counts


def _strong_peaks(centers: np.ndarray, areas: np.ndarray, min_frac: float,
                  separation_m: float) -> list[tuple[float, float]]:
    """Greedy area-ordered peak picking with a minimum separation. Returns
    [(offset_m, bin_area_m2), ...] sorted by offset."""
    if not len(areas) or areas.max() <= 0:
        return []
    cut = min_frac * float(areas.max())
    picked: list[tuple[float, float]] = []
    for i in np.argsort(-areas):
        if areas[i] < cut:
            break
        c = float(centers[i])
        if any(abs(c - p) < separation_m for p, _ in picked):
            continue
        picked.append((c, float(areas[i])))
    picked.sort()
    return picked


def _fit_plane(points: np.ndarray, weights: np.ndarray,
               ) -> tuple[np.ndarray, float, float]:
    wsum = float(weights.sum())
    mean = (points * weights[:, None]).sum(axis=0) / wsum
    centered = points - mean
    cov = (centered * weights[:, None]).T @ centered / wsum
    eigvals, eigvecs = np.linalg.eigh(cov)
    n = eigvecs[:, int(np.argmin(eigvals))]
    n = n / np.linalg.norm(n)
    d = -float(np.dot(n, mean))
    residual = points @ n + d
    rms = math.sqrt(float((weights * residual * residual).sum() / wsum))
    return n, d, rms


def _plane_at_peak(fg: FaceGeometry, xyz: np.ndarray, faces: np.ndarray,
                   up: np.ndarray, offsets: np.ndarray,
                   facing: np.ndarray, peak_offset: float,
                   cfg: FrameEstimatorConfig) -> PlaneEstimate | None:
    """Area-weighted least-squares plane over the triangle VERTICES in the
    band. Fitting vertices rather than face centroids keeps the fit
    well-posed for a plane made of very few large triangles."""
    sel = facing & (np.abs(offsets - peak_offset) <= cfg.plane_band_m)
    n_sel = int(np.count_nonzero(sel))
    if n_sel < 1:
        return None
    pts = xyz[faces[sel]].reshape(-1, 3)
    w = np.repeat(fg.area[sel].astype(np.float64) / 3.0, 3)
    if w.sum() <= 0:
        return None
    n, d, rms = _fit_plane(pts, w)
    if float(np.dot(n, up)) < 0:
        n, d = -n, -d
    face_area = fg.area[sel].astype(np.float64)
    mesh_area = float(face_area.sum())
    projected = float(
        (face_area * np.abs(fg.normal[sel].astype(np.float64) @ n)).sum())
    tilt = math.degrees(math.acos(min(1.0, abs(float(np.dot(n, up))))))
    centre_offset = float((pts @ up * w).sum() / w.sum())
    return PlaneEstimate(
        normal=tuple(float(v) for v in n),
        offset_m=centre_offset,
        d=d,
        mesh_area_m2=mesh_area,
        projected_area_m2=projected,
        rms_m=rms,
        n_faces=n_sel,
        tilt_to_up_deg=tilt,
    )


def _dominant_yaw(fg: FaceGeometry, up: np.ndarray,
                  cfg: FrameEstimatorConfig) -> tuple[float, np.ndarray, np.ndarray]:
    """Area-weighted 90-degree-symmetric circular mean of vertical-face
    azimuths. Returns (yaw_deg, e1, e2) with e1 the dominant horizontal
    wall-normal direction and e2 = up x e1."""
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(ref, up))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    e1_0 = ref - float(np.dot(ref, up)) * up
    e1_0 /= np.linalg.norm(e1_0)
    e2_0 = np.cross(up, e1_0)
    sin_vert = math.sin(math.radians(cfg.vertical_deg))
    proj = np.abs(fg.normal @ up.astype(np.float32))
    sel = proj <= np.float32(sin_vert)
    if not np.any(sel):
        return 0.0, e1_0, e2_0
    n = fg.normal[sel].astype(np.float64)
    w = fg.area[sel].astype(np.float64)
    theta = np.arctan2(n @ e2_0, n @ e1_0)
    sx = float((w * np.cos(4.0 * theta)).sum())
    sy = float((w * np.sin(4.0 * theta)).sum())
    if sx == 0.0 and sy == 0.0:
        return 0.0, e1_0, e2_0
    yaw = math.degrees(math.atan2(sy, sx) / 4.0)
    while yaw >= 45.0:
        yaw -= 90.0
    while yaw < -45.0:
        yaw += 90.0
    r = math.radians(yaw)
    e1 = math.cos(r) * e1_0 + math.sin(r) * e2_0
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    return yaw, e1, e2


# --------------------------------------------------------------------------
# public entry points
# --------------------------------------------------------------------------

def estimate_scene_frame(xyz: np.ndarray, faces: np.ndarray,
                         cfg: FrameEstimatorConfig = DEFAULT_CONFIG,
                         ) -> FrameEstimate:
    """Estimate up axis, floor/ceiling planes, and scene scale from geometry."""
    xyz = np.asarray(xyz, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be an [F,3] triangle array")
    if len(faces) < 4:
        raise ValueError("need at least 4 triangles to estimate a frame")

    fg = face_geometry(xyz, faces, cfg)
    total_area = float(fg.area.sum())
    if total_area <= 0:
        raise ValueError("mesh has zero total area")

    axis, candidates = _axis_candidates(fg, cfg, total_area)
    sign, sign_diag = _sign_cues(fg, axis, cfg)
    up = axis * sign
    if candidates:
        candidates[0] = AxisCandidate(
            direction=tuple(float(v) for v in up),
            area_m2=candidates[0].area_m2,
            area_frac=candidates[0].area_frac,
            angle_to_winner_deg=0.0,
        )

    cos_cone = math.cos(math.radians(cfg.cone_deg))
    proj = fg.normal @ up.astype(np.float32)
    offsets = (fg.centroid @ up.astype(np.float32)).astype(np.float64)
    up_facing = proj >= np.float32(cos_cone)
    down_facing = proj <= np.float32(-cos_cone)

    vproj = xyz @ up
    v_lo = float(np.percentile(vproj, cfg.extent_percentile))
    v_hi = float(np.percentile(vproj, 100.0 - cfg.extent_percentile))
    v_span = max(v_hi - v_lo, 1e-9)
    search = cfg.plane_search_frac * v_span

    centers, areas_hist = _horizontal_peaks(
        offsets[up_facing | down_facing],
        fg.area[up_facing | down_facing].astype(np.float64), cfg.offset_bin_m)
    peak_order = np.argsort(-areas_hist)[:cfg.n_reported_peaks]
    horizontal_peaks = [(float(centers[i]), float(areas_hist[i]))
                        for i in sorted(peak_order, key=lambda i: centers[i])
                        if areas_hist[i] > 0]

    floor = ceiling = None
    top_ceiling_offset = None
    fc, fa = _horizontal_peaks(offsets[up_facing],
                               fg.area[up_facing].astype(np.float64),
                               cfg.offset_bin_m)
    if len(fc):
        band = fc <= v_lo + search
        if np.any(band) and fa[band].max() > 0:
            peak = float(fc[band][int(np.argmax(fa[band]))])
            floor = _plane_at_peak(fg, xyz, faces, up, offsets,
                                   up_facing, peak, cfg)
    cc, ca = _horizontal_peaks(offsets[down_facing],
                               fg.area[down_facing].astype(np.float64),
                               cfg.offset_bin_m)
    ceiling_peaks = _strong_peaks(cc, ca, cfg.ceiling_peak_min_area_frac,
                                  cfg.peak_separation_m)
    if ceiling_peaks:
        top_ceiling_offset = ceiling_peaks[-1][0]
    peak = None
    if floor is not None and ceiling_peaks:
        # FIRST ceiling above the floor, not the topmost horizontal plane:
        # apartment_0 is a two-level capture, and its topmost down-facing
        # plane is 5.1 m above its lowest floor.
        above = [c for c, _ in ceiling_peaks
                 if c >= floor.offset_m + cfg.min_storey_height_m]
        if above:
            peak = above[0]
    if peak is None and len(cc):
        band = cc >= v_hi - search
        if np.any(band) and ca[band].max() > 0:
            peak = float(cc[band][int(np.argmax(ca[band]))])
    if peak is not None:
        ceiling = _plane_at_peak(fg, xyz, faces, up, offsets,
                                 down_facing, peak, cfg)

    yaw, e1, e2 = _dominant_yaw(fg, up, cfg)
    rot = np.vstack([e1, e2, up])
    canon = xyz @ rot.T
    extent_raw = canon.max(axis=0) - canon.min(axis=0)
    p_lo = np.percentile(canon, cfg.extent_percentile, axis=0)
    p_hi = np.percentile(canon, 100.0 - cfg.extent_percentile, axis=0)
    extent_rob = p_hi - p_lo

    storey = (ceiling.offset_m - floor.offset_m
              if (floor is not None and ceiling is not None) else None)
    scale = SceneScale(
        extent_raw_m=tuple(float(v) for v in extent_raw),
        extent_robust_m=tuple(float(v) for v in extent_rob),
        room_diagonal_m=float(np.linalg.norm(extent_rob)),
        floor_diagonal_m=float(math.hypot(extent_rob[0], extent_rob[1])),
        floor_area_m2=(floor.projected_area_m2 if floor is not None else None),
        storey_height_m=storey,
        vertical_extent_m=float(extent_rob[2]),
    )

    diagnostics: dict[str, Any] = {
        "estimator": "geometry_frame_v1",
        "config": cfg.to_dict(),
        "n_vertices": int(len(xyz)),
        "n_triangles": int(len(faces)),
        "n_degenerate_triangles": int(fg.n_degenerate),
        "total_mesh_area_m2": total_area,
        "axis_winner_area_frac": candidates[0].area_frac,
        "axis_runner_up_area_frac": (candidates[1].area_frac
                                     if len(candidates) > 1 else None),
        "axis_margin_ratio": (candidates[0].area_m2 / candidates[1].area_m2
                              if len(candidates) > 1 and candidates[1].area_m2 > 0
                              else None),
        "n_horizontal_bins": int(len(centers)),
        "floor_found": floor is not None,
        "ceiling_found": ceiling is not None,
        "ceiling_level_peaks_m": [round(c, 4) for c, _ in ceiling_peaks],
        "top_ceiling_offset_m": top_ceiling_offset,
        "multi_level_suspected": bool(
            ceiling is not None and top_ceiling_offset is not None
            and top_ceiling_offset - ceiling.offset_m
            >= cfg.min_storey_height_m),
    }
    diagnostics.update(sign_diag)
    return FrameEstimate(
        up_axis=tuple(float(v) for v in up),
        rotation=(tuple(float(v) for v in e1),
                  tuple(float(v) for v in e2),
                  tuple(float(v) for v in up)),
        yaw_deg=yaw,
        axis_candidates=candidates,
        floor=floor,
        ceiling=ceiling,
        scale=scale,
        horizontal_peaks=horizontal_peaks,
        diagnostics=diagnostics,
    )


def estimate_from_ply(path: Path,
                      cfg: FrameEstimatorConfig = DEFAULT_CONFIG) -> FrameEstimate:
    xyz, faces, provenance = load_mesh(Path(path))
    est = estimate_scene_frame(xyz, faces, cfg)
    est.diagnostics.update(provenance)
    est.diagnostics["mesh_path"] = str(path)
    return est


def angle_between_deg(a, b) -> float:
    """Angle between two directions, in degrees."""
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(av), np.linalg.norm(bv)
    if na == 0 or nb == 0:
        raise ValueError("cannot take the angle of a zero vector")
    return math.degrees(math.acos(
        max(-1.0, min(1.0, float(np.dot(av, bv)) / (na * nb)))))


def object_scale_stats(aabbs) -> dict[str, float | int]:
    """Scale statistics over a set of axis-aligned boxes, given as
    ((lo_x, lo_y, lo_z), (hi_x, hi_y, hi_z)) pairs.

    Pure geometry: this takes boxes from whatever produced them and reports
    their size distribution. It exists so a scale audit can express a
    threshold as a fraction of typical OBJECT size, not only of room size —
    a 0.3 m axis-dominance gate means something different in a room of
    wardrobes than in a room of mugs."""
    diagonals: list[float] = []
    max_extents: list[float] = []
    min_extents: list[float] = []
    for lo, hi in aabbs:
        ext = [float(hi[i]) - float(lo[i]) for i in range(3)]
        if min(ext) < 0:
            raise ValueError("aabb has hi < lo on some axis")
        diagonals.append(math.sqrt(sum(e * e for e in ext)))
        max_extents.append(max(ext))
        min_extents.append(min(ext))
    if not diagonals:
        return {"n": 0}
    return {
        "n": len(diagonals),
        "median_diagonal_m": float(np.median(diagonals)),
        "p25_diagonal_m": float(np.percentile(diagonals, 25)),
        "p75_diagonal_m": float(np.percentile(diagonals, 75)),
        "median_max_extent_m": float(np.median(max_extents)),
        "median_min_extent_m": float(np.median(min_extents)),
    }
