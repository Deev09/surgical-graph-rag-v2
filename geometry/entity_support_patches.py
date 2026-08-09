"""Geometry-only horizontal patch candidates inside segmented entities.

The input is deliberately narrow: canonical mesh vertices/faces plus a dense
per-vertex instance assignment.  No labels, oracle boxes, room annotations,
or graph relations are accepted.  The output retains continuous evidence for
each owner-local horizontal component so a later relation model can decide
whether another entity is actually supported by it.

This is not a support-relation classifier.  In particular, both the top and
underside of a closed object can be returned as horizontal patch candidates;
whether an observed entity rests above a patch is downstream evidence.

Faces enter an owner only when all three vertex assignments agree.  This
strict rule avoids manufacturing geometry across segment boundaries.  Within
an owner, near-horizontal faces are grouped by shared mesh edges, fitted with
an area-weighted plane, and represented by a convex footprint on that plane.
The convex footprint is explicitly named as such; mesh/projected area and
coverage are retained so holes and convex-hull overreach remain measurable.

All acceptance lengths are scale-relative to the owner.  Absolute outputs are
reported in the canonical bundle's declared metre units for inspection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from common.types import Plane, Vec3
from geometry.mesh_surfaces import load_raw_triangle_mesh


@dataclass(frozen=True)
class HorizontalPatchConfig:
    """Dimensionless patch gates plus a computational chunk size.

    ``min_projected_area_ratio`` divides projected patch area by the owner's
    XY AABB area. ``max_roughness_ratio`` divides plane-fit RMS by the owner's
    3D AABB diagonal.  These ratios preserve behaviour when a mesh is scaled.
    """

    horizontal_angle_deg: float = 15.0
    min_component_faces: int = 2
    min_projected_area_ratio: float = 0.005
    min_coverage_ratio: float = 0.25
    max_roughness_ratio: float = 0.01
    chunk_faces: int = 200_000

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


DEFAULT_CONFIG = HorizontalPatchConfig()


@dataclass(frozen=True)
class HorizontalPatchEvidence:
    patch_uid: str
    owner_instance_id: int
    plane: Plane
    normal: Vec3
    polygon: list[Vec3]
    footprint_kind: str
    height_m: float
    mesh_area_m2: float
    projected_area_m2: float
    footprint_area_m2: float
    coverage_ratio: float
    projected_area_ratio_owner_bbox: float
    roughness_rms_m: float
    roughness_ratio_owner_diagonal: float
    tilt_to_up_deg: float
    n_faces: int
    n_vertices: int
    source_face_ids_sha256: str
    # This qualifies only the owner-local geometric patch extraction. It is
    # never evidence that a target entity rests on the patch; target-relative
    # contact/containment must be evaluated separately.
    geometry_qualifies: bool
    geometry_rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch_uid": self.patch_uid,
            "owner_instance_id": self.owner_instance_id,
            "plane": {
                "a": self.plane.a, "b": self.plane.b,
                "c": self.plane.c, "d": self.plane.d,
            },
            "normal": list(self.normal),
            "polygon": [list(p) for p in self.polygon],
            "footprint_kind": self.footprint_kind,
            "height_m": self.height_m,
            "mesh_area_m2": self.mesh_area_m2,
            "projected_area_m2": self.projected_area_m2,
            "footprint_area_m2": self.footprint_area_m2,
            "coverage_ratio": self.coverage_ratio,
            "projected_area_ratio_owner_bbox": (
                self.projected_area_ratio_owner_bbox
            ),
            "roughness_rms_m": self.roughness_rms_m,
            "roughness_ratio_owner_diagonal": (
                self.roughness_ratio_owner_diagonal
            ),
            "tilt_to_up_deg": self.tilt_to_up_deg,
            "n_faces": self.n_faces,
            "n_vertices": self.n_vertices,
            "source_face_ids_sha256": self.source_face_ids_sha256,
            "geometry_qualifies": self.geometry_qualifies,
            "geometry_rejection_reasons": list(
                self.geometry_rejection_reasons
            ),
        }


@dataclass(frozen=True)
class OwnerPatchEvidence:
    owner_instance_id: int
    n_assigned_vertices: int
    n_strict_faces: int
    n_horizontal_faces: int
    n_horizontal_components: int
    n_components_below_face_min: int
    bbox_aabb: tuple[Vec3, Vec3]
    horizontal_diagonal_m: float
    owner_diagonal_m: float
    owner_xy_bbox_area_m2: float
    patches: list[HorizontalPatchEvidence]

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner_instance_id": self.owner_instance_id,
            "n_assigned_vertices": self.n_assigned_vertices,
            "n_strict_faces": self.n_strict_faces,
            "n_horizontal_faces": self.n_horizontal_faces,
            "n_horizontal_components": self.n_horizontal_components,
            "n_components_below_face_min": self.n_components_below_face_min,
            "bbox_aabb": [list(self.bbox_aabb[0]), list(self.bbox_aabb[1])],
            "horizontal_diagonal_m": self.horizontal_diagonal_m,
            "owner_diagonal_m": self.owner_diagonal_m,
            "owner_xy_bbox_area_m2": self.owner_xy_bbox_area_m2,
            "patches": [p.to_dict() for p in self.patches],
        }


@dataclass(frozen=True)
class EntityHorizontalPatchEstimate:
    owners: list[OwnerPatchEvidence]
    config: HorizontalPatchConfig
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "owners": [o.to_dict() for o in self.owners],
            "config": self.config.to_dict(),
            "diagnostics": dict(self.diagnostics),
        }


def _validate_config(cfg: HorizontalPatchConfig) -> None:
    if not 0.0 < cfg.horizontal_angle_deg < 90.0:
        raise ValueError("horizontal_angle_deg must be in (0, 90)")
    if cfg.min_component_faces < 1:
        raise ValueError("min_component_faces must be >= 1")
    for name in ("min_projected_area_ratio", "min_coverage_ratio",
                 "max_roughness_ratio"):
        value = float(getattr(cfg, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and >= 0")
    if cfg.chunk_faces < 1:
        raise ValueError("chunk_faces must be >= 1")


def _validate_inputs(
    xyz: np.ndarray, faces: np.ndarray, ids: np.ndarray, up: np.ndarray,
) -> None:
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape [V,3], got {xyz.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape [F,3], got {faces.shape}")
    if ids.ndim != 1 or len(ids) != len(xyz):
        raise ValueError(
            f"vertex_instance_ids must have shape ({len(xyz)},), got {ids.shape}")
    if not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("faces must contain integer vertex indices")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("vertex_instance_ids must be integer")
    if not np.isfinite(xyz).all():
        raise ValueError("xyz contains non-finite coordinates")
    if len(faces) and (int(faces.min()) < 0 or int(faces.max()) >= len(xyz)):
        raise ValueError("faces reference vertices outside xyz")
    if int(ids.min(initial=0)) < -1:
        raise ValueError("instance ids below -1 are invalid")
    if not np.isfinite(up).all() or not math.isclose(
        float(np.linalg.norm(up)), 1.0, rel_tol=0.0, abs_tol=1e-6,
    ):
        raise ValueError("up must be a finite unit vector")


def _strict_horizontal_faces(
    xyz: np.ndarray,
    faces: np.ndarray,
    ids: np.ndarray,
    up: np.ndarray,
    cfg: HorizontalPatchConfig,
) -> tuple[dict[int, np.ndarray], dict[int, int], int, int]:
    """Return horizontal face ids per owner without materializing all tris."""
    chunks: dict[int, list[np.ndarray]] = {}
    strict_counts: dict[int, int] = {}
    n_mixed = 0
    n_degenerate = 0
    cos_limit = math.cos(math.radians(cfg.horizontal_angle_deg))

    for start in range(0, len(faces), cfg.chunk_faces):
        stop = min(start + cfg.chunk_faces, len(faces))
        f = faces[start:stop]
        fid = ids[f]
        same = ((fid[:, 0] == fid[:, 1]) &
                (fid[:, 1] == fid[:, 2]) & (fid[:, 0] >= 0))
        n_mixed += int(len(f) - np.count_nonzero(same))
        if not np.any(same):
            continue
        local = np.flatnonzero(same)
        owner = fid[same, 0]
        tri = xyz[f[same]]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        twice_area = np.linalg.norm(cross, axis=1)
        good = twice_area > np.finfo(np.float64).eps
        n_degenerate += int(np.count_nonzero(~good))
        dot = np.zeros(len(twice_area), dtype=np.float64)
        # Explicit reduction avoids platform-BLAS floating-point warnings
        # observed on an otherwise finite 1M-vertex ARKitScenes mesh.
        dot[good] = np.abs(
            np.sum(cross[good] * up[None, :], axis=1) / twice_area[good]
        )
        horizontal = good & (dot >= cos_limit)

        for inst in np.unique(owner):
            inst_i = int(inst)
            owner_mask = owner == inst
            strict_counts[inst_i] = strict_counts.get(inst_i, 0) + int(
                np.count_nonzero(owner_mask))
            selected = local[owner_mask & horizontal] + start
            if len(selected):
                chunks.setdefault(inst_i, []).append(selected.astype(np.int64))

    return (
        {k: np.concatenate(v) for k, v in chunks.items()},
        strict_counts,
        n_mixed,
        n_degenerate,
    )


def _edge_components(faces: np.ndarray, face_ids: np.ndarray) -> list[np.ndarray]:
    """Connected components among selected triangles using shared edges."""
    n = len(face_ids)
    parent = np.arange(n, dtype=np.int64)

    def find(i: int) -> int:
        while int(parent[i]) != i:
            parent[i] = parent[int(parent[i])]
            i = int(parent[i])
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            if ri > rj:
                ri, rj = rj, ri
            parent[rj] = ri

    first_by_edge: dict[tuple[int, int], int] = {}
    for local_i, face_id in enumerate(face_ids):
        a, b, c = (int(x) for x in faces[int(face_id)])
        for x, y in ((a, b), (b, c), (c, a)):
            edge = (x, y) if x < y else (y, x)
            previous = first_by_edge.get(edge)
            if previous is None:
                first_by_edge[edge] = local_i
            else:
                union(local_i, previous)

    grouped: dict[int, list[int]] = {}
    for i, face_id in enumerate(face_ids):
        grouped.setdefault(find(i), []).append(int(face_id))
    return [
        np.asarray(ids_in_component, dtype=np.int64)
        for _, ids_in_component in sorted(
            grouped.items(), key=lambda item: min(item[1]),
        )
    ]


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.array((1.0, 0.0, 0.0), dtype=np.float64)
    if abs(float(ref @ normal)) > 0.9:
        ref = np.array((0.0, 1.0, 0.0), dtype=np.float64)
    u = ref - normal * float(ref @ normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Deterministic 2D monotone-chain hull, counter-clockwise."""
    unique = sorted({(float(p[0]), float(p[1])) for p in points})
    if len(unique) <= 1:
        return np.asarray(unique, dtype=np.float64).reshape((-1, 2))

    def cross(o, a, b) -> float:
        return ((a[0] - o[0]) * (b[1] - o[1]) -
                (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[tuple[float, float]] = []
    for p in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    return 0.5 * abs(float(
        np.dot(poly[:, 0], np.roll(poly[:, 1], -1)) -
        np.dot(poly[:, 1], np.roll(poly[:, 0], -1))
    ))


def _fit_patch(
    instance_id: int,
    component_face_ids: np.ndarray,
    xyz: np.ndarray,
    faces: np.ndarray,
    up: np.ndarray,
    owner_plan_area: float,
    owner_diagonal: float,
    cfg: HorizontalPatchConfig,
) -> HorizontalPatchEvidence:
    tri = xyz[faces[component_face_ids]]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    face_area = twice_area * 0.5
    flat = tri.reshape((-1, 3))
    weights = np.repeat(face_area / 3.0, 3)
    total_weight = float(weights.sum())
    centroid = np.average(flat, axis=0, weights=weights)
    centered = flat - centroid
    covariance = np.einsum(
        "ni,nj,n->ij", centered, centered, weights,
    ) / total_weight
    _, vectors = np.linalg.eigh(covariance)
    normal = vectors[:, 0]
    if float(normal @ up) < 0.0:
        normal = -normal
    normal /= np.linalg.norm(normal)
    signed = np.sum(centered * normal[None, :], axis=1)
    roughness = math.sqrt(float(np.average(signed * signed, weights=weights)))
    tilt = math.degrees(math.acos(float(np.clip(normal @ up, -1.0, 1.0))))

    u, v = _plane_basis(normal)
    unique_vertex_ids = np.unique(faces[component_face_ids])
    local = xyz[unique_vertex_ids] - centroid
    local_2d = np.column_stack((
        np.sum(local * u[None, :], axis=1),
        np.sum(local * v[None, :], axis=1),
    ))
    hull = _convex_hull(local_2d)
    footprint_area = _polygon_area(hull)
    polygon = [
        tuple(float(x) for x in centroid + p[0] * u + p[1] * v)
        for p in hull
    ]

    unit_face_normals = cross / twice_area[:, None]
    projected_area = float(np.sum(
        face_area * np.abs(np.sum(
            unit_face_normals * normal[None, :], axis=1,
        ))
    ))
    mesh_area = float(face_area.sum())
    coverage = (
        projected_area / footprint_area if footprint_area > 0.0 else 0.0
    )
    area_ratio = (
        projected_area / owner_plan_area if owner_plan_area > 0.0 else 0.0
    )
    roughness_ratio = (
        roughness / owner_diagonal if owner_diagonal > 0.0 else math.inf
    )

    reasons: list[str] = []
    if len(component_face_ids) < cfg.min_component_faces:
        reasons.append("too_few_faces")
    if footprint_area <= np.finfo(np.float64).eps:
        reasons.append("degenerate_footprint")
    if owner_plan_area <= np.finfo(np.float64).eps:
        reasons.append("degenerate_owner_plan_area")
    elif area_ratio < cfg.min_projected_area_ratio:
        reasons.append("projected_area_ratio_below_min")
    if coverage < cfg.min_coverage_ratio:
        reasons.append("coverage_below_min")
    if not math.isfinite(roughness_ratio):
        reasons.append("degenerate_owner_diagonal")
    elif roughness_ratio > cfg.max_roughness_ratio:
        reasons.append("roughness_ratio_above_max")
    if tilt > cfg.horizontal_angle_deg:
        reasons.append("fitted_plane_tilt_above_max")

    face_hash = hashlib.sha256(np.ascontiguousarray(
        np.sort(component_face_ids), dtype="<i8",
    ).tobytes()).hexdigest()
    return HorizontalPatchEvidence(
        patch_uid="",  # assigned after stable owner-local sorting
        owner_instance_id=instance_id,
        plane=Plane(a=float(normal[0]), b=float(normal[1]),
                    c=float(normal[2]), d=-float(normal @ centroid)),
        normal=tuple(float(x) for x in normal),
        polygon=polygon,
        footprint_kind="convex_hull_on_fitted_plane",
        height_m=float(centroid @ up),
        mesh_area_m2=mesh_area,
        projected_area_m2=projected_area,
        footprint_area_m2=footprint_area,
        coverage_ratio=coverage,
        projected_area_ratio_owner_bbox=area_ratio,
        roughness_rms_m=roughness,
        roughness_ratio_owner_diagonal=roughness_ratio,
        tilt_to_up_deg=tilt,
        n_faces=int(len(component_face_ids)),
        n_vertices=int(len(unique_vertex_ids)),
        source_face_ids_sha256=face_hash,
        geometry_qualifies=not reasons,
        geometry_rejection_reasons=tuple(reasons),
    )


def extract_entity_horizontal_patches(
    xyz: np.ndarray,
    faces: np.ndarray,
    vertex_instance_ids: np.ndarray,
    *,
    up: Vec3 = (0.0, 0.0, 1.0),
    config: HorizontalPatchConfig = DEFAULT_CONFIG,
) -> EntityHorizontalPatchEstimate:
    """Extract owner-local horizontal patches from canonical mesh arrays."""
    xyz = np.asarray(xyz, dtype=np.float64)
    faces = np.asarray(faces)
    ids = np.asarray(vertex_instance_ids)
    up_arr = np.asarray(up, dtype=np.float64)
    _validate_config(config)
    _validate_inputs(xyz, faces, ids, up_arr)

    horizontal_by_owner, strict_counts, n_mixed, n_degenerate = (
        _strict_horizontal_faces(xyz, faces, ids, up_arr, config)
    )
    owners: list[OwnerPatchEvidence] = []
    for instance_id in sorted(int(i) for i in np.unique(ids) if i >= 0):
        owner_points = xyz[ids == instance_id]
        lo_arr, hi_arr = owner_points.min(axis=0), owner_points.max(axis=0)
        extent = hi_arr - lo_arr
        horizontal_diagonal = float(np.linalg.norm(extent[:2]))
        owner_diagonal = float(np.linalg.norm(extent))
        owner_plan_area = float(extent[0] * extent[1])
        horizontal_face_ids = horizontal_by_owner.get(
            instance_id, np.zeros(0, dtype=np.int64),
        )
        components = _edge_components(faces, horizontal_face_ids)
        large_components = [
            component for component in components
            if len(component) >= config.min_component_faces
        ]
        patches = [
            _fit_patch(instance_id, component, xyz, faces, up_arr,
                       owner_plan_area, owner_diagonal, config)
            for component in large_components
        ]
        patches.sort(key=lambda p: (
            -p.height_m, -p.projected_area_m2, p.source_face_ids_sha256,
        ))
        patches = [
            replace(p, patch_uid=f"instance_{instance_id}_hpatch_{i:03d}")
            for i, p in enumerate(patches)
        ]
        owners.append(OwnerPatchEvidence(
            owner_instance_id=instance_id,
            n_assigned_vertices=int(len(owner_points)),
            n_strict_faces=int(strict_counts.get(instance_id, 0)),
            n_horizontal_faces=int(len(horizontal_face_ids)),
            n_horizontal_components=len(components),
            n_components_below_face_min=(
                len(components) - len(large_components)
            ),
            bbox_aabb=(tuple(float(x) for x in lo_arr),
                       tuple(float(x) for x in hi_arr)),
            horizontal_diagonal_m=horizontal_diagonal,
            owner_diagonal_m=owner_diagonal,
            owner_xy_bbox_area_m2=owner_plan_area,
            patches=patches,
        ))

    all_patches = [p for owner in owners for p in owner.patches]
    return EntityHorizontalPatchEstimate(
        owners=owners,
        config=config,
        diagnostics={
            "n_vertices": int(len(xyz)),
            "n_faces": int(len(faces)),
            "n_assigned_vertices": int(np.count_nonzero(ids >= 0)),
            "n_instances": len(owners),
            "n_mixed_or_unassigned_faces": n_mixed,
            "n_degenerate_strict_faces": n_degenerate,
            "n_horizontal_components": sum(
                o.n_horizontal_components for o in owners
            ),
            "n_components_below_face_min": sum(
                o.n_components_below_face_min for o in owners
            ),
            "n_fitted_patch_components": len(all_patches),
            "n_geometry_qualifying_patches": sum(
                p.geometry_qualifies for p in all_patches
            ),
            "uses_semantics": False,
            "uses_oracle": False,
            "frame_requirement": "canonical metres; caller-supplied up",
        },
    )


def extract_entity_horizontal_patches_from_files(
    mesh_path: Path,
    vertex_instance_ids_path: Path,
    *,
    up: Vec3 = (0.0, 0.0, 1.0),
    config: HorizontalPatchConfig = DEFAULT_CONFIG,
) -> EntityHorizontalPatchEstimate:
    """Load a canonical PLY and dense ``.npy`` assignment, then extract."""
    mesh = load_raw_triangle_mesh(Path(mesh_path))
    ids = np.load(Path(vertex_instance_ids_path), allow_pickle=False)
    return extract_entity_horizontal_patches(
        mesh.xyz, mesh.faces, ids, up=up, config=config,
    )
