"""Target-to-entity-patch resting evidence (experimental, geometry only).

This module is the deliberately separate second half of
``geometry.entity_support_patches``.  The patch extractor says only that an
owner contains a coherent horizontal mesh patch.  Here, every target-owner
pair is measured against those patches using the target AABB footprint:

* vertical gap to the fitted patch plane over the overlapping footprint,
* footprint intersection and target/patch containment fractions,
* target and owner relative-scale measurements, and
* floor-contact exclusion from a geometry-only floor-patch census.

No label, oracle annotation, answer key, or question is accepted.  This module
does not implement the generic RelationExtractor protocol and does not emit
stored graph edges: ``relation_candidate`` is experimental evidence only.
Every decision constant is provisional and transfer calibration is pending.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from extractors.base import EntityArtifact, EntityArtifacts
from geometry.entity_support_patches import (
    EntityHorizontalPatchEstimate, HorizontalPatchEvidence,
    OwnerPatchEvidence,
)


CALIBRATION_STATUS = "provisional_unvalidated_transfer_calibration_pending"


@dataclass(frozen=True)
class EntityPatchRestConfig:
    """Provisional, scale-relative candidate gates.

    Gap bands divide by ``min(target diagonal, owner diagonal)``.  Floor
    contact divides by the target diagonal.  No absolute metre threshold is
    used for a relation decision.
    """

    contact_gap_ratio: float = 0.05
    penetration_ratio: float = 0.02
    min_target_overlap_ratio: float = 0.50
    target_containment_ratio: float = 0.95
    floor_candidate_min_scene_area_ratio: float = 0.15
    floor_candidate_low_height_fraction: float = 0.20
    floor_contact_gap_ratio: float = 0.03
    floor_contact_min_target_overlap_ratio: float = 0.50
    reject_when_floor_unknown: bool = False

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["calibration_status"] = CALIBRATION_STATUS
        return out


DEFAULT_CONFIG = EntityPatchRestConfig()


@dataclass(frozen=True)
class PatchRestEvidence:
    target_entity_uid: str
    owner_entity_uid: str
    patch_uid: str
    vertical_gap_at_target_centroid_m: float
    vertical_gap_at_overlap_centroid_m: float | None
    vertical_gap_min_over_overlap_m: float | None
    vertical_gap_max_over_overlap_m: float | None
    vertical_gap_ratio_relation_scale: float | None
    target_footprint_area_m2: float
    patch_footprint_xy_area_m2: float
    footprint_overlap_area_m2: float
    overlap_ratio_target: float
    overlap_ratio_patch: float
    target_footprint_contained: bool
    relation_scale_m: float
    target_diagonal_m: float
    target_horizontal_diagonal_m: float
    target_height_m: float
    owner_diagonal_m: float
    target_to_owner_diagonal_ratio: float
    floor_contact_state: str
    relation_candidate: bool
    rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["rejection_reasons"] = list(self.rejection_reasons)
        return out


@dataclass(frozen=True)
class TargetOwnerRestEvidence:
    target_entity_uid: str
    owner_entity_uid: str
    owner_instance_id: int
    evaluated_patches: list[PatchRestEvidence]
    selected_patch_uid: str | None
    relation_candidate: bool
    rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_entity_uid": self.target_entity_uid,
            "owner_entity_uid": self.owner_entity_uid,
            "owner_instance_id": self.owner_instance_id,
            "evaluated_patches": [p.to_dict() for p in self.evaluated_patches],
            "selected_patch_uid": self.selected_patch_uid,
            "relation_candidate": self.relation_candidate,
            "rejection_reasons": list(self.rejection_reasons),
        }


@dataclass(frozen=True)
class EntityPatchRestEstimate:
    pairs: list[TargetOwnerRestEvidence]
    floor_evidence: dict[str, Any]
    config: EntityPatchRestConfig
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pairs": [p.to_dict() for p in self.pairs],
            "floor_evidence": dict(self.floor_evidence),
            "config": self.config.to_dict(),
            "diagnostics": dict(self.diagnostics),
        }


def _validate_config(config: EntityPatchRestConfig) -> None:
    nonnegative = (
        "contact_gap_ratio", "penetration_ratio",
        "floor_contact_gap_ratio",
    )
    fractions = (
        "min_target_overlap_ratio", "target_containment_ratio",
        "floor_candidate_min_scene_area_ratio",
        "floor_candidate_low_height_fraction",
        "floor_contact_min_target_overlap_ratio",
    )
    for name in nonnegative:
        value = float(getattr(config, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and >= 0")
    for name in fractions:
        value = float(getattr(config, name))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")


def _entity_instance_id(entity: EntityArtifact) -> int | None:
    prefix = "segmenter:"
    ref = entity.identity.source_instance_ref
    if not ref.startswith(prefix):
        return None
    try:
        value = int(ref[len(prefix):])
    except ValueError:
        return None
    return value if value >= 0 else None


def _signed_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    return 0.5 * float(
        np.dot(poly[:, 0], np.roll(poly[:, 1], -1)) -
        np.dot(poly[:, 1], np.roll(poly[:, 0], -1))
    )


def _area(poly: np.ndarray) -> float:
    return abs(_signed_area(poly))


def _ensure_ccw(poly: np.ndarray) -> np.ndarray:
    return poly[::-1].copy() if _signed_area(poly) < 0.0 else poly


def _line_intersection(
    p: np.ndarray, q: np.ndarray, a: np.ndarray, b: np.ndarray,
) -> np.ndarray:
    """Intersection of line pq with line ab; parallel falls back to q."""
    r, s = q - p, b - a
    denominator = float(r[0] * s[1] - r[1] * s[0])
    if abs(denominator) <= np.finfo(np.float64).eps:
        return q.copy()
    ap = a - p
    t = float((ap[0] * s[1] - ap[1] * s[0]) / denominator)
    return p + t * r


def _clip_convex(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman intersection for two convex CCW polygons."""
    output = [p.copy() for p in _ensure_ccw(subject)]
    clip = _ensure_ccw(clip)
    eps = 1e-12
    for i, a in enumerate(clip):
        b = clip[(i + 1) % len(clip)]
        incoming = output
        output = []
        if not incoming:
            break

        def inside(p: np.ndarray) -> bool:
            return float(
                (b[0] - a[0]) * (p[1] - a[1]) -
                (b[1] - a[1]) * (p[0] - a[0])
            ) >= -eps

        previous = incoming[-1]
        previous_inside = inside(previous)
        for current in incoming:
            current_inside = inside(current)
            if current_inside != previous_inside:
                output.append(_line_intersection(previous, current, a, b))
            if current_inside:
                output.append(current.copy())
            previous, previous_inside = current, current_inside
    return np.asarray(output, dtype=np.float64).reshape((-1, 2))


def _polygon_centroid(poly: np.ndarray) -> np.ndarray:
    """Area centroid; mean fallback handles line/point intersections."""
    signed = _signed_area(poly)
    if len(poly) < 3 or abs(signed) <= np.finfo(np.float64).eps:
        return poly.mean(axis=0)
    shifted = np.roll(poly, -1, axis=0)
    cross = poly[:, 0] * shifted[:, 1] - shifted[:, 0] * poly[:, 1]
    return np.asarray((
        float(np.sum((poly[:, 0] + shifted[:, 0]) * cross) / (6.0 * signed)),
        float(np.sum((poly[:, 1] + shifted[:, 1]) * cross) / (6.0 * signed)),
    ))


def _target_geometry(entity: EntityArtifact) -> dict[str, Any]:
    lo = np.asarray(entity.bbox_aabb[0], dtype=np.float64)
    hi = np.asarray(entity.bbox_aabb[1], dtype=np.float64)
    extent = hi - lo
    rect = np.asarray([
        (lo[0], lo[1]), (hi[0], lo[1]),
        (hi[0], hi[1]), (lo[0], hi[1]),
    ], dtype=np.float64)
    return {
        "lo": lo, "hi": hi, "extent": extent, "rect": rect,
        "area": float(extent[0] * extent[1]),
        "horizontal_diagonal": float(np.linalg.norm(extent[:2])),
        "diagonal": float(np.linalg.norm(extent)),
        "height": float(extent[2]),
    }


def _patch_measurements(
    target_geometry: dict[str, Any], patch: HorizontalPatchEvidence,
) -> dict[str, Any]:
    patch_xy = np.asarray([(p[0], p[1]) for p in patch.polygon],
                          dtype=np.float64)
    patch_xy_area = _area(patch_xy)
    target_area = float(target_geometry["area"])
    overlap = (
        _clip_convex(target_geometry["rect"], patch_xy)
        if len(patch_xy) >= 3 and target_area > 0.0 else
        np.zeros((0, 2), dtype=np.float64)
    )
    overlap_area = _area(overlap)
    target_ratio = overlap_area / target_area if target_area > 0.0 else 0.0
    patch_ratio = overlap_area / patch_xy_area if patch_xy_area > 0.0 else 0.0

    plane = patch.plane
    if abs(plane.c) <= np.finfo(np.float64).eps:
        raise ValueError(f"patch {patch.patch_uid} plane cannot solve z")

    def gap_at(xy: np.ndarray) -> float:
        patch_z = -(plane.a * float(xy[0]) +
                    plane.b * float(xy[1]) + plane.d) / plane.c
        return float(target_geometry["lo"][2] - patch_z)

    target_center_xy = np.asarray((
        (target_geometry["lo"][0] + target_geometry["hi"][0]) / 2.0,
        (target_geometry["lo"][1] + target_geometry["hi"][1]) / 2.0,
    ))
    gap_target_center = gap_at(target_center_xy)
    if overlap_area > 0.0:
        overlap_centroid = _polygon_centroid(overlap)
        overlap_gaps = [gap_at(p) for p in overlap]
        gap_overlap = gap_at(overlap_centroid)
        gap_min, gap_max = min(overlap_gaps), max(overlap_gaps)
    else:
        gap_overlap = gap_min = gap_max = None
    return {
        "patch_xy_area": patch_xy_area,
        "overlap_area": overlap_area,
        "overlap_ratio_target": target_ratio,
        "overlap_ratio_patch": patch_ratio,
        "gap_target_center": gap_target_center,
        "gap_overlap": gap_overlap,
        "gap_min": gap_min,
        "gap_max": gap_max,
    }


def _scene_bounds(entities: EntityArtifacts) -> tuple[np.ndarray, np.ndarray]:
    lo = np.min(np.asarray([e.bbox_aabb[0] for e in entities.entities]), axis=0)
    hi = np.max(np.asarray([e.bbox_aabb[1] for e in entities.entities]), axis=0)
    return lo, hi


def _floor_patch_census(
    entities: EntityArtifacts,
    patch_estimate: EntityHorizontalPatchEstimate,
    config: EntityPatchRestConfig,
) -> tuple[HorizontalPatchEvidence | None, dict[str, Any]]:
    """Select a scene-scale low patch using geometry only, or return unknown."""
    scene_lo, scene_hi = _scene_bounds(entities)
    extent = scene_hi - scene_lo
    scene_xy_area = float(extent[0] * extent[1])
    low_ceiling = float(
        scene_lo[2] + config.floor_candidate_low_height_fraction * extent[2]
    )
    candidates: list[tuple[HorizontalPatchEvidence, float]] = []
    for owner in patch_estimate.owners:
        for patch in owner.patches:
            if not patch.geometry_qualifies:
                continue
            area_ratio = (
                patch.projected_area_m2 / scene_xy_area
                if scene_xy_area > 0.0 else 0.0
            )
            if (area_ratio >= config.floor_candidate_min_scene_area_ratio and
                    patch.height_m <= low_ceiling):
                candidates.append((patch, area_ratio))
    candidates.sort(key=lambda item: (
        -item[1], item[0].height_m, item[0].patch_uid,
    ))
    selected = candidates[0] if candidates else None
    evidence = {
        "method": "geometry_only_low_scene_scale_patch_census",
        "status": "available" if selected else "unknown",
        "selected_patch_uid": selected[0].patch_uid if selected else None,
        "selected_area_ratio_scene_xy": selected[1] if selected else None,
        "scene_xy_bbox_area_m2": scene_xy_area,
        "scene_z_min_m": float(scene_lo[2]),
        "scene_z_extent_m": float(extent[2]),
        "candidate_low_height_max_m": low_ceiling,
        "n_candidates": len(candidates),
        "uses_semantics": False,
        "uses_oracle": False,
        "calibration_status": CALIBRATION_STATUS,
    }
    return (selected[0] if selected else None), evidence


def _floor_contact(
    target_geometry: dict[str, Any],
    floor_patch: HorizontalPatchEvidence | None,
    config: EntityPatchRestConfig,
) -> tuple[str, dict[str, Any]]:
    if floor_patch is None:
        return "unknown", {
            "state": "unknown", "floor_patch_uid": None,
            "reason": "no_geometry_floor_patch_candidate",
        }
    measured = _patch_measurements(target_geometry, floor_patch)
    scale = float(target_geometry["diagonal"])
    gap = measured["gap_overlap"]
    contact = bool(
        scale > 0.0 and gap is not None and
        measured["overlap_ratio_target"] >=
        config.floor_contact_min_target_overlap_ratio and
        -config.floor_contact_gap_ratio * scale <= gap <=
        config.floor_contact_gap_ratio * scale
    )
    return ("contact" if contact else "clear"), {
        "state": "contact" if contact else "clear",
        "floor_patch_uid": floor_patch.patch_uid,
        "target_overlap_ratio": measured["overlap_ratio_target"],
        "vertical_gap_at_overlap_centroid_m": gap,
        "target_diagonal_m": scale,
        "gap_ratio_target_diagonal": (
            gap / scale if gap is not None and scale > 0.0 else None
        ),
    }


def _patch_candidate(
    target: EntityArtifact,
    owner_uid: str,
    owner: OwnerPatchEvidence,
    patch: HorizontalPatchEvidence,
    floor_state: str,
    config: EntityPatchRestConfig,
) -> PatchRestEvidence:
    geometry = _target_geometry(target)
    measured = _patch_measurements(geometry, patch)
    relation_scale = min(float(geometry["diagonal"]), owner.owner_diagonal_m)
    gap = measured["gap_overlap"]
    reasons: list[str] = []
    if geometry["area"] <= np.finfo(np.float64).eps:
        reasons.append("target_footprint_degenerate")
    if measured["overlap_ratio_target"] < config.min_target_overlap_ratio:
        reasons.append("footprint_overlap_below_min")
    if relation_scale <= np.finfo(np.float64).eps:
        reasons.append("relation_scale_degenerate")
    elif gap is None:
        reasons.append("vertical_gap_unavailable_without_overlap")
    else:
        if gap < -config.penetration_ratio * relation_scale:
            reasons.append("vertical_gap_below_penetration_band")
        if gap > config.contact_gap_ratio * relation_scale:
            reasons.append("vertical_gap_above_contact_band")
    if floor_state == "contact":
        reasons.append("target_floor_contact")
    elif floor_state == "unknown" and config.reject_when_floor_unknown:
        reasons.append("target_floor_contact_unknown")

    target_diag = float(geometry["diagonal"])
    return PatchRestEvidence(
        target_entity_uid=target.identity.object_uid,
        owner_entity_uid=owner_uid,
        patch_uid=patch.patch_uid,
        vertical_gap_at_target_centroid_m=measured["gap_target_center"],
        vertical_gap_at_overlap_centroid_m=gap,
        vertical_gap_min_over_overlap_m=measured["gap_min"],
        vertical_gap_max_over_overlap_m=measured["gap_max"],
        vertical_gap_ratio_relation_scale=(
            gap / relation_scale
            if gap is not None and relation_scale > 0.0 else None
        ),
        target_footprint_area_m2=float(geometry["area"]),
        patch_footprint_xy_area_m2=measured["patch_xy_area"],
        footprint_overlap_area_m2=measured["overlap_area"],
        overlap_ratio_target=measured["overlap_ratio_target"],
        overlap_ratio_patch=measured["overlap_ratio_patch"],
        target_footprint_contained=(
            measured["overlap_ratio_target"] >=
            config.target_containment_ratio
        ),
        relation_scale_m=relation_scale,
        target_diagonal_m=target_diag,
        target_horizontal_diagonal_m=float(geometry["horizontal_diagonal"]),
        target_height_m=float(geometry["height"]),
        owner_diagonal_m=owner.owner_diagonal_m,
        target_to_owner_diagonal_ratio=(
            target_diag / owner.owner_diagonal_m
            if owner.owner_diagonal_m > 0.0 else math.inf
        ),
        floor_contact_state=floor_state,
        relation_candidate=not reasons,
        rejection_reasons=tuple(reasons),
    )


def evaluate_entity_patch_resting(
    entities: EntityArtifacts,
    patch_estimate: EntityHorizontalPatchEstimate,
    *,
    config: EntityPatchRestConfig = DEFAULT_CONFIG,
) -> EntityPatchRestEstimate:
    """Measure every target-owner pair; emit experimental candidates only."""
    _validate_config(config)
    if not entities.entities:
        raise ValueError("EntityArtifacts must contain at least one entity")
    if entities.frame.kind != "scene_canonical":
        raise ValueError("entity patch resting requires scene_canonical frame")
    gravity = np.asarray(entities.frame.gravity, dtype=np.float64)
    if not np.allclose(gravity, (0.0, 0.0, -1.0), rtol=0.0, atol=1e-6):
        raise ValueError("entity patch resting requires canonical gravity -z")

    by_instance: dict[int, EntityArtifact] = {}
    for entity in entities.entities:
        instance_id = _entity_instance_id(entity)
        if instance_id is None:
            continue
        if instance_id in by_instance:
            raise ValueError(f"duplicate segmenter instance id {instance_id}")
        by_instance[instance_id] = entity
    owners: list[tuple[OwnerPatchEvidence, EntityArtifact]] = []
    for owner in patch_estimate.owners:
        entity = by_instance.get(owner.owner_instance_id)
        if entity is None:
            raise ValueError(
                f"patch owner instance {owner.owner_instance_id} has no "
                "EntityArtifacts entity with source_instance_ref=segmenter:<id>"
            )
        owners.append((owner, entity))

    floor_patch, floor_evidence = _floor_patch_census(
        entities, patch_estimate, config,
    )
    floor_by_target: dict[str, tuple[str, dict[str, Any]]] = {}
    for target in entities.entities:
        floor_by_target[target.identity.object_uid] = _floor_contact(
            _target_geometry(target), floor_patch, config,
        )

    pairs: list[TargetOwnerRestEvidence] = []
    for target in sorted(entities.entities,
                         key=lambda entity: entity.identity.object_uid):
        target_uid = target.identity.object_uid
        floor_state, _ = floor_by_target[target_uid]
        for owner, owner_entity in sorted(
            owners, key=lambda item: item[1].identity.object_uid,
        ):
            owner_uid = owner_entity.identity.object_uid
            qualifying = [
                patch for patch in owner.patches if patch.geometry_qualifies
            ]
            if target_uid == owner_uid:
                pairs.append(TargetOwnerRestEvidence(
                    target_entity_uid=target_uid,
                    owner_entity_uid=owner_uid,
                    owner_instance_id=owner.owner_instance_id,
                    evaluated_patches=[], selected_patch_uid=None,
                    relation_candidate=False,
                    rejection_reasons=("self_support_excluded",),
                ))
                continue
            evaluated = [
                _patch_candidate(target, owner_uid, owner, patch,
                                 floor_state, config)
                for patch in qualifying
            ]
            evaluated.sort(key=lambda result: (
                not result.relation_candidate,
                -result.overlap_ratio_target,
                abs(result.vertical_gap_at_overlap_centroid_m)
                if result.vertical_gap_at_overlap_centroid_m is not None
                else math.inf,
                result.patch_uid,
            ))
            selected = next(
                (result for result in evaluated if result.relation_candidate),
                None,
            )
            pair_reasons: tuple[str, ...]
            if selected is not None:
                pair_reasons = ()
            elif not qualifying:
                pair_reasons = ("owner_has_no_geometry_qualifying_patch",)
            else:
                pair_reasons = ("no_patch_satisfies_resting_evidence",)
            pairs.append(TargetOwnerRestEvidence(
                target_entity_uid=target_uid,
                owner_entity_uid=owner_uid,
                owner_instance_id=owner.owner_instance_id,
                evaluated_patches=evaluated,
                selected_patch_uid=(selected.patch_uid if selected else None),
                relation_candidate=selected is not None,
                rejection_reasons=pair_reasons,
            ))

    candidate_pairs = [pair for pair in pairs if pair.relation_candidate]
    return EntityPatchRestEstimate(
        pairs=pairs,
        floor_evidence={
            **floor_evidence,
            "target_states": {
                uid: evidence for uid, (_, evidence) in
                sorted(floor_by_target.items())
            },
        },
        config=config,
        diagnostics={
            "n_entities": len(entities.entities),
            "n_patch_owners": len(owners),
            "n_target_owner_pairs": len(pairs),
            "n_relation_candidates": len(candidate_pairs),
            "candidate_semantics": (
                "experimental geometry evidence; not a calibrated "
                "ON_ENTITY_SURFACE claim"
            ),
            "calibration_status": CALIBRATION_STATUS,
            "uses_labels": False,
            "uses_oracle": False,
            "uses_keys": False,
        },
    )
