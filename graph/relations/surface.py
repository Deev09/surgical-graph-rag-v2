"""Surface-proximity relation extractor — NEAR_SURFACE.

Emits NEAR_SURFACE(entity, surface) edges where `surface` is a
StructuralSurface from the EntityArtifacts bundle and the entity's
AABB-to-surface distance is within a per-surface-type threshold.

Two modes, both selected via SurfaceProximityConfig.use_polygon_clip:

  Default (use_polygon_clip=False) — Phase 2 plane-mode (byte-frozen):
    - distance = bbox_to_plane(entity.bbox_aabb, surface.plane),
    - extractor_version = "0.1",
    - evidence schema = {distance_m, distance_metric="bbox_to_plane",
      threshold_m, surface_type, source} (unchanged from P2.09).

  Opt-in (use_polygon_clip=True) — Phase 3 polygon-mode:
    - distance, dispatcher_evidence = bbox_to_surface(
          entity.bbox_aabb, surface.plane, surface.polygon),
    - extractor_version = "0.2-near_surface_polygon_clipped"
      (applied to ALL opt-in edges, including polygon=None fallbacks —
      the mode is what changes, not the per-surface presence of a polygon),
    - evidence = dispatcher_evidence merged with {threshold_m, surface_type,
      source}. Per the dispatcher contract: polygon-present evidence carries
      {normal_gap_m, in_plane_gap_m, polygon_clipping_applied=True,
      distance_metric="polygon_clipped", distance_m}; polygon-None fallback
      carries {normal_gap_m, polygon_clipping_applied=False,
      distance_metric="bbox_to_plane", distance_m, fallback_reason="polygon_none"}.

A4 byte-equality: SurfaceProximityConfig.use_polygon_clip is declared with
`field(default=False, metadata={"hash_omit_if_default": True})` so that
default Phase 2 configs serialize to the Phase 2 _config_hash_payload byte-
for-byte. Tested in tests/relations/test_near_surface_polygon.py.

Canonical policy (A3, A7): the extractor refuses to emit NEAR_SURFACE
against surfaces whose `source == "synth_bbox_fallback"` unless
`config.include_synth_fallback` is True. Skipped surfaces are recorded
as EdgeRejection entries with rejected_reason="surface_source_excluded".

Phase 8 F4 (opt-in, exclude_room_scale_flat=True): room-scale flat entities
(wall-to-wall rugs) are never candidates against WALL surfaces, recorded as
rejected_reason="room_scale_flat_excluded"; floor/ceiling proximity is
untouched. Default off = frozen Phase 2/3 behavior in both modes.

Isolation from global builder density policy (P2.09 sign-off): this
module ships the extractor and tests, but is NOT yet wired into any
default GraphBuilder run. P2.10 makes the version-aware density-cap
decision before integration; the combined sparse-v2 graph already
exceeds 14/entity per the phase2_sparse_v2_telemetry artifact.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Literal

from common.types import FrameKind
from extractors.base import EntityArtifact, EntityArtifacts, StructuralSurface
from geometry.surface_distance import bbox_to_plane, bbox_to_surface
from graph.relations.base import (
    RelationExtractorConfig, RelationExtractorDiagnostics,
    count_logical_edges, edge_frame, make_edge_id, make_entity_ref, make_surface_ref,
    margin_confidence, ratio_margin, room_scale_flat, wall_xy_extent_area,
)
from graph.schema import Edge, EdgeRejection, EdgeType


NEAR_SURFACE_TYPES: frozenset[EdgeType] = frozenset({"NEAR_SURFACE"})

DEFAULT_FLOOR_THRESHOLD_M = 0.05
DEFAULT_WALL_THRESHOLD_M = 0.30
DEFAULT_CEILING_THRESHOLD_M = 0.10

PLANE_MODE_VERSION = "0.1"
POLYGON_CLIPPED_VERSION = "0.2-near_surface_polygon_clipped"


@dataclass(frozen=True)
class SurfaceProximityConfig:
    """Per-surface-type thresholds; provisional Replica-calibrated defaults
    per Q4. include_synth_fallback gates the synth-bbox-fallback policy
    (A3); leave False on the canonical path.

    `use_polygon_clip` (Phase 3 A4): opt-in flag for polygon-aware distance.
    Declared with metadata={"hash_omit_if_default": True} so that default
    configs serialize to the Phase 2 _config_hash_payload byte-for-byte
    (see graph/builder.py:_config_hash_payload). Enabling polygon clip
    switches the extractor to a distinct version string and a richer
    evidence schema; default Phase 2 plane-mode bytes are preserved."""
    mode: Literal["sparse"] = "sparse"
    floor_threshold_m: float = DEFAULT_FLOOR_THRESHOLD_M
    wall_threshold_m: float = DEFAULT_WALL_THRESHOLD_M
    ceiling_threshold_m: float = DEFAULT_CEILING_THRESHOLD_M
    include_synth_fallback: bool = False
    use_polygon_clip: bool = field(
        default=False,
        metadata={"hash_omit_if_default": True},
    )
    # Phase 8 F4 (opt-in): skip room-scale flat entities (wall-to-wall rugs)
    # as NEAR_SURFACE candidates against WALL surfaces only (floor/ceiling
    # proximity is untouched — a rug IS near the floor). Default False =
    # frozen Phase 2/3 behavior; hash_omit_if_default keeps default config
    # hashes byte-identical. Defaults mirror ContactsSurfaceConfig; enable on
    # BOTH configs with the same values so CONTACTS ⊆ NEAR(wall) still holds
    # (enabling here alone would break that subset).
    exclude_room_scale_flat: bool = field(
        default=False, metadata={"hash_omit_if_default": True})
    room_scale_flat_min_area_frac: float = field(
        default=0.60, metadata={"hash_omit_if_default": True})
    room_scale_flat_max_height_m: float = field(
        default=0.20, metadata={"hash_omit_if_default": True})
    # Calibration opt-in (see _near_surface_margin). False = frozen behavior
    # (confidence is literally 1.0); hash_omit_if_default keeps default config
    # hashes byte-identical in BOTH plane and polygon modes.
    emit_margins: bool = field(
        default=False, metadata={"hash_omit_if_default": True})


def _near_surface_margin(distance: float, threshold: float) -> float:
    """Normalized NEAR_SURFACE margin.

    Continuous quantity: the entity-to-surface distance (bbox_to_plane in
    plane mode, polygon-clipped in polygon mode). Threshold: the
    per-surface-type threshold (floor 0.05, wall 0.30, ceiling 0.10 by
    default). Margin = (threshold - distance) / threshold: the threshold is
    the per-type definition of "near", so it is also the right scale to
    measure comfort against, and the same margin means the same thing across
    surface types despite the 6x spread in absolute thresholds. Touching the
    surface reads 1.0; sitting exactly on the threshold reads 0."""
    return ratio_margin(distance, threshold)


def _threshold_for(config: SurfaceProximityConfig, surface_type: str) -> float:
    if surface_type == "floor":
        return config.floor_threshold_m
    if surface_type == "wall":
        return config.wall_threshold_m
    if surface_type == "ceiling":
        return config.ceiling_threshold_m
    raise ValueError(f"unknown surface_type {surface_type!r}")


def _validate_config(config: SurfaceProximityConfig) -> None:
    if config.mode != "sparse":
        raise ValueError(f"unknown mode {config.mode!r}; supported: 'sparse'")
    for name in (
        "floor_threshold_m",
        "wall_threshold_m",
        "ceiling_threshold_m",
        "room_scale_flat_max_height_m",
    ):
        value = getattr(config, name)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if (
        not math.isfinite(config.room_scale_flat_min_area_frac)
        or config.room_scale_flat_min_area_frac <= 0.0
    ):
        raise ValueError("room_scale_flat_min_area_frac must be finite and positive")


def _build_near_surface_edge(
    entity: EntityArtifact,
    surface: StructuralSurface,
    distance: float,
    threshold: float,
    *,
    extractor: str,
    version: str,
    frame: FrameKind,
    confidence: float = 1.0,
) -> Edge:
    source = make_entity_ref(entity.identity.object_uid)
    target = make_surface_ref(surface.surface_uid)
    return Edge(
        edge_id=make_edge_id(extractor, version, source, "NEAR_SURFACE", target),
        source=source,
        type="NEAR_SURFACE",
        target=target,
        frame=frame,
        weight=1.0,
        confidence=confidence,
        extractor=extractor,
        extractor_version=version,
        evidence={
            "distance_m": distance,
            "distance_metric": "bbox_to_plane",
            "threshold_m": threshold,
            "surface_type": surface.surface_type,
            "source": surface.source,
        },
    )


def _build_near_surface_edge_polygon(
    entity: EntityArtifact,
    surface: StructuralSurface,
    dispatcher_evidence: dict,
    threshold: float,
    *,
    extractor: str,
    version: str,
    frame: FrameKind,
    confidence: float = 1.0,
) -> Edge:
    """Polygon-mode edge builder (A5). The dispatcher_evidence already
    carries distance_m, distance_metric, normal_gap_m, polygon_clipping_applied,
    and (when polygon present) in_plane_gap_m; we layer the Phase 2
    extractor-side keys (threshold_m, surface_type, source) on top."""
    source = make_entity_ref(entity.identity.object_uid)
    target = make_surface_ref(surface.surface_uid)
    evidence = dict(dispatcher_evidence)
    evidence["threshold_m"] = threshold
    evidence["surface_type"] = surface.surface_type
    evidence["source"] = surface.source
    return Edge(
        edge_id=make_edge_id(extractor, version, source, "NEAR_SURFACE", target),
        source=source,
        type="NEAR_SURFACE",
        target=target,
        frame=frame,
        weight=1.0,
        confidence=confidence,
        extractor=extractor,
        extractor_version=version,
        evidence=evidence,
    )


class SurfaceProximityExtractor:
    name: str = "near_surface"
    version: str = PLANE_MODE_VERSION
    edge_types: frozenset[EdgeType] = NEAR_SURFACE_TYPES

    def extract(
        self,
        entities: EntityArtifacts,
        config: RelationExtractorConfig,
    ) -> tuple[list[Edge], RelationExtractorDiagnostics]:
        if not isinstance(config, SurfaceProximityConfig):
            raise TypeError(
                f"SurfaceProximityExtractor requires SurfaceProximityConfig, "
                f"got {type(config).__name__}"
            )
        _validate_config(config)
        start = time.perf_counter()
        effective_version = (
            POLYGON_CLIPPED_VERSION if config.use_polygon_clip else PLANE_MODE_VERSION
        )
        frame = edge_frame(entities)
        edges: list[Edge] = []
        rejections: list[EdgeRejection] = []
        rejection_counts: dict[EdgeType, int] = {}
        max_rejection_samples = 64

        active_surfaces: list[StructuralSurface] = []
        excluded_surfaces: list[StructuralSurface] = []
        for surface in entities.structural_surfaces:
            if surface.source == "synth_bbox_fallback" and not config.include_synth_fallback:
                excluded_surfaces.append(surface)
                continue
            active_surfaces.append(surface)

        # Phase 8 F4 (opt-in): room-scale flat entities (wall-to-wall rugs)
        # are never NEAR_SURFACE candidates against WALL surfaces. Room area
        # computed once; floor/ceiling proximity is untouched.
        room_xy_area: float | None = None
        if config.exclude_room_scale_flat:
            room_xy_area = wall_xy_extent_area(entities.structural_surfaces)

        for entity in entities.entities:
            entity_flat = False
            flat_evidence: dict = {}
            if config.exclude_room_scale_flat:
                entity_flat, flat_evidence = room_scale_flat(
                    entity, room_xy_area,
                    min_area_frac=config.room_scale_flat_min_area_frac,
                    max_height_m=config.room_scale_flat_max_height_m,
                )
            for surface in excluded_surfaces:
                rejection_counts["NEAR_SURFACE"] = (
                    rejection_counts.get("NEAR_SURFACE", 0) + 1
                )
                if len(rejections) < max_rejection_samples:
                    rejections.append(EdgeRejection(
                        source=make_entity_ref(entity.identity.object_uid),
                        type="NEAR_SURFACE",
                        target=make_surface_ref(surface.surface_uid),
                        extractor=self.name,
                        rejected_reason="surface_source_excluded",
                        evidence={
                            "source": surface.source,
                            "include_synth_fallback": False,
                            "policy": (
                                "canonical_extractor_excludes_synth_bbox_fallback"
                            ),
                        },
                    ))
            for surface in active_surfaces:
                if entity_flat and surface.surface_type == "wall":
                    rejection_counts["NEAR_SURFACE"] = (
                        rejection_counts.get("NEAR_SURFACE", 0) + 1
                    )
                    if len(rejections) < max_rejection_samples:
                        rejections.append(EdgeRejection(
                            source=make_entity_ref(entity.identity.object_uid),
                            type="NEAR_SURFACE",
                            target=make_surface_ref(surface.surface_uid),
                            extractor=self.name,
                            rejected_reason="room_scale_flat_excluded",
                            evidence=dict(flat_evidence) | {
                                "surface_type": surface.surface_type,
                                "source": surface.source,
                                "policy": (
                                    "room_scale_flat_never_a_wall_proximity_candidate"
                                ),
                            },
                        ))
                    continue
                threshold = _threshold_for(config, surface.surface_type)
                if config.use_polygon_clip:
                    distance, dispatcher_evidence = bbox_to_surface(
                        entity.bbox_aabb, surface.plane, surface.polygon,
                    )
                    if distance <= threshold:
                        edges.append(_build_near_surface_edge_polygon(
                            entity, surface, dispatcher_evidence, threshold,
                            extractor=self.name, version=effective_version,
                            frame=frame,
                            confidence=margin_confidence(_near_surface_margin(
                                distance, threshold)) if config.emit_margins else 1.0,
                        ))
                    else:
                        rejection_counts["NEAR_SURFACE"] = (
                            rejection_counts.get("NEAR_SURFACE", 0) + 1
                        )
                        if len(rejections) < max_rejection_samples:
                            rejection_evidence = dict(dispatcher_evidence)
                            rejection_evidence["threshold_m"] = threshold
                            rejection_evidence["surface_type"] = surface.surface_type
                            rejection_evidence["source"] = surface.source
                            if config.emit_margins:
                                rejection_evidence["margin_confidence"] = \
                                    margin_confidence(_near_surface_margin(
                                        distance, threshold))
                            rejections.append(EdgeRejection(
                                source=make_entity_ref(entity.identity.object_uid),
                                type="NEAR_SURFACE",
                                target=make_surface_ref(surface.surface_uid),
                                extractor=self.name,
                                rejected_reason="distance_exceeds_surface_threshold",
                                evidence=rejection_evidence,
                            ))
                else:
                    distance = bbox_to_plane(entity.bbox_aabb, surface.plane)
                    if distance <= threshold:
                        edges.append(_build_near_surface_edge(
                            entity, surface, distance, threshold,
                            extractor=self.name, version=effective_version,
                            frame=frame,
                            confidence=margin_confidence(_near_surface_margin(
                                distance, threshold)) if config.emit_margins else 1.0,
                        ))
                    else:
                        rejection_counts["NEAR_SURFACE"] = (
                            rejection_counts.get("NEAR_SURFACE", 0) + 1
                        )
                        if len(rejections) < max_rejection_samples:
                            rejection_evidence = {
                                "distance_m": distance,
                                "distance_metric": "bbox_to_plane",
                                "threshold_m": threshold,
                                "surface_type": surface.surface_type,
                                "source": surface.source,
                            }
                            if config.emit_margins:
                                rejection_evidence["margin_confidence"] = \
                                    margin_confidence(_near_surface_margin(
                                        distance, threshold))
                            rejections.append(EdgeRejection(
                                source=make_entity_ref(entity.identity.object_uid),
                                type="NEAR_SURFACE",
                                target=make_surface_ref(surface.surface_uid),
                                extractor=self.name,
                                rejected_reason="distance_exceeds_surface_threshold",
                                evidence=rejection_evidence,
                            ))

        counts: dict[EdgeType, int] = {"NEAR_SURFACE": len(edges)}
        runtime_ms = int((time.perf_counter() - start) * 1000)
        return edges, RelationExtractorDiagnostics(
            extractor=self.name,
            version=effective_version,
            mode=config.mode,
            physical_edges_per_type=counts,
            physical_edges_total=len(edges),
            logical_edges_total=count_logical_edges(edges),
            rejections_per_type=rejection_counts,
            rejection_samples=rejections,
            runtime_ms=runtime_ms,
        )
