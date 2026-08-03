"""CONTACTS_SURFACE relation extractor — wall contact (P5.02).

Emits CONTACTS_SURFACE(entity -> surface) edges where an entity is in
normal-side contact with a WALL surface, per the Phase 5 wall-contact
predicate (geometry/wall_contact.py). The direct analog of the ON_SURFACE
extractor, with the orientation gate flipped (wall, not floor).

Edge semantics:
  - verdict computed by wall_contact(entity.bbox_aabb, entity.centroid,
    surface.plane, surface.polygon, frame.gravity, cfg),
  - emit iff result.contacts_surface,
  - source = entity ref, target = surface ref (same direction as NEAR_SURFACE).

Provenance / evidence:
  - extractor = "contacts_surface", version = "0.1",
  - evidence = wall_contact evidence (wall_capable, on_interior_side,
    sd_centroid_m, wall_gap_m, sd_min_m, sd_max_m, in_plane_gap_m,
    footprint_ok, contact, wall_normal_dot_up, up, contact_threshold_m,
    penetration_tolerance_m, max_wall_tilt_deg, footprint_tolerance_m)
    plus near_surface_threshold_m, surface_type, source, polygon_clip_required=True.
    `up` is normalized tuple -> list at this boundary (the P4.06 serde lesson).

Policy (explicit, semantic-layer):
  - The relation is "contact with WALL-like surfaces". Surfaces with
    surface_type != "wall" are skipped (rejected_reason="surface_type_not_wall")
    -- floors/ceilings are NOT run through wall_capable here; the extractor
    is wall-scoped by intent, not just by the geometry gate.
  - source == "synth_bbox_fallback" skipped unless include_synth_fallback
    (mirrors NEAR_SURFACE / ON_SURFACE).
  - polygon is None skipped (footprint clipping required):
    rejected_reason="polygon_required_for_contacts_surface".
  - opt-in (Phase 8 F4, exclude_room_scale_flat=True): room-scale flat
    entities (wall-to-wall rugs: footprint >= room_scale_flat_min_area_frac
    of the wall-bounded XY extent AND height <= room_scale_flat_max_height_m)
    are never candidates (rejected_reason="room_scale_flat_excluded") — a rug
    spanning the room touches every wall geometrically, but "against the
    wall" never means the floor covering. Default off = frozen P5 behavior.

Cross-relation subset guard (D-subset): ContactsSurfaceConfig validates
  hypot(contact_threshold_m, footprint_tolerance_m) <= near_surface_threshold_m
so every CONTACTS_SURFACE edge corresponds to a polygon-mode NEAR_SURFACE
edge on the same wall surface.

Isolation: ships extractor + tests, NOT wired into any default GraphBuilder
run (same as ON_SURFACE / NEAR_SURFACE). ATTACHED_TO is NOT emitted (deferred
in P5; "against" != "attached").
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Literal

from extractors.base import EntityArtifact, EntityArtifacts, StructuralSurface
from geometry.wall_contact import WallContactConfig, wall_contact
from graph.relations.base import (
    sample_rejections,
    RelationExtractorConfig, RelationExtractorDiagnostics,
    count_logical_edges, edge_frame, make_edge_id, make_entity_ref, make_surface_ref,
    margin_confidence, room_scale_flat, wall_contact_margin,
    wall_xy_extent_area,
)
from graph.schema import Edge, EdgeRejection, EdgeType


CONTACTS_SURFACE_TYPES: frozenset[EdgeType] = frozenset({"CONTACTS_SURFACE"})

CONTACTS_SURFACE_VERSION = "0.1"

DEFAULT_CONTACT_THRESHOLD_M = 0.02
DEFAULT_PENETRATION_TOLERANCE_M = 0.02
DEFAULT_MAX_WALL_TILT_DEG = 30.0
DEFAULT_FOOTPRINT_TOLERANCE_M = 0.0
DEFAULT_NEAR_SURFACE_THRESHOLD_M = 0.30  # Phase 2 wall NEAR threshold

# Phase 8 F4 (opt-in): room-scale flat exclusion defaults. Chosen from the
# six-scene measurement (2026-07-12): the only true wall-to-wall flat object
# is office_0's rug at area_frac 0.975; the next-largest flat object anywhere
# is room_1's rug at 0.483, so 0.60 has wide margins on both sides. 0.20 m
# keeps tall-but-large furniture (room_1's bed: frac 0.911, height 1.27 m).
DEFAULT_ROOM_SCALE_FLAT_MIN_AREA_FRAC = 0.60
DEFAULT_ROOM_SCALE_FLAT_MAX_HEIGHT_M = 0.20


@dataclass(frozen=True)
class ContactsSurfaceConfig:
    """Wall-contact thresholds + the cross-relation subset guard. The
    wall-contact fields mirror WallContactConfig; near_surface_threshold_m
    is the wall NEAR_SURFACE proximity threshold used by the subset guard."""
    mode: Literal["sparse"] = "sparse"
    contact_threshold_m: float = DEFAULT_CONTACT_THRESHOLD_M
    penetration_tolerance_m: float = DEFAULT_PENETRATION_TOLERANCE_M
    max_wall_tilt_deg: float = DEFAULT_MAX_WALL_TILT_DEG
    footprint_tolerance_m: float = DEFAULT_FOOTPRINT_TOLERANCE_M
    near_surface_threshold_m: float = DEFAULT_NEAR_SURFACE_THRESHOLD_M
    include_synth_fallback: bool = False
    # Phase 8 F4 (opt-in): skip room-scale flat entities (wall-to-wall rugs)
    # as CONTACTS_SURFACE candidates. Default False = frozen P5 behavior;
    # hash_omit_if_default keeps default config hashes byte-identical. When
    # enabling, enable the SAME fields on SurfaceProximityConfig too, or the
    # NEAR⊇CONTACTS subset guarantee weakens to "excluded pairs missing from
    # NEAR only" (enabling here alone keeps the subset direction intact).
    exclude_room_scale_flat: bool = field(
        default=False, metadata={"hash_omit_if_default": True})
    room_scale_flat_min_area_frac: float = field(
        default=DEFAULT_ROOM_SCALE_FLAT_MIN_AREA_FRAC,
        metadata={"hash_omit_if_default": True})
    room_scale_flat_max_height_m: float = field(
        default=DEFAULT_ROOM_SCALE_FLAT_MAX_HEIGHT_M,
        metadata={"hash_omit_if_default": True})
    # Calibration opt-in. When True the emitted edge's confidence is
    # margin_confidence(wall_contact_margin(evidence)) — the min over the four
    # wall-contact clause margins, whose discriminating member is the contact
    # band on wall_gap_m: [-penetration_tolerance_m (0.02),
    # +contact_threshold_m (0.02)], normalized by the band's half-width (0.02).
    # An object flush against the wall reads ~0.98, one 0.018 m off it ~0.55.
    # NOTE this is the CONTACT band, not the 0.20 m room_scale_flat height cut
    # or the 0.30 m wall NEAR threshold; those two are policy/superset gates
    # and do not enter the score. False = frozen P5 behavior (confidence is
    # literally 1.0); hash_omit_if_default keeps config hashes byte-identical.
    emit_margins: bool = field(
        default=False, metadata={"hash_omit_if_default": True})

    def __post_init__(self) -> None:
        # Delegate wall-contact field sanity to WallContactConfig.
        WallContactConfig(
            contact_threshold_m=self.contact_threshold_m,
            penetration_tolerance_m=self.penetration_tolerance_m,
            max_wall_tilt_deg=self.max_wall_tilt_deg,
            footprint_tolerance_m=self.footprint_tolerance_m,
        )
        if (
            not math.isfinite(self.near_surface_threshold_m)
            or self.near_surface_threshold_m < 0.0
        ):
            raise ValueError(
                "near_surface_threshold_m must be finite and non-negative, "
                f"got {self.near_surface_threshold_m!r}"
            )
        combined = math.hypot(self.contact_threshold_m, self.footprint_tolerance_m)
        if combined > self.near_surface_threshold_m:
            raise ValueError(
                "threshold ordering violated: "
                f"hypot(contact_threshold_m={self.contact_threshold_m}, "
                f"footprint_tolerance_m={self.footprint_tolerance_m}) = {combined!r} "
                f"> near_surface_threshold_m={self.near_surface_threshold_m}. "
                "CONTACTS_SURFACE would not be a subset of polygon-mode NEAR_SURFACE."
            )
        if (
            not math.isfinite(self.room_scale_flat_min_area_frac)
            or self.room_scale_flat_min_area_frac <= 0.0
        ):
            raise ValueError(
                "room_scale_flat_min_area_frac must be finite and positive, "
                f"got {self.room_scale_flat_min_area_frac!r}"
            )
        if (
            not math.isfinite(self.room_scale_flat_max_height_m)
            or self.room_scale_flat_max_height_m < 0.0
        ):
            raise ValueError(
                "room_scale_flat_max_height_m must be finite and non-negative, "
                f"got {self.room_scale_flat_max_height_m!r}"
            )
        if self.mode != "sparse":
            raise ValueError(f"unknown mode {self.mode!r}; supported: 'sparse'")


def _wall_config(config: ContactsSurfaceConfig) -> WallContactConfig:
    return WallContactConfig(
        contact_threshold_m=config.contact_threshold_m,
        penetration_tolerance_m=config.penetration_tolerance_m,
        max_wall_tilt_deg=config.max_wall_tilt_deg,
        footprint_tolerance_m=config.footprint_tolerance_m,
    )


def _evidence_for(
    result_evidence: dict,
    surface: StructuralSurface,
    config: ContactsSurfaceConfig,
) -> dict:
    evidence = dict(result_evidence)
    # Edge evidence is JSON-serialized; normalize the `up` tuple to a list so
    # dump==load holds (the P4.06 round-trip lesson).
    if isinstance(evidence.get("up"), tuple):
        evidence["up"] = list(evidence["up"])
    evidence["near_surface_threshold_m"] = config.near_surface_threshold_m
    evidence["surface_type"] = surface.surface_type
    evidence["source"] = surface.source
    evidence["polygon_clip_required"] = True
    return evidence


def _build_edge(
    entity: EntityArtifact,
    surface: StructuralSurface,
    evidence: dict,
    *,
    extractor: str,
    version: str,
    frame: FrameKind,
    confidence: float = 1.0,
) -> Edge:
    source = make_entity_ref(entity.identity.object_uid)
    target = make_surface_ref(surface.surface_uid)
    return Edge(
        edge_id=make_edge_id(extractor, version, source, "CONTACTS_SURFACE", target),
        source=source,
        type="CONTACTS_SURFACE",
        target=target,
        frame=frame,
        weight=1.0,
        confidence=confidence,
        extractor=extractor,
        extractor_version=version,
        evidence=evidence,
    )


class ContactsSurfaceExtractor:
    name: str = "contacts_surface"
    version: str = CONTACTS_SURFACE_VERSION
    edge_types: frozenset[EdgeType] = CONTACTS_SURFACE_TYPES

    def extract(
        self,
        entities: EntityArtifacts,
        config: RelationExtractorConfig,
    ) -> tuple[list[Edge], RelationExtractorDiagnostics]:
        if not isinstance(config, ContactsSurfaceConfig):
            raise TypeError(
                f"ContactsSurfaceExtractor requires ContactsSurfaceConfig, "
                f"got {type(config).__name__}"
            )
        start = time.perf_counter()
        wall_cfg = _wall_config(config)
        gravity = entities.frame.gravity
        frame = edge_frame(entities)
        edges: list[Edge] = []
        rejections: list[EdgeRejection] = []
        rejection_counts: dict[EdgeType, int] = {}
        max_rejection_samples = 64

        def record_rejection(entity, surface, reason: str, evidence: dict) -> None:
            rejection_counts["CONTACTS_SURFACE"] = (
                rejection_counts.get("CONTACTS_SURFACE", 0) + 1
            )
            if config.emit_margins or len(rejections) < max_rejection_samples:
                rejections.append(EdgeRejection(
                    source=make_entity_ref(entity.identity.object_uid),
                    type="CONTACTS_SURFACE",
                    target=make_surface_ref(surface.surface_uid),
                    extractor=self.name,
                    rejected_reason=reason,
                    evidence=evidence,
                ))

        # Partition surfaces: wall-only (semantic policy), then synth-skip,
        # then polygon-required.
        active: list[StructuralSurface] = []
        non_wall: list[StructuralSurface] = []
        excluded_synth: list[StructuralSurface] = []
        no_polygon: list[StructuralSurface] = []
        for surface in entities.structural_surfaces:
            if surface.surface_type != "wall":
                non_wall.append(surface)
            elif surface.source == "synth_bbox_fallback" and not config.include_synth_fallback:
                excluded_synth.append(surface)
            elif surface.polygon is None:
                no_polygon.append(surface)
            else:
                active.append(surface)

        # Phase 8 F4 (opt-in): room-scale flat entities (wall-to-wall rugs)
        # are never CONTACTS_SURFACE candidates. Room area computed once.
        room_xy_area: float | None = None
        if config.exclude_room_scale_flat:
            room_xy_area = wall_xy_extent_area(entities.structural_surfaces)

        for entity in entities.entities:
            for surface in non_wall:
                record_rejection(entity, surface, "surface_type_not_wall", {
                    "surface_type": surface.surface_type,
                    "source": surface.source,
                    "policy": "contacts_surface_evaluates_wall_surfaces_only",
                })
            for surface in excluded_synth:
                record_rejection(entity, surface, "surface_source_excluded", {
                    "source": surface.source,
                    "include_synth_fallback": False,
                    "policy": "contacts_surface_excludes_synth_bbox_fallback",
                })
            for surface in no_polygon:
                record_rejection(entity, surface, "polygon_required_for_contacts_surface", {
                    "source": surface.source,
                    "surface_type": surface.surface_type,
                    "polygon_clip_required": True,
                    "policy": "contacts_surface_requires_polygon_footprint",
                })
            if config.exclude_room_scale_flat:
                flat, flat_evidence = room_scale_flat(
                    entity, room_xy_area,
                    min_area_frac=config.room_scale_flat_min_area_frac,
                    max_height_m=config.room_scale_flat_max_height_m,
                )
                if flat:
                    for surface in active:
                        record_rejection(
                            entity, surface, "room_scale_flat_excluded",
                            dict(flat_evidence) | {
                                "surface_type": surface.surface_type,
                                "source": surface.source,
                                "policy": "room_scale_flat_never_a_wall_contact_candidate",
                            })
                    continue
            for surface in active:
                result = wall_contact(
                    entity.bbox_aabb, entity.centroid,
                    surface.plane, surface.polygon, gravity, wall_cfg,
                )
                evidence = _evidence_for(result.evidence, surface, config)
                margin = (wall_contact_margin(result.evidence)
                          if config.emit_margins else None)
                if result.contacts_surface:
                    edges.append(_build_edge(
                        entity, surface, evidence,
                        extractor=self.name, version=self.version, frame=frame,
                        confidence=1.0 if margin is None
                        else margin_confidence(margin),
                    ))
                else:
                    rej = dict(evidence)
                    rej["failed_clauses"] = result.failed_clauses
                    # EdgeRejection has no confidence field; the near-miss
                    # margin rides in evidence. The policy rejections above
                    # (non-wall, synth source, no polygon, room-scale flat)
                    # measure nothing comparable and get none.
                    if margin is not None:
                        rej["margin_confidence"] = margin_confidence(margin)
                    record_rejection(
                        entity, surface, "wall_contact_clauses_failed", rej,
                    )

        counts: dict[EdgeType, int] = {"CONTACTS_SURFACE": len(edges)}
        runtime_ms = int((time.perf_counter() - start) * 1000)
        return edges, RelationExtractorDiagnostics(
            extractor=self.name,
            version=self.version,
            mode=config.mode,
            physical_edges_per_type=counts,
            physical_edges_total=len(edges),
            logical_edges_total=count_logical_edges(edges),
            rejections_per_type=rejection_counts,
            rejection_samples=sample_rejections(
                rejections, margin_aware=config.emit_margins),
            runtime_ms=runtime_ms,
        )
