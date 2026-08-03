"""ON_SURFACE relation extractor — gravity-supported rest-contact (P4.02).

Emits ON_SURFACE(entity -> surface) edges where the entity's AABB rests on
a support-capable (up-facing) finite surface, per the Phase 4 rest-contact
predicate (Design B). The geometry lives in geometry/rest_contact.py; this
module is the extractor translation layer: policy, evidence assembly,
provenance, diagnostics.

Edge semantics:
  - distance/verdict computed by rest_contact(entity.bbox_aabb,
    entity.centroid, surface.plane, surface.polygon, frame.gravity, cfg),
  - emit iff result.on_surface (support_capable AND centroid_on_support_side
    AND contact AND footprint_ok),
  - source = entity ref, target = surface ref (same direction as NEAR_SURFACE).

Provenance / evidence on each edge:
  - extractor = "on_surface", version = "0.1",
  - evidence = rest_contact evidence
      (support_capable, centroid_on_support_side, sd_centroid_m, bottom_gap_m,
       sd_min_m, sd_max_m, in_plane_gap_m, footprint_ok, contact,
       support_normal_dot_up, up, contact_threshold_m, penetration_tolerance_m,
       max_tilt_deg, footprint_tolerance_m)
    plus near_surface_threshold_m, surface_type, source, polygon_clip_required=True.

Policy:
  - Surfaces with source == "synth_bbox_fallback" are skipped unless
    config.include_synth_fallback (mirrors NEAR_SURFACE A3/A7).
  - Surfaces with polygon is None are skipped: ON_SURFACE requires footprint
    clipping. Recorded as rejected_reason="polygon_required_for_on_surface".

Cross-relation subset guard (D4a/G8): OnSurfaceConfig validates
  hypot(contact_threshold_m, footprint_tolerance_m) <= near_surface_threshold_m
so that every ON_SURFACE edge corresponds to a polygon-mode NEAR_SURFACE edge
on the same surface. This is the layer that knows near_surface_threshold_m;
the pure geometry helper deliberately does not.

Isolation: like SurfaceProximityExtractor at P2.09, this ships extractor +
tests but is NOT wired into any default GraphBuilder run. SUPPORTS is not
emitted here — that is the P4.03 derived view (no materialized SUPPORTS edges).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Literal

from extractors.base import EntityArtifact, EntityArtifacts, StructuralSurface
from geometry.rest_contact import RestContactConfig, rest_contact
from graph.relations.base import (
    sample_rejections,
    RelationExtractorConfig, RelationExtractorDiagnostics,
    count_logical_edges, edge_frame, make_edge_id, make_entity_ref, make_surface_ref,
    margin_confidence, rest_contact_margin,
)
from graph.schema import Edge, EdgeRejection, EdgeType


ON_SURFACE_TYPES: frozenset[EdgeType] = frozenset({"ON_SURFACE"})

ON_SURFACE_VERSION = "0.1"

DEFAULT_CONTACT_THRESHOLD_M = 0.02
DEFAULT_PENETRATION_TOLERANCE_M = 0.03
DEFAULT_MAX_TILT_DEG = 30.0
DEFAULT_FOOTPRINT_TOLERANCE_M = 0.0
DEFAULT_NEAR_SURFACE_THRESHOLD_M = 0.05  # floor NEAR threshold (Design B: only up-facing surfaces qualify)


@dataclass(frozen=True)
class OnSurfaceConfig:
    """Rest-contact thresholds + the cross-relation subset guard.

    The rest-contact fields mirror RestContactConfig; near_surface_threshold_m
    is the NEAR_SURFACE proximity threshold used by the D4a/G8 guard. Under
    Design B only floor (up-facing) surfaces qualify, so the default is the
    floor NEAR threshold; the empirical G2 subset test validates against the
    actual polygon-mode NEAR_SURFACE extractor regardless of this value.
    """
    mode: Literal["sparse"] = "sparse"
    contact_threshold_m: float = DEFAULT_CONTACT_THRESHOLD_M
    penetration_tolerance_m: float = DEFAULT_PENETRATION_TOLERANCE_M
    max_tilt_deg: float = DEFAULT_MAX_TILT_DEG
    footprint_tolerance_m: float = DEFAULT_FOOTPRINT_TOLERANCE_M
    near_surface_threshold_m: float = DEFAULT_NEAR_SURFACE_THRESHOLD_M
    include_synth_fallback: bool = False
    # Calibration opt-in. When True the emitted edge's confidence is
    # margin_confidence(rest_contact_margin(evidence)) — the min over the four
    # rest-contact clause margins, whose discriminating member is the contact
    # band on bottom_gap_m: [-penetration_tolerance_m (0.03),
    # +contact_threshold_m (0.02)], normalized by the band's half-width. So a
    # box resting dead centre of the band reads ~0.98 and one 0.019 m above the
    # floor plane (1 mm inside the upper edge) reads ~0.54. False = frozen
    # behavior (confidence is literally 1.0); hash_omit_if_default keeps
    # default config hashes byte-identical.
    emit_margins: bool = field(
        default=False, metadata={"hash_omit_if_default": True})

    def __post_init__(self) -> None:
        # Delegate rest-contact field sanity to RestContactConfig (raises on
        # negative tolerances / max_tilt out of (0,90)).
        RestContactConfig(
            contact_threshold_m=self.contact_threshold_m,
            penetration_tolerance_m=self.penetration_tolerance_m,
            max_tilt_deg=self.max_tilt_deg,
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
        # D4a/G8 subset guard: ON_SURFACE must imply polygon-mode NEAR_SURFACE.
        combined = math.hypot(self.contact_threshold_m, self.footprint_tolerance_m)
        if combined > self.near_surface_threshold_m:
            raise ValueError(
                "threshold ordering violated: "
                f"hypot(contact_threshold_m={self.contact_threshold_m}, "
                f"footprint_tolerance_m={self.footprint_tolerance_m}) = {combined!r} "
                f"> near_surface_threshold_m={self.near_surface_threshold_m}. "
                "ON_SURFACE would not be a subset of polygon-mode NEAR_SURFACE."
            )
        if self.mode != "sparse":
            raise ValueError(f"unknown mode {self.mode!r}; supported: 'sparse'")


def _rest_config(config: OnSurfaceConfig) -> RestContactConfig:
    return RestContactConfig(
        contact_threshold_m=config.contact_threshold_m,
        penetration_tolerance_m=config.penetration_tolerance_m,
        max_tilt_deg=config.max_tilt_deg,
        footprint_tolerance_m=config.footprint_tolerance_m,
    )


def _build_on_surface_edge(
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
        edge_id=make_edge_id(extractor, version, source, "ON_SURFACE", target),
        source=source,
        type="ON_SURFACE",
        target=target,
        frame=frame,
        weight=1.0,
        confidence=confidence,
        extractor=extractor,
        extractor_version=version,
        evidence=evidence,
    )


def _evidence_for(
    result_evidence: dict,
    surface: StructuralSurface,
    config: OnSurfaceConfig,
) -> dict:
    evidence = dict(result_evidence)
    # Edge evidence is JSON-serialized into the bundle, so it must be
    # JSON-native: rest_contact returns `up` as a tuple, which would
    # round-trip to a list and break dump==load equality. Normalize to a
    # list here (the edge-evidence assembly boundary) so both emitted and
    # rejected evidence serialize faithfully.
    if isinstance(evidence.get("up"), tuple):
        evidence["up"] = list(evidence["up"])
    evidence["near_surface_threshold_m"] = config.near_surface_threshold_m
    evidence["surface_type"] = surface.surface_type
    evidence["source"] = surface.source
    evidence["polygon_clip_required"] = True
    return evidence


class OnSurfaceExtractor:
    name: str = "on_surface"
    version: str = ON_SURFACE_VERSION
    edge_types: frozenset[EdgeType] = ON_SURFACE_TYPES

    def extract(
        self,
        entities: EntityArtifacts,
        config: RelationExtractorConfig,
    ) -> tuple[list[Edge], RelationExtractorDiagnostics]:
        if not isinstance(config, OnSurfaceConfig):
            raise TypeError(
                f"OnSurfaceExtractor requires OnSurfaceConfig, "
                f"got {type(config).__name__}"
            )
        start = time.perf_counter()
        rest_cfg = _rest_config(config)
        gravity = entities.frame.gravity
        frame = edge_frame(entities)
        edges: list[Edge] = []
        rejections: list[EdgeRejection] = []
        rejection_counts: dict[EdgeType, int] = {}
        max_rejection_samples = 64

        def record_rejection(entity, surface, reason: str, evidence: dict) -> None:
            rejection_counts["ON_SURFACE"] = (
                rejection_counts.get("ON_SURFACE", 0) + 1
            )
            if config.emit_margins or len(rejections) < max_rejection_samples:
                rejections.append(EdgeRejection(
                    source=make_entity_ref(entity.identity.object_uid),
                    type="ON_SURFACE",
                    target=make_surface_ref(surface.surface_uid),
                    extractor=self.name,
                    rejected_reason=reason,
                    evidence=evidence,
                ))

        # Partition surfaces: synth-fallback skip (A3/A7), polygon-required.
        active_surfaces: list[StructuralSurface] = []
        excluded_synth: list[StructuralSurface] = []
        no_polygon: list[StructuralSurface] = []
        for surface in entities.structural_surfaces:
            if surface.source == "synth_bbox_fallback" and not config.include_synth_fallback:
                excluded_synth.append(surface)
            elif surface.polygon is None:
                no_polygon.append(surface)
            else:
                active_surfaces.append(surface)

        for entity in entities.entities:
            for surface in excluded_synth:
                record_rejection(entity, surface, "surface_source_excluded", {
                    "source": surface.source,
                    "include_synth_fallback": False,
                    "policy": "on_surface_excludes_synth_bbox_fallback",
                })
            for surface in no_polygon:
                record_rejection(entity, surface, "polygon_required_for_on_surface", {
                    "source": surface.source,
                    "surface_type": surface.surface_type,
                    "polygon_clip_required": True,
                    "policy": "on_surface_requires_polygon_footprint",
                })
            for surface in active_surfaces:
                result = rest_contact(
                    entity.bbox_aabb, entity.centroid,
                    surface.plane, surface.polygon, gravity, rest_cfg,
                )
                evidence = _evidence_for(result.evidence, surface, config)
                margin = (rest_contact_margin(result.evidence)
                          if config.emit_margins else None)
                if result.on_surface:
                    edges.append(_build_on_surface_edge(
                        entity, surface, evidence,
                        extractor=self.name, version=self.version, frame=frame,
                        confidence=1.0 if margin is None
                        else margin_confidence(margin),
                    ))
                else:
                    rej = dict(evidence)
                    rej["failed_clauses"] = result.failed_clauses
                    # Near-misses are half the calibration data, but
                    # EdgeRejection has no confidence field, so the margin
                    # rides in evidence. Policy rejections above (synth
                    # source, missing polygon) measure nothing and get none.
                    if margin is not None:
                        rej["margin_confidence"] = margin_confidence(margin)
                    record_rejection(
                        entity, surface, "rest_contact_clauses_failed", rej,
                    )

        counts: dict[EdgeType, int] = {"ON_SURFACE": len(edges)}
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
