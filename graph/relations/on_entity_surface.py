"""ON_ENTITY_SURFACE relation extractor — entity-on-entity support (P6).

Emits stored entity -> entity edges where a portable entity rests on the
derived top face of a support-capable owner entity. The derived top face is
recorded in edge evidence; it is not a structural surface and is never
referenced via GraphRef(kind="surface").
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from common.serde import plane_to_dict, vec3_to_list
from common.types import FrameKind
from extractors.base import EntityArtifact, EntityArtifacts
from extractors.entity_surfaces import (
    DEFAULT_SUPPORT_CLASS_ALLOWLIST,
    EntitySurface,
    derive_entity_top_surfaces,
    support_class_for_entity,
)
from geometry.rest_contact import RestContactConfig, rest_contact
from graph.relations.base import (
    sample_rejections,
    RelationExtractorConfig, RelationExtractorDiagnostics,
    count_logical_edges, edge_frame, make_edge_id, make_entity_ref, margin_confidence,
    rest_contact_margin,
)
from graph.relations.on_surface import (
    DEFAULT_CONTACT_THRESHOLD_M,
    DEFAULT_FOOTPRINT_TOLERANCE_M,
    DEFAULT_MAX_TILT_DEG,
    DEFAULT_PENETRATION_TOLERANCE_M,
)
from graph.schema import Edge, EdgeRejection, EdgeType


ON_ENTITY_SURFACE_TYPES: frozenset[EdgeType] = frozenset({"ON_ENTITY_SURFACE"})
ON_ENTITY_SURFACE_VERSION = "0.1"


@dataclass(frozen=True)
class OnEntitySurfaceConfig:
    """Entity-top rest-contact thresholds and support-class policy."""
    mode: Literal["sparse"] = "sparse"
    contact_threshold_m: float = DEFAULT_CONTACT_THRESHOLD_M
    penetration_tolerance_m: float = DEFAULT_PENETRATION_TOLERANCE_M
    max_tilt_deg: float = DEFAULT_MAX_TILT_DEG
    footprint_tolerance_m: float = DEFAULT_FOOTPRINT_TOLERANCE_M
    support_class_allowlist: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_SUPPORT_CLASS_ALLOWLIST
    )
    # Calibration opt-in. Same predicate and therefore the same normalization
    # as ON_SURFACE (rest_contact_margin: min over the four clause margins,
    # discriminated by the contact band on bottom_gap_m), only the plane is a
    # DERIVED entity top rather than a structural surface — so these
    # confidences also inherit that top's estimation error, which the
    # structural-surface case does not have. False = frozen P6 behavior
    # (confidence is literally 1.0); hash_omit_if_default keeps default config
    # hashes byte-identical.
    emit_margins: bool = field(
        default=False, metadata={"hash_omit_if_default": True})

    def __post_init__(self) -> None:
        RestContactConfig(
            contact_threshold_m=self.contact_threshold_m,
            penetration_tolerance_m=self.penetration_tolerance_m,
            max_tilt_deg=self.max_tilt_deg,
            footprint_tolerance_m=self.footprint_tolerance_m,
        )
        if self.mode != "sparse":
            raise ValueError(f"unknown mode {self.mode!r}; supported: 'sparse'")
        if not self.support_class_allowlist:
            raise ValueError("support_class_allowlist must not be empty")


def _rest_config(config: OnEntitySurfaceConfig) -> RestContactConfig:
    return RestContactConfig(
        contact_threshold_m=config.contact_threshold_m,
        penetration_tolerance_m=config.penetration_tolerance_m,
        max_tilt_deg=config.max_tilt_deg,
        footprint_tolerance_m=config.footprint_tolerance_m,
    )


def _surface_evidence(surface: EntitySurface) -> dict:
    return {
        "entity_surface_uid": surface.surface_uid,
        "owner_entity_uid": surface.owner_entity_uid,
        "owner_class": surface.owner_class,
        "entity_surface_source": surface.source,
        "entity_surface_confidence": surface.confidence,
        "entity_surface_plane": plane_to_dict(surface.plane),
        "entity_surface_polygon": [vec3_to_list(v) for v in surface.polygon],
    }


def _evidence_for(
    result_evidence: dict,
    surface: EntitySurface,
    config: OnEntitySurfaceConfig,
) -> dict:
    evidence = dict(result_evidence)
    if isinstance(evidence.get("up"), tuple):
        evidence["up"] = list(evidence["up"])
    evidence.update(_surface_evidence(surface))
    evidence["polygon_clip_required"] = True
    evidence["support_class_allowlist"] = list(config.support_class_allowlist)
    return evidence


def _build_edge(
    entity: EntityArtifact,
    surface: EntitySurface,
    evidence: dict,
    *,
    extractor: str,
    version: str,
    frame: FrameKind,
    confidence: float = 1.0,
) -> Edge:
    source = make_entity_ref(entity.identity.object_uid)
    target = make_entity_ref(surface.owner_entity_uid)
    if target.uid != evidence.get("owner_entity_uid"):
        raise AssertionError(
            "ON_ENTITY_SURFACE target/evidence invariant failed: "
            f"{target.uid!r} != {evidence.get('owner_entity_uid')!r}"
        )
    return Edge(
        edge_id=make_edge_id(extractor, version, source, "ON_ENTITY_SURFACE", target),
        source=source,
        type="ON_ENTITY_SURFACE",
        target=target,
        frame=frame,
        weight=1.0,
        confidence=confidence,
        extractor=extractor,
        extractor_version=version,
        evidence=evidence,
    )


class OnEntitySurfaceExtractor:
    name: str = "on_entity_surface"
    version: str = ON_ENTITY_SURFACE_VERSION
    edge_types: frozenset[EdgeType] = ON_ENTITY_SURFACE_TYPES

    def extract(
        self,
        entities: EntityArtifacts,
        config: RelationExtractorConfig,
    ) -> tuple[list[Edge], RelationExtractorDiagnostics]:
        if not isinstance(config, OnEntitySurfaceConfig):
            raise TypeError(
                f"OnEntitySurfaceExtractor requires OnEntitySurfaceConfig, "
                f"got {type(config).__name__}"
            )

        start = time.perf_counter()
        rest_cfg = _rest_config(config)
        gravity = entities.frame.gravity
        frame = edge_frame(entities)
        entity_surfaces = derive_entity_top_surfaces(
            entities.entities,
            frame=entities.frame,
            support_class_allowlist=config.support_class_allowlist,
        )
        edges: list[Edge] = []
        rejections: list[EdgeRejection] = []
        rejection_counts: dict[EdgeType, int] = {}
        max_rejection_samples = 64

        def record_rejection(
            entity: EntityArtifact,
            surface: EntitySurface,
            reason: str,
            evidence: dict,
        ) -> None:
            rejection_counts["ON_ENTITY_SURFACE"] = (
                rejection_counts.get("ON_ENTITY_SURFACE", 0) + 1
            )
            if config.emit_margins or len(rejections) < max_rejection_samples:
                rejections.append(EdgeRejection(
                    source=make_entity_ref(entity.identity.object_uid),
                    type="ON_ENTITY_SURFACE",
                    target=make_entity_ref(surface.owner_entity_uid),
                    extractor=self.name,
                    rejected_reason=reason,
                    evidence=evidence,
                ))

        for entity in entities.entities:
            supported_class = support_class_for_entity(
                entity, config.support_class_allowlist
            )
            for surface in entity_surfaces:
                if entity.identity.object_uid == surface.owner_entity_uid:
                    record_rejection(
                        entity, surface, "self_support_excluded",
                        {**_surface_evidence(surface), "policy": "owner cannot rest on its own derived top"},
                    )
                    continue
                if supported_class is not None:
                    record_rejection(
                        entity, surface, "supported_class_excluded",
                        {
                            **_surface_evidence(surface),
                            "supported_entity_uid": entity.identity.object_uid,
                            "supported_class": supported_class,
                            "policy": (
                                "support-class furniture owners are excluded as "
                                "supported tabletop objects in P6"
                            ),
                        },
                    )
                    continue
                result = rest_contact(
                    entity.bbox_aabb,
                    entity.centroid,
                    surface.plane,
                    surface.polygon,
                    gravity,
                    rest_cfg,
                )
                evidence = _evidence_for(result.evidence, surface, config)
                margin = (rest_contact_margin(result.evidence)
                          if config.emit_margins else None)
                if result.on_surface:
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
                    # margin rides in evidence. The self-support and
                    # supported-class rejections above are policy, not
                    # measurement, and get none.
                    if margin is not None:
                        rej["margin_confidence"] = margin_confidence(margin)
                    record_rejection(
                        entity, surface, "rest_contact_clauses_failed", rej,
                    )

        counts: dict[EdgeType, int] = {"ON_ENTITY_SURFACE": len(edges)}
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
