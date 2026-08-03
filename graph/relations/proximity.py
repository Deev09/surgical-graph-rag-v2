"""Proximity relation extractor (compat + sparse v1 + sparse v2).

Compat: faithful port of relations/compute.py NEAR logic.
  - LEGACY_NEAR_THRESHOLD = 1.0 (centroid distance, strict less-than)
  - Each unordered pair within threshold contributes TWO edges:
    NEAR(a, b) and NEAR(b, a). This duplication is preserved exactly
    for legacy reproduction.

Sparse v1 (default): centroid-to-centroid distance. Emits NEAR ONCE per
unordered pair within threshold with evidence flag symmetric=True.
**Byte-frozen** — do not modify. Phase 1 compat reproduction and the
sparse-density gate depend on this code path.

Sparse v2 (opt-in via ProximityConfig.sparse_version=2): exact
AABB-to-AABB surface distance via geometry.surface_distance.
aabb_to_aabb_surface (P2.07). Physically separate function
(extract_sparse_v2) so the v1 path cannot drift. Edges carry
distance_metric="aabb_surface", sparse_version=2 in evidence, and the
truthful extractor_version SPARSE_V2_VERSION.

FAR is intentionally not emitted by any mode. FAR is a query-time
operator over a centroid index (phase0_design.md §6).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from math import sqrt
from typing import Literal

from common.types import FrameKind
from extractors.base import EntityArtifact, EntityArtifacts
from geometry.surface_distance import aabb_to_aabb_surface
from graph.relations.base import (
    sample_rejections,
    RelationExtractorConfig, RelationExtractorDiagnostics,
    count_logical_edges, edge_frame, make_edge_id, make_entity_ref,
    margin_confidence, ratio_margin,
)
from graph.schema import Edge, EdgeRejection, EdgeType


LEGACY_NEAR_THRESHOLD = 1.0   # frozen for compat; do not change
SPARSE_V2_VERSION = "0.2-sparse_v2"   # truthful v2 algorithm version

PROXIMITY_TYPES: frozenset[EdgeType] = frozenset({"NEAR"})


@dataclass(frozen=True)
class ProximityConfig:
    mode: Literal["compat", "sparse"]
    sparse_near_threshold: float = 1.0   # threshold; interpretation depends on sparse_version
    sparse_version: Literal[1, 2] = field(
        default=1,
        metadata={"hash_omit_if_default": True},
    )   # 1 = centroid (Phase 1, byte-frozen); 2 = aabb_surface
    # Calibration opt-in. False = frozen behavior (confidence is literally 1.0);
    # hash_omit_if_default keeps default config hashes byte-identical.
    emit_margins: bool = field(
        default=False, metadata={"hash_omit_if_default": True})


def _euclid_3d(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _build_near(
    src: EntityArtifact,
    tgt: EntityArtifact,
    *,
    extractor: str,
    version: str,
    frame: FrameKind,
    evidence: dict,
    confidence: float = 1.0,
) -> Edge:
    source = make_entity_ref(src.identity.object_uid)
    target = make_entity_ref(tgt.identity.object_uid)
    return Edge(
        edge_id=make_edge_id(extractor, version, source, "NEAR", target),
        source=source,
        type="NEAR",
        target=target,
        frame=frame,
        weight=1.0,
        confidence=confidence,
        extractor=extractor,
        extractor_version=version,
        evidence=evidence,
    )


# ----- compat path -----

def extract_compat(
    entities: list[EntityArtifact],
    *,
    extractor: str,
    version: str,
    frame: FrameKind,
    emit_margins: bool = False,
) -> tuple[list[Edge], dict[EdgeType, int]]:
    """Legacy NEAR. Emits both NEAR(a,b) and NEAR(b,a) per unordered
    pair within LEGACY_NEAR_THRESHOLD. Iteration order matches the
    legacy compute_relations loop.

    Continuous quantity: centroid distance, thresholded at
    LEGACY_NEAR_THRESHOLD. Margin = (threshold - distance) / threshold —
    the threshold IS the relation's length scale ("near" is defined by it),
    so a coincident pair reads 1.0, a pair at half the threshold ~0.88, and
    a pair exactly on it 0.5. Both physical edges of the pair share the one
    measurement, hence the one confidence."""
    edges: list[Edge] = []
    counts: dict[EdgeType, int] = {"NEAR": 0}
    n = len(entities)
    for i in range(n):
        for j in range(i + 1, n):
            a = entities[i]
            b = entities[j]
            distance = _euclid_3d(a.centroid, b.centroid)
            if distance < LEGACY_NEAR_THRESHOLD:
                evidence = {"distance_m": distance}
                confidence = 1.0
                if emit_margins:
                    confidence = margin_confidence(
                        ratio_margin(distance, LEGACY_NEAR_THRESHOLD))
                edges.append(_build_near(a, b, extractor=extractor, version=version, frame=frame, evidence=evidence, confidence=confidence))
                edges.append(_build_near(b, a, extractor=extractor, version=version, frame=frame, evidence=evidence, confidence=confidence))
                counts["NEAR"] += 2
    return edges, counts


# ----- sparse path -----

def extract_sparse(
    entities: list[EntityArtifact],
    *,
    extractor: str,
    version: str,
    frame: FrameKind,
    sparse_near_threshold: float,
    emit_margins: bool = False,
) -> tuple[list[Edge], dict[EdgeType, int], list[EdgeRejection], dict[EdgeType, int]]:
    """Sparse NEAR. Emits ONCE per unordered pair within threshold with
    evidence symmetric=True.

    Continuous quantity: centroid distance vs sparse_near_threshold; margin =
    (threshold - distance) / threshold, as in extract_compat. Rejected pairs
    carry the same (now negative) margin in evidence["margin_confidence"] —
    a pair just past the threshold is the most informative near-miss there
    is, and EdgeRejection has no confidence field to put it in."""
    edges: list[Edge] = []
    counts: dict[EdgeType, int] = {"NEAR": 0}
    rejections: list[EdgeRejection] = []
    rejection_counts: dict[EdgeType, int] = {}
    max_rejection_samples = 64
    n = len(entities)
    for i in range(n):
        for j in range(i + 1, n):
            a = entities[i]
            b = entities[j]
            distance = _euclid_3d(a.centroid, b.centroid)
            if distance < sparse_near_threshold:
                confidence = 1.0
                if emit_margins:
                    confidence = margin_confidence(
                        ratio_margin(distance, sparse_near_threshold))
                edges.append(_build_near(
                    a, b, extractor=extractor, version=version, frame=frame,
                    evidence={"distance_m": distance, "symmetric": True},
                    confidence=confidence,
                ))
                counts["NEAR"] += 1
            else:
                rejection_counts["NEAR"] = rejection_counts.get("NEAR", 0) + 1
                if emit_margins or len(rejections) < max_rejection_samples:
                    rej_evidence = {
                        "distance_m": distance,
                        "sparse_near_threshold": sparse_near_threshold,
                    }
                    if emit_margins:
                        rej_evidence["margin_confidence"] = margin_confidence(
                            ratio_margin(distance, sparse_near_threshold))
                    rejections.append(EdgeRejection(
                        source=make_entity_ref(a.identity.object_uid),
                        type="NEAR",
                        target=make_entity_ref(b.identity.object_uid),
                        extractor=extractor,
                        rejected_reason="distance_exceeds_sparse_near_threshold",
                        evidence=rej_evidence,
                    ))
    return edges, counts, rejections, rejection_counts


# ----- sparse v2 path (opt-in, physically separate from v1) -----

def extract_sparse_v2(
    entities: list[EntityArtifact],
    *,
    extractor: str,
    frame: FrameKind,
    sparse_near_threshold: float,
    emit_margins: bool = False,
) -> tuple[list[Edge], dict[EdgeType, int], list[EdgeRejection], dict[EdgeType, int]]:
    """Sparse NEAR using exact AABB-to-AABB surface distance (P2.07 A4).

    Margin normalization is identical to v1 — (threshold - distance) /
    threshold — only the metric under it changes (aabb_surface, not
    centroid). Surface distance is bounded above by centroid distance and
    bottoms out at exactly 0 for touching/overlapping boxes, so v2
    confidences pile up harder at the top of the range than v1's do.
    Emits ONCE per unordered pair, symmetric=True. Edge provenance:
      - extractor_version = SPARSE_V2_VERSION ("0.2-sparse_v2")
      - evidence["distance_metric"] = "aabb_surface"
      - evidence["sparse_version"] = 2

    Threshold semantics: strict-less-than against the surface-to-surface
    distance (not centroid-to-centroid). Per A2 the surface distance is
    bounded above by the centroid distance, so v2 typically yields more
    edges than v1 at the same threshold. Do not retune silently — log
    telemetry first.

    Requires every entity to have a populated bbox_aabb (Phase 1 default
    AABBs work for the math; the v2 enriched bundle provides tight
    world-frame AABBs)."""
    edges: list[Edge] = []
    counts: dict[EdgeType, int] = {"NEAR": 0}
    rejections: list[EdgeRejection] = []
    rejection_counts: dict[EdgeType, int] = {}
    max_rejection_samples = 64
    n = len(entities)
    for i in range(n):
        for j in range(i + 1, n):
            a = entities[i]
            b = entities[j]
            distance = aabb_to_aabb_surface(a.bbox_aabb, b.bbox_aabb)
            if distance < sparse_near_threshold:
                confidence = 1.0
                if emit_margins:
                    confidence = margin_confidence(
                        ratio_margin(distance, sparse_near_threshold))
                edges.append(_build_near(
                    a, b, extractor=extractor, version=SPARSE_V2_VERSION,
                    frame=frame,
                    evidence={
                        "distance_m": distance,
                        "distance_metric": "aabb_surface",
                        "sparse_version": 2,
                        "symmetric": True,
                    },
                    confidence=confidence,
                ))
                counts["NEAR"] += 1
            else:
                rejection_counts["NEAR"] = rejection_counts.get("NEAR", 0) + 1
                if emit_margins or len(rejections) < max_rejection_samples:
                    rej_evidence = {
                        "distance_m": distance,
                        "distance_metric": "aabb_surface",
                        "sparse_near_threshold": sparse_near_threshold,
                        "sparse_version": 2,
                    }
                    if emit_margins:
                        rej_evidence["margin_confidence"] = margin_confidence(
                            ratio_margin(distance, sparse_near_threshold))
                    rejections.append(EdgeRejection(
                        source=make_entity_ref(a.identity.object_uid),
                        type="NEAR",
                        target=make_entity_ref(b.identity.object_uid),
                        extractor=extractor,
                        rejected_reason="distance_exceeds_sparse_near_threshold",
                        evidence=rej_evidence,
                    ))
    return edges, counts, rejections, rejection_counts


class ProximityExtractor:
    name: str = "proximity"
    version: str = "0.1"
    edge_types: frozenset[EdgeType] = PROXIMITY_TYPES

    def extract(
        self,
        entities: EntityArtifacts,
        config: RelationExtractorConfig,
    ) -> tuple[list[Edge], RelationExtractorDiagnostics]:
        if not isinstance(config, ProximityConfig):
            raise TypeError(
                f"ProximityExtractor requires ProximityConfig, "
                f"got {type(config).__name__}"
            )
        start = time.perf_counter()
        if config.mode == "compat":
            edges, counts = extract_compat(
                entities.entities, extractor=self.name, version=self.version,
                frame=edge_frame(entities), emit_margins=config.emit_margins,
            )
            rejections: list[EdgeRejection] = []
            rejection_counts: dict[EdgeType, int] = {}
            reported_version = self.version
        elif config.mode == "sparse":
            if config.sparse_version == 1:
                edges, counts, rejections, rejection_counts = extract_sparse(
                    entities.entities, extractor=self.name, version=self.version,
                    frame=edge_frame(entities),
                    sparse_near_threshold=config.sparse_near_threshold,
                    emit_margins=config.emit_margins,
                )
                reported_version = self.version
            elif config.sparse_version == 2:
                edges, counts, rejections, rejection_counts = extract_sparse_v2(
                    entities.entities, extractor=self.name,
                    frame=edge_frame(entities),
                    sparse_near_threshold=config.sparse_near_threshold,
                    emit_margins=config.emit_margins,
                )
                reported_version = SPARSE_V2_VERSION
            else:
                raise ValueError(
                    f"unknown sparse_version {config.sparse_version!r}; supported: 1, 2"
                )
        else:
            raise ValueError(f"unknown mode {config.mode!r}")
        runtime_ms = int((time.perf_counter() - start) * 1000)
        return edges, RelationExtractorDiagnostics(
            extractor=self.name,
            version=reported_version,
            mode=config.mode,
            physical_edges_per_type=counts,
            physical_edges_total=len(edges),
            logical_edges_total=count_logical_edges(edges),
            rejections_per_type=rejection_counts,
            rejection_samples=sample_rejections(
                rejections, margin_aware=config.emit_margins),
            runtime_ms=runtime_ms,
        )
