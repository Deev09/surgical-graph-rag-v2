"""Directional relation extractor (compat + sparse).

Compat is a faithful port of relations/compute.py:
  - LEGACY_MIN_DELTA = 0.3
  - One edge per ordered pair when the dominant axis exceeds the threshold
  - Iteration: outer i, inner j over EntityArtifacts.entities (matches
    legacy)
  - Type chosen by dominant axis sign (LEFT_OF if dx < 0, etc.)

Sparse is a fresh code path:
  - Tighter axis-dominance margin (default 0.5)
  - Inter-object distance cap (default 3.0 m)
  - Canonical-only storage: one edge per unordered pair in the canonical
    direction (LEFT_OF, BEHIND, ABOVE). The executor derives the inverse
    (RIGHT_OF, IN_FRONT_OF, BELOW) at query time.

Compat and sparse are two separate functions. They share only the
EntityArtifact reading shape and the build_edge helper. Legacy
reproduction is therefore not contaminated by sparse-mode tweaks.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from math import sqrt
from typing import Literal

from common.types import FrameKind
from extractors.base import EntityArtifact, EntityArtifacts
from graph.relations.base import (
    RelationExtractorConfig, RelationExtractorDiagnostics,
    count_logical_edges, edge_frame, make_edge_id, make_entity_ref,
    margin_confidence, ratio_margin,
)
from graph.schema import Edge, EdgeRejection, EdgeType


LEGACY_MIN_DELTA = 0.3   # frozen for compat; do not change

DIRECTIONAL_TYPES: frozenset[EdgeType] = frozenset({
    "LEFT_OF", "RIGHT_OF", "BEHIND", "IN_FRONT_OF", "ABOVE", "BELOW",
})

# Canonical types stored in sparse mode. Their inverses (RIGHT_OF,
# IN_FRONT_OF, BELOW) are derived by the executor at query time.
SPARSE_CANONICAL_TYPES: frozenset[EdgeType] = frozenset({
    "LEFT_OF", "BEHIND", "ABOVE",
})


@dataclass(frozen=True)
class DirectionalConfig:
    """sparse_max_distance=2.5 is a Replica-calibrated provisional
    default. It was chosen so room_0's 73-entity scene lands under the
    GraphBuilder's sparse density guardrail (logical_edges/entity_count
    <= 14) — NOT as evidence that 2.5m is a generally good threshold.
    Different scene scales / object densities may need different values.
    Phase 3's hand-labeled relation evaluation will replace this density
    heuristic with measured precision and recall."""
    mode: Literal["compat", "sparse"]
    sparse_min_delta: float = 0.5
    sparse_max_distance: float = 2.5
    # Calibration opt-in. False = frozen behavior (confidence is literally 1.0);
    # hash_omit_if_default keeps default config hashes byte-identical.
    emit_margins: bool = field(
        default=False, metadata={"hash_omit_if_default": True})


def _euclid_3d(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _dominance_margin(
    dx: float, dy: float, dz: float, min_delta: float,
) -> float:
    """Normalized axis-dominance margin for one directional edge.

    The extractor reads two decisions off the same three numbers: (1) is the
    separation worth reporting at all — `dominant >= min_delta` — and (2) which
    axis owns it — `dominant == max(|dx|, |dy|, |dz|)`. Both must hold for THIS
    edge to be the one emitted, so the binding competitor is whichever is
    larger: the threshold, or the runner-up axis. The margin is the gap to that
    competitor, normalized by min_delta (the extractor's own notion of "a
    separation worth reporting").

    Consequence, and the reason to prefer this over `dominant - min_delta`
    alone: two nearly-tied axes land near 0.5 even when both are far past the
    threshold. "left of" chosen by 1 cm over "behind" is exactly the ambiguity
    a downstream risk-coverage sweep needs to be able to see.

    Sign invariant: >= 0 iff the dominant axis both clears min_delta and is the
    max, i.e. iff an edge of this type is emitted.
    """
    deltas = sorted((abs(dx), abs(dy), abs(dz)))
    dominant, runner_up = deltas[2], deltas[1]
    if min_delta <= 0.0:
        return math.inf
    return (dominant - max(min_delta, runner_up)) / min_delta


def _build_edge(
    src: EntityArtifact,
    tgt: EntityArtifact,
    type_: EdgeType,
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
        edge_id=make_edge_id(extractor, version, source, type_, target),
        source=source,
        type=type_,
        target=target,
        frame=frame,
        weight=1.0,
        confidence=confidence,
        extractor=extractor,
        extractor_version=version,
        evidence=evidence,
    )


# ----- compat path (legacy port) -----

def _legacy_dominant_axis_relation(
    a: EntityArtifact, b: EntityArtifact,
) -> tuple[EdgeType, dict] | None:
    """Faithful port of relations/compute.py::_directional_edge.

    Returns (type, evidence) for source=a, target=b, or None if the axis
    delta does not exceed LEGACY_MIN_DELTA."""
    ax, ay, az = a.centroid
    bx, by, bz = b.centroid
    dx, dy, dz = ax - bx, ay - by, az - bz
    abs_dx, abs_dy, abs_dz = abs(dx), abs(dy), abs(dz)
    dominant = max(abs_dx, abs_dy, abs_dz)
    if dominant < LEGACY_MIN_DELTA:
        return None
    if dominant == abs_dx:
        rel: EdgeType = "LEFT_OF" if dx < 0 else "RIGHT_OF"
    elif dominant == abs_dy:
        rel = "BEHIND" if dy < 0 else "IN_FRONT_OF"
    else:
        rel = "BELOW" if dz < 0 else "ABOVE"
    return rel, {"dx": dx, "dy": dy, "dz": dz, "dominant": dominant}


def extract_compat(
    entities: list[EntityArtifact],
    *,
    extractor: str,
    version: str,
    frame: FrameKind,
    emit_margins: bool = False,
) -> tuple[list[Edge], dict[EdgeType, int]]:
    """Legacy reproduction. Iteration order matches relations/compute.py.

    Continuous quantity: the dominant centroid-axis delta, thresholded at
    LEGACY_MIN_DELTA. With emit_margins, confidence =
    margin_confidence(_dominance_margin(dx, dy, dz, LEGACY_MIN_DELTA))."""
    edges: list[Edge] = []
    counts: dict[EdgeType, int] = {}
    n = len(entities)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a = entities[i]
            b = entities[j]
            result = _legacy_dominant_axis_relation(a, b)
            if result is None:
                continue
            rel, evidence = result
            confidence = 1.0
            if emit_margins:
                confidence = margin_confidence(_dominance_margin(
                    evidence["dx"], evidence["dy"], evidence["dz"],
                    LEGACY_MIN_DELTA,
                ))
            edges.append(_build_edge(
                a, b, rel,
                extractor=extractor, version=version, frame=frame,
                evidence=evidence, confidence=confidence,
            ))
            counts[rel] = counts.get(rel, 0) + 1
    return edges, counts


# ----- sparse path -----

def _canonical_direction_for_pair(
    a: EntityArtifact,
    b: EntityArtifact,
    sparse_min_delta: float,
) -> tuple[EntityArtifact, EntityArtifact, EdgeType, dict] | None:
    """Pick the canonical (src, tgt, type, evidence) for unordered pair
    {a, b}. Canonical types are LEFT_OF, BEHIND, ABOVE. Returns None if
    no axis dominance exceeds sparse_min_delta."""
    ax, ay, az = a.centroid
    bx, by, bz = b.centroid
    dx, dy, dz = ax - bx, ay - by, az - bz
    abs_dx, abs_dy, abs_dz = abs(dx), abs(dy), abs(dz)
    dominant = max(abs_dx, abs_dy, abs_dz)
    if dominant < sparse_min_delta:
        return None
    evidence = {"dx": dx, "dy": dy, "dz": dz, "dominant": dominant}
    if dominant == abs_dx:
        # LEFT_OF canonical: src is left of tgt (negative dx from src to tgt)
        if dx < 0:
            return a, b, "LEFT_OF", evidence
        return b, a, "LEFT_OF", evidence
    if dominant == abs_dy:
        # BEHIND canonical: src is behind tgt
        if dy < 0:
            return a, b, "BEHIND", evidence
        return b, a, "BEHIND", evidence
    # z dominant. ABOVE canonical: src is above tgt (positive dz from src to tgt)
    if dz > 0:
        return a, b, "ABOVE", evidence
    return b, a, "ABOVE", evidence


def extract_sparse(
    entities: list[EntityArtifact],
    *,
    extractor: str,
    version: str,
    frame: FrameKind,
    sparse_min_delta: float,
    sparse_max_distance: float,
    emit_margins: bool = False,
) -> tuple[list[Edge], dict[EdgeType, int], list[EdgeRejection], dict[EdgeType, int]]:
    """Sparse: one canonical edge per unordered pair, gated on axis
    dominance AND inter-object distance.

    Continuous quantities and thresholds (emit_margins only):
      - emitted edge: dominant axis delta vs sparse_min_delta AND vs the
        runner-up axis -> _dominance_margin (normalized by sparse_min_delta),
      - distance rejection: centroid distance vs sparse_max_distance ->
        ratio_margin (normalized by the threshold),
      - dominance rejection: the same _dominance_margin, now negative.
    Rejection margins land in evidence["margin_confidence"] because
    EdgeRejection has no confidence field (see module note in the margin
    test)."""
    edges: list[Edge] = []
    counts: dict[EdgeType, int] = {}
    rejections: list[EdgeRejection] = []
    rejection_counts: dict[EdgeType, int] = {}
    max_rejection_samples = 64
    n = len(entities)
    for i in range(n):
        for j in range(i + 1, n):
            a = entities[i]
            b = entities[j]
            distance = _euclid_3d(a.centroid, b.centroid)
            if distance > sparse_max_distance:
                rejection_counts["LEFT_OF"] = rejection_counts.get("LEFT_OF", 0) + 1
                if len(rejections) < max_rejection_samples:
                    rej_evidence = {
                        "distance_m": distance,
                        "sparse_max_distance": sparse_max_distance,
                    }
                    if emit_margins:
                        rej_evidence["margin_confidence"] = margin_confidence(
                            ratio_margin(distance, sparse_max_distance))
                    rejections.append(EdgeRejection(
                        source=make_entity_ref(a.identity.object_uid),
                        type="LEFT_OF",
                        target=make_entity_ref(b.identity.object_uid),
                        extractor=extractor,
                        rejected_reason="distance_exceeds_sparse_max_distance",
                        evidence=rej_evidence,
                    ))
                continue
            result = _canonical_direction_for_pair(a, b, sparse_min_delta)
            if result is None:
                rejection_counts["LEFT_OF"] = rejection_counts.get("LEFT_OF", 0) + 1
                if len(rejections) < max_rejection_samples:
                    rej_evidence = {
                        "distance_m": distance,
                        "sparse_min_delta": sparse_min_delta,
                    }
                    if emit_margins:
                        rej_evidence["margin_confidence"] = margin_confidence(
                            _dominance_margin(
                                a.centroid[0] - b.centroid[0],
                                a.centroid[1] - b.centroid[1],
                                a.centroid[2] - b.centroid[2],
                                sparse_min_delta,
                            ))
                    rejections.append(EdgeRejection(
                        source=make_entity_ref(a.identity.object_uid),
                        type="LEFT_OF",
                        target=make_entity_ref(b.identity.object_uid),
                        extractor=extractor,
                        rejected_reason="axis_dominance_below_sparse_min_delta",
                        evidence=rej_evidence,
                    ))
                continue
            src, tgt, rel, evidence = result
            confidence = 1.0
            if emit_margins:
                confidence = margin_confidence(_dominance_margin(
                    evidence["dx"], evidence["dy"], evidence["dz"],
                    sparse_min_delta,
                ))
            edges.append(_build_edge(
                src, tgt, rel,
                extractor=extractor, version=version, frame=frame,
                evidence={**evidence, "distance_m": distance},
                confidence=confidence,
            ))
            counts[rel] = counts.get(rel, 0) + 1
    return edges, counts, rejections, rejection_counts


class DirectionalExtractor:
    """RelationExtractor for directional edges. Dispatches to compat or
    sparse based on config.mode."""
    name: str = "directional"
    version: str = "0.1"
    edge_types: frozenset[EdgeType] = DIRECTIONAL_TYPES

    def extract(
        self,
        entities: EntityArtifacts,
        config: RelationExtractorConfig,
    ) -> tuple[list[Edge], RelationExtractorDiagnostics]:
        if not isinstance(config, DirectionalConfig):
            raise TypeError(
                f"DirectionalExtractor requires DirectionalConfig, "
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
        elif config.mode == "sparse":
            edges, counts, rejections, rejection_counts = extract_sparse(
                entities.entities,
                extractor=self.name, version=self.version,
                frame=edge_frame(entities),
                sparse_min_delta=config.sparse_min_delta,
                sparse_max_distance=config.sparse_max_distance,
                emit_margins=config.emit_margins,
            )
        else:
            raise ValueError(f"unknown mode {config.mode!r}")
        runtime_ms = int((time.perf_counter() - start) * 1000)
        return edges, RelationExtractorDiagnostics(
            extractor=self.name,
            version=self.version,
            mode=config.mode,
            physical_edges_per_type=counts,
            physical_edges_total=len(edges),
            logical_edges_total=count_logical_edges(edges),
            rejections_per_type=rejection_counts,
            rejection_samples=rejections,
            runtime_ms=runtime_ms,
        )
