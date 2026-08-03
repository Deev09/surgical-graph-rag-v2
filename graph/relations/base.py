"""RelationExtractor Protocol and shared types for per-family extractors.

Each relation family (directional, proximity, support, containment, ...)
lives in its own module. The GraphBuilder (P1.07) orchestrates them but
does not understand any single family's internals.

Phase 1 implements only directional and proximity. Support / containment /
attached / NEAR_SURFACE are Phase 3 work; the EdgeType schema already
reserves slots for them.

Modes (extractor-defined; each family knows its own compat and sparse
code paths):
  - 'compat' — legacy reproduction. Faithful port of relations/compute.py.
    Designed to recreate the existing Replica edge artifact
    (scenes/replica_room_0/computed_relations/scene_graph.json) edge-for-edge.
  - 'sparse' — desired graph going forward. Tighter sparsity, canonical-
    only edge storage. The executor derives inverse types at query time.

The pre-imported Replica scene_graph.json is the Phase 1 replay fixture;
importers/replica.py remains the raw-data ingestion path. See
adapters/oracle_replica.py module docstring for details.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal, Protocol

from common.types import FrameKind
from extractors.base import EntityArtifact, EntityArtifacts, StructuralSurface
from graph.schema import Edge, EdgeRejection, EdgeType, GraphRef


# Canonical → inverse type mapping. Sparse mode stores only canonical
# types; the executor derives the inverse at query time. Canonical chosen
# alphabetically within each pair so the rule is mechanical and easy to
# audit.
CANONICAL_INVERSE_PAIRS: dict[EdgeType, EdgeType] = {
    "ABOVE": "BELOW",
    "BEHIND": "IN_FRONT_OF",
    "CONTAINS": "INSIDE",
    "LEFT_OF": "RIGHT_OF",
    "ON_TOP_OF": "SUPPORTS",
}

INVERSE_TO_CANONICAL: dict[EdgeType, EdgeType] = {
    v: k for k, v in CANONICAL_INVERSE_PAIRS.items()
}

SYMMETRIC_EDGE_TYPES: frozenset[EdgeType] = frozenset({"NEAR"})


class RelationExtractorConfig(Protocol):
    """Structural protocol for relation-extractor configs. Concrete
    family configs are dataclasses that happen to expose `.mode`."""
    mode: Literal["compat", "sparse"]


@dataclass(frozen=True)
class RelationExtractorDiagnostics:
    """Per-extractor diagnostics. Aggregated into BuildDiagnostics by the
    GraphBuilder.

    physical_edges_total counts every emitted Edge object.
    logical_edges_total normalizes: symmetric pairs counted once, inverse
    pairs counted once. In compat mode physical ≈ 2 × logical because
    compat duplicates NEAR and emits both directions of inverse pairs;
    in sparse mode the two should be equal."""
    extractor: str
    version: str
    mode: str
    physical_edges_per_type: dict[EdgeType, int]
    physical_edges_total: int
    logical_edges_total: int
    rejections_per_type: dict[EdgeType, int]
    rejection_samples: list[EdgeRejection]
    runtime_ms: int


class RelationExtractor(Protocol):
    name: str
    version: str
    edge_types: frozenset[EdgeType]

    def extract(
        self,
        entities: EntityArtifacts,
        config: RelationExtractorConfig,
    ) -> tuple[list[Edge], RelationExtractorDiagnostics]: ...


def edge_frame(entities: EntityArtifacts) -> FrameKind:
    """The frame label every edge extracted from `entities` must carry.

    An edge is computed in whatever frame the importer put the geometry in,
    so the label is a property of the ENTITY BUNDLE, never a constant an
    extractor gets to pick. Every extractor in this package routes its
    `Edge.frame` through here; tests/graph/test_edge_frame_label.py fails if
    one stops doing so.

    Historically all nine edge constructors hardcoded frame="world" while the
    live importer (demo/replica_habitat_import.py) was handing them a
    gravity-aligned, yaw-de-rotated frame. That is not a cosmetic mislabel:
    recomputing directional edges in the raw world frame changes 26.9% of
    room_1's edges and 9.7% of room_2's (docs/frame_and_scale_audit.md,
    Finding 4). It stayed invisible because on room_0 — the one scene the
    thresholds were calibrated on — the two frames agree on 2593 of 2594
    edges.
    """
    return entities.frame.kind


def count_logical_edges(edges: list[Edge]) -> int:
    """Normalize an edge list to count distinct logical facts.

    - Symmetric edge types (NEAR): an unordered pair {a, b} counts once
      regardless of how many physical edges encode it.
    - Inverse pair types (LEFT_OF/RIGHT_OF, ABOVE/BELOW, ...): a fact
      stored as either direction counts once. The non-canonical
      direction is mapped to its canonical via INVERSE_TO_CANONICAL.
    - All other edge types (ATTACHED_TO, NEAR_SURFACE, ...): each
      ordered (source, type, target) counts once.
    """
    canonical_keys: set[tuple[str, str, str, str, str]] = set()
    symmetric_keys: set[tuple[str, tuple]] = set()

    for e in edges:
        if e.type in SYMMETRIC_EDGE_TYPES:
            a = (e.source.kind, e.source.uid)
            b = (e.target.kind, e.target.uid)
            pair = tuple(sorted([a, b]))
            symmetric_keys.add((e.type, pair))
        elif e.type in CANONICAL_INVERSE_PAIRS:
            canonical_keys.add((
                e.type,
                e.source.kind, e.source.uid,
                e.target.kind, e.target.uid,
            ))
        elif e.type in INVERSE_TO_CANONICAL:
            canonical = INVERSE_TO_CANONICAL[e.type]
            canonical_keys.add((
                canonical,
                e.target.kind, e.target.uid,
                e.source.kind, e.source.uid,
            ))
        else:
            canonical_keys.add((
                e.type,
                e.source.kind, e.source.uid,
                e.target.kind, e.target.uid,
            ))

    return len(canonical_keys) + len(symmetric_keys)


def edge_key(e: Edge) -> tuple[str, str, str, str, str]:
    """Stable identity for set-comparison and gate diffs.
    (source.kind, source.uid, type, target.kind, target.uid)."""
    return (e.source.kind, e.source.uid, e.type, e.target.kind, e.target.uid)


def make_edge_id(
    extractor: str,
    version: str,
    source: GraphRef,
    type_: EdgeType,
    target: GraphRef,
) -> str:
    """Deterministic edge id over extractor + key."""
    payload = f"{extractor}|{version}|{source.kind}:{source.uid}|{type_}|{target.kind}:{target.uid}"
    return f"e_{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


# ---------------------------------------------------------------------------
# Calibration margins (opt-in per extractor via its config's `emit_margins`).
#
# Every extractor in this package already computes a continuous quantity and
# then thresholds it to a boolean. These helpers keep that quantity: each
# extractor converts its own measurement into a signed, scale-normalized
# MARGIN (>= 0 iff the test passed) and squashes it once through
# `margin_confidence`. The shared invariant every extractor documents and
# tests/graph/test_relation_margins.py asserts: 0.5 IS the decision boundary —
# emitted edges score >= 0.5, rejected pairs score <= 0.5. The score is a
# faithful relaxation of the gate, not a second, disagreeing opinion.
#
# Equality holds only for a measurement sitting exactly on a threshold, and
# such a case is genuinely a coin flip rather than a defect: the clearest
# example is directional's legacy tie-break, which resolves |dx| == |dy| in
# favour of x arbitrarily and now scores that edge exactly 0.5.
# ---------------------------------------------------------------------------

MARGIN_STEEPNESS = 4.0


def margin_confidence(margin: float, *, k: float = MARGIN_STEEPNESS) -> float:
    """Squash a signed, scale-normalized margin into a confidence in (0, 1).

    `margin` is (measured - threshold) / scale, signed so that positive means
    the test passed and larger means it passed more comfortably. Each
    extractor picks and documents its own `scale`; this helper only fixes the
    shared calibration anchors via the logistic squash:

        margin  0  (exactly on the decision boundary)  -> 0.500
        margin +1  (one full scale past it)            -> 0.982
        margin -1  (one full scale short of it)        -> 0.018

    k = 4 is chosen so a one-scale margin reads ~0.98 rather than saturating:
    the output has to stay a usable continuous score for a risk-coverage sweep
    instead of collapsing back onto {0, 1}.

    `math.inf` is a legal input meaning "this clause carries no discriminative
    information here" (a saturated or degenerate metric); it maps to 1.0 and
    under `min()` never binds. NaN maps to 0.5 (no information).
    """
    z = k * margin
    if z != z:                       # NaN
        return 0.5
    z = max(-50.0, min(50.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def band_margin(measured: float, lo: float, hi: float) -> float:
    """Normalized margin for a two-sided acceptance band lo <= measured <= hi.

    Distance to the NEARER band edge, normalized by the band's half-width: 0 on
    either edge, +1 dead centre, negative outside. A zero/negative-width band
    has no scale to normalize by and returns inf."""
    half = (hi - lo) / 2.0
    if half <= 0.0:
        return math.inf
    return min(measured - lo, hi - measured) / half


def ratio_margin(measured: float, threshold: float) -> float:
    """Normalized margin for a one-sided `measured <= threshold` test whose
    threshold IS the natural scale: (threshold - measured) / threshold.

    A non-finite threshold means the test is vacuous (inf). A non-positive
    threshold leaves no scale to normalize by, so the verdict degenerates to
    +/-inf."""
    if not math.isfinite(threshold):
        return math.inf
    if threshold <= 0.0:
        return math.inf if measured <= threshold else -math.inf
    return (threshold - measured) / threshold


def _clause_margins(
    ev: dict,
    *,
    orientation: float,
    gap_key: str,
) -> float:
    """Shared body of `rest_contact_margin` / `wall_contact_margin`.

    Both geometry predicates are the same four-clause conjunction with one
    orientation gate swapped, so the remaining three clauses normalize
    identically. `orientation` is the caller's already-normalized orientation
    margin; `gap_key` selects bottom_gap_m vs wall_gap_m.
    """
    contact_thresh = float(ev["contact_threshold_m"])
    # contact_threshold_m is the predicate's own "what counts as touching"
    # length scale; it is what the side and footprint gaps are measured
    # against. Guard the degenerate zero case so scale stays positive.
    scale = contact_thresh if contact_thresh > 0.0 else 1.0
    m_side = float(ev["sd_centroid_m"]) / scale
    m_contact = band_margin(
        float(ev[gap_key]),
        -float(ev["penetration_tolerance_m"]),
        contact_thresh,
    )
    in_plane_gap = float(ev["in_plane_gap_m"])
    # aabb_to_polygon_planar saturates at 0 for ANY interior overlap, so a zero
    # gap says "inside the footprint" and nothing about how comfortably. A
    # saturated clause must not bind the min.
    m_footprint = (
        math.inf if in_plane_gap <= 0.0
        else (float(ev["footprint_tolerance_m"]) - in_plane_gap) / scale
    )
    return min(orientation, m_side, m_contact, m_footprint)


def rest_contact_margin(ev: dict) -> float:
    """Normalized margin for geometry.rest_contact's four-clause conjunction.

    One margin per clause, then min() — a conjunction is only as comfortable as
    its tightest clause, and min >= 0 holds exactly when
    RestContactResult.on_surface is True.

    Scales:
      support_capable  (dot - cos(max_tilt)) / (1 - cos(max_tilt)) — the full
          span of orientations the gate admits, so +1 means dead level.
      centroid side    sd_centroid_m / contact_threshold_m.
      contact          two-sided band [-penetration_tolerance, contact_threshold]
          on bottom_gap_m, normalized by half the band width. This is the clause
          that actually discriminates in practice; the others are near-binary.
      footprint        (footprint_tolerance - in_plane_gap) / contact_threshold_m,
          saturating to inf at in_plane_gap == 0 (see `_clause_margins`).
    """
    cos_max = math.cos(math.radians(float(ev["max_tilt_deg"])))
    span = 1.0 - cos_max
    m_orient = (
        (float(ev["support_normal_dot_up"]) - cos_max) / span
        if span > 0.0 else math.inf
    )
    return _clause_margins(ev, orientation=m_orient, gap_key="bottom_gap_m")


def wall_contact_margin(ev: dict) -> float:
    """Normalized margin for geometry.wall_contact's four-clause conjunction.

    Identical in structure to `rest_contact_margin` with the orientation gate
    flipped: wall_capable is `abs(dot) <= sin(max_wall_tilt)`, so its margin is
    (sin_max - abs(dot)) / sin_max — +1 for a perfectly vertical wall, 0 at the
    tilt limit. The discriminating clause is the contact band on wall_gap_m.
    """
    sin_max = math.sin(math.radians(float(ev["max_wall_tilt_deg"])))
    m_orient = (
        (sin_max - abs(float(ev["wall_normal_dot_up"]))) / sin_max
        if sin_max > 0.0 else math.inf
    )
    return _clause_margins(ev, orientation=m_orient, gap_key="wall_gap_m")


def make_entity_ref(uid: str) -> GraphRef:
    return GraphRef(kind="entity", uid=uid)


def make_surface_ref(uid: str) -> GraphRef:
    return GraphRef(kind="surface", uid=uid)


def wall_xy_extent_area(surfaces: list[StructuralSurface]) -> float | None:
    """XY bounding-rectangle area over all wall polygons — a proxy for room
    footprint that does not trust floor labels (Replica office_0's `floor`
    instance covers ~25% of the real floor). None when no wall has a polygon."""
    xs: list[float] = []
    ys: list[float] = []
    for s in surfaces:
        if s.surface_type != "wall" or not s.polygon:
            continue
        for p in s.polygon:
            xs.append(p[0])
            ys.append(p[1])
    if not xs:
        return None
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def room_scale_flat(
    entity: EntityArtifact,
    room_xy_area_m2: float | None,
    *,
    min_area_frac: float,
    max_height_m: float,
) -> tuple[bool, dict]:
    """Phase 8 F4 predicate: is this entity a room-scale flat object (a
    wall-to-wall rug)? Such objects satisfy wall contact/proximity geometry
    trivially — a rug spanning the room touches every wall — but are never
    what a human means by "against/near the wall". Returns (verdict, evidence)
    so extractors can record the exclusion. Verdict is False when room area is
    unknown (no wall polygons): no silent exclusions without a room estimate."""
    lo, hi = entity.bbox_aabb
    footprint = (hi[0] - lo[0]) * (hi[1] - lo[1])
    height = hi[2] - lo[2]
    evidence = {
        "footprint_xy_m2": round(footprint, 4),
        "height_m": round(height, 4),
        "room_xy_area_m2": None if room_xy_area_m2 is None else round(room_xy_area_m2, 4),
        "min_area_frac": min_area_frac,
        "max_height_m": max_height_m,
    }
    if room_xy_area_m2 is None or room_xy_area_m2 <= 0.0:
        return False, evidence
    frac = footprint / room_xy_area_m2
    evidence["area_frac"] = round(frac, 4)
    return frac >= min_area_frac and height <= max_height_m, evidence
