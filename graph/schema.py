"""SceneGraph schema — phase0_design.md §5.4 and §6.

Frozen meanings for EdgeType; changing one requires bumping schema_version
and rebuilding cached bundles. Edges use GraphRef on both sides so a target
can be either an entity or a structural surface (ATTACHED_TO, NEAR_SURFACE).

SceneGraphBundle is pure data (no methods). Indexed-access helpers are
module-level functions below.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from common.types import FrameKind, JSON, OrientedBBox, Plane, SceneFrame, Vec3


EdgeType = Literal[
    "LEFT_OF", "RIGHT_OF", "IN_FRONT_OF", "BEHIND",
    "ABOVE", "BELOW",
    "ON_TOP_OF", "SUPPORTS", "ATTACHED_TO",
    "INSIDE", "CONTAINS",
    "NEAR_SURFACE",
    "ON_SURFACE",
    "CONTACTS_SURFACE",
    "ON_ENTITY_SURFACE",
    "NEAR",
    # FAR is intentionally not in this list — it is a query-time operator
    # over a centroid index, never stored. See reasoner/executor.
]


@dataclass(frozen=True)
class GraphRef:
    """Typed reference into a SceneGraphBundle. kind discriminates entities
    from structural surfaces; uid is stable within the EntityArtifacts
    bundle that produced this graph."""
    kind: Literal["entity", "surface"]
    uid: str


@dataclass(frozen=True)
class Node:
    """Graph-level entity node. Selected attributes copied from
    EntityArtifact at build time. Structural surfaces are NOT nodes; they
    are referenced via SceneGraphBundle.structural_surface_refs."""
    id: str
    label: str
    label_confidence: float
    centroid: Vec3
    bbox_aabb: tuple[Vec3, Vec3]
    bbox_obb: OrientedBBox | None
    embedding_ref: str | None
    attributes: dict[str, JSON] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    edge_id: str
    source: GraphRef
    type: EdgeType
    target: GraphRef
    # Same domain as SceneFrame.kind, and required to equal the frame of the
    # bundle the edge was extracted from. Extractors must not hardcode it —
    # route it through graph.relations.base.edge_frame.
    frame: FrameKind
    weight: float
    confidence: float
    extractor: str
    extractor_version: str
    evidence: dict[str, JSON] = field(default_factory=dict)


@dataclass(frozen=True)
class EdgeRejection:
    source: GraphRef
    type: EdgeType
    target: GraphRef
    extractor: str
    rejected_reason: str
    evidence: dict[str, JSON] = field(default_factory=dict)


@dataclass(frozen=True)
class BuildDiagnostics:
    """Build-level diagnostics aggregated by the GraphBuilder.

    physical_edges_total counts every stored Edge. logical_edges_total
    normalizes per graph.relations.base.count_logical_edges: symmetric
    pairs once, inverse pairs once. mode is the single mode shared by
    all extractor runs (mixed-mode runs are rejected upstream).

    per_extractor preserves the full per-family RelationExtractorDiagnostics
    so callers can inspect physical and logical totals per family without
    re-running anything.
    """
    extractor_versions: dict[str, str]
    edges_emitted_per_type: dict[EdgeType, int]
    rejections_per_type: dict[EdgeType, int]
    rejection_samples: list[EdgeRejection]
    runtime_ms_per_extractor: dict[str, int]
    # P1.07 additions:
    per_extractor: list["RelationExtractorDiagnostics"] = field(default_factory=list)  # type: ignore[name-defined]  # noqa: F821
    physical_edges_total: int = 0
    logical_edges_total: int = 0
    mode: Literal["compat", "sparse"] = "sparse"
    # P2.10 additions: explicit density-policy bookkeeping.
    density_policy: Literal["phase1_block", "phase2_telemetry_only"] = "phase1_block"
    density_ratio: float | None = None
    sparse_density_limit: float = 14.0


@dataclass(frozen=True)
class SurfaceRecord:
    """Graph-level structural-surface record (C1).

    Surfaces are NOT graph nodes — edges reference them via
    GraphRef(kind="surface", uid=...). This record carries the geometry
    and provenance forward from EntityArtifacts.StructuralSurface so the
    graph alone is sufficient for downstream consumers (no need to
    re-load the entity bundle to look up a surface's plane or source).

    NEAR_SURFACE limitation: the Phase 2 NEAR_SURFACE extractor measures
    distance to the INFINITE plane defined by `plane`, NOT clipped to
    `polygon` extents. An entity above the same plane but well outside
    the polygon footprint will still register as NEAR. This is recorded
    as a Phase 2 limitation; Phase 3 may add polygon-clipped variants.
    """
    uid: str
    surface_type: Literal["floor", "wall", "ceiling"]
    plane: Plane
    polygon: list[Vec3] | None
    source: Literal[
        "habitat_label", "mesh_ransac", "mesh_region_fit",
        "synth_bbox_fallback",
    ]
    confidence: float


@dataclass(frozen=True)
class SceneGraphBundle:
    schema_version: int            # bump on breaking changes; loader checks
    bundle_hash: str
    scene_id: str
    frame: SceneFrame
    entity_bundle_hash: str
    nodes: list[Node]
    edges: list[Edge]
    structural_surface_refs: list[str]
    # C1 (P2.10): full structural-surface retention. Populated from
    # EntityArtifacts.structural_surfaces by the builder (NOT derived
    # from the edge set). `structural_surface_refs` is kept as a derived
    # view = [s.uid for s in structural_surfaces] for back-compat read
    # paths; the builder enforces the equality.
    structural_surfaces: list[SurfaceRecord] = field(default_factory=list)


def find_node(bundle: SceneGraphBundle, uid: str) -> Node | None:
    for n in bundle.nodes:
        if n.id == uid:
            return n
    return None


def edges_from(
    bundle: SceneGraphBundle,
    src: GraphRef,
    *,
    type: EdgeType | None = None,
) -> list[Edge]:
    return [
        e for e in bundle.edges
        if e.source == src and (type is None or e.type == type)
    ]


def edges_to(
    bundle: SceneGraphBundle,
    tgt: GraphRef,
    *,
    type: EdgeType | None = None,
) -> list[Edge]:
    return [
        e for e in bundle.edges
        if e.target == tgt and (type is None or e.type == type)
    ]
