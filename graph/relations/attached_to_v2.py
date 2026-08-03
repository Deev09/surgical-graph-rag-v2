"""ATTACHED_TO v2 — semantics_v2 track wall-mounted attachment (D1).

Protocol: docs/semantics_v2_track_protocol.md (signed off 2026-08-02;
constants FROZEN). This extractor is OPT-IN: no default battery/gate
path constructs it, and the frozen v1 `attached_to.py` is untouched.

D1, per-pair (frozen): evaluate EVERY entity-wall pair with the frozen
`geometry/wall_contact.py::wall_contact` predicate exactly as v1
CONTACTS_SURFACE consumes it, with ONE changed parameter
(`contact_threshold_m = 0.12`). Per pair, depth = the width of the
entity AABB projected onto THAT pair's unit wall normal. A pair
satisfies D1 iff wall_contact passes, depth <= 0.35 m, and one of:

  (a) elevated mount:  AABB bottom >= 0.30 m above the calibrated floor
      plane (evaluated at the entity centroid XY);
  (b) low thin panel:  AABB bottom < 0.30 m (including below-floor
      boxes) AND depth <= 0.12 m.

An ATTACHED_TO edge is emitted for every satisfying pair; the entity is
cited iff at least one edge exists. There is NO floor-support
disqualifier in v2 (D1 revision 1/3: the keys rule floor-reaching thin
panels attached — room_2 obj_57). Declared consequence: thin doors can
fire under (b) and count as false positives against the exhaustive
attachment keys.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Literal

from extractors.base import EntityArtifacts, StructuralSurface
from geometry.wall_contact import WallContactConfig, wall_contact
from graph.relations import contacts_surface as _cs
from graph.relations.attached_to import _active, _build_edge
from graph.relations.base import (
    RelationExtractorConfig, RelationExtractorDiagnostics,
    count_logical_edges, edge_frame, make_entity_ref, make_surface_ref,
    margin_confidence, ratio_margin, wall_contact_margin,
)
from graph.schema import Edge, EdgeRejection, EdgeType

ATTACHED_TO_V2_VERSION = "2.0"
ATTACHED_TO_V2_TYPES: frozenset[EdgeType] = frozenset({"ATTACHED_TO"})


@dataclass(frozen=True)
class AttachedToV2Config:
    """D1 frozen constants. Only contact_threshold_m differs from the v1
    wall geometry; the remaining wall_contact parameters are inherited
    verbatim from CONTACTS_SURFACE."""
    mode: Literal["sparse"] = "sparse"
    contact_threshold_m: float = 0.12          # D1: the ONE changed parameter
    penetration_tolerance_m: float = _cs.DEFAULT_PENETRATION_TOLERANCE_M
    max_wall_tilt_deg: float = _cs.DEFAULT_MAX_WALL_TILT_DEG
    footprint_tolerance_m: float = _cs.DEFAULT_FOOTPRINT_TOLERANCE_M
    depth_max_m: float = 0.35                  # D1 base condition
    elevated_bottom_min_m: float = 0.30        # disjunct (a) / partition point
    thin_panel_depth_max_m: float = 0.12       # disjunct (b)
    include_synth_fallback: bool = False
    # Calibration opt-in (see _d1_margin). False = frozen D1 behavior
    # (confidence is literally 1.0); hash_omit_if_default keeps default config
    # hashes byte-identical.
    emit_margins: bool = field(
        default=False, metadata={"hash_omit_if_default": True})

    def __post_init__(self) -> None:
        self._wall_config()
        for name in ("depth_max_m", "elevated_bottom_min_m",
                     "thin_panel_depth_max_m"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.mode != "sparse":
            raise ValueError(f"unknown mode {self.mode!r}")

    def _wall_config(self) -> WallContactConfig:
        return WallContactConfig(
            contact_threshold_m=self.contact_threshold_m,
            penetration_tolerance_m=self.penetration_tolerance_m,
            max_wall_tilt_deg=self.max_wall_tilt_deg,
            footprint_tolerance_m=self.footprint_tolerance_m,
        )


def _depth_along_normal(aabb, plane) -> float:
    n = (plane.a, plane.b, plane.c)
    norm = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
    if norm <= 0:
        raise ValueError("degenerate wall plane normal")
    lo, hi = aabb
    return sum(abs(n[i] / norm) * (hi[i] - lo[i]) for i in range(3))


def _d1_margin(
    wall_evidence: dict,
    depth: float,
    bottom_elev: float | None,
    config: AttachedToV2Config,
) -> float:
    """Normalized D1 margin: min over the conjunction, max over the disjunction.

    D1 is `wall_contact AND depth <= 0.35 AND (elevated OR thin panel)`. The
    conjunction takes the min of its parts (only as comfortable as the tightest
    clause); the mount disjunction takes the MAX of its two disjuncts (a
    disjunction is as comfortable as its best-satisfied branch), which also
    means the score does not jump when a pair crosses the 0.30 m partition
    point from disjunct (a) to disjunct (b).

      wall     wall_contact_margin(evidence), with D1's widened
               contact_threshold_m = 0.12 already inside it, so the contact
               band is [-0.02, +0.12] and its half-width 0.07 m is the scale.
      depth    (depth_max_m - depth) / depth_max_m, depth being the AABB width
               projected onto that pair's wall normal. depth_max_m = 0.35 m is
               both the threshold and the natural scale ("how deep may an
               attached thing be").
      mount    (a) (bottom_elevation - 0.30) / 0.30 — the elevation test the
               protocol names, normalized by its own threshold so a sconce at
               1.5 m saturates and a shelf at 0.31 m sits just above 0.5;
               (b) (0.12 - depth) / 0.12 — the thin-panel test.

    bottom_elev is None (no calibrated floor plane) -> -inf: D1 rejects such
    pairs outright, and the score has to agree rather than paper over a
    missing input.

    Sign invariant: >= 0 iff the pair satisfies D1, i.e. iff an edge is
    emitted.
    """
    m_wall = wall_contact_margin(wall_evidence)
    m_depth = ratio_margin(depth, config.depth_max_m)
    if bottom_elev is None:
        m_mount = -math.inf
    else:
        m_elevated = ((bottom_elev - config.elevated_bottom_min_m)
                      / config.elevated_bottom_min_m)
        m_thin = ratio_margin(depth, config.thin_panel_depth_max_m)
        m_mount = max(m_elevated, m_thin)
    return min(m_wall, m_depth, m_mount)


def _floor_z_at(floors: list[StructuralSurface], x: float, y: float) -> float | None:
    """Calibrated floor-plane height at (x, y), from the first active floor
    (every keyed scene in this track has exactly one)."""
    for f in floors:
        p = f.plane
        if abs(p.c) < 1e-9:
            continue
        return -(p.a * x + p.b * y + p.d) / p.c
    return None


class AttachedToV2Extractor:
    name: str = "attached_to_v2"
    version: str = ATTACHED_TO_V2_VERSION
    edge_types: frozenset[EdgeType] = ATTACHED_TO_V2_TYPES

    def extract(
        self,
        entities: EntityArtifacts,
        config: RelationExtractorConfig,
    ) -> tuple[list[Edge], RelationExtractorDiagnostics]:
        if not isinstance(config, AttachedToV2Config):
            raise TypeError("AttachedToV2Extractor requires AttachedToV2Config")
        start = time.perf_counter()
        wall_cfg = config._wall_config()
        gravity = entities.frame.gravity
        frame = edge_frame(entities)
        edges: list[Edge] = []
        rejections: list[EdgeRejection] = []
        rejection_counts: dict[EdgeType, int] = {}

        def reject(entity, surface, reason: str, evidence: dict) -> None:
            rejection_counts["ATTACHED_TO"] = rejection_counts.get("ATTACHED_TO", 0) + 1
            if len(rejections) < 64:
                rejections.append(EdgeRejection(
                    source=make_entity_ref(entity.identity.object_uid),
                    type="ATTACHED_TO",
                    target=make_surface_ref(surface.surface_uid),
                    extractor=self.name,
                    rejected_reason=reason,
                    evidence=evidence,
                ))

        walls = _active(entities.structural_surfaces, "wall",
                        config.include_synth_fallback)
        floors = _active(entities.structural_surfaces, "floor",
                         config.include_synth_fallback)

        for entity in entities.entities:
            lo, hi = entity.bbox_aabb
            floor_z = _floor_z_at(floors, entity.centroid[0],
                                  entity.centroid[1])
            for surface in walls:
                result = wall_contact(entity.bbox_aabb, entity.centroid,
                                      surface.plane, surface.polygon,
                                      gravity, wall_cfg)
                depth = _depth_along_normal(entity.bbox_aabb, surface.plane)
                evidence = dict(result.evidence)
                if isinstance(evidence.get("up"), tuple):
                    evidence["up"] = list(evidence["up"])
                bottom_elev = (lo[2] - floor_z) if floor_z is not None else None
                evidence.update({
                    "semantics_track": "v2",
                    "surface_type": surface.surface_type,
                    "source": surface.source,
                    "depth_along_wall_normal_m": round(depth, 4),
                    "bottom_elevation_m": (round(bottom_elev, 4)
                                           if bottom_elev is not None else None),
                    "contact_threshold_m": config.contact_threshold_m,
                    "depth_max_m": config.depth_max_m,
                    "elevated_bottom_min_m": config.elevated_bottom_min_m,
                    "thin_panel_depth_max_m": config.thin_panel_depth_max_m,
                })
                if config.emit_margins:
                    # EdgeRejection has no confidence field, so every branch
                    # below — emitted or rejected — carries the margin in
                    # evidence; the emitted branch additionally puts it on
                    # Edge.confidence.
                    evidence["margin_confidence"] = margin_confidence(
                        _d1_margin(result.evidence, depth, bottom_elev, config))
                if not result.contacts_surface:
                    evidence["failed_clauses"] = result.failed_clauses
                    reject(entity, surface, "wall_contact_clauses_failed",
                           evidence)
                    continue
                if depth > config.depth_max_m:
                    reject(entity, surface, "depth_exceeds_max", evidence)
                    continue
                if bottom_elev is None:
                    reject(entity, surface, "no_calibrated_floor_plane",
                           evidence)
                    continue
                if bottom_elev >= config.elevated_bottom_min_m:
                    evidence["mount_disjunct"] = "a_elevated"
                elif depth <= config.thin_panel_depth_max_m:
                    evidence["mount_disjunct"] = "b_low_thin_panel"
                else:
                    reject(entity, surface, "low_but_not_thin_panel",
                           evidence)
                    continue
                edges.append(_build_edge(
                    entity, surface, evidence,
                    extractor=self.name, version=self.version, frame=frame,
                    confidence=evidence.get("margin_confidence", 1.0),
                ))

        runtime_ms = int((time.perf_counter() - start) * 1000)
        return edges, RelationExtractorDiagnostics(
            extractor=self.name, version=self.version, mode=config.mode,
            physical_edges_per_type={"ATTACHED_TO": len(edges)},
            physical_edges_total=len(edges),
            logical_edges_total=count_logical_edges(edges),
            rejections_per_type=rejection_counts,
            rejection_samples=rejections,
            runtime_ms=runtime_ms,
        )
