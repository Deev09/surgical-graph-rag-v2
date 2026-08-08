"""Frame + scale audit over every available Replica scene. DIAGNOSTIC ONLY.

This tool measures three things and changes nothing:

  1. FRAME. geometry/frame.py estimates the up axis, floor plane and yaw
     from mesh geometry alone. Those estimates are then compared against
     what the pipeline currently assumes:
        - Replica's declared `gravity_dir` in habitat/info_semantic.json,
        - the hard `+Z up` guard in importers/replica.py,
        - the gravity-aligned frame and the CALIBRATED FLOOR PLANE that
          demo/replica_habitat_import.py hands to
          graph/relations/attached_to_v2.py.
     Agreement and disagreement are both reported, numerically.

  2. SCALE. Per scene: room diagonal, floor footprint diagonal, storey
     height (all mesh-derived), and object-size statistics (from the
     imported entity AABBs, i.e. the boxes the extractors actually see).

  3. THRESHOLD TRANSFER RISK. Every absolute-metre constant in
     graph/relations/** is read straight out of the source modules (never
     re-typed here) and expressed as a fraction of each scene's own scale.
     The spread of that fraction across scenes is the transfer risk the
     constant carries. For the distance-family constants the audit also
     reports the constant's EFFECT: what share of this scene's actual
     entity pairs / floor gaps falls on each side of it.

Nothing here imports into the extraction path, and nothing here writes
outside runs/frame_audit/.

Usage:
    python3 tools/frame_scale_audit.py [--out DIR] [--scene NAME ...]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from geometry.frame import (
    DEFAULT_CONFIG, angle_between_deg, estimate_from_ply, object_scale_stats,
)

# Oracle side of the comparison. These are READ, never modified.
from demo.replica_habitat_import import (
    ROOM_0_Z_TRANSLATION, STRUCTURAL_CLASSES, YAW_DEROTATE_GUARD_DEG,
    _aligned_structural_surfaces, _dominant_yaw_deg, _gravity_align_matrix,
    _matvec, _structural_surfaces, _world_aabb, import_habitat_room,
)
import importers.replica as replica_importer
from graph.relations import (
    attached_to_v2 as _atv2,
    contacts_surface as _cs,
    directional as _dir,
    on_entity_surface_v2 as _oes2,
    on_surface as _os,
    proximity as _prox,
    surface as _surf,
)

DATA_ROOT = Path.home() / "Desktop/datasets/replica"
DEFAULT_OUT = REPO_ROOT / "runs" / "frame_audit"
SCENES: tuple[str, ...] = (
    "room_0", "room_1", "room_2", "office_0",
    "frl_apartment_0", "apartment_0",
)

# Scale denominators. Every absolute-metre constant is audited against ONE
# of these, chosen by what the constant is physically comparing.
DENOMS = ("room_diag", "floor_diag", "storey_height", "obj_diag")

# (module.attr, value, denominator, one-line role). Values are pulled from
# the live modules so this table cannot drift from the code.
def _constants() -> list[dict]:
    dcfg = _dir.DirectionalConfig(mode="sparse")
    pcfg = _prox.ProximityConfig(mode="compat")
    a2 = _atv2.AttachedToV2Config()
    rows = [
        ("graph.relations.directional.LEGACY_MIN_DELTA",
         _dir.LEGACY_MIN_DELTA, "obj_diag",
         "compat axis-dominance floor between object centroids"),
        ("graph.relations.directional.DirectionalConfig.sparse_min_delta",
         dcfg.sparse_min_delta, "obj_diag",
         "sparse axis-dominance floor between object centroids"),
        ("graph.relations.directional.DirectionalConfig.sparse_max_distance",
         dcfg.sparse_max_distance, "room_diag",
         "sparse directional edge distance cap"),
        ("graph.relations.proximity.LEGACY_NEAR_THRESHOLD",
         _prox.LEGACY_NEAR_THRESHOLD, "room_diag",
         "compat NEAR radius (centroid)"),
        ("graph.relations.proximity.ProximityConfig.sparse_near_threshold",
         pcfg.sparse_near_threshold, "room_diag",
         "sparse NEAR radius"),
        ("graph.relations.contacts_surface.DEFAULT_CONTACT_THRESHOLD_M",
         _cs.DEFAULT_CONTACT_THRESHOLD_M, "obj_diag",
         "wall contact band (the 2 cm band)"),
        ("graph.relations.contacts_surface.DEFAULT_PENETRATION_TOLERANCE_M",
         _cs.DEFAULT_PENETRATION_TOLERANCE_M, "obj_diag",
         "allowed wall penetration"),
        ("graph.relations.contacts_surface.DEFAULT_NEAR_SURFACE_THRESHOLD_M",
         _cs.DEFAULT_NEAR_SURFACE_THRESHOLD_M, "obj_diag",
         "wall NEAR_SURFACE radius used by the subset guard"),
        ("graph.relations.contacts_surface.DEFAULT_ROOM_SCALE_FLAT_MAX_HEIGHT_M",
         _cs.DEFAULT_ROOM_SCALE_FLAT_MAX_HEIGHT_M, "storey_height",
         "max height for the room-scale-flat exclusion (rugs)"),
        ("graph.relations.on_surface.DEFAULT_CONTACT_THRESHOLD_M",
         _os.DEFAULT_CONTACT_THRESHOLD_M, "obj_diag",
         "horizontal-surface contact band"),
        ("graph.relations.on_surface.DEFAULT_PENETRATION_TOLERANCE_M",
         _os.DEFAULT_PENETRATION_TOLERANCE_M, "obj_diag",
         "allowed horizontal-surface penetration"),
        ("graph.relations.on_surface.DEFAULT_NEAR_SURFACE_THRESHOLD_M",
         _os.DEFAULT_NEAR_SURFACE_THRESHOLD_M, "obj_diag",
         "floor NEAR radius"),
        ("graph.relations.surface.DEFAULT_FLOOR_THRESHOLD_M",
         _surf.DEFAULT_FLOOR_THRESHOLD_M, "obj_diag",
         "NEAR_SURFACE floor threshold"),
        ("graph.relations.surface.DEFAULT_WALL_THRESHOLD_M",
         _surf.DEFAULT_WALL_THRESHOLD_M, "obj_diag",
         "NEAR_SURFACE wall threshold"),
        ("graph.relations.surface.DEFAULT_CEILING_THRESHOLD_M",
         _surf.DEFAULT_CEILING_THRESHOLD_M, "obj_diag",
         "NEAR_SURFACE ceiling threshold"),
        ("graph.relations.attached_to_v2.AttachedToV2Config.contact_threshold_m",
         a2.contact_threshold_m, "obj_diag",
         "v2 wall contact band"),
        ("graph.relations.attached_to_v2.AttachedToV2Config.depth_max_m",
         a2.depth_max_m, "obj_diag",
         "max wall-normal depth for a mounted object"),
        ("graph.relations.attached_to_v2.AttachedToV2Config.elevated_bottom_min_m",
         a2.elevated_bottom_min_m, "storey_height",
         "elevation above the calibrated floor for disjunct (a)"),
        ("graph.relations.attached_to_v2.AttachedToV2Config.thin_panel_depth_max_m",
         a2.thin_panel_depth_max_m, "obj_diag",
         "thin-panel depth for disjunct (b)"),
    ]
    return [{"name": n, "value": float(v), "denom": d, "role": r}
            for n, v, d, r in rows]


def _short(name: str) -> str:
    """`graph.relations.on_surface.DEFAULT_CONTACT_THRESHOLD_M` ->
    `on_surface.DEFAULT_CONTACT_THRESHOLD_M`. Two modules define constants
    with identical leaf names, so the module qualifier has to survive."""
    parts = [p for p in name.split(".") if not p.endswith("Config")]
    return ".".join(parts[-2:])


def _scale_free_constants() -> list[dict]:
    o2 = _oes2.OnEntitySurfaceV2Config()
    return [
        {"name": "graph.relations.contacts_surface.DEFAULT_MAX_WALL_TILT_DEG",
         "value": _cs.DEFAULT_MAX_WALL_TILT_DEG, "kind": "angle_deg"},
        {"name": "graph.relations.on_surface.DEFAULT_MAX_TILT_DEG",
         "value": _os.DEFAULT_MAX_TILT_DEG, "kind": "angle_deg"},
        {"name": "graph.relations.contacts_surface.DEFAULT_FOOTPRINT_TOLERANCE_M",
         "value": _cs.DEFAULT_FOOTPRINT_TOLERANCE_M, "kind": "zero_length"},
        {"name": "graph.relations.on_surface.DEFAULT_FOOTPRINT_TOLERANCE_M",
         "value": _os.DEFAULT_FOOTPRINT_TOLERANCE_M, "kind": "zero_length"},
        {"name": "graph.relations.contacts_surface.DEFAULT_ROOM_SCALE_FLAT_MIN_AREA_FRAC",
         "value": _cs.DEFAULT_ROOM_SCALE_FLAT_MIN_AREA_FRAC, "kind": "fraction"},
        {"name": "graph.relations.on_entity_surface_v2.OnEntitySurfaceV2Config"
                 ".footprint_area_max_frac",
         "value": o2.footprint_area_max_frac, "kind": "fraction"},
    ]


# --------------------------------------------------------------------------

def _plane_z_at(plane, x: float, y: float) -> float | None:
    if abs(plane.c) < 1e-9:
        return None
    return -(plane.a * x + plane.b * y + plane.d) / plane.c


def _compat_directional_type(a, b, min_delta: float) -> str | None:
    """Read-only port of graph.relations.directional._legacy_dominant_axis
    _relation, used ONLY to measure how much the answer depends on the
    frame. The extractor itself is untouched."""
    dx, dy, dz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    adx, ady, adz = abs(dx), abs(dy), abs(dz)
    dom = max(adx, ady, adz)
    if dom < min_delta:
        return None
    if dom == adx:
        return "LEFT_OF" if dx < 0 else "RIGHT_OF"
    if dom == ady:
        return "BEHIND" if dy < 0 else "IN_FRONT_OF"
    return "BELOW" if dz < 0 else "ABOVE"


def _frame_sensitivity(info: dict, r_align) -> dict:
    """How many directional edges change if the extractor runs in the raw
    Habitat world frame instead of the gravity-aligned, yaw-de-rotated frame
    the importer actually builds? Every edge is stamped frame="world"; this
    measures what that label is worth."""
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    drop = set(("undefined", "non-plane", "plane")) | set(STRUCTURAL_CLASSES)
    raw, aligned = [], []
    for o in info["objects"]:
        cls = str(o.get("class_name", "")).strip()
        if not cls or cls in drop:
            continue
        for target, rot, dz in ((raw, identity, 0.0),
                                (aligned, r_align, ROOM_0_Z_TRANSLATION)):
            lo, hi = _world_aabb(o, dz, rot)
            target.append(tuple((lo[i] + hi[i]) / 2.0 for i in range(3)))
    md = _dir.LEGACY_MIN_DELTA
    n_pairs = n_edge_aligned = n_differ = n_axis_differ = 0
    axis_of = {"LEFT_OF": "x", "RIGHT_OF": "x", "BEHIND": "y",
               "IN_FRONT_OF": "y", "ABOVE": "z", "BELOW": "z"}
    for i in range(len(aligned)):
        for j in range(i + 1, len(aligned)):
            n_pairs += 1
            ta = _compat_directional_type(aligned[i], aligned[j], md)
            tr = _compat_directional_type(raw[i], raw[j], md)
            if ta is not None:
                n_edge_aligned += 1
            if ta != tr:
                n_differ += 1
            if (ta is not None and tr is not None
                    and axis_of[ta] != axis_of[tr]):
                n_axis_differ += 1
    return {
        "n_pairs": n_pairs,
        "n_edges_in_aligned_frame": n_edge_aligned,
        "n_pairs_whose_directional_type_changes": n_differ,
        "frac_pairs_whose_directional_type_changes": (
            n_differ / n_edge_aligned if n_edge_aligned else None),
        "n_pairs_whose_dominant_axis_changes": n_axis_differ,
        "frac_pairs_whose_dominant_axis_changes": (
            n_axis_differ / n_edge_aligned if n_edge_aligned else None),
    }


def _percentile_of(values: list[float], threshold: float) -> float:
    """Share of `values` strictly below `threshold`."""
    if not values:
        return float("nan")
    return sum(1 for v in values if v < threshold) / len(values)


def audit_scene(scene: str, out_dir: Path, data_root: Path = DATA_ROOT) -> dict:
    room_dir = data_root / scene
    mesh_path = room_dir / "habitat" / "mesh_semantic.ply"
    info_path = room_dir / "habitat" / "info_semantic.json"
    scene_id = f"replica_{scene}"

    t0 = time.time()
    est = estimate_from_ply(mesh_path)
    mesh_seconds = time.time() - t0

    info = json.loads(info_path.read_text(encoding="utf-8"))
    gravity_dir = [float(v) for v in info["gravity_dir"]]
    declared_up = [-v for v in gravity_dir]

    # --- frame comparison -------------------------------------------------
    up = list(est.up_axis)
    frame_cmp = {
        "estimated_up_axis": [round(v, 6) for v in up],
        "declared_gravity_dir": [round(v, 6) for v in gravity_dir],
        "angle_estimated_up_to_world_plus_z_deg":
            angle_between_deg(up, (0.0, 0.0, 1.0)),
        "angle_declared_up_to_world_plus_z_deg":
            angle_between_deg(declared_up, (0.0, 0.0, 1.0)),
        "angle_estimated_up_to_declared_up_deg":
            angle_between_deg(up, declared_up),
        "importers_replica_plus_z_guard_passes":
            bool(replica_importer._gravity_is_neg_z(gravity_dir)),
        "axis_margin_ratio_winner_over_runner_up":
            est.diagnostics["axis_margin_ratio"],
        "axis_winner_area_frac": est.diagnostics["axis_winner_area_frac"],
        "sign_votes": est.diagnostics["sign_votes"],
        "sign_unanimous": est.diagnostics["sign_unanimous"],
        "sign_cue_a_vote": est.diagnostics["sign_cue_a_vote"],
        "sign_cue_b_vote": est.diagnostics["sign_cue_b_vote"],
        "sign_cue_c_vote": est.diagnostics["sign_cue_c_vote"],
    }

    # --- the pipeline's actual frame + calibrated floor -------------------
    r0 = _gravity_align_matrix(tuple(gravity_dir))
    r_align, _pre, _diag, importer_yaw = _aligned_structural_surfaces(
        info, r0, ROOM_0_Z_TRANSLATION)
    arts = import_habitat_room(room_dir, scene_id)
    floors = [s for s in arts.structural_surfaces if s.surface_type == "floor"]

    # Mesh floor plane, mapped into the importer's frame.
    fn = np.asarray(est.floor.normal, dtype=np.float64)
    p0 = -est.floor.d * fn
    q = _matvec(r_align, tuple(p0))
    q = (q[0], q[1], q[2] + ROOM_0_Z_TRANSLATION)
    m = np.asarray(_matvec(r_align, tuple(fn)), dtype=np.float64)
    m = m / np.linalg.norm(m)
    d_mesh = -float(np.dot(m, np.asarray(q)))

    calibration = arts.notes.get("floor_calibration", {}) or {}
    floor_rows = []
    for s in floors:
        cx = sum(p[0] for p in s.polygon) / len(s.polygon)
        cy = sum(p[1] for p in s.polygon) / len(s.polygon)
        z_imp = _plane_z_at(s.plane, cx, cy)
        z_mesh = (-(m[0] * cx + m[1] * cy + d_mesh) / m[2]
                  if abs(m[2]) > 1e-9 else None)
        delta = (z_imp - z_mesh
                 if (z_imp is not None and z_mesh is not None) else None)
        snap = float(calibration.get(s.surface_uid, 0.0))
        floor_rows.append({
            "surface_uid": s.surface_uid,
            "polygon_centroid_xy": [round(cx, 4), round(cy, 4)],
            "importer_floor_z_m": round(z_imp, 5) if z_imp is not None else None,
            "mesh_floor_z_m": round(z_mesh, 5) if z_mesh is not None else None,
            "delta_importer_minus_mesh_m": (round(delta, 5)
                                            if delta is not None else None),
            # Undo the F2 floor-calibration snap to show what the raw Habitat
            # `floor` label said before the pipeline corrected it.
            "floor_calibration_snap_m": round(snap, 5),
            "delta_before_calibration_m": (round(delta - snap, 5)
                                           if delta is not None else None),
            "normal_angle_deg": angle_between_deg(
                m, (s.plane.a, s.plane.b, s.plane.c)),
        })
    deltas = [abs(r["delta_importer_minus_mesh_m"]) for r in floor_rows
              if r["delta_importer_minus_mesh_m"] is not None]
    pre_deltas = [abs(r["delta_before_calibration_m"]) for r in floor_rows
                  if r["delta_before_calibration_m"] is not None]
    # The importer's own yaw estimate BEFORE the 5-degree guard clamps it to
    # zero, so the mesh-derived yaw can be compared against the same quantity.
    pre_surfaces, _ = _structural_surfaces(info, r0, ROOM_0_Z_TRANSLATION)
    label_yaw = _dominant_yaw_deg(
        [s for s in pre_surfaces if s.surface_type == "wall"])
    floor_cmp = {
        "n_importer_floors": len(floors),
        "per_floor": floor_rows,
        "best_abs_delta_m": min(deltas) if deltas else None,
        "worst_abs_delta_m": max(deltas) if deltas else None,
        "best_abs_delta_before_calibration_m": (min(pre_deltas)
                                                if pre_deltas else None),
        "importer_yaw_derotation_deg": round(importer_yaw, 4),
        "importer_label_yaw_pre_guard_deg": round(label_yaw, 4),
        "yaw_guard_deg": YAW_DEROTATE_GUARD_DEG,
        "mesh_dominant_yaw_deg": round(est.yaw_deg, 4),
        "mesh_vs_label_yaw_disagreement_deg": round(
            abs(est.yaw_deg - label_yaw), 4),
        "importer_floor_calibration_offsets_m": calibration,
        "z_translation_applied_m": ROOM_0_Z_TRANSLATION,
    }

    # --- scale ------------------------------------------------------------
    aabbs = [e.bbox_aabb for e in arts.entities]
    obj_stats = object_scale_stats(aabbs)
    lo = [min(b[0][i] for b in aabbs) for i in range(3)]
    hi = [max(b[1][i] for b in aabbs) for i in range(3)]
    entity_diag = math.sqrt(sum((hi[i] - lo[i]) ** 2 for i in range(3)))

    centroids = [e.centroid for e in arts.entities]
    pair_d: list[float] = []
    dom_delta: list[float] = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            a, b = centroids[i], centroids[j]
            dx, dy, dz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
            pair_d.append(math.sqrt(dx * dx + dy * dy + dz * dz))
            dom_delta.append(max(abs(dx), abs(dy), abs(dz)))

    # bottom elevation above the calibrated floor: exactly what
    # attached_to_v2 compares against elevated_bottom_min_m
    bottom_elev: list[float] = []
    if floors:
        p = floors[0].plane
        for e in arts.entities:
            z = _plane_z_at(p, e.centroid[0], e.centroid[1])
            if z is not None:
                bottom_elev.append(e.bbox_aabb[0][2] - z)

    scale = {
        "room_diag": est.scale.room_diagonal_m,
        "floor_diag": est.scale.floor_diagonal_m,
        "storey_height": est.scale.storey_height_m,
        "obj_diag": obj_stats.get("median_diagonal_m"),
        "mesh_extent_robust_m": [round(v, 4) for v in est.scale.extent_robust_m],
        "mesh_extent_raw_m": [round(v, 4) for v in est.scale.extent_raw_m],
        "observed_floor_area_m2": est.scale.floor_area_m2,
        "entity_bbox_diagonal_m": entity_diag,
        "n_entities": len(arts.entities),
        "object_stats": obj_stats,
        "pair_distance_median_m": statistics.median(pair_d) if pair_d else None,
        "pair_distance_p10_m": (float(np.percentile(pair_d, 10))
                                if pair_d else None),
        "pair_distance_p90_m": (float(np.percentile(pair_d, 90))
                                if pair_d else None),
        "n_pairs": len(pair_d),
        "multi_level_suspected": est.diagnostics["multi_level_suspected"],
        "ceiling_level_peaks_m": est.diagnostics["ceiling_level_peaks_m"],
    }

    effect = {
        "frac_pairs_within_sparse_max_distance": _percentile_of(
            pair_d, _dir.DirectionalConfig(mode="sparse").sparse_max_distance),
        "frac_pairs_within_legacy_near_threshold": _percentile_of(
            pair_d, _prox.LEGACY_NEAR_THRESHOLD),
        "frac_pairs_dominant_delta_below_legacy_min_delta": _percentile_of(
            dom_delta, _dir.LEGACY_MIN_DELTA),
        "frac_pairs_dominant_delta_below_sparse_min_delta": _percentile_of(
            dom_delta, _dir.DirectionalConfig(mode="sparse").sparse_min_delta),
        "frac_entities_bottom_below_elevated_min": _percentile_of(
            bottom_elev, _atv2.AttachedToV2Config().elevated_bottom_min_m),
        "n_bottom_elevation_samples": len(bottom_elev),
    }

    est_dump = {
        "up_axis": est.up_axis,
        "yaw_deg": est.yaw_deg,
        "rotation_rows_e1_e2_up": est.rotation,
        "floor": asdict(est.floor) if est.floor else None,
        "ceiling": asdict(est.ceiling) if est.ceiling else None,
        "scale": asdict(est.scale),
        "axis_candidates": [asdict(c) for c in est.axis_candidates],
        "horizontal_peaks": est.horizontal_peaks,
        "diagnostics": est.diagnostics,
    }
    (out_dir / f"{scene}_frame_estimate.json").write_text(
        json.dumps(est_dump, indent=2, default=str), encoding="utf-8")

    return {
        "scene": scene,
        "scene_id": scene_id,
        "mesh_path": str(mesh_path),
        "mesh_seconds": round(mesh_seconds, 2),
        "habitat_class_counts": {
            k: sum(1 for o in info["objects"] if o.get("class_name") == k)
            for k in ("floor", "wall", "ceiling")
        },
        "n_habitat_objects": len(info["objects"]),
        "frame": frame_cmp,
        "frame_sensitivity": _frame_sensitivity(info, r_align),
        "floor_agreement": floor_cmp,
        "scale": scale,
        "threshold_effect": effect,
    }


# --------------------------------------------------------------------------

def _fmt(v, nd=4):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "NO"
    if isinstance(v, float):
        if math.isnan(v):
            return "n/a"
        return f"{v:.{nd}f}"
    return str(v)


def build_tables(rows: list[dict]) -> tuple[str, dict]:
    scenes = [r["scene"] for r in rows]
    out: list[str] = []
    summary: dict = {}

    out.append("## Table 1 — per-scene frame estimate vs. what the pipeline assumes\n")
    out.append("| scene | est. up vs world +Z (deg) | declared gravity vs -Z (deg) "
               "| est. up vs declared up (deg) | `+Z up` guard in importers/replica.py "
               "| axis margin (win/runner-up) | sign votes A/B/C |")
    out.append("|---|---|---|---|---|---|---|")
    for r in rows:
        f = r["frame"]
        out.append(
            f"| {r['scene']} | {_fmt(f['angle_estimated_up_to_world_plus_z_deg'], 3)} "
            f"| {_fmt(f['angle_declared_up_to_world_plus_z_deg'], 3)} "
            f"| {_fmt(f['angle_estimated_up_to_declared_up_deg'], 3)} "
            f"| {_fmt(f['importers_replica_plus_z_guard_passes'])} "
            f"| {_fmt(f['axis_margin_ratio_winner_over_runner_up'], 3)} "
            f"| {'/'.join('+' if v > 0 else '-' for v in f['sign_votes'])} |")
    out.append("")

    out.append("## Table 2 — mesh floor plane vs. the calibrated floor plane "
               "attached_to_v2 uses\n")
    out.append("| scene | importer floors | best abs delta (m) | worst abs delta (m) "
               "| same delta BEFORE the F2 floor snap (m) | F2 snap applied (m) "
               "| floor-normal angle (deg) |")
    out.append("|---|---|---|---|---|---|---|")
    for r in rows:
        g = r["floor_agreement"]
        angles = [p["normal_angle_deg"] for p in g["per_floor"]]
        out.append(
            f"| {r['scene']} | {g['n_importer_floors']} "
            f"| {_fmt(g['best_abs_delta_m'], 4)} | {_fmt(g['worst_abs_delta_m'], 4)} "
            f"| {_fmt(g['best_abs_delta_before_calibration_m'], 4)} "
            f"| {g['importer_floor_calibration_offsets_m'] or '(none)'} "
            f"| {_fmt(min(angles) if angles else None, 3)} |")
    out.append("")

    out.append("## Table 2b — yaw: mesh geometry vs. the wall-label estimator "
               "and its 5-degree guard\n")
    out.append("| scene | mesh dominant yaw (deg) | importer label yaw, pre-guard "
               "(deg) | disagreement (deg) | yaw actually applied (deg) "
               "| guard would flip on the other estimate |")
    out.append("|---|---|---|---|---|---|")
    for r in rows:
        g = r["floor_agreement"]
        guard = g["yaw_guard_deg"]
        flip = ((abs(g["mesh_dominant_yaw_deg"]) >= guard)
                != (abs(g["importer_label_yaw_pre_guard_deg"]) >= guard))
        out.append(
            f"| {r['scene']} | {_fmt(g['mesh_dominant_yaw_deg'], 3)} "
            f"| {_fmt(g['importer_label_yaw_pre_guard_deg'], 3)} "
            f"| {_fmt(g['mesh_vs_label_yaw_disagreement_deg'], 3)} "
            f"| {_fmt(g['importer_yaw_derotation_deg'], 3)} "
            f"| {'YES' if flip else 'no'} |")
    out.append("")

    out.append("## Table 2c — what the `frame=\"world\"` stamp is worth\n")
    out.append("Directional edges recomputed in the raw Habitat world frame "
               "vs. the gravity-aligned, yaw-de-rotated frame the importer "
               "actually builds. Every edge in graph/relations/** is stamped "
               "`frame=\"world\"` regardless.\n")
    out.append("| scene | pairs with an edge (aligned frame) | type changes "
               "| share | dominant axis changes | share |")
    out.append("|---|---|---|---|---|---|")
    for r in rows:
        s = r["frame_sensitivity"]
        out.append(
            f"| {r['scene']} | {s['n_edges_in_aligned_frame']} "
            f"| {s['n_pairs_whose_directional_type_changes']} "
            f"| {_fmt(s['frac_pairs_whose_directional_type_changes'], 3)} "
            f"| {s['n_pairs_whose_dominant_axis_changes']} "
            f"| {_fmt(s['frac_pairs_whose_dominant_axis_changes'], 3)} |")
    out.append("")

    out.append("## Table 3 — measured scene scale (mesh-derived unless noted)\n")
    out.append("| scene | room diag (m) | floor diag (m) | storey height (m) "
               "| median object diag (m) | entity-set diag (m) | n entities "
               "| median pair dist (m) | observed floor area (m2) |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        s = r["scale"]
        out.append(
            f"| {r['scene']} | {_fmt(s['room_diag'], 3)} | {_fmt(s['floor_diag'], 3)} "
            f"| {_fmt(s['storey_height'], 3)} | {_fmt(s['obj_diag'], 3)} "
            f"| {_fmt(s['entity_bbox_diagonal_m'], 3)} | {s['n_entities']} "
            f"| {_fmt(s['pair_distance_median_m'], 3)} "
            f"| {_fmt(s['observed_floor_area_m2'], 2)} |")
    spread = {}
    for d in DENOMS:
        vals = [r["scale"][d] for r in rows if r["scale"][d]]
        spread[d] = {
            "min": min(vals), "max": max(vals),
            "max_over_min": max(vals) / min(vals),
            "mean": statistics.fmean(vals),
            "cv": statistics.pstdev(vals) / statistics.fmean(vals),
        }
    summary["scale_spread"] = spread
    out.append("")
    out.append("Spread of each denominator across the measured scenes:\n")
    out.append("| denominator | min | max | max/min | mean | CV |")
    out.append("|---|---|---|---|---|---|")
    for d in DENOMS:
        s = spread[d]
        out.append(f"| {d} | {_fmt(s['min'], 3)} | {_fmt(s['max'], 3)} "
                   f"| {_fmt(s['max_over_min'], 2)} | {_fmt(s['mean'], 3)} "
                   f"| {_fmt(s['cv'], 3)} |")
    out.append("")

    out.append("## Table 4 — MAIN DELIVERABLE: each hardcoded metre constant "
               "as a fraction of each scene's own scale\n")
    out.append("Read the max/min and CV columns honestly: they are a property "
               "of the DENOMINATOR, so every row sharing a denominator shares "
               "them. What differs per constant is WHERE its fraction sits — "
               "whether it is a few-percent-of-room quantity or a "
               "comparable-to-one-object quantity, and therefore whether the "
               "spread moves it across a decision boundary.\n")
    out.append("| constant | value (m) | denominator | "
               + " | ".join(scenes) + " | min | max | max/min | CV |")
    out.append("|---|---|---|" + "---|" * (len(scenes) + 4))
    const_rows = []
    for c in _constants():
        fracs = []
        for r in rows:
            s = r["scale"][c["denom"]]
            fracs.append(c["value"] / s if s else float("nan"))
        good = [f for f in fracs if not math.isnan(f)]
        rec = {
            **c,
            "fraction_by_scene": {sc: fr for sc, fr in zip(scenes, fracs)},
            "min": min(good), "max": max(good),
            "max_over_min": max(good) / min(good),
            "cv": statistics.pstdev(good) / statistics.fmean(good),
        }
        const_rows.append(rec)
        out.append(
            f"| `{_short(c['name'])}` | {c['value']:g} | {c['denom']} | "
            + " | ".join(_fmt(f, 4) for f in fracs)
            + f" | {_fmt(rec['min'], 4)} | {_fmt(rec['max'], 4)} "
            f"| {_fmt(rec['max_over_min'], 2)} | {_fmt(rec['cv'], 3)} |")
    summary["constants"] = const_rows
    out.append("")

    out.append("## Table 5 — the same constants re-expressed per scene, holding "
               "room_0's fraction fixed\n")
    out.append("If a constant is the *right* value for room_0, and it is really a "
               "fraction of scene scale, then transferring it means these values — "
               "not the frozen one.\n")
    out.append("| constant | frozen value (m) | denominator | "
               + " | ".join(scenes) + " |")
    out.append("|---|---|---|" + "---|" * len(scenes))
    ref_idx = scenes.index("room_0")
    for rec in const_rows:
        ref_frac = list(rec["fraction_by_scene"].values())[ref_idx]
        scaled = [ref_frac * rows[i]["scale"][rec["denom"]]
                  for i in range(len(rows))]
        rec["value_if_scaled_from_room_0"] = dict(zip(scenes, scaled))
        out.append(f"| `{_short(rec['name'])}` | {rec['value']:g} "
                   f"| {rec['denom']} | "
                   + " | ".join(_fmt(v, 3) for v in scaled) + " |")
    out.append("")

    out.append("## Table 6 — what the distance constants actually DO in each scene\n")
    out.append("Fraction of this scene's own entity pairs / entities on the "
               "permissive side of the threshold. Same constant, different "
               "selectivity per scene.\n")
    out.append("| measure | " + " | ".join(scenes) + " |")
    out.append("|---|" + "---|" * len(scenes))
    keys = [
        ("frac_pairs_within_sparse_max_distance",
         "pairs within sparse_max_distance = 2.5 m"),
        ("frac_pairs_within_legacy_near_threshold",
         "pairs within NEAR threshold = 1.0 m"),
        ("frac_pairs_dominant_delta_below_legacy_min_delta",
         "pairs REJECTED by LEGACY_MIN_DELTA = 0.3 m"),
        ("frac_pairs_dominant_delta_below_sparse_min_delta",
         "pairs REJECTED by sparse_min_delta = 0.5 m"),
        ("frac_entities_bottom_below_elevated_min",
         "entities below elevated_bottom_min_m = 0.30 m"),
    ]
    for k, label in keys:
        out.append(f"| {label} | "
                   + " | ".join(_fmt(r["threshold_effect"][k], 3) for r in rows)
                   + " |")
    out.append("")

    out.append("## Table 7 — constants in graph/relations/** that carry NO scale risk\n")
    out.append("| constant | value | why it is scale-free |")
    out.append("|---|---|---|")
    why = {"angle_deg": "angular, dimensionless under uniform scaling",
           "zero_length": "zero length; scale-invariant",
           "fraction": "already a ratio of two same-scene quantities"}
    for c in _scale_free_constants():
        out.append(f"| `{_short(c['name'])}` | {c['value']} "
                   f"| {why[c['kind']]} |")
    out.append("")
    summary["scale_free_constants"] = _scale_free_constants()
    return "\n".join(out) + "\n", summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--scene", action="append", default=None)
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = ap.parse_args(argv)

    data_root = args.data_root
    args.out.mkdir(parents=True, exist_ok=True)

    wanted = tuple(args.scene) if args.scene else SCENES
    rows: list[dict] = []
    missing: list[str] = []
    for scene in wanted:
        mesh = data_root / scene / "habitat" / "mesh_semantic.ply"
        info = data_root / scene / "habitat" / "info_semantic.json"
        if not (mesh.is_file() and info.is_file()):
            missing.append(scene)
            print(f"SKIP {scene}: missing {mesh if not mesh.is_file() else info}")
            continue
        print(f"measuring {scene} ...", flush=True)
        rows.append(audit_scene(scene, args.out, data_root))

    if not rows:
        print("no scene geometry available; nothing measured")
        return 1

    tables, summary = build_tables(rows)
    payload = {
        "schema": "frame_scale_audit",
        "schema_version": 1,
        "note": ("diagnostic only; no threshold, extractor or frame in the "
                 "pipeline was modified by this run"),
        "estimator_config": DEFAULT_CONFIG.to_dict(),
        "scenes_measured": [r["scene"] for r in rows],
        "scenes_missing_geometry": missing,
        "per_scene": rows,
        "summary": summary,
    }
    (args.out / "frame_scale_audit.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (args.out / "tables.md").write_text(tables, encoding="utf-8")
    print(tables)
    print(f"wrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
