"""Where does a missing support relation actually fail? Probe, do not fix.

  python3 tools/arkitscenes_support_evidence_probe.py --scene 41069025 \
      --target obj_1 --owner obj_14

Answers one question for a human-confirmed support pair the stage missed, by
looking at MESH VERTICES rather than at extracted patches:

  segmentation_evidence_missing  the owner segment does not contain the
                                 physical supporting surface. Nothing in the
                                 relation module can fix this.
  patch_extraction_failure       the owner segment DOES contain a usable
                                 supporting surface, but no qualifying patch
                                 was extracted from it. Extraction is the place
                                 to change.
  patch_selection_failure        a usable patch exists and was not chosen.

WHY THIS PROBE EXISTS
---------------------
An earlier analysis called `obj_1 -> obj_14` a patch-selection problem because
the stage reports the highest-overlap patch and that patch was 0.80 m from
contact. That was wrong twice over. First, if no extracted patch lies near
contact then selection cannot recover the pair whatever it picks. Second, the
relation module's pair test is ALREADY existential -- `graph/relations/
entity_patch_rest.py` evaluates every qualifying patch and accepts the pair if
any one of them passes; the highest-overlap ordering only decides which passing
patch gets reported. The select-then-report behaviour was in the analysis
tooling, not in the module under study.

So the question is not which patch was chosen. It is whether the supporting
surface is in the owner's vertices at all.

THE MEASUREMENT
---------------
Take the target's underside (1st percentile of its z, robust to stray points).
Take a slab of CONTACT_BAND_M around it, clipped to the target's XY footprint.
Then ask who owns the vertices in that slab, and how much of the footprint each
owner covers on a COVERAGE_CELL_M grid. Coverage is the honest statistic: a
handful of fringe vertices is not a supporting surface, and a raw count would
suggest otherwise.

Read-only. Imports nothing from the relations package and writes no config.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.arkitscenes import MESH_SUFFIX, read_mesh

DEFAULT_DATA_ROOT = Path.home() / "Desktop/datasets/arkitscenes/Validation"
DEFAULT_IDS = (REPO_ROOT / "runs" / "arkitscenes_mask3d_transfer"
               / "bundle_arkitscenes_41069025" / "vertex_instance_ids.npy")
DEFAULT_ENTITIES = (REPO_ROOT / "runs" / "arkit_label_image_ab_41069025"
                    / "rgb_tight" / "entities" / "manifest.json")
DEFAULT_OUT = REPO_ROOT / "runs" / "arkit_support_calibration"

CONTACT_BAND_M = 0.05
COVERAGE_CELL_M = 0.02
FOOTPRINT_PAD_M = 0.02
# Share of the target's footprint the owner must cover at contact height for
# the surface to count as present in the owner segment. Matches the patch
# stage's own min_coverage_ratio, so the probe asks the question extraction
# would ask rather than inventing a softer one.
OWNER_COVERAGE_MIN = 0.25


def coverage(points: np.ndarray, origin: np.ndarray, span: np.ndarray) -> tuple:
    """(cells occupied, cells available, ratio) on a fixed grid."""
    nx = max(1, int(np.ceil(span[0] / COVERAGE_CELL_M)))
    ny = max(1, int(np.ceil(span[1] / COVERAGE_CELL_M)))
    if not len(points):
        return 0, nx * ny, 0.0
    gx = np.clip(((points[:, 0] - origin[0]) / COVERAGE_CELL_M).astype(int),
                 0, nx - 1)
    gy = np.clip(((points[:, 1] - origin[1]) / COVERAGE_CELL_M).astype(int),
                 0, ny - 1)
    occupied = len(set(zip(gx.tolist(), gy.tolist())))
    return occupied, nx * ny, occupied / (nx * ny)


def probe(xyz: np.ndarray, ids: np.ndarray, target_id: int,
          owner_id: int) -> dict:
    target = xyz[ids == target_id]
    if not len(target):
        raise ValueError(f"instance {target_id} has no vertices")
    underside = float(np.percentile(target[:, 2], 1.0))
    origin = target[:, :2].min(axis=0)
    span = target[:, :2].max(axis=0) - origin

    in_footprint = (
        (xyz[:, 0] >= origin[0] - FOOTPRINT_PAD_M)
        & (xyz[:, 0] <= origin[0] + span[0] + FOOTPRINT_PAD_M)
        & (xyz[:, 1] >= origin[1] - FOOTPRINT_PAD_M)
        & (xyz[:, 1] <= origin[1] + span[1] + FOOTPRINT_PAD_M))
    slab = in_footprint & (np.abs(xyz[:, 2] - underside) <= CONTACT_BAND_M)

    owners = {}
    for name, mask in (("owner", slab & (ids == owner_id)),
                       ("target_itself", slab & (ids == target_id)),
                       ("unassigned", slab & (ids < 0)),
                       ("other_instances", slab & (ids >= 0)
                        & (ids != owner_id) & (ids != target_id))):
        pts = xyz[mask]
        cells, total, ratio = coverage(pts, origin, span)
        owners[name] = {
            "n_vertices": int(len(pts)),
            "footprint_cells_covered": cells,
            "footprint_cells_total": total,
            "footprint_coverage": round(ratio, 4),
            "z_spread_m": round(float(pts[:, 2].max() - pts[:, 2].min()), 4)
            if len(pts) else None,
        }

    owner_cov = owners["owner"]["footprint_coverage"]
    if owner_cov >= OWNER_COVERAGE_MIN:
        verdict = "patch_extraction_failure"
        because = (f"the owner covers {owner_cov:.1%} of the target footprint "
                   f"at contact height, at or above the {OWNER_COVERAGE_MIN:.0%} "
                   "the patch stage itself requires, so a usable surface is "
                   "present in the owner segment and extraction missed it")
    elif owners["unassigned"]["footprint_coverage"] >= OWNER_COVERAGE_MIN:
        verdict = "segmentation_evidence_missing"
        because = (f"the owner covers only {owner_cov:.1%} of the target "
                   "footprint at contact height, while UNASSIGNED mesh covers "
                   f"{owners['unassigned']['footprint_coverage']:.1%}. The "
                   "supporting surface exists in the mesh but was not given to "
                   "the owner segment, so no change inside the relation module "
                   "can recover this pair")
    else:
        verdict = "no_supporting_surface_in_mesh"
        because = ("no owner and no unassigned surface reaches the coverage "
                   "minimum at contact height; the reconstruction itself lacks "
                   "the supporting geometry here")

    return {
        "target_underside_z": round(underside, 4),
        "target_footprint_m2": round(float(span[0] * span[1]), 4),
        "contact_band_m": CONTACT_BAND_M,
        "coverage_cell_m": COVERAGE_CELL_M,
        "owner_coverage_min": OWNER_COVERAGE_MIN,
        "slab_vertices": int(slab.sum()),
        "by_owner": owners,
        "verdict": verdict,
        "because": because,
    }


def render(xyz, rgb, ids, target_id, owner_id) -> dict:
    from tools.arkitscenes_uid_visual_sheet import SceneRenderer
    renderer = SceneRenderer(xyz, rgb)
    t = np.flatnonzero(ids == target_id)
    o = np.flatnonzero(ids == owner_id)
    return {"side": renderer.isolated_pair(t, o),
            "ctx": renderer.context_pair(t, o)}


def rgb_crop(scene_dir: Path, xyz_canonical, rotation, vertices) -> str | None:
    """Best real photograph containing the pair, via the validated crop path."""
    try:
        from extractors.arkitscenes_rgb_crops import RgbCropSource
        source = RgbCropSource(scene_dir, xyz_canonical, rotation,
                               stride=6, n_views=1, context_pad=0.9)
        crops = source.crops_for(np.asarray(vertices))
    except Exception as exc:                      # dataset or pose gaps
        print(f"    (rgb crop unavailable: {exc})")
        return None
    if not crops:
        return None
    buffer = io.BytesIO()
    crops[0].save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", default="41069025")
    ap.add_argument("--target", default="obj_1")
    ap.add_argument("--owner", default="obj_14")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--ids", type=Path, default=DEFAULT_IDS)
    ap.add_argument("--entities", type=Path, default=DEFAULT_ENTITIES)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-visual", action="store_true")
    args = ap.parse_args(argv)

    scene_dir = args.data_root / args.scene
    mesh = read_mesh(scene_dir / f"{args.scene}{MESH_SUFFIX}")
    ids = np.load(args.ids)
    manifest = json.loads(args.entities.read_text())
    index_of = {e["identity"]["object_uid"]:
                int(e["geometry_handle"].rsplit("#", 1)[1])
                for e in manifest["entities"]}
    target_id, owner_id = index_of[args.target], index_of[args.owner]

    result = probe(mesh.xyz, ids, target_id, owner_id)
    result.update({
        "scene_id": f"arkitscenes_{args.scene}",
        "pair_id": f"{args.target}->{args.owner}",
        "read_only": True, "logic_changed": False, "thresholds_changed": False,
        "delivered_partition_unassigned_rate": round(float((ids < 0).mean()), 4),
        "pair_test_is_existential": (
            "existential: graph/relations/entity_patch_rest.py evaluates every "
            "qualifying "
            "patch and accepts the pair if any passes; the highest-overlap "
            "ordering only picks which passing patch is reported"),
    })

    out_dir = args.out_root
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{result['scene_id']}_{args.target}_{args.owner}_evidence_probe"
    (out_dir / f"{stem}.json").write_text(json.dumps(result, indent=1) + "\n")

    if not args.no_visual:
        pics = render(mesh.xyz, mesh.rgb, ids, target_id, owner_id)
        rotation = np.asarray(json.loads(
            (args.entities.parent / "manifest.json").read_text()
        ).get("frame", {}).get("rotation_row_major", np.eye(3).tolist()))
        both = np.flatnonzero((ids == target_id) | (ids == owner_id))
        photo = rgb_crop(scene_dir, mesh.xyz, np.eye(3), both)
        rows = "".join(
            f'<div><img src="{src}"><span>{cap}</span></div>'
            for src, cap in (
                (pics["side"], "side-on: pink target, teal owner"),
                (pics["ctx"][0], "room A"), (pics["ctx"][1], "room B"))
            + (((photo, "real capture frame"),) if photo else ()))
        by = result["by_owner"]
        table = "".join(
            f"<tr><td>{k}</td><td>{v['n_vertices']}</td>"
            f"<td>{v['footprint_coverage']:.1%}</td>"
            f"<td>{'' if v['z_spread_m'] is None else v['z_spread_m']}</td></tr>"
            for k, v in by.items())
        (out_dir / f"{stem}.html").write_text(f"""<title>{stem}</title>
<style>body{{font:15px/1.5 ui-sans-serif,system-ui,sans-serif;max-width:900px;
margin:0 auto;padding:24px}}img{{width:100%;border:1px solid #ccc;border-radius:5px}}
.imgs{{display:flex;gap:8px}}.imgs div{{flex:1;text-align:center}}
.imgs span{{font-size:11px;color:#666}}table{{border-collapse:collapse;
margin-top:14px}}td,th{{border-bottom:1px solid #ddd;padding:5px 10px;
text-align:left}}.v{{border-left:3px solid #c0158f;padding:8px 14px;
margin:16px 0}}</style>
<h1>{result['pair_id']} — support evidence probe</h1>
<div class="imgs">{rows}</div>
<div class="v"><b>{result['verdict']}</b><br>{result['because']}</div>
<table><tr><th>owner of the contact slab</th><th>vertices</th>
<th>footprint coverage</th><th>z spread m</th></tr>{table}</table>
<p style="color:#666;font-size:13px">Contact slab = ±{CONTACT_BAND_M} m about
the target underside, clipped to its XY footprint. Coverage on a
{COVERAGE_CELL_M} m grid. Read-only; no logic or threshold changed.</p>
""")

    print(f"=== {result['pair_id']}  support evidence probe   (read-only)")
    print(f"    target underside z : {result['target_underside_z']}")
    print(f"    contact slab       : {result['slab_vertices']} vertices "
          f"within ±{CONTACT_BAND_M} m of it, inside the target footprint")
    for name, stats in result["by_owner"].items():
        print(f"      {name:16s} {stats['n_vertices']:5d} verts  "
              f"covers {stats['footprint_coverage']:6.1%} of the footprint")
    print(f"    VERDICT: {result['verdict']}")
    print(f"      {result['because']}")
    print(f"    delivered partition is {result['delivered_partition_unassigned_rate']:.1%} "
          "unassigned")
    print(f"    -> {(out_dir / stem).relative_to(REPO_ROOT)}.json/.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
