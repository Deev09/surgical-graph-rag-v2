"""Visual, label-free UID confirmation sheet: raw mesh context + isolation.

  python3 tools/arkitscenes_uid_visual_sheet.py --scene 41069025

Renders every delivered region three ways and writes ONE self-contained HTML
page with the answer form embedded. No external files, no CDN, no fonts.

WHY VISUAL
----------
The text sheet asked the owner to recognise "a thin thing lying on the floor"
from `1.35 x 4.24 x 0.24 m, underside 2.26 m`. That is a puzzle, not a question.
Three renders per region answer it directly: where the thing sits in the room,
and what it looks like on its own.

  context x2   the whole captured mesh in its own vertex colours, desaturated,
               seen from above with the ceiling clipped away, and this region
               alone painted bright over it, from two azimuths. Shows WHERE it
               is and what surrounds it. The highlight is drawn over the scene
               rather than depth-tested, so a region behind a wall still shows.
  isolated     only this region, in its true captured colours, auto-framed.
               Shows WHAT it is.

LABEL-FREE, ENFORCED
--------------------
The page never shows a predicted class, and this module never reads
`display_label`. Asking "is `obj_12` the trash can?" *because the labeler said
trash-can* would launder a model guess into ground truth; the key exists to be
independent of the model. A test AST-checks that `display_label` appears
nowhere in this file.

Region order is by vertex count, not by any semantic score.

THE FOUR OUTCOMES ARE ALL FIRST-CLASS
-------------------------------------
`uid`, `none/missing`, `overmerged into <uid>`, `ambiguous`. All four occur in
this scene; collapsing any of them loses information the owner actually has.
The form emits JSON matching `arkitscenes_uid_confirmation_v1`.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.arkitscenes import MESH_SUFFIX, read_mesh

DEFAULT_DATA_ROOT = Path.home() / "Desktop/datasets/arkitscenes/Validation"
DEFAULT_ENTITIES = (REPO_ROOT / "runs" / "arkit_label_image_ab_41069025"
                    / "rgb_tight" / "entities" / "manifest.json")
DEFAULT_IDS = (REPO_ROOT / "runs" / "arkitscenes_mask3d_transfer"
               / "bundle_arkitscenes_41069025" / "vertex_instance_ids.npy")
DEFAULT_OUT = REPO_ROOT / "runs" / "arkit_uid_confirmation"

CONTEXT_PX = 260
ISOLATED_PX = 260
AZIMUTHS = (35.0, 215.0)
ELEVATION = 22.0
RENDER_STRIDE = 4          # scene points used for the context base
HIGHLIGHT = np.array([255, 40, 190], dtype=np.uint8)
# Base points above this height over the lowest point are dropped, so the
# ceiling does not cover the whole room in a view from above. The HIGHLIGHTED
# region is never clipped -- if the region is the ceiling, it must still show.
CEILING_CLIP_M = 2.2
# Below this many highlighted pixels a region is effectively invisible in the
# room view, so a crosshair marks it instead. obj_29 (39 vertices) rendered at
# 0.0% and would otherwise be unanswerable.
MIN_HIGHLIGHT_PX = 60


def _splat_radius(n_points: int) -> int:
    """Sparse regions need fatter points or they vanish into single pixels."""
    return 1 if n_points > 2000 else (2 if n_points > 200 else 3)


def _crosshair(canvas: np.ndarray, u: int, v: int,
               colour: np.ndarray, arm: int = 9, gap: int = 4) -> None:
    size = canvas.shape[0]
    for d in range(gap, arm + 1):
        for uu, vv in ((u + d, v), (u - d, v), (u, v + d), (u, v - d)):
            if 0 <= uu < size and 0 <= vv < size:
                canvas[vv, uu] = colour


def _basis(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Rows: right, up, forward. +z is up in the canonical frame."""
    # forward is the direction the camera looks ALONG, so a negative z means
    # the viewer is above the scene looking down. It was positive, which put
    # the camera under the floor: the ceiling slab rendered 97.7% occluded and
    # the largest region in the scene highlighted 1% of its frame.
    a, e = np.radians(azimuth_deg), np.radians(elevation_deg)
    forward = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), -np.sin(e)])
    forward /= np.linalg.norm(forward)
    right = np.cross(np.array([0.0, 0.0, 1.0]), forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    return np.stack([right, up, forward])


def _project(xyz: np.ndarray, basis: np.ndarray, centre: np.ndarray,
             scale: float, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local = (xyz - centre) @ basis.T
    u = np.rint(local[:, 0] * scale + size / 2).astype(np.int64)
    v = np.rint(-local[:, 1] * scale + size / 2).astype(np.int64)
    return u, v, local[:, 2]


def _paint(canvas: np.ndarray, depth_buf: np.ndarray, u, v, depth, colours,
           radius: int = 1) -> None:
    """Painter's algorithm, far to near, with a small square splat."""
    size = canvas.shape[0]
    order = np.argsort(depth)[::-1]
    u, v, depth = u[order], v[order], depth[order]
    colours = colours[order] if colours.ndim == 2 else colours
    for dy in range(-radius, radius + 1):
        vv = v + dy
        for dx in range(-radius, radius + 1):
            uu = u + dx
            ok = (uu >= 0) & (uu < size) & (vv >= 0) & (vv < size)
            canvas[vv[ok], uu[ok]] = colours[ok] if colours.ndim == 2 else colours
            depth_buf[vv[ok], uu[ok]] = depth[ok]


def _png(canvas: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(canvas).save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class SceneRenderer:
    """Renders context views against a cached, desaturated base per azimuth."""

    def __init__(self, xyz: np.ndarray, rgb: np.ndarray):
        self.xyz = xyz
        self.rgb = rgb
        lo, hi = xyz.min(axis=0), xyz.max(axis=0)
        self.centre = (lo + hi) / 2.0
        self.scale = 0.85 * CONTEXT_PX / max(float((hi - lo).max()), 1e-6)
        self._base: dict[float, tuple] = {}
        keep = np.flatnonzero(
            xyz[:, 2] - xyz[:, 2].min() <= CEILING_CLIP_M)[::RENDER_STRIDE]
        for azimuth in AZIMUTHS:
            basis = _basis(azimuth, ELEVATION)
            u, v, depth = _project(xyz[keep], basis, self.centre, self.scale,
                                   CONTEXT_PX)
            grey = (0.35 * rgb[keep].astype(np.float64)
                    + 0.65 * 190.0).astype(np.uint8)
            canvas = np.full((CONTEXT_PX, CONTEXT_PX, 3), 248, dtype=np.uint8)
            depth_buf = np.full((CONTEXT_PX, CONTEXT_PX), np.inf)
            _paint(canvas, depth_buf, u, v, depth, grey)
            self._base[azimuth] = (basis, canvas, depth_buf)

    def context(self, vertices: np.ndarray) -> list[str]:
        out = []
        for azimuth in AZIMUTHS:
            basis, base, _ = self._base[azimuth]
            canvas = base.copy()
            u, v, depth = _project(self.xyz[vertices], basis, self.centre,
                                   self.scale, CONTEXT_PX)
            # Drawn OVER the scene with no occlusion test, on purpose: a
            # highlight the room can hide is useless for confirmation, and the
            # two azimuths supply the parallax instead. Stated in the caption.
            scratch = np.full((CONTEXT_PX, CONTEXT_PX), np.inf)
            _paint(canvas, scratch, u, v, depth, HIGHLIGHT,
                   radius=_splat_radius(len(vertices)))
            painted = int((np.abs(canvas.astype(np.int32)
                                  - HIGHLIGHT.astype(np.int32)).sum(axis=2)
                           < 30).sum())
            if painted < MIN_HIGHLIGHT_PX:
                inside = ((u >= 0) & (u < CONTEXT_PX)
                          & (v >= 0) & (v < CONTEXT_PX))
                if inside.any():
                    _crosshair(canvas, int(np.median(u[inside])),
                               int(np.median(v[inside])), HIGHLIGHT)
            out.append(_png(canvas))
        return out

    def isolated(self, vertices: np.ndarray) -> str:
        points = self.xyz[vertices]
        colours = self.rgb[vertices]
        lo, hi = points.min(axis=0), points.max(axis=0)
        centre = (lo + hi) / 2.0
        scale = 0.80 * ISOLATED_PX / max(float((hi - lo).max()), 1e-6)
        basis = _basis(AZIMUTHS[0], ELEVATION)
        u, v, depth = _project(points, basis, centre, scale, ISOLATED_PX)
        canvas = np.full((ISOLATED_PX, ISOLATED_PX, 3), 255, dtype=np.uint8)
        depth_buf = np.full((ISOLATED_PX, ISOLATED_PX), np.inf)
        _paint(canvas, depth_buf, u, v, depth, colours,
               radius=_splat_radius(len(points)))
        return _png(canvas)


def build_html(scene_id: str, objects: list[dict], regions: list[dict],
               provenance: dict) -> str:
    uid_options = "".join(
        f'<option value="{r["uid"]}">{r["uid"]}</option>' for r in regions)
    rows = []
    for obj in objects:
        for i in range(obj["confirmed_count"]):
            suffix = f" #{i + 1}" if obj["confirmed_count"] > 1 else ""
            slot = f'{obj["object"]}{suffix}'
            rows.append(f"""
      <tr data-object="{obj['object']}" data-slot="{slot}">
        <td class="obj">{slot}</td>
        <td><select class="pick">
          <option value="">— choose —</option>
          <option value="none_missing">none / missing (not delivered)</option>
          <option value="ambiguous">ambiguous (cannot tell)</option>
          {uid_options}
        </select></td>
        <td><label class="om"><input type="checkbox" class="overmerged">
          overmerged into the region above</label></td>
      </tr>""")

    cards = []
    for r in regions:
        cards.append(f"""
    <figure class="card" id="{r['uid']}">
      <figcaption><b>{r['uid']}</b>
        <span class="dim">{r['n_vertices']} verts ·
        {r['width_m']}×{r['depth_m']}×{r['height_m']} m ·
        {r['footprint_m2']} m² · underside {r['underside_above_floor_m']} m</span>
      </figcaption>
      <div class="imgs">
        <div><img src="{r['ctx'][0]}" alt="context view A"><span>context A</span></div>
        <div><img src="{r['ctx'][1]}" alt="context view B"><span>context B</span></div>
        <div><img src="{r['iso']}" alt="isolated"><span>isolated</span></div>
      </div>
    </figure>""")

    facts = "".join(
        f"<tr><td>{o['object']}</td><td>{o['confirmed_count']}</td>"
        f"<td>{o['source']}</td></tr>" for o in objects)

    return f"""<title>UID confirmation — {scene_id}</title>
<style>
  :root {{ --bg:#fff; --fg:#15171a; --mut:#666; --line:#e3e5e8; --hi:#c0158f; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#14161a; --fg:#e9ebee; --mut:#9aa1aa; --line:#2b2f36; --hi:#ff5cc8; }} }}
  :root[data-theme="dark"] {{
    --bg:#14161a; --fg:#e9ebee; --mut:#9aa1aa; --line:#2b2f36; --hi:#ff5cc8; }}
  :root[data-theme="light"] {{
    --bg:#fff; --fg:#15171a; --mut:#666; --line:#e3e5e8; --hi:#c0158f; }}
  body {{ background:var(--bg); color:var(--fg); margin:0 auto; padding:24px;
    max-width:1000px; font:15px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  h2 {{ font-size:16px; margin:28px 0 8px; padding-top:14px;
    border-top:1px solid var(--line); }}
  .sub {{ color:var(--mut); font-size:13px; margin:0 0 18px; }}
  table {{ border-collapse:collapse; width:100%; font-size:14px; }}
  th,td {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--line);
    vertical-align:middle; }}
  th {{ color:var(--mut); font-weight:600; }}
  td.obj {{ font-weight:600; white-space:nowrap; }}
  select {{ font:inherit; padding:4px; max-width:260px; background:var(--bg);
    color:var(--fg); border:1px solid var(--line); border-radius:5px; }}
  .om {{ color:var(--mut); font-size:13px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr));
    gap:14px; }}
  .card {{ margin:0; border:1px solid var(--line); border-radius:8px; padding:8px; }}
  .card figcaption {{ font-size:13px; margin-bottom:6px; }}
  .dim {{ color:var(--mut); display:block; font-size:11.5px; }}
  .imgs {{ display:flex; gap:5px; }}
  .imgs div {{ flex:1; text-align:center; }}
  .imgs img {{ width:100%; max-width:100%; border:1px solid var(--line);
    border-radius:4px; display:block; }}
  .imgs span {{ font-size:10.5px; color:var(--mut); }}
  button {{ font:inherit; padding:7px 14px; border-radius:6px; cursor:pointer;
    border:1px solid var(--line); background:var(--hi); color:#fff; }}
  pre {{ background:rgba(128,128,128,.10); padding:12px; border-radius:6px;
    overflow-x:auto; font-size:12.5px; }}
  .note {{ border-left:3px solid var(--hi); padding:6px 12px; color:var(--mut);
    font-size:13.5px; }}
</style>

<h1>UID confirmation — {scene_id}</h1>
<p class="sub">{provenance['n_regions']} delivered regions ·
bundle <code>{provenance['bundle_hash']}</code></p>

<p class="note">Predicted class labels are deliberately not shown anywhere on
this page. Regions are ordered by size. Pink = the region; grey = the rest of
the captured mesh, seen from above with the ceiling clipped away. The pink is
drawn <em>over</em> the scene rather than hidden behind it, so a region behind
a wall still shows — use the two azimuths to judge depth. A region too small
to paint visibly is marked with a crosshair instead.</p>

<h2>1 · Human object facts <span class="dim">confirmed; not in question here</span></h2>
<table><tr><th>object</th><th>count</th><th>source</th></tr>{facts}</table>

<h2>2 · Predicted-UID correspondence <span class="dim">what needs your answer</span></h2>
<p class="sub">Separate from section 1 on purpose: this mapping is revisable and
dies with the current segmentation.</p>
<table>
  <tr><th>object</th><th>choose one</th><th>or mark</th></tr>{''.join(rows)}
</table>
<p style="margin-top:14px"><button onclick="emit()">Build JSON</button></p>
<pre id="out">(choose above, then press Build JSON)</pre>

<h2>3 · Delivered regions</h2>
<div class="grid">{''.join(cards)}</div>

<script>
function emit() {{
  const byObject = {{}};
  document.querySelectorAll('tr[data-slot]').forEach(tr => {{
    const object = tr.dataset.object;
    const pick = tr.querySelector('.pick').value;
    const overmerged = tr.querySelector('.overmerged').checked;
    const a = {{uid:null, none_missing:null, overmerged_into:null, ambiguous:null}};
    if (pick === 'none_missing') a.none_missing = true;
    else if (pick === 'ambiguous') a.ambiguous = true;
    else if (pick && overmerged) a.overmerged_into = pick;
    else if (pick) a.uid = pick;
    (byObject[object] = byObject[object] || []).push(a);
  }});
  const out = {{
    schema: 'arkitscenes_uid_confirmation_v1',
    scene_id: '{scene_id}',
    status: 'OWNER_CONFIRMED',
    predicted_uid_correspondence:
      Object.entries(byObject).map(([object, assignments]) =>
        ({{object, assignments}}))
  }};
  document.getElementById('out').textContent = JSON.stringify(out, null, 1);
}}
</script>
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", default="41069025")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--entities", type=Path, default=DEFAULT_ENTITIES)
    ap.add_argument("--ids", type=Path, default=DEFAULT_IDS)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    scene_id = f"arkitscenes_{args.scene}"
    scene_dir = args.data_root / args.scene
    mesh = read_mesh(scene_dir / f"{args.scene}{MESH_SUFFIX}")
    instance_ids = np.load(args.ids)
    if len(instance_ids) != len(mesh.xyz):
        raise ValueError(
            f"{len(instance_ids)} instance ids against {len(mesh.xyz)} mesh "
            "vertices; the sidecar does not belong to this mesh")

    manifest = json.loads(args.entities.read_text())
    floor_z = min(e["bbox_aabb"][0][2] for e in manifest["entities"])
    renderer = SceneRenderer(mesh.xyz, mesh.rgb)

    regions = []
    for entity in manifest["entities"]:
        uid = entity["identity"]["object_uid"]
        index = int(entity["geometry_handle"].rsplit("#", 1)[1])
        vertices = np.flatnonzero(instance_ids == index)
        if len(vertices) != entity["extraction_diagnostics"]["n_vertices"]:
            raise ValueError(
                f"{uid}: {len(vertices)} vertices in the sidecar, manifest says "
                f"{entity['extraction_diagnostics']['n_vertices']}")
        (x0, y0, z0), (x1, y1, z1) = entity["bbox_aabb"]
        regions.append({
            "uid": uid,
            "n_vertices": int(len(vertices)),
            "width_m": round(x1 - x0, 2), "depth_m": round(y1 - y0, 2),
            "height_m": round(z1 - z0, 2),
            "footprint_m2": round((x1 - x0) * (y1 - y0), 2),
            "underside_above_floor_m": round(z0 - floor_z, 2),
            "ctx": renderer.context(vertices),
            "iso": renderer.isolated(vertices),
        })
    regions.sort(key=lambda r: -r["n_vertices"])

    objects = [
        {"object": "rug", "confirmed_count": 1,
         "source": "owner confirmation 2026-08-10"},
        {"object": "trash can", "confirmed_count": 1,
         "source": "owner confirmation 2026-08-10"},
        {"object": "main kitchen counter", "confirmed_count": 1,
         "source": "owner confirmation 2026-08-10"},
        {"object": "visible sofa cushion", "confirmed_count": 2,
         "source": "owner confirmation 2026-08-10"},
    ]
    provenance = {
        "n_regions": len(regions),
        "bundle_hash": manifest.get("bundle_hash"),
        "entity_manifest_sha256": hashlib.sha256(
            args.entities.read_bytes()).hexdigest(),
        "mesh": f"{args.scene}{MESH_SUFFIX}",
    }
    html = build_html(scene_id, objects, regions, provenance)
    args.out_root.mkdir(parents=True, exist_ok=True)
    path = args.out_root / f"{scene_id}_uid_confirmation.html"
    path.write_text(html)

    print(f"=== {scene_id} visual UID confirmation sheet")
    print(f"    regions rendered : {len(regions)} × (2 context + 1 isolated)")
    print(f"    slots to assign  : "
          f"{sum(o['confirmed_count'] for o in objects)}")
    print(f"    predicted labels : not read, not shown")
    print(f"    page             : {path.stat().st_size / 1e6:.1f} MB self-contained")
    print(f"    -> {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
