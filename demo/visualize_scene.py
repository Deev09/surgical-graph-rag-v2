"""Render a top-down floorplan of a Replica scene with the system's answers drawn
on it -- walls, every object footprint, the support furniture, and each
ON_ENTITY_SURFACE support pair as a labeled connector. A visual sanity check that
the relations land where they should.

  python3 demo/visualize_scene.py                       # room_0
  python3 demo/visualize_scene.py /path/to/room scene_id

Writes demo/<scene_id>_floorplan.png. Pure PIL (no matplotlib).
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.replica_habitat_import import import_habitat_room
from graph.builder import ExtractorRun, build_graph
from graph.relations.contacts_surface import ContactsSurfaceConfig, ContactsSurfaceExtractor
from graph.relations.directional import DirectionalConfig, DirectionalExtractor
from graph.relations.on_entity_surface import OnEntitySurfaceConfig, OnEntitySurfaceExtractor
from graph.relations.on_surface import OnSurfaceConfig, OnSurfaceExtractor
from graph.relations.surface import SurfaceProximityConfig, SurfaceProximityExtractor

DEFAULT_ROOM = Path.home() / "Desktop/datasets/replica/room_0"
PLOT = 1500          # plot area px (longest world axis maps to this)
MARGIN = 60
LEGEND_H = 150

# colors
C_BG = (255, 255, 255)
C_WALL = (40, 40, 40)
C_OBJ = (210, 210, 210)         # generic object footprint
C_OBJ_EDGE = (170, 170, 170)
C_SUPPORTER = (40, 110, 220)    # furniture that supports (table/desk/...)
C_SUPPORTED = (235, 130, 30)    # object resting on furniture
C_FLOOR_OBJ = (60, 170, 90)     # object resting on the floor
C_WALL_OBJ = (200, 40, 60)      # object against a wall
C_LINK = (235, 130, 30)


def _font(sz):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def _runs():
    return [
        ExtractorRun(DirectionalExtractor(), DirectionalConfig(mode="sparse")),
        ExtractorRun(SurfaceProximityExtractor(),
                     SurfaceProximityConfig(use_polygon_clip=True)),
        ExtractorRun(OnSurfaceExtractor(), OnSurfaceConfig()),
        ExtractorRun(ContactsSurfaceExtractor(), ContactsSurfaceConfig()),
        ExtractorRun(OnEntitySurfaceExtractor(), OnEntitySurfaceConfig()),
    ]


def main() -> int:
    room = Path(sys.argv[1]) if len(sys.argv) >= 2 else DEFAULT_ROOM
    scene = sys.argv[2] if len(sys.argv) >= 3 else "replica_room_0"
    if not (room / "habitat" / "info_semantic.json").exists():
        print(f"Refusing: {room}/habitat/info_semantic.json not found.")
        return 1

    arts = import_habitat_room(room, scene)
    bundle, _ = build_graph(arts, _runs(), density_policy="phase2_telemetry_only")
    ent = {e.identity.object_uid: e for e in arts.entities}
    label = {u: e.identity.display_label for u, e in ent.items()}

    support = [(e.source.uid, e.target.uid, e.evidence.get("owner_class"))
               for e in bundle.edges if e.type == "ON_ENTITY_SURFACE"]
    supporters = {t for _, t, _ in support}
    supported = {s for s, _, _ in support}
    floor_objs = {e.source.uid for e in bundle.edges
                  if e.type == "ON_SURFACE" and e.target.uid.startswith("floor")}
    wall_objs = {e.source.uid for e in bundle.edges if e.type == "CONTACTS_SURFACE"}
    walls = [s for s in arts.structural_surfaces if s.surface_type == "wall"]

    # ---- world bounds (objects + walls), XY ----
    xs, ys = [], []
    for e in arts.entities:
        (lo, hi) = e.bbox_aabb
        xs += [lo[0], hi[0]]; ys += [lo[1], hi[1]]
    for w in walls:
        for p in (w.polygon or []):
            xs.append(p[0]); ys.append(p[1])
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    span = max(maxx - minx, maxy - miny) or 1.0
    scale = PLOT / span
    W = int((maxx - minx) * scale) + 2 * MARGIN
    H = int((maxy - miny) * scale) + 2 * MARGIN + LEGEND_H

    def px(x, y):
        return (MARGIN + (x - minx) * scale,
                MARGIN + (maxy - y) * scale)   # flip Y so north is up

    img = Image.new("RGB", (W, H), C_BG)
    d = ImageDraw.Draw(img)
    f_sm, f_md, f_lg = _font(13), _font(16), _font(22)

    # walls
    for w in walls:
        pts = [px(p[0], p[1]) for p in (w.polygon or [])]
        if len(pts) >= 2:
            d.line(pts + [pts[0]], fill=C_WALL, width=5)

    def rect(uid, fill, edge, width=1):
        lo, hi = ent[uid].bbox_aabb
        x0, y0 = px(lo[0], hi[1]); x1, y1 = px(hi[0], lo[1])
        d.rectangle([x0, y0, x1, y1], fill=fill, outline=edge, width=width)

    # generic objects first (background)
    for u in ent:
        if u in supporters or u in supported:
            continue
        col = (C_FLOOR_OBJ if u in floor_objs else
               C_WALL_OBJ if u in wall_objs else C_OBJ)
        rect(u, fill=None if (u in floor_objs or u in wall_objs) else C_OBJ,
             edge=col, width=3 if (u in floor_objs or u in wall_objs) else 1)

    # supporters (furniture)
    for u in supporters:
        rect(u, fill=None, edge=C_SUPPORTER, width=4)
        cx, cy = px(*ent[u].centroid[:2])
        d.text((cx + 3, cy), label.get(u, "?"), fill=C_SUPPORTER, font=f_md)

    # support links + supported objects
    for s, t, oc in support:
        sc = px(*ent[s].centroid[:2]); tc = px(*ent[t].centroid[:2])
        d.line([sc, tc], fill=C_LINK, width=2)
        rect(s, fill=C_SUPPORTED, edge=(120, 60, 0), width=1)
        d.text((sc[0] + 3, sc[1] - 14), label.get(s, "?"), fill=(120, 60, 0), font=f_sm)

    # ---- title + legend ----
    d.text((MARGIN, 10), f"{scene}   top-down floorplan", fill=(0, 0, 0), font=f_lg)
    ly = H - LEGEND_H + 20
    d.text((MARGIN, ly), f"objects: {len(ent)}   walls: {len(walls)}   "
           f"support pairs: {len(support)}   on-floor: {len(floor_objs)}   "
           f"against-wall: {len(wall_objs)}", fill=(0, 0, 0), font=f_md)
    items = [(C_WALL, "wall"), (C_SUPPORTER, "support furniture"),
             (C_SUPPORTED, "ON furniture (support pair)"),
             (C_FLOOR_OBJ, "on floor"), (C_WALL_OBJ, "against wall"),
             (C_OBJ, "other object")]
    lx = MARGIN
    for col, txt in items:
        d.rectangle([lx, ly + 40, lx + 26, ly + 60], fill=col, outline=(80, 80, 80))
        d.text((lx + 32, ly + 42), txt, fill=(0, 0, 0), font=f_sm)
        lx += 40 + int(d.textlength(txt, font=f_sm)) + 28

    out = REPO_ROOT / "demo" / f"{scene}_floorplan.png"
    img.save(out)
    print(f"wrote {out}  ({W}x{H})")
    print(f"  support pairs drawn: {len(support)}")
    for s, t, oc in support:
        print(f"    {label.get(s,'?')} ({s})  ON  {label.get(t,'?')} ({t}) [{oc}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
