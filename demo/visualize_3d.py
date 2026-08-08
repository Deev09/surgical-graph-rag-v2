"""Oblique (isometric) 3D view of a Replica scene: every object drawn as a shaded
3D box, walls as vertical planes, support furniture and the objects resting on it
highlighted. Same data as the floorplan, seen from the side so height reads.

  python3 demo/visualize_3d.py                          # room_0
  python3 demo/visualize_3d.py /path/to/room scene_id

Writes demo/<scene_id>_iso3d.png. Pure PIL, painter's-algorithm depth sort.
"""
from __future__ import annotations

import math
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
FIT = 1500
MARGIN = 60
LEGEND_H = 110
COS30, SIN30 = math.cos(math.radians(30)), math.sin(math.radians(30))


def _font(sz):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def _shade(c, k):
    return tuple(max(0, min(255, int(v * k))) for v in c)


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
    support = [(e.source.uid, e.target.uid)
               for e in bundle.edges if e.type == "ON_ENTITY_SURFACE"]
    supporters = {t for _, t in support}
    supported = {s for s, _ in support}
    walls = [s for s in arts.structural_surfaces if s.surface_type == "wall"]

    def iso(p):
        x, y, z = p
        return ((x - y) * COS30, (x + y) * SIN30 - z)

    # projected bounds over all box corners + wall corners
    us, vs = [], []
    def acc(p):
        u, v = iso(p); us.append(u); vs.append(v)
    for e in arts.entities:
        lo, hi = e.bbox_aabb
        for cx in (lo[0], hi[0]):
            for cy in (lo[1], hi[1]):
                for cz in (lo[2], hi[2]):
                    acc((cx, cy, cz))
    for w in walls:
        for p in (w.polygon or []):
            acc(p)
    umin, umax, vmin, vmax = min(us), max(us), min(vs), max(vs)
    scale = FIT / (max(umax - umin, vmax - vmin) or 1.0)
    W = int((umax - umin) * scale) + 2 * MARGIN
    H = int((vmax - vmin) * scale) + 2 * MARGIN + LEGEND_H

    def scr(p):
        u, v = iso(p)
        return (MARGIN + (u - umin) * scale, MARGIN + (v - vmin) * scale)

    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img, "RGBA")

    # walls first (background), as translucent vertical planes
    for w in walls:
        pts = [scr(p) for p in (w.polygon or [])]
        if len(pts) >= 3:
            d.polygon(pts, fill=(150, 150, 150, 60), outline=(90, 90, 90))

    def box_color(uid):
        if uid in supporters:
            return (40, 110, 220)
        if uid in supported:
            return (235, 130, 30)
        return (200, 200, 200)

    # depth sort: far (large x+y) first
    order = sorted(ent.values(),
                   key=lambda e: -(e.centroid[0] + e.centroid[1]))
    for e in order:
        lo, hi = e.bbox_aabb
        x0, y0, z0 = lo; x1, y1, z1 = hi
        base = box_color(e.identity.object_uid)
        is_hi = e.identity.object_uid in supporters or e.identity.object_uid in supported
        edge = (60, 60, 60) if is_hi else (175, 175, 175)
        a = 255 if is_hi else 150
        top = [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        fx = [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)]
        fy = [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)]
        d.polygon([scr(p) for p in fy], fill=_shade(base, 0.62) + (a,), outline=edge)
        d.polygon([scr(p) for p in fx], fill=_shade(base, 0.80) + (a,), outline=edge)
        d.polygon([scr(p) for p in top], fill=base + (a,), outline=edge)

    # support links (draw on top)
    for s, t in support:
        d.line([scr(ent[s].centroid), scr(ent[t].centroid)],
               fill=(235, 130, 30, 255), width=3)

    f_lg, f_sm = _font(24), _font(15)
    d.text((MARGIN, 14), f"{scene}   oblique 3D view", fill=(0, 0, 0), font=f_lg)
    ly = H - LEGEND_H + 24
    d.text((MARGIN, ly), f"objects: {len(ent)}   walls: {len(walls)}   "
           f"support pairs: {len(support)}   (height is up; viewed from above-side)",
           fill=(0, 0, 0), font=f_sm)
    lx = MARGIN
    for col, txt in [((40, 110, 220), "support furniture"),
                     ((235, 130, 30), "object ON furniture"),
                     ((200, 200, 200), "other object"),
                     ((150, 150, 150), "wall")]:
        d.rectangle([lx, ly + 36, lx + 24, ly + 56], fill=col, outline=(80, 80, 80))
        d.text((lx + 30, ly + 38), txt, fill=(0, 0, 0), font=f_sm)
        lx += 30 + int(d.textlength(txt, font=f_sm)) + 34

    out = REPO_ROOT / "demo" / f"{scene}_iso3d.png"
    img.save(out)
    print(f"wrote {out}  ({W}x{H})  support_pairs={len(support)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
