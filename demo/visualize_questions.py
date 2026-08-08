"""Per-question floorplan panels: one small top-down map per relation, so each
question's answer is isolated instead of all overlaid at once.

  python3 demo/visualize_questions.py                    # room_0
  python3 demo/visualize_questions.py /path/to/room scene_id

Writes demo/<scene_id>_questions.png  (a 2x2 grid of panels).
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
FIT = 660          # longest world axis -> px, per panel
PAD = 26
TITLE_H = 64

C_BG = (255, 255, 255)
C_WALL = (35, 35, 35)
C_OBJ_EDGE = (214, 214, 214)
C_SUPPORTER = (40, 110, 220)
GREEN = (55, 165, 85)
RED = (205, 45, 60)
YELLOW = (225, 170, 25)
ORANGE = (235, 130, 30)


def _font(sz):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def _runs():
    # Must mirror demo/question_battery._runs() (minus AttachedTo) so the
    # panels show exactly what the battery answers — including the Phase 8
    # F4 room-scale-flat exclusion on both wall-relation configs.
    return [
        ExtractorRun(DirectionalExtractor(), DirectionalConfig(mode="sparse")),
        ExtractorRun(SurfaceProximityExtractor(),
                     SurfaceProximityConfig(use_polygon_clip=True,
                                            exclude_room_scale_flat=True)),
        ExtractorRun(OnSurfaceExtractor(), OnSurfaceConfig()),
        ExtractorRun(ContactsSurfaceExtractor(),
                     ContactsSurfaceConfig(exclude_room_scale_flat=True)),
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

    floor_objs = {e.source.uid for e in bundle.edges
                  if e.type == "ON_SURFACE" and e.target.uid.startswith("floor")}
    wall_objs = {e.source.uid for e in bundle.edges if e.type == "CONTACTS_SURFACE"}
    near_wall = {e.source.uid for e in bundle.edges
                 if e.type == "NEAR_SURFACE" and e.target.uid.startswith("wall")}
    support = [(e.source.uid, e.target.uid)
               for e in bundle.edges if e.type == "ON_ENTITY_SURFACE"]
    supporters = {t for _, t in support}
    supported = {s for s, _ in support}
    walls = [s for s in arts.structural_surfaces if s.surface_type == "wall"]

    xs, ys = [], []
    for e in arts.entities:
        lo, hi = e.bbox_aabb
        xs += [lo[0], hi[0]]; ys += [lo[1], hi[1]]
    for w in walls:
        for p in (w.polygon or []):
            xs.append(p[0]); ys.append(p[1])
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    scale = FIT / (max(maxx - minx, maxy - miny) or 1.0)
    CW, CH = int((maxx - minx) * scale), int((maxy - miny) * scale)
    PW, PH = CW + 2 * PAD, CH + 2 * PAD + TITLE_H

    def px(x, y, ox, oy):
        return (ox + PAD + (x - minx) * scale, oy + PAD + TITLE_H + (maxy - y) * scale)

    f_title, f_sub = _font(22), _font(15)

    panels = [
        ("what is on the floor?", floor_objs, GREEN, "fill", []),
        ("what is against the wall?", wall_objs, RED, "fill", []),
        ("what is near the wall?", near_wall, YELLOW, "fill", []),
        ("what is on furniture? (table/desk/plant-stand)",
         supported, ORANGE, "support", support),
    ]

    canvas = Image.new("RGB", (2 * PW, 2 * PH), C_BG)
    d = ImageDraw.Draw(canvas)

    for i, (title, hi_set, color, mode, links) in enumerate(panels):
        ox, oy = (i % 2) * PW, (i // 2) * PH
        # walls
        for w in walls:
            pts = [px(p[0], p[1], ox, oy) for p in (w.polygon or [])]
            if len(pts) >= 2:
                d.line(pts + [pts[0]], fill=C_WALL, width=3)
        # all objects faint
        for e in arts.entities:
            lo, hi = e.bbox_aabb
            x0, y0 = px(lo[0], hi[1], ox, oy); x1, y1 = px(hi[0], lo[1], ox, oy)
            d.rectangle([x0, y0, x1, y1], outline=C_OBJ_EDGE, width=1)
        # support furniture outline (only in support panel)
        if mode == "support":
            for u in supporters:
                lo, hi = ent[u].bbox_aabb
                x0, y0 = px(lo[0], hi[1], ox, oy); x1, y1 = px(hi[0], lo[1], ox, oy)
                d.rectangle([x0, y0, x1, y1], outline=C_SUPPORTER, width=4)
            for s, t in links:
                d.line([px(*ent[s].centroid[:2], ox, oy),
                        px(*ent[t].centroid[:2], ox, oy)], fill=ORANGE, width=2)
        # highlighted answers
        for u in hi_set:
            if u not in ent:
                continue
            lo, hi = ent[u].bbox_aabb
            x0, y0 = px(lo[0], hi[1], ox, oy); x1, y1 = px(hi[0], lo[1], ox, oy)
            # ensure a minimum visible size
            if x1 - x0 < 7:
                x0, x1 = (x0 + x1) / 2 - 4, (x0 + x1) / 2 + 4
            if y1 - y0 < 7:
                y0, y1 = (y0 + y1) / 2 - 4, (y0 + y1) / 2 + 4
            d.rectangle([x0, y0, x1, y1], fill=color, outline=(90, 90, 90))
        # title
        d.text((ox + PAD, oy + 12), title, fill=(0, 0, 0), font=f_title)
        d.text((ox + PAD, oy + 40), f"{len(hi_set)} object(s) answered",
               fill=color, font=f_sub)
        d.rectangle([ox + 1, oy + 1, ox + PW - 2, oy + PH - 2],
                    outline=(225, 225, 225), width=1)

    out = REPO_ROOT / "demo" / f"{scene}_questions.png"
    canvas.save(out)
    print(f"wrote {out}  ({2*PW}x{2*PH})")
    print(f"  on-floor={len(floor_objs)} against-wall={len(wall_objs)} "
          f"near-wall={len(near_wall)} support={len(support)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
