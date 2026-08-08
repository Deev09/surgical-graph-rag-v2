"""Per-scene UID index sheet for answer-key review.

  python3 demo/visualize_uid_index.py /path/to/room scene_id

Writes demo/<scene_id>_uid_index.png: the same top-down map as the question
sheets, but with EVERY object annotated by its obj number, plus a legend
column (uid -> label, sorted by number). This is the missing link between
the draft JSON's candidate_labels (uid -> class) and the question-sheet
boxes: it tells you WHICH chair is obj_39 vs obj_41.

Big objects get their number centered inside the box; small objects get a
leader dot. Numbers only (not 'obj_') to keep the map readable — read the
legend for labels.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.replica_habitat_import import import_habitat_room

DEFAULT_ROOM = Path.home() / "Desktop/datasets/replica/room_0"
FIT = 1150
PAD = 30
TITLE_H = 54
LEGEND_W = 340
C_BG = (255, 255, 255)
C_WALL = (35, 35, 35)
C_OBJ = (120, 150, 200)
C_TXT = (170, 40, 40)


def _font(sz):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def main() -> int:
    room = Path(sys.argv[1]) if len(sys.argv) >= 2 else DEFAULT_ROOM
    scene = sys.argv[2] if len(sys.argv) >= 3 else "replica_room_0"
    arts = import_habitat_room(room, scene)
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

    # legend: multiple columns if needed
    ents = sorted(arts.entities,
                  key=lambda e: int(e.identity.object_uid.split("_")[1]))
    lg_font = _font(13)
    row_h = 17
    rows_per_col = max(1, (CH + 2 * PAD) // row_h)
    n_cols = (len(ents) + rows_per_col - 1) // rows_per_col
    PW = CW + 2 * PAD + n_cols * LEGEND_W
    PH = CH + 2 * PAD + TITLE_H

    img = Image.new("RGB", (PW, PH), C_BG)
    d = ImageDraw.Draw(img)
    d.text((PAD, 14), f"UID index — {scene}  ({len(ents)} objects; "
                      "numbers = obj_<n>, see legend)", fill=(0, 0, 0),
           font=_font(22))

    def px(x, y):
        return (PAD + (x - minx) * scale, PAD + TITLE_H + (maxy - y) * scale)

    for w in walls:
        pts = [px(p[0], p[1]) for p in (w.polygon or [])]
        if len(pts) >= 2:
            d.line(pts + [pts[0]], fill=C_WALL, width=3)

    num_font = _font(14)
    # draw big objects first so small-object numbers land on top
    for e in sorted(ents, key=lambda e: -(
            (e.bbox_aabb[1][0] - e.bbox_aabb[0][0])
            * (e.bbox_aabb[1][1] - e.bbox_aabb[0][1]))):
        lo, hi = e.bbox_aabb
        x0, y0 = px(lo[0], hi[1]); x1, y1 = px(hi[0], lo[1])
        d.rectangle([x0, y0, x1, y1], outline=C_OBJ, width=1)
        n = e.identity.object_uid.split("_")[1]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        tw = d.textlength(n, font=num_font)
        # halo for readability
        d.rectangle([cx - tw / 2 - 2, cy - 9, cx + tw / 2 + 2, cy + 9],
                    fill=(255, 255, 255))
        d.text((cx - tw / 2, cy - 8), n, fill=C_TXT, font=num_font)

    lx0 = CW + 2 * PAD
    for i, e in enumerate(ents):
        col, row = divmod(i, rows_per_col)
        x = lx0 + col * LEGEND_W
        y = PAD + TITLE_H + row * row_h
        c = e.centroid
        d.text((x, y), f"obj_{e.identity.object_uid.split('_')[1]:>4}  "
                       f"{e.identity.display_label:<18} "
                       f"({c[0]:5.1f},{c[1]:5.1f})",
               fill=(60, 60, 60), font=lg_font)

    out = REPO_ROOT / "demo" / f"{scene}_uid_index.png"
    img.save(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
