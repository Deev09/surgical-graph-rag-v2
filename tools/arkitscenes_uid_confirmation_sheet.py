"""Build the owner UID-confirmation sheet for a human spatial-QA key.

  python3 tools/arkitscenes_uid_confirmation_sheet.py --scene 41069025

Emits a compact fillable sheet listing every delivered region with
**label-free** descriptors, plus one answer block per human-confirmed object.

TWO THINGS THIS DELIBERATELY DOES NOT DO
----------------------------------------
1. **It never uses the system's predicted label to nominate a candidate.** No
   shortlist is ranked by `display_label`, no class name is matched, and the
   predicted label is not printed anywhere in the candidate table. Asking the
   owner "is `obj_12` the trash can?" *because the labeler called it
   trash-can* would launder a model guess into ground truth, and the whole
   point of the key is to be independent of the model. A test AST-checks that
   this module never reads `display_label`.

2. **It does not mix the human facts with the correspondence table.** The
   confirmed object facts (one rug, one trash can, one main kitchen counter,
   two visible sofa cushions) are scene truth and stand on their own. Which
   delivered region -- if any -- corresponds to each is a separate, revisable
   mapping that dies the moment segmentation changes. They are written as two
   separate sections and two separate JSON blocks.

WHAT THE OWNER GETS
-------------------
Every delivered region, sorted by size, described only by geometry: vertex
count, width x depth x height in metres, footprint, and the height of its
underside above the lowest point in the scene. That is enough to recognise "a
thin thing lying on the floor" or "a small tall thing" without being told what
the model thinks it is.

Each answer block offers the same four outcomes, because all four occur in this
scene and collapsing them loses information:

  uid            one delivered region is this object
  none/missing   the object exists in the room but was not delivered at all
  overmerged     the object is inside a larger region that also covers other
                 things; name that region
  ambiguous      the owner cannot tell from the available views
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_ENTITIES = (REPO_ROOT / "runs" / "arkit_label_image_ab_41069025"
                    / "rgb_tight" / "entities" / "manifest.json")
DEFAULT_OUT = REPO_ROOT / "eval" / "human_feedback"

OUTCOMES = ("uid", "none_missing", "overmerged_into", "ambiguous")


def load_regions(path: Path) -> tuple[list[dict], dict]:
    """Delivered regions with geometry only. `display_label` is never read."""
    manifest = json.loads(path.read_text())
    floor_z = min(e["bbox_aabb"][0][2] for e in manifest["entities"])
    rows = []
    for entity in manifest["entities"]:
        (x0, y0, z0), (x1, y1, z1) = entity["bbox_aabb"]
        rows.append({
            "uid": entity["identity"]["object_uid"],
            "n_vertices": entity["extraction_diagnostics"]["n_vertices"],
            "width_m": round(x1 - x0, 2),
            "depth_m": round(y1 - y0, 2),
            "height_m": round(z1 - z0, 2),
            "footprint_m2": round((x1 - x0) * (y1 - y0), 2),
            "underside_above_floor_m": round(z0 - floor_z, 2),
        })
    rows.sort(key=lambda r: -r["n_vertices"])
    provenance = {
        "entity_manifest": str(path.relative_to(REPO_ROOT)),
        "entity_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bundle_hash": manifest.get("bundle_hash"),
        "n_regions": len(rows),
        "scene_floor_z": round(floor_z, 3),
        "descriptors": "geometry only; predicted labels deliberately omitted",
    }
    return rows, provenance


def build(objects: list[dict], regions: list[dict], provenance: dict,
          scene_id: str) -> tuple[str, dict]:
    skeleton = {
        "schema": "arkitscenes_uid_confirmation_v1",
        "scene_id": scene_id,
        "status": "AWAITING_OWNER",
        "instructions": (
            "For each object set exactly one of: uid, none_missing (true), "
            "overmerged_into (a uid), ambiguous (true). Leave the rest null."),
        "human_object_facts": objects,
        "predicted_uid_correspondence": [
            {"object": obj["object"], "expected_count": obj["confirmed_count"],
             "assignments": [
                 {"uid": None, "none_missing": None, "overmerged_into": None,
                  "ambiguous": None}
                 for _ in range(obj["confirmed_count"])]}
            for obj in objects
        ],
        "region_provenance": provenance,
    }

    lines = [
        f"# UID confirmation sheet — {scene_id}",
        "",
        f"Regions: **{provenance['n_regions']}** delivered. "
        f"Bundle `{provenance['bundle_hash']}`.",
        "",
        "Candidate descriptors are **geometry only**. The system's predicted "
        "labels are deliberately not shown, so a model guess cannot become "
        "ground truth by suggestion.",
        "",
        "## Section 1 — human object facts (confirmed; not up for revision here)",
        "",
        "| object | confirmed count | source |",
        "|---|---|---|",
    ]
    for obj in objects:
        lines.append(f"| {obj['object']} | {obj['confirmed_count']} | "
                     f"{obj['source']} |")
    lines += [
        "",
        "These are scene facts. They stay true whatever the segmentation does.",
        "",
        "## Section 2 — predicted-UID correspondence (what needs your answer)",
        "",
        "Separate from Section 1 on purpose: this mapping is revisable and "
        "dies with the current segmentation.",
        "",
        "For each row choose ONE:",
        "",
        "- `uid = obj_N` — this delivered region is the object",
        "- `none/missing` — the object is in the room but was not delivered",
        "- `overmerged into obj_N` — it is inside a bigger region covering "
        "other things too",
        "- `ambiguous` — cannot tell from the views available",
        "",
    ]
    for obj in objects:
        lines.append(f"**{obj['object']}** (expect {obj['confirmed_count']})")
        lines.append("")
        for i in range(obj["confirmed_count"]):
            suffix = f" #{i + 1}" if obj["confirmed_count"] > 1 else ""
            lines.append(
                f"- [ ] instance{suffix}: uid = `______`  |  "
                "[ ] none/missing  |  [ ] overmerged into `______`  |  "
                "[ ] ambiguous")
        lines.append("")
    lines += [
        "## Section 3 — delivered regions, geometry only (reference)",
        "",
        "Sorted largest first. `under` is the height of the region's underside "
        "above the lowest point in the scene — 0.00 means it sits on the floor.",
        "",
        "| uid | verts | w×d×h (m) | footprint m² | under (m) |",
        "|---|---|---|---|---|",
    ]
    for r in regions:
        lines.append(
            f"| `{r['uid']}` | {r['n_vertices']} | "
            f"{r['width_m']}×{r['depth_m']}×{r['height_m']} | "
            f"{r['footprint_m2']} | {r['underside_above_floor_m']} |")
    lines += [
        "",
        "---",
        "",
        "Return the filled Section 2 (or the JSON skeleton beside this file). "
        "The key is finalized and scored once, after that.",
        "",
    ]
    return "\n".join(lines), skeleton


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", default="41069025")
    ap.add_argument("--entities", type=Path, default=DEFAULT_ENTITIES)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    scene_id = f"arkitscenes_{args.scene}"
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
    regions, provenance = load_regions(args.entities)
    sheet, skeleton = build(objects, regions, provenance, scene_id)

    args.out_root.mkdir(parents=True, exist_ok=True)
    md_path = args.out_root / f"{scene_id}_uid_confirmation_sheet.md"
    json_path = args.out_root / f"{scene_id}_uid_confirmation.json"
    md_path.write_text(sheet)
    json_path.write_text(json.dumps(skeleton, indent=1) + "\n")

    print(f"=== {scene_id} UID confirmation sheet")
    print(f"    objects to map : {len(objects)} "
          f"({sum(o['confirmed_count'] for o in objects)} instances)")
    print(f"    regions listed : {provenance['n_regions']} (geometry only)")
    print(f"    sheet -> {md_path.relative_to(REPO_ROOT)}")
    print(f"    form  -> {json_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
