#!/usr/bin/env python3
"""Build the qualitative real-scene figure for the paper.

The diagnostic figures are honest but abstract: a reader can finish the paper
without ever seeing the room. This assembles one figure from evidence that
already exists --- real capture frames and delivered-region crops taken from the
committed human-review sheet, and per-arm outcomes taken from the packed
relation report --- around a single question whose fate is the paper's thesis in
miniature.

The question is `q25_bin_basket_near_desk`. Its two referents are stored NEAR
each other (surface distance 0.0 m, threshold 1.0 m); a human resolving both
objects answers it correctly; the delivered graph cannot address it; and the
oracle-free bridge binds the basket but abstains on the desk.

Runs no model and re-scores nothing. Every number and image is read from a
committed artifact.

    tools/paper_qualitative_figure.py [--check]
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REVIEW = REPO / "runs" / "arkit_relation_challenge" / "arkitscenes_41069025_relation_review.html"
REPORT = REPO / "eval" / "results" / "project_census_v1" / "arkit_relation_challenge_report.json"
OUT = REPO / "docs" / "3dv" / "figures" / "fig5_qualitative.pdf"

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

QID = "q25_bin_basket_near_desk"
SCENE = "arkitscenes_41069025"
BASKET, DESK = "obj_1", "obj_14"
# Frames 4 and 5 of the sheet's twelve evenly-spaced orientation frames are the
# two that actually show both referents -- the desk with the basket beneath it.
# Chosen by looking at all twelve, not by any score.
CONTEXT_FRAMES = (4, 5)

INK, MUTED, FAINT = "#15181c", "#5b646e", "#8b939c"
ACCENT, OK, BAD, ORACLE = "#0d6a70", "#1d7548", "#a2382c", "#7c6118"


def review_html() -> str:
    if not REVIEW.is_file():
        raise SystemExit(f"review sheet not found: {REVIEW}")
    return REVIEW.read_text(encoding="utf-8")


def context_frames(html: str, want: tuple[int, ...]) -> list[str]:
    """Named raw-capture frames from the sheet's evenly-spaced orientation set."""
    start = html.index("Raw capture context")
    end = html.index("Which delivered region is which object?")
    frames = re.findall(r'src="(data:image/jpeg;base64,[^"]+)"', html[start:end])
    if max(want) >= len(frames):
        raise SystemExit(f"sheet has {len(frames)} orientation frames; wanted {want}")
    return [frames[i] for i in want]


def region_crop(html: str, uid: str) -> str:
    """The hero crop of one delivered region card."""
    m = re.search(rf'<figure class="card" id="{uid}">(.*?)</figure>', html, flags=re.S)
    if not m:
        raise SystemExit(f"no delivered-region card for {uid}")
    img = re.search(r'src="(data:image/jpeg;base64,[^"]+)"', m.group(1))
    if not img:
        raise SystemExit(f"card {uid} carries no image")
    return img.group(1)


def facts() -> dict:
    doc = json.loads(REPORT.read_text())
    arms = doc["arms"]

    def row(arm: str) -> dict:
        return next(r for r in arms[arm]["rows"] if r["id"] == QID)

    stored = row("stored_graph_human_identity")
    anchors = {a["anchor"]: a for a in doc["anchor_resolution"]["rows"]
               if a["scene_id"] == SCENE}
    return {
        "stored_distance_m": stored["stored_distance_m"],
        "threshold_m": stored["threshold_m"],
        "arms": [
            ("stored relations, human identity", row("stored_graph_human_identity"), True),
            ("delivered graph, learned labels", row("delivered_graph"), False),
            ("grounded graph, CLIP bridge", row("grounded_delivered_graph"), False),
            ("direct multiview RGB", row("blinded_rgb_vlm"), False),
        ],
        "anchors": anchors,
    }


def build_html() -> str:
    html = review_html()
    frames = context_frames(html, CONTEXT_FRAMES)
    basket, desk = region_crop(html, BASKET), region_crop(html, DESK)
    f = facts()

    basket_anchor = next(a for k, a in f["anchors"].items() if "basket" in k)
    desk_anchor = next(a for k, a in f["anchors"].items() if "long narrow desk" in k)

    rows = []
    for label, row, is_oracle in f["arms"]:
        outcome = row["outcome"]
        colour = OK if outcome == "correct" else (BAD if outcome == "wrong" else MUTED)
        mark = "correct" if outcome == "correct" else (
            "wrong" if outcome == "wrong" else "could not answer")
        why = row.get("reason") or ""
        badge = (f'<span class="oracle">NOT DEPLOYABLE</span>' if is_oracle else "")
        rows.append(
            f'<tr><td class="arm">{label} {badge}</td>'
            f'<td class="out" style="color:{colour}">{mark}</td>'
            f'<td class="why">{why}</td></tr>')

    frame_imgs = "".join(f'<img class="frame" src="{s}">' for s in frames)
    dist = f["stored_distance_m"]
    thr = f["threshold_m"]

    return f"""<!doctype html><meta charset="utf-8">
<style>
  @page {{ size: 1240px 402px; margin: 0; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font: 20px -apple-system, "Helvetica Neue", Arial, sans-serif;
          color: {INK}; background: #fff; width: 1240px; height: 402px; padding: 14px 16px; }}
  .cols {{ display: flex; gap: 18px; align-items: flex-start; }}
  .col {{ flex: 0 0 auto; }}
  .lbl {{ font-size: 18px; color: {FAINT}; margin: 0 0 5px;
          letter-spacing: .04em; text-transform: uppercase; }}
  img.frame {{ height: 148px; border: 1px solid #d3d9e0; border-radius: 3px; margin-right: 7px; }}
  img.crop {{ height: 124px; border: 2px solid {ACCENT}; border-radius: 3px; }}
  .cap {{ font-size: 17px; color: {MUTED}; margin-top: 4px; max-width: 250px; line-height: 1.3; }}
  .uid {{ font-family: ui-monospace, Menlo, monospace; font-size: 17px; color: {INK}; }}
  .edge {{ margin: 9px 0 0; font-size: 19px; }}
  .edge b {{ color: {ACCENT}; }}
  table {{ border-collapse: collapse; margin-top: 6px; width: 100%; }}
  td {{ padding: 4px 7px; font-size: 18px; vertical-align: top;
        border-top: 1px solid #e8ecf0; }}
  td.arm {{ width: 400px; }} td.out {{ width: 175px; font-weight: 700; }}
  td.why {{ color: {FAINT}; font-size: 16px; }}
  .oracle {{ color: {ORACLE}; font-weight: 700; font-size: 14px;
             letter-spacing: .04em; white-space: nowrap; }}
</style>
<div class="cols">
  <div class="col">
    <p class="lbl">the capture &mdash; both referents visible</p>
    {frame_imgs}
  </div>
  <div class="col">
    <p class="lbl">the two referents, as delivered</p>
    <div style="display:flex; gap:12px">
      <div><img class="crop" src="{basket}">
        <div class="cap"><span class="uid">{BASKET}</span> &mdash; the basket.<br>
        Bridge resolved it: <b style="color:{OK}">correct</b>.</div></div>
      <div><img class="crop" src="{desk}">
        <div class="cap"><span class="uid">{DESK}</span> &mdash; the desk.<br>
        Bridge <b style="color:{BAD}">abstained</b>: no entity won two view slots.</div></div>
    </div>
    <p class="edge">Stored edge <span class="uid">{BASKET}</span>&nbsp;&harr;&nbsp;<span
      class="uid">{DESK}</span>: surface distance <b>{dist} m</b> &lt; {thr} m &mdash; the
      serialized <b>NEAR</b> edge matches recomputed geometry under the same convention.</p>
  </div>
</div>
<table>{''.join(rows)}</table>
"""


def content_ops(pdf: bytes) -> bytes:
    blobs = []
    for m in re.finditer(rb"stream\r?\n", pdf):
        start = m.end()
        end = pdf.find(b"endstream", start)
        if end != -1:
            try:
                blobs.append(zlib.decompress(pdf[start:end]))
            except zlib.error:
                pass
    return b"".join(sorted(blobs))


def render() -> bytes:
    with tempfile.TemporaryDirectory() as td:
        page = Path(td) / "fig.html"
        page.write_text(build_html(), encoding="utf-8")
        pdf = Path(td) / "fig.pdf"
        r = subprocess.run(
            [CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
             f"--print-to-pdf={pdf}", f"file://{page}"],
            capture_output=True, text=True, timeout=240)
        if not pdf.is_file():
            raise SystemExit(f"chrome produced no PDF: {r.stderr[-300:]}")
        return pdf.read_bytes()


def main(argv: list[str]) -> int:
    fresh = render()
    if "--check" in argv:
        if not OUT.is_file() or content_ops(OUT.read_bytes()) != content_ops(fresh):
            print("fig5 is stale; re-run tools/paper_qualitative_figure.py")
            return 1
        print("fig5 is current")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(fresh)
    print(f"    fig5_qualitative.pdf  {len(fresh):>9,} B")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
