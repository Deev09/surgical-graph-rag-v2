#!/usr/bin/env python3
"""Render the figure SVGs to tightly-cropped vector PDFs for LaTeX.

LaTeX cannot include SVG directly and this machine has no rsvg/cairosvg/inkscape,
so we drive headless Chrome, which is already present. The SVG is wrapped in a
page whose CSS `@page` size matches its own viewBox, which is what makes the
output a cropped figure rather than an SVG stranded on US Letter.

Output stays vector: Chrome's print pipeline keeps the paths and text as PDF
drawing operators, so the figures scale in the compiled paper.

    tools/paper_figures_pdf.py [--check]

`--check` re-renders and fails if a committed PDF would change. Chrome embeds a
creation date, so the check compares the *drawing content* rather than raw bytes.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIGS = REPO / "docs" / "figures"
OUT = REPO / "docs" / "3dv" / "figures"

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# Only the figures the main paper actually embeds. Figures 2 and 3 still
# generate as SVG for the supplement; they are not needed as PDFs.
WANTED = ("fig1_evaluation_ladder", "fig4_reachability")

PAGE = """<!doctype html><meta charset="utf-8">
<style>
  @page {{ size: {w}px {h}px; margin: 0; }}
  html, body {{ margin: 0; padding: 0; }}
  svg {{ display: block; }}
</style>
{svg}
"""


def viewbox(svg: str) -> tuple[float, float]:
    m = re.search(r'viewBox="([\d.\-]+) ([\d.\-]+) ([\d.]+) ([\d.]+)"', svg)
    if not m:
        raise SystemExit("figure has no viewBox; cannot size the page")
    return float(m.group(3)), float(m.group(4))


def content_ops(pdf: bytes) -> bytes:
    """The PDF's drawing operators, with metadata and dates stripped out."""
    blobs = []
    for m in re.finditer(rb"stream\r?\n", pdf):
        start = m.end()
        end = pdf.find(b"endstream", start)
        if end == -1:
            continue
        try:
            blobs.append(zlib.decompress(pdf[start:end]))
        except zlib.error:
            pass
    return b"".join(sorted(blobs))


def render(name: str, outdir: Path) -> bytes:
    svg = (FIGS / f"{name}.svg").read_text()
    w, h = viewbox(svg)
    with tempfile.TemporaryDirectory() as td:
        page = Path(td) / "page.html"
        page.write_text(PAGE.format(w=w, h=h, svg=svg))
        pdf = Path(td) / "out.pdf"
        r = subprocess.run(
            [CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
             f"--print-to-pdf={pdf}", f"file://{page}"],
            capture_output=True, text=True, timeout=180)
        if not pdf.is_file():
            raise SystemExit(f"chrome produced no PDF for {name}: {r.stderr[-300:]}")
        return pdf.read_bytes()


def main(argv: list[str]) -> int:
    if not Path(CHROME).is_file():
        print(f"chrome not found at {CHROME}", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    stale = []
    for name in WANTED:
        fresh = render(name, OUT)
        target = OUT / f"{name}.pdf"
        if "--check" in argv:
            if not target.is_file() or content_ops(target.read_bytes()) != content_ops(fresh):
                stale.append(name)
            continue
        target.write_bytes(fresh)
        print(f"    {name}.pdf  {len(fresh):>8,} B")
    if "--check" in argv:
        if stale:
            print(f"figure PDFs are stale: {stale}; re-run tools/paper_figures_pdf.py")
            return 1
        print("figure PDFs are current")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
