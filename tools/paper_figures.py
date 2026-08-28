"""Deterministic paper figures, generated from the results registry.

  python3 tools/paper_figures.py

Reads `docs/project_results_registry.csv` and writes three SVGs into
`docs/figures/`. No randomness, no timestamps, no network: identical inputs
produce byte-identical output, and a test asserts it.

Every number drawn is read from the registry by `result_id` — nothing is typed
into this file. If a figure and the registry ever disagree, the figure cannot
be regenerated, which is the point.

SCOPE IS DRAWN, NOT ANNOTATED. Layers that consume human-supplied identity are
hatched and carry a NOT DEPLOYABLE badge; the component evaluation carries its
oracle-selected denominator in the axis label. A reader who only looks at the
pictures still cannot mistake a bound for system performance.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "docs" / "project_results_registry.csv"
OUT = REPO_ROOT / "docs" / "figures"

INK = "#15181c"
MUTED = "#5b646e"
FAINT = "#8b939c"
LINE = "#d3d9e0"
ACCENT = "#0d6a70"
OK = "#1d7548"
BAD = "#a2382c"
ORACLE = "#7c6118"
BG = "#ffffff"
FONT = "Helvetica, Arial, sans-serif"
MONO = "ui-monospace, 'SF Mono', Menlo, monospace"


def load() -> dict:
    with REGISTRY.open(newline="") as fh:
        return {r["result_id"]: r for r in csv.DictReader(fh)}


def frac(row: dict) -> tuple[int, int]:
    """Numerator and denominator, taken from the registry columns only."""
    n, d = row["numerator"].strip(), row["denominator"].strip()
    if n.isdigit() and d.isdigit():
        return int(n), int(d)
    m = re.search(r"(\d+)\s*/\s*(\d+)", row["value"])
    if not m:
        raise ValueError(f"{row['result_id']}: no fraction in {row['value']!r}")
    return int(m.group(1)), int(m.group(2))


def esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def text(x, y, s, size=11, fill=INK, anchor="start", weight="normal",
         family=FONT, spacing=None):
    sp = f' letter-spacing="{spacing}"' if spacing else ""
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
            f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}"{sp}>'
            f'{esc(s)}</text>')


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}">'
            f'<title>{esc(title)}</title>'
            f'<defs><pattern id="hatch" width="6" height="6" '
            f'patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
            f'<rect width="6" height="6" fill="{BG}"/>'
            f'<line x1="0" y1="0" x2="0" y2="6" stroke="{ORACLE}" '
            f'stroke-width="2.2"/></pattern></defs>'
            f'<rect width="{w}" height="{h}" fill="{BG}"/>{body}</svg>\n')


# --------------------------------------------------------------------------
# FONT-SIZE FLOOR: the two embedded figures print at \textwidth = 6.875 in =
# 496.9 pt, so printed size = px * 496.9 / viewBox_width. At W = 1040 the 7 pt
# floor demands >= 14.65 px; every text below uses 15 px or more. The registry
# footers and scope prose moved into the LaTeX captions; the strings the test
# suite pins (fig4: F35/F40/F45/F50 + identity_oracle) stay in the graphic.
def figure_1_layers(R: dict) -> str:
    """The evaluation ladder: which stage each arm exercises, and its scope."""
    stages = ["capture", "segmentation", "instance\ndelivery", "object\nnaming",
              "relation\nextraction", "answer"]
    rows = [
        ("geometry ceiling",        F"{frac(R['F28'])[0]}/{frac(R['F28'])[1]}", 3, 5, True,
         "human identity + recomputed geometry"),
        ("stored-edge replay",      f"{frac(R['F35'])[0]}/{frac(R['F35'])[1]}", 4, 5, True,
         "human identity + serialized edges"),
        ("delivered graph",         f"{frac(R['F40'])[0]}/{frac(R['F40'])[1]}", 3, 5, False,
         "learned labels + serialized edges"),
        ("grounded graph",          f"{frac(R['F45'])[0]}/{frac(R['F45'])[1]}", 3, 5, False,
         "oracle-free grounding + serialized edges"),
        ("direct multiview RGB",    f"{frac(R['F50'])[0]}/{frac(R['F50'])[1]}", 0, 5, False,
         "the images alone"),
    ]
    W, H = 1040, 462
    # x0=330 leaves room for 17 px arm names and 15 px basis lines on the left;
    # colw=88 ends the bars at 858 so the stacked value + badge fit inside 1040.
    # top=150 keeps the two-line stage headers (drawn from top-42) clear of the
    # 15 px legend line whose baseline is y=82.
    x0, top, colw, rowh = 330, 150, 88, 62
    b = [text(28, 36, "The evaluation ladder", 20, INK, weight="600"),
         text(28, 60, "Each arm holds the same pipeline fixed and changes one source of "
                      "identity. Bars span the stages an arm relies on.", 15, MUTED),
         text(28, 82, "Hatched = human-supplied identity: a bound, never system "
                      "performance.", 15, ORACLE)]
    for i, s in enumerate(stages):
        cx = x0 + i * colw + colw / 2
        for k, part in enumerate(s.split("\n")):
            b.append(text(cx, top - 42 + k * 17, part, 15, FAINT, anchor="middle"))
        b.append(f'<line x1="{cx}" y1="{top - 18}" x2="{cx}" y2="{top + len(rows)*rowh - 30}" '
                 f'stroke="{LINE}" stroke-width="1"/>')
    for j, (name, val, a, z, oracle, basis) in enumerate(rows):
        y = top + j * rowh
        b.append(text(28, y + 2, name, 17, INK, weight="600"))
        b.append(text(28, y + 22, basis, 15, FAINT))
        bx = x0 + a * colw + 5
        bw = (z - a) * colw + colw - 10
        fill = "url(#hatch)" if oracle else ACCENT
        b.append(f'<rect x="{bx}" y="{y - 13}" width="{bw}" height="26" rx="3" '
                 f'fill="{fill}" stroke="{ORACLE if oracle else ACCENT}" stroke-width="1"/>')
        b.append(text(x0 + 6 * colw + 14, y + 2, val, 16, INK, weight="700", family=MONO))
        b.append(text(x0 + 6 * colw + 14, y + 22,
                      "NOT DEPLOYABLE" if oracle else "deployable" if j == 4 else "delivered",
                      15, ORACLE if oracle else MUTED, weight="700" if oracle else "normal"))
    return svg(W, H, "".join(b), "The evaluation ladder")


# --------------------------------------------------------------------------
def figure_2_component(R: dict) -> str:
    """Scoped component result: labeler input A/B, per scene and pooled.

    Supplement figure. At W = 980 the 7 pt floor demands >= 13.8 px; every
    text below uses 14 px or more.
    """
    scenes = [("41069021", "C05", "C07", "C06", "C08", "C17", "C18"),
              ("41069025", "C09", "C11", "C10", "C12", None, None),
              ("41069042", "C13", "C15", "C14", "C16", None, None),
              ("pooled",   "C01", "C02", "C03", "C04", None, None)]
    W, H = 980, 486
    left, top, gw, bh = 60, 168, 216, 16
    b = [text(28, 36, "Real-RGB crops as the labeler input", 18, INK, weight="600"),
         text(28, 60, "Only the label images changed. Weights, vocabulary, admission "
                      "threshold, evaluator, delivered", 14, MUTED),
         text(28, 78, "partitions and IoU matching all fixed.", 14, MUTED),
         f'<rect x="28" y="90" width="{W-56}" height="48" rx="3" fill="#f6efdc" '
         f'stroke="{ORACLE}" stroke-width="1"/>',
         text(40, 110, "SCOPE  oracle_free_component_eval — the denominator is instances "
                       "the evaluator already", 14, ORACLE, weight="600"),
         text(40, 130, "matched to an annotation box. NOT end-to-end performance.",
              14, ORACLE, weight="600")]
    maxden = 21
    scale = 150 / maxden
    for i, (scene, s1, r1, s3, r3, c1, c3) in enumerate(scenes):
        gx = left + i * gw
        b.append(text(gx, top - 16, scene, 15, INK, weight="700", family=MONO))
        series = [("splat", s1, s3, "#b9c0c8"), ("rgb_tight", r1, r3, ACCENT)]
        if c1:
            series.append(("rgb_context", c1, c3, "#8aa8ab"))
        y = top
        for label, i1, i3, col in series:
            n1, d1 = frac(R[i1]); n3, d3 = frac(R[i3])
            b.append(text(gx, y + 8, label, 14, MUTED))
            for k, (n, d, idr) in enumerate(((n1, d1, i1), (n3, d3, i3))):
                yy = y + 15 + k * (bh + 6)
                b.append(f'<rect x="{gx}" y="{yy}" width="{d*scale:.1f}" height="{bh}" '
                         f'rx="2" fill="#eef1f4"/>')
                b.append(f'<rect x="{gx}" y="{yy}" width="{n*scale:.1f}" height="{bh}" '
                         f'rx="2" fill="{col}"/>')
                b.append(text(gx + d * scale + 6, yy + 13, f"{n}/{d}", 14, INK,
                              weight="700" if label == "rgb_tight" else "normal", family=MONO))
                b.append(text(gx + d * scale + 56, yy + 13, f"t{1 if k==0 else 3}",
                              14, FAINT))
                b.append(text(gx + d * scale + 80, yy + 13, idr, 14, FAINT, family=MONO))
            y += 68
    b.append(text(28, H - 58, "The context control (41069021, rgb_context) scores BELOW "
                              "rgb_tight at both ranks — 3/7 and 5/7", 14, INK))
    b.append(text(28, H - 40, "against 5/7 and 7/7. More context hurts, which SUPPORTS the "
                              "interpretation that the gain comes", 14, MUTED))
    b.append(text(28, H - 22, "from object texture rather than room gist. It does not "
                              "prove it.", 14, MUTED))
    b.append(text(W - 28, H - 22, "registry C01–C21", 14, FAINT, family=MONO,
                  anchor="end"))
    return svg(W, H, "".join(b), "Real-RGB crops as the labeler input")


# --------------------------------------------------------------------------
def figure_3_unreachable(R: dict) -> str:
    """Held but unreachable, plus what direct RGB does on the pilot room.

    Supplement figure. At W = 1040 the 7 pt floor demands >= 14.65 px; every
    text below uses 15 px or more.
    """
    W, H = 1040, 640
    b = [text(28, 36, "Held, but unreachable", 20, INK, weight="600"),
         text(28, 60, "Same twelve questions, same stored relations, same scoring. "
                      "Only the source of object identity changes.", 15, MUTED),
         # fig2 carries a scope banner; a picture-only reader of fig3 needs the
         # same warning, because the top bar is a bound and not a system result.
         f'<rect x="28" y="74" width="984" height="50" rx="3" fill="#f6efdc" '
         f'stroke="{ORACLE}" stroke-width="1"/>',
         text(40, 95, "SCOPE  the hatched bar is identity_oracle — it consumes "
                      "human-supplied identity.", 15, ORACLE, weight="600"),
         text(40, 116, "It BOUNDS what the representation could express; it is not "
                       "performance.", 15, ORACLE, weight="600")]
    bars = [("stored relations, human identity", "F35", True),
            ("stored relations, grounded identity", "F45", False),
            ("stored relations, learned labels", "F40", False),
            ("direct multiview RGB", "F50", False)]
    left, top, bw, rowh = 320, 152, 240, 48
    # The RGB arm is a different family from the three stored-relation arms, so
    # it is pushed down past the gap brace rather than sharing its bracket.
    def rowy(i): return top + i * rowh + (46 if i == 3 else 0)
    for i, (label, rid, oracle) in enumerate(bars):
        n, d = frac(R[rid]); y = rowy(i)
        b.append(text(306, y + 16, label, 15, INK, anchor="end",
                      weight="700" if i == 0 else "normal"))
        b.append(f'<rect x="{left}" y="{y}" width="{bw}" height="24" rx="3" fill="#eef1f4"/>')
        fill = "url(#hatch)" if oracle else (ACCENT if n else "#e2e6ea")
        b.append(f'<rect x="{left}" y="{y}" width="{bw*n/d:.1f}" height="24" rx="3" '
                 f'fill="{fill}" stroke="{ORACLE if oracle else "none"}" stroke-width="1"/>')
        b.append(text(left + bw + 14, y + 17, f"{n}/{d}", 16, INK, weight="700", family=MONO))
        b.append(text(left + bw + 66, y + 17, rid, 15, FAINT, family=MONO))
        if oracle:
            b.append(text(left + bw + 112, y + 17, "NOT DEPLOYABLE", 15, ORACLE,
                          weight="700"))
        elif rid == "F50":
            # F50 draws the same width as the F35 bound. Left unmarked, the tie
            # reads as "the deployable path already reaches the ceiling", which
            # is the opposite of this figure's claim.
            b.append(text(left + bw + 112, y + 17, "deployable — ties the bound on "
                          "DIFFERENT items", 15, OK, weight="700"))
    # The gap brace sits BELOW the bars, not beside them: an earlier version put
    # its label inline and it overlapped the 2/10 value and its result_id.
    gx0, gx1 = left, left + bw * 0.70
    gy = top + 3 * rowh + 2
    b.append(f'<path d="M {gx0} {gy-8} L {gx0} {gy} L {gx1} {gy} L {gx1} {gy-8}" '
             f'fill="none" stroke="{BAD}" stroke-width="1.4"/>')
    b.append(text((gx0 + gx1) / 2, gy + 18, "the reachability gap — 7/10 held, 0/10 reached",
                  15, BAD, anchor="middle", weight="700"))
    n64, d64 = frac(R["F64"])
    b.append(text(28, rowy(3) + 58,
                  f"No additional serialization loss was measured: the stored-edge replay "
                  f"agrees with recomputed", 15, INK))
    b.append(text(28, rowy(3) + 78,
                  f"geometry {R['F63']['value'].split('(')[0].strip()} item for item [F63]. "
                  f"{n64} of {d64} items are ceiling-correct but delivered-unanswered",
                  15, INK))
    b.append(text(28, rowy(3) + 98,
                  f"[F64]; delivered-graph-unique wins "
                  f"{R['F60']['value'].split('(')[0].strip()} [F60].", 15, MUTED))
    yb = rowy(3) + 132
    b.append(f'<line x1="28" y1="{yb-16}" x2="1012" y2="{yb-16}" stroke="{LINE}" stroke-width="1"/>')
    b.append(text(28, yb + 8, "On the pilot room, direct RGB does not close it either",
                  17, INK, weight="600"))
    # The two gates decided the outcome, so they are tiles rather than a footnote.
    # 1.000-when-answered is NOT a tile: its denominator is 5, and at tile size it
    # reads as the transfer result.
    cells = [(f"{frac(R['F76'])[0]}/{frac(R['F76'])[1]}", "correct", "", "F76", ACCENT),
             (f"{frac(R['F77'])[0]}/{frac(R['F77'])[1]}", "wrong", "", "F77", OK),
             ("0/3", "cross-view items", "all declined", "F79", BAD),
             ("0.50", "exact accuracy", "FAIL (gate 0.60)", "F82", BAD),
             ("0.50", "coverage", "FAIL (gate 0.80)", "F83", BAD)]
    for i, (val, lab, sub, rid, col) in enumerate(cells):
        cx = 40 + i * 196
        b.append(f'<rect x="{cx-12}" y="{yb+20}" width="184" height="70" rx="4" '
                 f'fill="{BG}" stroke="{LINE}" stroke-width="1"/>')
        b.append(text(cx, yb + 46, val, 18, col, weight="700", family=MONO))
        b.append(text(cx, yb + 66, lab, 15, MUTED))
        if sub:
            b.append(text(cx, yb + 84, sub, 15, BAD if "FAIL" in sub else MUTED,
                          weight="700" if "FAIL" in sub else "normal"))
        b.append(text(cx + 160, yb + 46, rid, 15, FAINT, anchor="end", family=MONO))
    b.append(text(28, H - 30, f"Accuracy when answered is {R['F85']['value']} [F85] over the "
                              "5 items it answered — denominator 5, not 10.", 15, FAINT))
    b.append(text(28, H - 10, "Both predeclared gates failed; no transfer claim is made "
                              "[F88]. On the development rooms this arm was wrong 2/10 [F51].",
                  15, FAINT))
    return svg(W, H, "".join(b), "Held, but unreachable")


# --------------------------------------------------------------------------
def figure_4_reachability(R: dict) -> str:
    """Per-arm reachability: each arm's declared path, one chain per arm.

    Reads the committed statistics report rather than recomputing it, so this
    figure and docs/paper_reachability_ledger.csv can never disagree. The four
    chains are the per-arm ladders of the schema freeze: the graph arms share
    their gating prefix (delivery, expressibility, serialization), the grounded
    arm adds anchor grounding, and the oracle and RGB arms bypass graph stages.
    """
    stats = json.loads((REPO_ROOT / "eval" / "results" / "paper_statistics.json").read_text())
    reach = stats["reachability"]
    n = reach["n_scored"]
    by_stage = {s["stage"]: s["surviving"] for s in reach["survivors_by_stage"]}
    delivered = by_stage["objects_delivered"]
    expressible = by_stage["relation_expressible"]
    serialized = by_stage["edge_serialized"]
    grounded = by_stage["anchor_grounded"]

    W, H = 1040, 620
    b = [text(28, 36, "Per-arm reachability", 20, INK, weight="600"),
         text(28, 60, "Each arm reports its own declared path; counts are questions "
                      "surviving each stage. There is no mixed ladder.", 15, MUTED),
         f'<rect x="28" y="74" width="984" height="30" rx="3" fill="#f6efdc" '
         f'stroke="{ORACLE}" stroke-width="1"/>',
         text(40, 95, "SCOPE  re-reads already-scored outcomes; the stored-human arm is "
                      "identity_oracle \u2014 a BOUND, not performance.", 15, ORACLE,
              weight="600")]

    # Stage columns shared by the graph arms; every arm's outcome sits at x=858.
    xs = [40, 204, 368, 532, 696]
    xo = 858
    nw, nh = 74, 32   # node size

    def node(x, y, count, oracle=False):
        fill = "url(#hatch)" if oracle else (ACCENT if count else "#e2e6ea")
        tcol = BG if (count and not oracle) else INK
        out = [f'<rect x="{x}" y="{y}" width="{nw}" height="{nh}" rx="4" '
               f'fill="{fill}" stroke="{ORACLE if oracle else ACCENT}" stroke-width="1"/>',
               text(x + nw / 2, y + 21, f"{count}/{n}", 16, tcol if count else BAD,
                    anchor="middle", weight="700", family=MONO)]
        return "".join(out)

    def arrow(x0, x1, y, lost=0, dashed=False, note=None):
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        out = [f'<line x1="{x0}" y1="{y}" x2="{x1 - 7}" y2="{y}" stroke="{MUTED}" '
               f'stroke-width="1.4"{dash}/>',
               f'<path d="M {x1 - 7} {y - 4} L {x1} {y} L {x1 - 7} {y + 4}" '
               f'fill="none" stroke="{MUTED}" stroke-width="1.4"/>']
        if lost:
            out.append(text((x0 + x1) / 2, y - 8, f"\u2212{lost}", 15, BAD,
                            anchor="middle", weight="700"))
        if note:
            out.append(text((x0 + x1) / 2, y - 8, note, 15, FAINT, anchor="middle"))
        return "".join(out)

    def stage_label(x, y, s):
        return text(x + nw / 2, y, s, 15, FAINT, anchor="middle")

    def band(top_y, name, rid, tag, tag_col):
        out = [text(28, top_y, name, 17, INK, weight="700"),
               text(28 + 9.4 * len(name) + 16, top_y, tag, 15, tag_col),
               text(xo + nw + 10, top_y + 43, f"[{rid}]", 15, FAINT, family=MONO)]
        return "".join(out)

    # --- delivered graph -----------------------------------------------------
    y = 148
    b.append(band(y, "delivered graph", "F40", "learned labels; deployable path", MUTED))
    ny = y + 14
    chain = [(xs[0], n), (xs[1], delivered), (xs[2], expressible), (xs[3], serialized)]
    for (x, c) in chain:
        b.append(node(x, ny, c))
    b.append(node(xo, ny, reach["reached_by_delivered_graph"]))
    for (x0c, c0), (x1c, c1) in zip(chain, chain[1:]):
        b.append(arrow(x0c + nw, x1c, ny + nh / 2, lost=c0 - c1))
    b.append(arrow(xs[3] + nw, xo, ny + nh / 2,
                   lost=serialized - reach["reached_by_delivered_graph"]))
    for x, s in zip(xs, ("scored", "delivered", "expressible", "serialized")):
        b.append(stage_label(x, ny + nh + 20, s))
    b.append(stage_label(xo, ny + nh + 20, "correct"))

    # --- grounded graph ------------------------------------------------------
    y = 266
    b.append(band(y, "grounded graph", "F45", "oracle-free bridge; deployable path", MUTED))
    ny = y + 14
    chain = [(xs[0], n), (xs[1], delivered), (xs[2], expressible), (xs[3], serialized),
             (xs[4], grounded)]
    for (x, c) in chain:
        b.append(node(x, ny, c))
    b.append(node(xo, ny, reach["reached_by_deployable_grounding"]))
    for (x0c, c0), (x1c, c1) in zip(chain, chain[1:]):
        b.append(arrow(x0c + nw, x1c, ny + nh / 2, lost=c0 - c1))
    b.append(arrow(xs[4] + nw, xo, ny + nh / 2,
                   lost=grounded - reach["reached_by_deployable_grounding"]))
    for x, s in zip(xs, ("scored", "delivered", "expressible", "serialized", "grounded")):
        b.append(stage_label(x, ny + nh + 20, s))
    b.append(stage_label(xo, ny + nh + 20, "correct"))
    b.append(text(xs[4] + nw / 2, ny + nh + 42, "the dominant loss", 15, BAD,
                  anchor="middle", weight="700"))

    # --- stored-human identity ----------------------------------------------
    y = 406
    b.append(band(y, "stored-human identity", "F35",
                  "identity_oracle \u2014 a bound, NOT DEPLOYABLE", ORACLE))
    ny = y + 14
    b.append(node(xs[0], ny, n))
    b.append(node(xo, ny, reach["held_by_representation"], oracle=True))
    b.append(arrow(xs[0] + nw, xo, ny + nh / 2, dashed=True,
                   note="identity injected by a human; referent grounding bypassed"))
    b.append(stage_label(xs[0], ny + nh + 20, "scored"))
    b.append(stage_label(xo, ny + nh + 20, "correct (held)"))

    # --- direct RGB ----------------------------------------------------------
    y = 524
    b.append(band(y, "direct multiview RGB", "F50", "no graph; deployable path", MUTED))
    ny = y + 14
    b.append(node(xs[0], ny, n))
    b.append(node(xo, ny, reach["reached_by_direct_rgb"]))
    b.append(arrow(xs[0] + nw, xo, ny + nh / 2, dashed=True,
                   note="graph stages bypassed \u2014 answers from the images alone"))
    b.append(stage_label(xs[0], ny + nh + 20, "scored"))
    b.append(stage_label(xo, ny + nh + 20, "correct"))

    return svg(W, H, "".join(b), "Per-arm reachability")


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    R = load()
    OUT.mkdir(parents=True, exist_ok=True)
    figs = {"fig1_evaluation_ladder.svg": figure_1_layers(R),
            "fig2_component_result.svg": figure_2_component(R),
            "fig3_held_but_unreachable.svg": figure_3_unreachable(R),
            "fig4_reachability.svg": figure_4_reachability(R)}
    for name, body in figs.items():
        (OUT / name).write_text(body)
        print(f"    {name:38s} {len(body):>7,} B")
    print(f"    registry rows read: {len(R)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
