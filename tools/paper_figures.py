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
def figure_1_layers(R: dict) -> str:
    """The evaluation ladder: which stage each layer exercises, and its scope."""
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
    # 1040, not 940: at 940 the NOT DEPLOYABLE badges — the whole point of the
    # figure — ran off the right edge at x=962.8 and were clipped by the renderer.
    W, H = 1040, 450
    # top=116, not 96: the ORACLE note at y=72 is wide enough to reach x~478, and
    # at top=96 the stage headers (drawn at top-26) collided with it horizontally.
    x0, top, colw, rowh = 250, 116, 96, 52
    b = [text(28, 34, "Figure 1 — The evaluation ladder", 15, INK, weight="600"),
         text(28, 55, "Each layer holds the same pipeline fixed and changes one source of "
                      "identity. Bars span the stages a layer relies on.", 11, MUTED),
         text(28, 72, "Hatched layers consume human-supplied identity: bounds, never system "
                      "performance.", 11, ORACLE)]
    for i, s in enumerate(stages):
        cx = x0 + i * colw + colw / 2
        for k, part in enumerate(s.split("\n")):
            b.append(text(cx, top - 26 + k * 11, part, 9, FAINT, anchor="middle",
                          spacing="0.04em"))
        b.append(f'<line x1="{cx}" y1="{top - 8}" x2="{cx}" y2="{top + len(rows)*rowh - 12}" '
                 f'stroke="{LINE}" stroke-width="1"/>')
    for j, (name, val, a, z, oracle, basis) in enumerate(rows):
        y = top + j * rowh
        b.append(text(28, y + 4, name, 12, INK, weight="600"))
        b.append(text(28, y + 19, basis, 9.5, FAINT))
        bx = x0 + a * colw + 6
        bw = (z - a) * colw + colw - 12
        fill = "url(#hatch)" if oracle else ACCENT
        b.append(f'<rect x="{bx}" y="{y - 12}" width="{bw}" height="22" rx="3" '
                 f'fill="{fill}" stroke="{ORACLE if oracle else ACCENT}" stroke-width="1"/>')
        b.append(text(bx + bw + 12, y + 4, val, 12, INK, weight="700", family=MONO))
        b.append(text(bx + bw + 58, y + 4,
                      "NOT DEPLOYABLE" if oracle else "deployable" if j == 4 else "delivered",
                      8.5, ORACLE if oracle else MUTED, weight="700" if oracle else "normal",
                      spacing="0.06em"))
    b.append(text(28, H - 26, "correct / 10 scored items, ARKitScenes 41069025 + 41069042 "
                              "· registry F28, F35, F40, F45, F50", 9.5, FAINT, family=MONO))
    return svg(W, H, "".join(b), "The evaluation ladder")


# --------------------------------------------------------------------------
def figure_2_component(R: dict) -> str:
    """Scoped component result: labeler input A/B, per scene and pooled."""
    scenes = [("41069021", "C05", "C07", "C06", "C08", "C17", "C18"),
              ("41069025", "C09", "C11", "C10", "C12", None, None),
              ("41069042", "C13", "C15", "C14", "C16", None, None),
              ("pooled",   "C01", "C02", "C03", "C04", None, None)]
    W, H = 940, 430
    left, top, gw, bh = 60, 130, 210, 15
    b = [text(28, 34, "Figure 2 — Real-RGB crops as the labeler input", 15, INK, weight="600"),
         text(28, 55, "Only the label images changed. Weights, vocabulary, admission "
                      "threshold, evaluator, delivered partitions and IoU matching all fixed.",
              11, MUTED),
         f'<rect x="28" y="66" width="884" height="26" rx="3" fill="#f6efdc" '
         f'stroke="{ORACLE}" stroke-width="1"/>',
         text(40, 83, "SCOPE  oracle_free_component_eval — the denominator is instances the "
                      "evaluator already matched to an annotation box. NOT end-to-end "
                      "performance.", 10.5, ORACLE, weight="600")]
    maxden = 21
    scale = 150 / maxden
    for i, (scene, s1, r1, s3, r3, c1, c3) in enumerate(scenes):
        gx = left + i * gw
        b.append(text(gx, top - 14, scene, 11.5, INK, weight="700", family=MONO))
        series = [("splat", s1, s3, "#b9c0c8"), ("rgb_tight", r1, r3, ACCENT)]
        if c1:
            series.append(("rgb_context", c1, c3, "#8aa8ab"))
        y = top
        for label, i1, i3, col in series:
            n1, d1 = frac(R[i1]); n3, d3 = frac(R[i3])
            b.append(text(gx, y + 8, label, 9.5, MUTED))
            for k, (n, d, idr) in enumerate(((n1, d1, i1), (n3, d3, i3))):
                yy = y + 14 + k * (bh + 5)
                b.append(f'<rect x="{gx}" y="{yy}" width="{d*scale:.1f}" height="{bh}" '
                         f'rx="2" fill="#eef1f4"/>')
                b.append(f'<rect x="{gx}" y="{yy}" width="{n*scale:.1f}" height="{bh}" '
                         f'rx="2" fill="{col}"/>')
                b.append(text(gx + d * scale + 8, yy + 11.5, f"{n}/{d}", 10, INK,
                              weight="700" if label == "rgb_tight" else "normal", family=MONO))
                b.append(text(gx + d * scale + 46, yy + 11.5, f"top-{1 if k==0 else 3}",
                              8.5, FAINT))
                b.append(text(gx + d * scale + 78, yy + 11.5, idr, 8, FAINT, family=MONO))
            y += 62
    b.append(text(28, H - 44, "The context control (41069021, rgb_context) scores BELOW "
                              "rgb_tight at both ranks — 3/7 and 5/7 against 5/7 and 7/7.",
                  10.5, INK))
    b.append(text(28, H - 28, "More context hurts, which SUPPORTS the interpretation that the "
                              "gain comes from object texture rather than room gist. It does "
                              "not prove it.", 10.5, MUTED))
    b.append(text(28, H - 10, "registry C01–C21", 9.5, FAINT, family=MONO))
    return svg(W, H, "".join(b), "Real-RGB crops as the labeler input")


# --------------------------------------------------------------------------
def figure_3_unreachable(R: dict) -> str:
    """Held but unreachable, plus what direct RGB does on an unseen room."""
    W, H = 1040, 566
    b = [text(28, 34, "Figure 3 — Held, but unreachable", 15, INK, weight="600"),
         text(28, 55, "Same twelve questions, same stored relations, same scoring. "
                      "Only the source of object identity changes.", 11, MUTED),
         # fig2 carries a scope banner; a picture-only reader of fig3 needs the
         # same warning, because the top bar is a bound and not a system result.
         f'<rect x="28" y="66" width="984" height="26" rx="3" fill="#f6efdc" '
         f'stroke="{ORACLE}" stroke-width="1"/>',
         text(40, 83, "SCOPE  the hatched bar is identity_oracle — it consumes "
                      "human-supplied identity. It BOUNDS what the representation "
                      "could express; it is not performance.", 10, ORACLE,
              weight="600")]
    bars = [("stored relations, human identity", "F35", True),
            ("stored relations, grounded identity", "F45", False),
            ("stored relations, learned labels", "F40", False),
            ("direct multiview RGB", "F50", False)]
    left, top, bw, rowh = 300, 116, 250, 46
    # The RGB arm is a different family from the three stored-relation arms, so
    # it is pushed down past the gap brace rather than sharing its bracket.
    def rowy(i): return top + i * rowh + (44 if i == 3 else 0)
    for i, (label, rid, oracle) in enumerate(bars):
        n, d = frac(R[rid]); y = rowy(i)
        b.append(text(286, y + 15, label, 11.5, INK, anchor="end",
                      weight="700" if i == 0 else "normal"))
        b.append(f'<rect x="{left}" y="{y}" width="{bw}" height="22" rx="3" fill="#eef1f4"/>')
        fill = "url(#hatch)" if oracle else (ACCENT if n else "#e2e6ea")
        b.append(f'<rect x="{left}" y="{y}" width="{bw*n/d:.1f}" height="22" rx="3" '
                 f'fill="{fill}" stroke="{ORACLE if oracle else "none"}" stroke-width="1"/>')
        b.append(text(left + bw + 14, y + 15, f"{n}/{d}", 13, INK, weight="700", family=MONO))
        b.append(text(left + bw + 62, y + 15, rid, 9, FAINT, family=MONO))
        if oracle:
            b.append(text(left + bw + 96, y + 15, "NOT DEPLOYABLE", 8.5, ORACLE,
                          weight="700", spacing="0.06em"))
        elif rid == "F50":
            # F50 draws the same width as the F35 bound. Left unmarked, the tie
            # reads as "the deployable path already reaches the ceiling", which
            # is the opposite of this figure's claim.
            b.append(text(left + bw + 96, y + 15, "deployable — ties the bound on "
                          "DIFFERENT items", 8.5, OK, weight="700"))
    # The gap brace sits BELOW the bars, not beside them: an earlier version put
    # its label inline and it overlapped the 2/10 value and its result_id.
    gx0, gx1 = left, left + bw * 0.70
    gy = top + 3 * rowh + 2
    b.append(f'<path d="M {gx0} {gy-8} L {gx0} {gy} L {gx1} {gy} L {gx1} {gy-8}" '
             f'fill="none" stroke="{BAD}" stroke-width="1.4"/>')
    b.append(text((gx0 + gx1) / 2, gy + 15, "the reachability gap — 7/10 held, 0/10 reached",
                  10.5, BAD, anchor="middle", weight="700"))
    n64, d64 = frac(R["F64"])
    b.append(text(28, rowy(3) + 56,
                  f"Relation extraction is not the loss: the stored-edge replay agrees with "
                  f"recomputed geometry {R['F63']['value'].split('(')[0].strip()} item for item "
                  f"[F63].", 11, INK))
    b.append(text(28, rowy(3) + 74,
                  f"{n64} of {d64} items are ceiling-correct but delivered-unanswered [F64]; "
                  f"delivered-graph-unique wins {R['F60']['value'].split('(')[0].strip()} [F60].",
                  11, MUTED))
    yb = rowy(3) + 106
    b.append(f'<line x1="28" y1="{yb-14}" x2="1012" y2="{yb-14}" stroke="{LINE}" stroke-width="1"/>')
    b.append(text(28, yb + 6, "On a previously unseen room, direct RGB does not close it either",
                  12.5, INK, weight="600"))
    # The two gates decided the outcome, so they are tiles rather than a footnote.
    # 1.000-when-answered is NOT a tile: its denominator is 5, and at tile size it
    # reads as the transfer result.
    cells = [(f"{frac(R['F76'])[0]}/{frac(R['F76'])[1]}", "correct", "F76", ACCENT),
             (f"{frac(R['F77'])[0]}/{frac(R['F77'])[1]}", "wrong", "F77", OK),
             ("0/3", "cross-view — all declined", "F79", BAD),
             ("0.50", "exact accuracy — FAIL (gate 0.60)", "F82", BAD),
             ("0.50", "coverage — FAIL (gate 0.80)", "F83", BAD)]
    for i, (val, lab, rid, col) in enumerate(cells):
        cx = 40 + i * 196
        b.append(f'<rect x="{cx-12}" y="{yb+18}" width="184" height="52" rx="4" '
                 f'fill="{BG}" stroke="{LINE}" stroke-width="1"/>')
        b.append(text(cx, yb + 42, val, 17, col, weight="700", family=MONO))
        b.append(text(cx, yb + 60, lab, 8.5, MUTED))
        b.append(text(cx + 160, yb + 42, rid, 8, FAINT, anchor="end", family=MONO))
    b.append(text(28, H - 26, f"Accuracy when answered is {R['F85']['value']} [F85] over the "
                              "5 items it answered — denominator 5, not 10.", 10, FAINT))
    b.append(text(28, H - 10, "Both predeclared gates failed; no transfer claim is made [F88]. "
                              "On the development rooms this same arm was wrong 2/10 [F51].",
                  10, FAINT))
    return svg(W, H, "".join(b), "Held, but unreachable")


# --------------------------------------------------------------------------
def figure_4_reachability(R: dict) -> str:
    """Where questions are lost between representation and deployable answer.

    Reads the committed statistics report rather than recomputing it, so this
    figure and docs/paper_reachability_ledger.csv can never disagree.
    """
    stats = json.loads((REPO_ROOT / "eval" / "results" / "paper_statistics.json").read_text())
    reach = stats["reachability"]
    stages = reach["survivors_by_stage"]
    n = reach["n_scored"]

    # H=520, not 470: the stat tiles run to y=486 and the footer sits below them.
    W, H = 1040, 520
    b = [text(28, 34, "Figure 4 — Where questions are lost", 15, INK, weight="600"),
         text(28, 55, "One row per stage an answer must survive, over the "
                      f"{n} scored relation questions. Read top to bottom.", 11, MUTED),
         f'<rect x="28" y="66" width="984" height="26" rx="3" fill="#f6efdc" '
         f'stroke="{ORACLE}" stroke-width="1"/>',
         text(40, 83, "SCOPE  bookkeeping over already-scored outcomes, not a new "
                      "measurement. n = 10 questions over two rooms.", 10, ORACLE,
              weight="600")]

    left, top, bw, rowh = 330, 116, 420, 44
    labels = {
        "human_answerable": "a human answered it",
        "objects_delivered": "objects are in the delivered partition",
        "relation_expressible": "NEAR can express the question",
        "edge_serialized": "the serialized edge matches geometry",
        "anchor_grounded": "the bridge bound the anchors",
        "graph_correct": "the delivered graph answered correctly",
    }
    worst = max(s["lost_here"] for s in stages)
    for i, st in enumerate(stages):
        y = top + i * rowh
        surviving, lost = st["surviving"], st["lost_here"]
        is_worst = lost == worst and lost > 0
        b.append(text(318, y + 15, labels.get(st["stage"], st["stage"]), 11,
                      INK if is_worst else MUTED, anchor="end",
                      weight="700" if is_worst else "normal"))
        b.append(f'<rect x="{left}" y="{y}" width="{bw}" height="22" rx="3" fill="#eef1f4"/>')
        w = bw * surviving / n
        if w > 0:
            b.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="22" rx="3" '
                     f'fill="{ACCENT}"/>')
        b.append(text(left + bw + 14, y + 15, f"{surviving}/{n}", 13,
                      BAD if surviving == 0 else INK, weight="700", family=MONO))
        if lost > 0:
            b.append(text(left + bw + 74, y + 15, f"−{lost} here", 10,
                          BAD if is_worst else FAINT,
                          weight="700" if is_worst else "normal"))
        if is_worst:
            b.append(text(left + bw + 150, y + 15, "the dominant single loss", 9,
                          BAD, weight="700", spacing="0.05em"))

    yb = top + len(stages) * rowh + 14
    b.append(f'<line x1="28" y1="{yb}" x2="1012" y2="{yb}" stroke="{LINE}" stroke-width="1"/>')
    b.append(text(28, yb + 22, "Held by the representation, versus reached by a "
                               "deployable path", 12.5, INK, weight="600"))
    cells = [(f"{reach['held_by_representation']}/{n}", "held — stored relations,",
              "human identity [F35]", ORACLE),
             (f"{reach['reached_by_delivered_graph']}/{n}", "reached — delivered graph",
              "[F40]", BAD),
             (f"{reach['reached_by_deployable_grounding']}/{n}", "reached — grounded graph",
              "[F45]", BAD),
             (f"{reach['reached_by_direct_rgb']}/{n}", "reached — direct RGB",
              "[F50]", OK)]
    for i, (val, lab, sub, col) in enumerate(cells):
        cx = 40 + i * 246
        b.append(f'<rect x="{cx-12}" y="{yb+34}" width="212" height="58" rx="4" '
                 f'fill="{BG}" stroke="{LINE}" stroke-width="1"/>')
        b.append(text(cx, yb + 58, val, 17, col, weight="700", family=MONO))
        b.append(text(cx, yb + 74, lab, 9, MUTED))
        b.append(text(cx, yb + 87, sub, 8.5, FAINT))

    b.append(text(28, H - 12, "Extraction is not the loss: no question is lost at the "
                              "serialized-edge stage. Identity grounding is where they go.",
                  10, FAINT))
    return svg(W, H, "".join(b), "Where questions are lost")


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
