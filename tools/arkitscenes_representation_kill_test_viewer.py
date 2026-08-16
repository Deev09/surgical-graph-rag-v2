"""Static result viewer for the ARKitScenes representation kill test.

  python3 tools/arkitscenes_representation_kill_test_viewer.py \
      --report runs/arkit_representation_kill_test/41069025_common6/report.json \
      --packet runs/arkit_representation_kill_test/41069025_common6/packet.json \
      --contact-sheet runs/arkit_representation_kill_test/41069025_common6/contact_sheet.jpg \
      --out runs/arkit_representation_kill_test/41069025_common6/result.html

Emits ONE offline HTML file with no scripts and no external requests: the
contact sheet is inlined as a data URI. Every answer, outcome, tally, rate and
decision value is rendered VERBATIM from the committed report; this tool
computes no metric and contains no JavaScript, so nothing on the page can
disagree with report.json.

A partial report (direct/hybrid arms not yet scored) renders those columns as
explicitly pending rather than blank or fabricated.

This is a presentation layer over an evaluation-only screening experiment. It
is not a benchmark-definition change and not an end-to-end product claim.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

SCHEMA = "arkitscenes_representation_kill_test_v1"

# Display order and human labels for the four arms. Keys must match the scorer.
ARMS = (
    ("object_retrieval", "RGB-labelled object map", "no graph edges"),
    ("typed_graph", "Current typed graph", "same instances + typed edges"),
    ("direct_vlm", "Direct visual VLM", "blinded, RGB views only"),
    ("hybrid", "Evidence-aware hybrid", "question-kind router + 2-view gate"),
)

OUTCOME_LABEL = {
    "correct": "correct",
    "wrong": "wrong",
    "unanswered": "unanswered",
}

HYPOTHESIS = [
    ("Visible semantics and inventory",
     "Direct real-RGB reasoning should beat the current object map, because the "
     "delivered labels are still unreliable."),
    ("Persistent spatial facts",
     "A typed 3D graph should help when a question requires cross-view identity, "
     "metric geometry or a materialized relation such as support — not merely "
     "object appearance."),
    ("Hybrid",
     "Routing visible-appearance questions to RGB and materialized-relation "
     "questions to the graph, while abstaining when evidence is insufficient, "
     "should reduce confident errors without collapsing coverage."),
]

FALSIFIER = (
    "The hypothesis is falsifiable. If the graph contributes no uniquely correct "
    "answer and the router does not beat the best single arm, do not build a "
    "larger hybrid — improve the missing relation evidence first, or use "
    "direct visual QA for the demo."
)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def fmt_answer(value: object) -> str:
    """Render a key/arm answer without changing its type semantics."""
    if value is None:
        return '<span class="none">—</span>'
    if isinstance(value, bool):
        return f"<code>{'true' if value else 'false'}</code>"
    return f"<code>{esc(value)}</code>"


def fmt_number(value: object, digits: int = 3) -> str:
    """Format a committed report number; never derive a new one."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def summary_cells(summary: dict) -> list[str]:
    tally = summary["tally"]
    return [
        str(tally["correct"]),
        str(tally["wrong"]),
        str(tally["unanswered"]),
        fmt_number(summary["coverage"]),
        fmt_number(summary["exact_accuracy_all"]),
        fmt_number(summary["accuracy_when_answered"]),
        fmt_number(summary["false_confident_rate"]),
    ]


def arm_summary_table(report: dict) -> str:
    head = ("arm", "correct", "wrong", "unanswered", "coverage",
            "exact accuracy", "accuracy when answered", "false-confident rate")
    rows = []
    for key, label, note in ARMS:
        arm = report["arms"].get(key)
        name = f'<th scope="row">{esc(label)}<span class="note">{esc(note)}</span></th>'
        if arm is None:
            cells = '<td class="pending" colspan="7">pending — arm not scored</td>'
        else:
            cells = "".join(f"<td>{c}</td>" for c in summary_cells(arm["summary"]))
        rows.append(f"<tr>{name}{cells}</tr>")
    header = "".join(f"<th>{esc(h)}</th>" for h in head)
    return (f'<table class="summary"><thead><tr>{header}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def evidence_html(row: dict) -> str:
    """Per-arm provenance: cited views, matched instances, route, refusal reason."""
    parts = []
    uids = row.get("matched_uids")
    if uids is not None:
        joined = ", ".join(esc(u) for u in uids) if uids else "none"
        parts.append(f'<div class="ev">matched instances: <code>{joined}</code></div>')
    frames = row.get("evidence_frame_ids")
    if frames is not None:
        joined = ", ".join(esc(f) for f in frames) if frames else "none"
        parts.append(f'<div class="ev">cited views: <code>{joined}</code></div>')
        parts.append('<div class="ev">two-view evidence: '
                     f'<code>{"yes" if row.get("evidence_sufficient") else "no"}</code></div>')
    if row.get("confidence") is not None:
        parts.append('<div class="ev">stated confidence: '
                     f'<code>{fmt_number(row["confidence"], 2)}</code></div>')
    if row.get("route"):
        parts.append(f'<div class="ev">route: <code>{esc(row["route"])}</code></div>')
        parts.append(f'<div class="ev">answered by: <code>{esc(row["source"])}</code></div>')
    if row.get("insufficiency"):
        parts.append(f'<div class="ev why">withheld: {esc(row["insufficiency"])}</div>')
    if row.get("reason"):
        parts.append(f'<div class="ev why">{esc(row["reason"])}</div>')
    return "".join(parts)


def question_blocks(report: dict, packet: dict) -> str:
    """One block per question; arms side by side, in the packet's question order."""
    by_arm = {
        key: {r["id"]: r for r in report["arms"][key]["rows"]}
        for key, _, _ in ARMS if key in report["arms"]
    }
    blocks = []
    for question in packet["questions"]:
        qid = question["id"]
        expected = None
        for key, _, _ in ARMS:
            row = by_arm.get(key, {}).get(qid)
            if row is not None and "expected" in row:
                expected = row["expected"]
                break
        cards = []
        for key, label, _ in ARMS:
            row = by_arm.get(key, {}).get(qid)
            if row is None:
                cards.append(
                    f'<div class="card pending"><h4>{esc(label)}</h4>'
                    '<p class="pending">pending — arm not scored</p></div>')
                continue
            outcome = row["outcome"]
            cards.append(
                f'<div class="card {esc(outcome)}"><h4>{esc(label)}</h4>'
                f'<div class="verdict">{esc(OUTCOME_LABEL[outcome])}</div>'
                f'<div class="ans">answer: {fmt_answer(row.get("answer"))}</div>'
                f'{evidence_html(row)}</div>')
        blocks.append(
            f'<section class="question"><h3>{esc(question["question"])}</h3>'
            f'<div class="qmeta"><code>{esc(qid)}</code>'
            f'<span>kind: <code>{esc(question["kind"])}</code></span>'
            f'<span>answer type: <code>{esc(question["answer_type"])}</code></span>'
            f'<span>human key: {fmt_answer(expected)}</span></div>'
            f'<div class="cards">{"".join(cards)}</div></section>')
    return "".join(blocks)


def decision_html(report: dict) -> str:
    dec = report["decision"]
    if dec.get("status") != "measured":
        return ('<p class="pending">Decision pending: the direct-visual arm has '
                'not been scored, so no proceed/stop value exists yet.</p>')
    verdict = "PROCEED" if dec["proceed"] else "STOP"
    rules = dec.get("rules", {})
    measured = [
        ("accuracy gain vs best single arm",
         rules.get("accuracy", ""),
         fmt_number(dec["absolute_accuracy_gain_vs_best_single"], 4)),
        ("wrong-answer reduction vs lowest single arm",
         rules.get("safety", ""),
         fmt_number(dec["wrong_answer_reduction_vs_lowest_single"], 4)),
        ("hybrid coverage", "", fmt_number(dec["hybrid_coverage"])),
    ]
    rows = "".join(
        f'<tr><th scope="row">{esc(name)}</th><td>{esc(rule)}</td>'
        f'<td class="val">{value}</td></tr>'
        for name, rule, value in measured)
    return (f'<p class="verdict-line {"go" if dec["proceed"] else "stop"}">'
            f'Predeclared screening rule: <strong>{verdict}</strong></p>'
            f'<table class="decision"><thead><tr><th>measured value</th>'
            f'<th>predeclared threshold</th><th>measured</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>'
            f'<p class="interp">{esc(rules.get("interpretation", ""))}</p>')


def integrity_html(report: dict) -> str:
    inputs = report["inputs"]
    edges = inputs["graph"].get("edge_types", {})
    edge_text = ", ".join(f"{k}: {v}" for k, v in sorted(edges.items())) or "none"
    items = [
        ("scene", report["scene_id"]),
        ("report stage", report["stage"]),
        ("human key SHA-256", inputs["key_sha256"]),
        ("answer-free packet SHA-256", inputs["packet_sha256"]),
        ("entity/graph geometry check",
         inputs["graph_geometry_compatibility"]["status"]),
        ("geometry signature",
         inputs["graph_geometry_compatibility"]["geometry_signature"]),
        ("delivered entities", inputs["entities"]["n_entities"]),
        ("graph edge types", edge_text),
        ("direct-response sidecar", inputs.get("direct_responses") or "not supplied"),
    ]
    rows = "".join(
        f'<tr><th scope="row">{esc(k)}</th><td><code>{esc(v)}</code></td></tr>'
        for k, v in items)
    return f'<table class="integrity"><tbody>{rows}</tbody></table>'


def scope_html(report: dict) -> str:
    scope = report["question_scope"]
    included = ", ".join(f"<code>{esc(q)}</code>" for q in scope["included"])
    excluded = ", ".join(f"<code>{esc(q)}</code>" for q in scope["excluded"])
    return (f'<p>Scored: {included}</p>'
            f'<p>Excluded: {excluded or "none"} — {esc(scope["reason"])}.</p>')


def build_html(report: dict, packet: dict, contact_sheet: Path) -> str:
    if report.get("schema") != SCHEMA:
        raise ValueError(f"unexpected report schema: {report.get('schema')!r}")
    if report["scene_id"] != packet["scene_id"]:
        raise ValueError("report and packet scene differ")
    if report["inputs"]["packet_sha256"] != packet["packet_sha256"]:
        raise ValueError("report was not scored against this packet")

    sheet_b64 = base64.b64encode(contact_sheet.read_bytes()).decode("ascii")
    hypothesis = "".join(
        f"<li><strong>{esc(name)}:</strong> {esc(text)}</li>"
        for name, text in HYPOTHESIS)
    limits = "".join(f"<li>{esc(item)}</li>" for item in report["limits"])
    n_frames = len(packet["frames"])
    selection = packet["selection"]["method"]

    return f"""<!DOCTYPE html>
<meta charset="utf-8">
<title>Representation kill test — {esc(report['scene_id'])}</title>
<style>
:root {{ color-scheme: light dark;
  --bg:#fbfbfa; --fg:#1a1a19; --muted:#6b6b66; --line:#dcdcd6; --card:#fff;
  --ok:#1a7f4b; --okbg:#e8f5ee; --bad:#a3372c; --badbg:#fbecea;
  --idk:#6b6b66; --idkbg:#f0f0ec; }}
@media (prefers-color-scheme: dark) {{ :root {{
  --bg:#16161a; --fg:#e8e8e4; --muted:#9a9a94; --line:#33333a; --card:#1e1e24;
  --ok:#6ede9f; --okbg:#163023; --bad:#f0938a; --badbg:#331d1a;
  --idk:#9a9a94; --idkbg:#26262c; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0 auto; padding:2.5rem 1.25rem 5rem; max-width:78rem;
  background:var(--bg); color:var(--fg);
  font:15px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif; }}
h1 {{ font-size:1.6rem; margin:0 0 .3rem; }}
h2 {{ font-size:1.1rem; margin:2.75rem 0 .75rem; padding-bottom:.35rem;
  border-bottom:1px solid var(--line); }}
h3 {{ font-size:1rem; margin:0 0 .5rem; }}
h4 {{ font-size:.8rem; margin:0 0 .5rem; color:var(--muted);
  text-transform:uppercase; letter-spacing:.04em; }}
p.sub {{ color:var(--muted); margin:0 0 .25rem; }}
code {{ font:12.5px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;
  background:var(--idkbg); padding:.08em .35em; border-radius:3px;
  overflow-wrap:anywhere; }}
table {{ border-collapse:collapse; width:100%; margin:.5rem 0; font-size:14px; }}
th,td {{ text-align:left; padding:.5rem .6rem; border-bottom:1px solid var(--line);
  vertical-align:top; }}
thead th {{ font-size:.72rem; text-transform:uppercase; letter-spacing:.04em;
  color:var(--muted); }}
tbody th {{ font-weight:600; }}
.summary td, .decision .val {{ font-variant-numeric:tabular-nums; }}
.summary tbody th .note {{ display:block; font-weight:400; font-size:.75rem;
  color:var(--muted); }}
.integrity th {{ width:16rem; color:var(--muted); font-weight:400; }}
.scroll {{ overflow-x:auto; }}
.callout {{ border-left:3px solid var(--line); padding:.1rem 0 .1rem 1rem;
  margin:1rem 0; color:var(--muted); }}
.question {{ border:1px solid var(--line); border-radius:8px; padding:1rem;
  margin:0 0 1rem; background:var(--card); }}
.qmeta {{ display:flex; flex-wrap:wrap; gap:.4rem 1rem; color:var(--muted);
  font-size:.8rem; margin-bottom:.9rem; }}
.cards {{ display:grid; gap:.75rem;
  grid-template-columns:repeat(auto-fit,minmax(15rem,1fr)); }}
.card {{ border:1px solid var(--line); border-radius:6px; padding:.7rem .8rem;
  background:var(--bg); }}
.card.correct {{ background:var(--okbg); border-color:var(--ok); }}
.card.wrong {{ background:var(--badbg); border-color:var(--bad); }}
.card.unanswered, .card.pending {{ background:var(--idkbg); }}
.verdict {{ font-weight:700; font-size:.85rem; letter-spacing:.03em;
  text-transform:uppercase; }}
.card.correct .verdict {{ color:var(--ok); }}
.card.wrong .verdict {{ color:var(--bad); }}
.card.unanswered .verdict {{ color:var(--idk); }}
.ans {{ margin:.35rem 0 .5rem; }}
.ev {{ font-size:.78rem; color:var(--muted); margin-top:.2rem; }}
.ev.why {{ font-style:italic; }}
.none {{ color:var(--muted); }}
.pending {{ color:var(--muted); font-style:italic; }}
.verdict-line {{ font-size:1.05rem; }}
.verdict-line.go strong {{ color:var(--ok); }}
.verdict-line.stop strong {{ color:var(--bad); }}
.interp {{ color:var(--muted); font-size:.85rem; }}
figure {{ margin:.5rem 0; }}
figure img {{ width:100%; height:auto; border:1px solid var(--line);
  border-radius:6px; display:block; }}
figcaption {{ color:var(--muted); font-size:.8rem; margin-top:.4rem; }}
ul {{ padding-left:1.15rem; }}
li {{ margin:.3rem 0; }}
footer {{ margin-top:3rem; padding-top:1rem; border-top:1px solid var(--line);
  color:var(--muted); font-size:.8rem; }}
</style>

<h1>Representation kill test — {esc(report['scene_id'])}</h1>
<p class="sub">Does explicit, evidence-bearing 3D structure add measurable value
over direct multi-view visual reasoning for spatial QA on one real handheld
room capture?</p>
<p class="sub">Screening result on <strong>one</strong> human-reviewed scene and
<strong>{len(packet['questions'])}</strong> room-observable questions. This page
displays committed values from <code>report.json</code> only; it computes no
metric and runs no script.</p>

<h2>Hypothesis under test</h2>
<ul>{hypothesis}</ul>
<p class="callout">{esc(FALSIFIER)}</p>

<h2>Result by arm</h2>
<div class="scroll">{arm_summary_table(report)}</div>

<h2>Predeclared decision</h2>
{decision_html(report)}

<h2>Question-by-question comparison</h2>
{question_blocks(report, packet)}

<h2>Views supplied to the direct visual arm</h2>
<figure>
  <img src="data:image/jpeg;base64,{sheet_b64}"
       alt="Contact sheet of {n_frames} RGB views selected from the capture.">
  <figcaption>{n_frames} views, selected by <code>{esc(selection)}</code> from
  {esc(packet['selection']['n_usable_strided_frames'])} usable strided frames.
  The direct arm saw these RGB views and nothing else — no mesh, no graph,
  no human key. An object outside these views is unobservable to it.</figcaption>
</figure>

<h2>Question scope</h2>
{scope_html(report)}

<h2>Inputs and integrity</h2>
<div class="scroll">{integrity_html(report)}</div>

<h2>Limits</h2>
<ul>{limits}</ul>

<footer>Generated from <code>report.json</code> by
<code>tools/arkitscenes_representation_kill_test_viewer.py</code>. No value on
this page is recomputed here. This is a screening comparison on a single scene:
it is not a generalization claim, not an embodied-planning result, and not a
claim of human-level room understanding.</footer>
"""


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--packet", type=Path, required=True)
    ap.add_argument("--contact-sheet", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = json.loads(args.report.read_text())
    packet = json.loads(args.packet.read_text())
    page = build_html(report, packet, args.contact_sheet)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page)
    print(f"stage: {report['stage']}")
    print(f"decision: {report['decision']['status']} "
          f"proceed={report['decision']['proceed']}")
    print(f"viewer -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
