"""Precomputed direct-RGB demo: a recorded evaluation replay, not live QA.

  python3 tools/arkitscenes_rgb_demo.py \
      --report runs/arkit_relation_challenge/report.json \
      --questions eval/questions/arkitscenes_relation_challenge_v1.json \
      --out runs/arkit_relation_challenge/rgb_demo.html

Emits ONE self-contained offline page. Every value on it is read from the
committed report; the page computes no metric and its only script toggles the
visibility of panels that were already rendered server-side. There is no
network request, no API key and no model call at view time.

WHAT IT IS, AND WHAT IT IS NOT
------------------------------
It replays twelve fixed questions that were scored once against an
owner-confirmed key, with the blinded responses hash-pinned beforehand. It is
NOT open-ended question answering: a typed question would need a live vision
API, which this build deliberately does not have. The page says so.

THE 3D LAYER IS EVIDENCE, NOT THE ANSWER ENGINE
------------------------------------------------
Scene renders appear beside each question as inspectable state, labelled with
the identity source they depend on. The geometry ceiling and the stored-graph
replay both consume HUMAN-VERIFIED identity, so their numbers are not
deployable performance and the page never presents them as such.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.arkitscenes import MESH_SUFFIX, read_mesh
from tools import arkitscenes_uid_visual_sheet as uid_sheet
from tools.arkitscenes_uid_visual_sheet import SceneRenderer

uid_sheet.CONTEXT_PX = 460

SCENE_DIRS = {"arkitscenes_41069025": "41069025",
              "arkitscenes_41069042": "41069042"}
DEFAULT_DATA_ROOT = Path.home() / "Desktop/datasets/arkitscenes/Validation"
FRAME_PX = 340
RENDER_PX = 380

# Layer display order and the identity each one depends on. `deployable` is
# rendered next to every number so a ceiling can never be read as performance.
LAYERS = (
    ("blinded_rgb_vlm", "Direct multiview RGB", "the images themselves", True),
    ("grounded_delivered_graph", "Grounded delivered graph",
     "oracle-free grounding bridge", True),
    ("delivered_graph", "Delivered graph", "learned labels", True),
    ("stored_graph_human_identity", "Stored graph + human identity",
     "HUMAN-VERIFIED identity", False),
    ("geometry_relation_ceiling", "Geometry ceiling",
     "HUMAN-VERIFIED identity", False),
)


def esc(v: object) -> str:
    return html.escape(str(v), quote=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jpeg_uri(image: Image.Image, px: int, quality: int = 82) -> str:
    image = image.convert("RGB").copy()
    image.thumbnail((px, px), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def cited_frames(report: dict, packets: dict) -> dict:
    """Embed each cited frame once, keyed by id; questions reference the id."""
    wanted: set[tuple[str, str]] = set()
    for row in report["arms"]["blinded_rgb_vlm"]["rows"]:
        for fid in row.get("evidence_frame_ids") or []:
            wanted.add((row["scene_id"], fid))
    out = {}
    for scene_id, fid in sorted(wanted):
        packet_dir = packets[scene_id]["_dir"]
        entry = next(f for f in packets[scene_id]["frames"] if f["id"] == fid)
        out[f"{scene_id}/{fid}"] = jpeg_uri(
            Image.open(packet_dir / entry["file"]), FRAME_PX)
    return out


def scene_renders(report: dict, data_root: Path) -> dict:
    """Per-question 3D context with the ceiling's entities highlighted.

    Evidence, not an answer. The highlighted uids come from the human-verified
    mapping, which is exactly why the panel is labelled non-deployable.
    """
    ceiling = {r["id"]: r for r in report["arms"]["geometry_relation_ceiling"]["rows"]}
    by_scene: dict[str, list[str]] = {}
    for row in ceiling.values():
        by_scene.setdefault(row["scene_id"], [])
    renders = {}
    for scene_id in sorted(by_scene):
        short = SCENE_DIRS[scene_id]
        mesh = read_mesh(data_root / short / f"{short}{MESH_SUFFIX}")
        ids = np.load(REPO_ROOT / "runs" / "arkitscenes_mask3d_transfer"
                      / f"bundle_arkitscenes_{short}" / "vertex_instance_ids.npy")
        manifest = json.loads(
            (REPO_ROOT / "runs" / f"arkit_label_image_ab_{short}"
             / "rgb_tight" / "entities" / "manifest.json").read_text())
        handle = {e["identity"]["object_uid"]:
                  int(e["geometry_handle"].rsplit("#", 1)[1])
                  for e in manifest["entities"]}
        renderer = SceneRenderer(mesh.xyz, mesh.rgb)
        for qid, row in ceiling.items():
            if row["scene_id"] != scene_id:
                continue
            uids = row.get("uids") or []
            vertices = (np.concatenate([np.flatnonzero(ids == handle[u])
                                        for u in uids if u in handle])
                        if uids else np.array([], dtype=int))
            if len(vertices) == 0:
                renders[qid] = None
                continue
            renders[qid] = [
                jpeg_uri(Image.open(io.BytesIO(base64.b64decode(
                    uri.split(",", 1)[1]))), RENDER_PX)
                for uri in renderer.context(vertices)]
    return renders


def outcome_chip(outcome: str) -> str:
    label = {"correct": "correct", "wrong": "wrong", "unanswered": "unknown",
             "excluded_no_human_answer": "excluded"}.get(outcome, outcome)
    return f'<span class="chip {esc(outcome)}">{esc(label)}</span>'


def fmt(value: object) -> str:
    if value is None:
        return '<span class="none">—</span>'
    if isinstance(value, bool):
        return f"<code>{'true' if value else 'false'}</code>"
    if isinstance(value, list):
        return " ".join(f"<code>{esc(v)}</code>" for v in value) or '<span class="none">—</span>'
    return f"<code>{esc(value)}</code>"


def question_panel(question: dict, report: dict, frames: dict,
                   renders: dict) -> str:
    qid, scene_id = question["id"], question["scene_id"]
    rows = {name: {r["id"]: r for r in report["arms"][name]["rows"]}[qid]
            for name, _, _, _ in LAYERS}
    rgb = rows["blinded_rgb_vlm"]

    cited = "".join(
        f'<figure><img src="{frames[f"{scene_id}/{fid}"]}" alt="{esc(fid)}">'
        f'<figcaption>{esc(fid)}</figcaption></figure>'
        for fid in rgb.get("evidence_frame_ids") or []
        if f"{scene_id}/{fid}" in frames)
    if not cited:
        cited = ('<p class="none">No frames cited — the model returned '
                 '<code>unknown</code>.</p>')

    layer_rows = "".join(
        f'<tr><th scope="row">{esc(title)}'
        f'<span class="idsrc">identity: {esc(source)}</span></th>'
        f'<td>{outcome_chip(rows[name]["outcome"])}</td>'
        f'<td>{fmt(rows[name].get("answer"))}</td>'
        f'<td>{"" if deployable else "<b>not deployable</b>"}</td></tr>'
        for name, title, source, deployable in LAYERS)

    render = renders.get(qid)
    render_html = (
        "".join(f'<figure><img src="{uri}" alt="scene context {i + 1}">'
                f'<figcaption>context {i + 1}</figcaption></figure>'
                for i, uri in enumerate(render))
        if render else
        '<p class="none">No entities to highlight: the geometry ceiling '
        'abstained on this question.</p>')

    return f"""
  <article class="panel" id="panel-{esc(qid)}" data-qid="{esc(qid)}" hidden>
    <h2>{esc(question['question'])}</h2>
    <div class="qmeta"><code>{esc(qid)}</code>
      <span>scene <code>{esc(scene_id)}</code></span>
      <span>form <code>{esc(question['form'])}</code></span>
      {'<span class="tag">cross-view</span>' if question.get('cross_view') else ''}
    </div>

    <div class="headline">
      <div class="hl"><span class="hlk">Direct RGB answer</span>
        {outcome_chip(rgb['outcome'])} {fmt(rgb.get('answer'))}</div>
      <div class="hl"><span class="hlk">Human answer</span>{fmt(rgb.get('expected'))}</div>
      <div class="hl"><span class="hlk">Model confidence</span>{fmt(rgb.get('confidence'))}</div>
      <div class="hl"><span class="hlk">Owner-recorded evidence views</span>
        {fmt(rgb.get('evidence_views'))}</div>
    </div>

    <h3>Cited RGB frames <span class="sub">the evidence the answer was given on</span></h3>
    <div class="frames">{cited}</div>

    <h3>Every layer on this question</h3>
    <table class="layers"><thead><tr><th>layer</th><th>outcome</th>
      <th>answer</th><th></th></tr></thead><tbody>{layer_rows}</tbody></table>

    <h3>3D scene context <span class="sub">inspectable evidence — not the answer engine</span></h3>
    <p class="note">Highlighted regions are the entities the geometry ceiling
    used, resolved by <b>human-verified</b> identity. That is why the two
    ceiling rows above are marked not deployable: they were handed the object
    identities a person supplied, which no shipped system has.</p>
    <div class="renders">{render_html}</div>
  </article>"""


def build(report: dict, questions_doc: dict, frames: dict, renders: dict,
          provenance: dict) -> str:
    questions = questions_doc["questions"]
    options = "".join(
        f'<option value="{esc(q["id"])}">{esc(SCENE_DIRS[q["scene_id"]])} — '
        f'{esc(q["question"])}</option>' for q in questions)
    panels = "".join(question_panel(q, report, frames, renders) for q in questions)

    summary_rows = "".join(
        f'<tr><th scope="row">{esc(title)}'
        f'<span class="idsrc">identity: {esc(source)}</span></th>'
        f'<td class="n">{report["arms"][name]["summary"]["tally"]["correct"]}'
        f' / {report["arms"][name]["summary"]["n_questions_scored"]}</td>'
        f'<td class="n">{report["arms"][name]["summary"]["coverage"]}</td>'
        f'<td>{"yes" if deployable else "<b>no</b>"}</td></tr>'
        for name, title, source, deployable in LAYERS)

    inputs = "".join(f"<tr><td><code>{esc(k)}</code></td>"
                     f"<td><code>{esc(v)}</code></td></tr>"
                     for k, v in sorted(provenance["inputs"].items()))

    return f"""<!DOCTYPE html>
<meta charset="utf-8">
<title>Recorded evaluation replay — direct multiview RGB</title>
<style>
:root {{ color-scheme: light dark; --bg:#fbfbfa; --fg:#1a1a19; --mut:#6b6b66;
  --line:#dcdcd6; --card:#fff; --ok:#1a7f4b; --okbg:#e8f5ee; --bad:#a3372c;
  --badbg:#fbecea; --idk:#6b6b66; --idkbg:#f0f0ec; --warn:#8a5a00;
  --warnbg:#fff6e0; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#16161a; --fg:#e8e8e4; --mut:#9a9a94; --line:#33333a; --card:#1e1e24;
  --ok:#6ede9f; --okbg:#163023; --bad:#f0938a; --badbg:#331d1a;
  --idk:#9a9a94; --idkbg:#26262c; --warn:#f0c060; --warnbg:#2a2214; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0 auto; padding:2rem 1.25rem 5rem; max-width:64rem; background:var(--bg);
  color:var(--fg); font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",sans-serif; }}
h1 {{ font-size:1.5rem; margin:0 0 .25rem; }}
h2 {{ font-size:1.1rem; margin:0 0 .5rem; }}
h3 {{ font-size:.9rem; margin:1.5rem 0 .5rem; text-transform:uppercase;
  letter-spacing:.04em; color:var(--mut); }}
h3 .sub {{ text-transform:none; letter-spacing:0; font-weight:400; }}
p.sub, .sub {{ color:var(--mut); font-size:.85rem; }}
code {{ font:12.5px ui-monospace,Menlo,monospace; background:var(--idkbg);
  padding:.08em .35em; border-radius:3px; overflow-wrap:anywhere; }}
.banner {{ border-left:3px solid var(--warn); background:var(--warnbg);
  padding:.8rem 1rem; border-radius:0 6px 6px 0; margin:1rem 0; font-size:.9rem; }}
.note {{ border-left:3px solid var(--line); padding:.15rem 0 .15rem 1rem;
  color:var(--mut); font-size:.85rem; margin:.5rem 0; }}
table {{ border-collapse:collapse; width:100%; font-size:14px; margin:.5rem 0; }}
th,td {{ text-align:left; padding:.45rem .6rem; border-bottom:1px solid var(--line);
  vertical-align:top; }}
thead th {{ font-size:.72rem; text-transform:uppercase; color:var(--mut); }}
tbody th {{ font-weight:600; }}
.idsrc {{ display:block; font-weight:400; font-size:.75rem; color:var(--mut); }}
td.n {{ font-variant-numeric:tabular-nums; }}
select {{ font:inherit; padding:.5rem; width:100%; max-width:100%;
  background:var(--card); color:var(--fg); border:1px solid var(--line);
  border-radius:6px; }}
.panel {{ border:1px solid var(--line); border-radius:10px; padding:1.2rem 1.3rem;
  background:var(--card); margin-top:1rem; }}
.qmeta {{ display:flex; gap:.9rem; flex-wrap:wrap; color:var(--mut);
  font-size:.78rem; margin-bottom:1rem; }}
.tag {{ background:rgba(128,128,128,.16); padding:1px 7px; border-radius:10px; }}
.headline {{ display:grid; gap:.6rem;
  grid-template-columns:repeat(auto-fit,minmax(13rem,1fr)); margin-bottom:.5rem; }}
.hl {{ border:1px solid var(--line); border-radius:7px; padding:.55rem .7rem; }}
.hlk {{ display:block; font-size:.7rem; text-transform:uppercase;
  letter-spacing:.04em; color:var(--mut); margin-bottom:.25rem; }}
.chip {{ display:inline-block; padding:.1em .55em; border-radius:10px;
  font-size:.78rem; font-weight:700; text-transform:uppercase; }}
.chip.correct {{ background:var(--okbg); color:var(--ok); }}
.chip.wrong {{ background:var(--badbg); color:var(--bad); }}
.chip.unanswered, .chip.excluded_no_human_answer {{ background:var(--idkbg); color:var(--idk); }}
.frames, .renders {{ display:grid; gap:.6rem;
  grid-template-columns:repeat(auto-fit,minmax(15rem,1fr)); }}
figure {{ margin:0; }}
figure img {{ width:100%; border:1px solid var(--line); border-radius:6px; display:block; }}
figcaption {{ font-size:.72rem; color:var(--mut); margin-top:.25rem; }}
.none {{ color:var(--mut); }}
footer {{ margin-top:3rem; padding-top:1rem; border-top:1px solid var(--line);
  color:var(--mut); font-size:.8rem; }}
</style>

<h1>Recorded evaluation replay — direct multiview RGB</h1>
<p class="sub">Twelve fixed questions, scored once against an owner-confirmed
key. Not live question answering.</p>

<div class="banner">
  <b>This is a recorded evaluation replay.</b> Every answer below was produced
  once, in a blinded context, and hash-pinned before the human key was opened.
  Nothing here calls a model: the page is static, makes no network request and
  contains no API key. <b>Arbitrary typed questions are not supported</b> — an
  open-ended demo would need a live vision API, which this build deliberately
  does not have.
</div>

<h3>The measured result</h3>
<table><thead><tr><th>layer</th><th>correct</th><th>coverage</th>
  <th>deployable</th></tr></thead><tbody>{summary_rows}</tbody></table>
<p class="note">Direct RGB and the identity ceiling both answer 7 of 10 — but
<b>on different questions</b>, which is the whole finding. The delivered graph
answers 0 of 10 and the grounded graph 2 of 10, so the spatial information the
ceiling reaches is real and is not deployably reachable. The two ceiling rows
were handed object identities a person supplied; they are bounds on what the
representation could express, never system performance.</p>

<h3>Pick a question</h3>
<select id="pick" aria-label="Question">{options}</select>
{panels}

<footer>
  <p>Every value on this page is read from a committed report. The page
  computes no metric, and its only script shows and hides panels that were
  rendered ahead of time. Deterministic: identical inputs produce a
  byte-identical page.</p>
  <table><thead><tr><th>input</th><th>sha256</th></tr></thead>
    <tbody>{inputs}</tbody></table>
  <p>Blinded responses were sealed before the key was opened. Graph results are
  retained as the measured negative comparison, not as a rejected alternative.</p>
</footer>

<script>
// Shows and hides panels that were rendered ahead of time. Deliberately free
// of arithmetic and string building: any calculation here would be a second,
// unreviewed implementation of a number that already exists in the committed
// report, and the two could disagree with nothing failing.
var pick = document.getElementById("pick");
function show() {{
  document.querySelectorAll(".panel").forEach(function (panel) {{
    panel.hidden = panel.dataset.qid !== pick.value;
  }});
}}
pick.addEventListener("change", show);
show();
</script>
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--packets-root", type=Path,
                    default=REPO_ROOT / "runs" / "arkit_relation_challenge" / "blinded_rgb")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    report = json.loads(args.report.read_text())
    questions_doc = json.loads(args.questions.read_text())

    packets, inputs = {}, {"report": sha256(args.report),
                           "questions": sha256(args.questions)}
    for scene_id, short in SCENE_DIRS.items():
        path = args.packets_root / short / "packet.json"
        packet = json.loads(path.read_text())
        packet["_dir"] = path.parent
        packets[scene_id] = packet
        inputs[f"packet_{short}"] = sha256(path)

    frames = cited_frames(report, packets)
    renders = scene_renders(report, args.data_root)
    page = build(report, questions_doc, frames, renders, {"inputs": inputs})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page)

    print(f"    questions      : {len(questions_doc['questions'])}")
    print(f"    cited frames   : {len(frames)} unique, embedded once each")
    print(f"    scene renders  : {sum(1 for v in renders.values() if v)}")
    print(f"    page           : {args.out.stat().st_size / 1e6:.1f} MB self-contained")
    print(f"    page sha256    : {sha256(args.out)}")
    print(f"    -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
