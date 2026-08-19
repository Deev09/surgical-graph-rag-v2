"""Answer-free review kit for the two-scene NEAR relation challenge.

  python3 tools/arkitscenes_relation_challenge_review.py sheet
  python3 tools/arkitscenes_relation_challenge_review.py packets
  python3 tools/arkitscenes_relation_challenge_review.py validate --returned KEY.json

Three commands, deliberately separate:

  sheet     one self-contained HTML page per scene plus an index, embedding raw
            RGB context and label-free mesh renders of every delivered region.
            The owner maps anchors to UIDs and answers the questions on it.
  packets   an answer-free multi-view RGB packet and prompt per scene for the
            blinded VLM layer, to be run later in a fresh context.
  validate  schema-check a returned key against the question manifest, without
            scoring anything.

NOTHING PREDICTED APPEARS ON THE OWNER'S PAGE
----------------------------------------------
This module never reads `display_label`, `semantic_hypotheses`, `node.label`,
any graph edge, or any expected answer. It reads geometry only: `bbox_aabb`,
`geometry_handle`, and the mesh. Tests AST-check that the forbidden names
appear nowhere in this file and that no graph manifest is opened.

Showing the owner "is obj_12 near obj_8?" *because the extractor linked them*
would launder a system guess into ground truth. The page therefore shows what
each region looks like and where it sits, and asks the owner to name it. UIDs
appear only as mapping handles, never as the question.

FOUR MAPPING OUTCOMES, ALL FIRST-CLASS
---------------------------------------
`uid`, `none / missing`, `overmerged into <uid>`, `ambiguous`. The previous
round used all four: in 41069025 the owner returned the rug as none/missing and
both the trash can and the counter as ambiguous. Collapsing any outcome would
discard information the owner actually has, and the geometry ceiling abstains
rather than guessing wherever identity is unresolved.

EVIDENCE VISIBILITY IS RECORDED, NOT MANUFACTURED
--------------------------------------------------
For every question the owner independently records whether the evidence is
visible in 0, 1, or 2+ RGB views. That produces a NATURAL thin-evidence slice
for the secondary sufficiency subtest. No one-view case is constructed to order.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
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
from extractors.arkitscenes_rgb_crops import RgbCropSource
from tools.arkitscenes_eval import load_canonical_geometry
from tools.arkitscenes_representation_kill_test import image_information_score
from tools import arkitscenes_uid_visual_sheet as uid_sheet
from tools.arkitscenes_uid_visual_sheet import SceneRenderer

# The shared renderer sizes its canvases from module constants at construction
# time. 260 px of whole-flat top-down with a small highlight is a puzzle, not a
# question -- the owner reported being unable to tell what they were looking
# at. Raised here, for this process only; the uid sheet runs in its own process
# and its output is unchanged.
uid_sheet.CONTEXT_PX = 520
uid_sheet.ISOLATED_PX = 520

N_RGB_CROPS = 3
# Renders are rasterised large, then downsampled for the page: sampling at 520
# and showing at 400 keeps the splat legible without paying 4x the bytes.
RENDER_PX = 400
MAX_PAGE_MB = 9.0

QUESTIONS_SCHEMA = "arkitscenes_relation_challenge_questions_v1"
KEY_SCHEMA = "arkitscenes_relation_challenge_key_v1"
PACKET_SCHEMA = "arkitscenes_relation_challenge_packet_v1"
RESPONSE_SCHEMA = "arkitscenes_relation_challenge_rgb_responses_v1"

DEFAULT_QUESTIONS = REPO_ROOT / "eval" / "questions" / "arkitscenes_relation_challenge_v1.json"
DEFAULT_DATA_ROOT = Path.home() / "Desktop/datasets/arkitscenes/Validation"
DEFAULT_OUT = REPO_ROOT / "runs" / "arkit_relation_challenge"

SCENE_DIRS = {
    "arkitscenes_41069025": "41069025",
    "arkitscenes_41069042": "41069042",
}
N_CONTEXT_FRAMES = 12
N_PACKET_FRAMES = 18
FRAME_PX = 384
EVIDENCE_CHOICES = (
    ("2+", "2 or more independent views"),
    ("1", "exactly 1 view"),
    ("0", "0 views (not visible in the capture)"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def scene_paths(scene_id: str, data_root: Path) -> dict:
    short = SCENE_DIRS[scene_id]
    return {
        "short": short,
        "scene_dir": data_root / short,
        "frames_dir": data_root / short / "lowres_wide",
        "mesh": data_root / short / f"{short}{MESH_SUFFIX}",
        "entities": (REPO_ROOT / "runs" / f"arkit_label_image_ab_{short}"
                     / "rgb_tight" / "entities" / "manifest.json"),
        "ids": (REPO_ROOT / "runs" / "arkitscenes_mask3d_transfer"
                / f"bundle_arkitscenes_{short}" / "vertex_instance_ids.npy"),
    }


def even_frames(frames_dir: Path, n: int) -> list[Path]:
    """Deterministic even spread across the capture; no scoring, no randomness."""
    files = sorted(frames_dir.glob("*.png"))
    if not files:
        raise ValueError(f"no frames under {frames_dir}")
    if len(files) <= n:
        return files
    step = len(files) / n
    return [files[min(len(files) - 1, int(i * step))] for i in range(n)]


def survey_frames(frames_dir: Path, n: int) -> list[Path]:
    """Equal temporal bins, best-information frame inside each bin.

    This is the SAME answer-free rule the previous kill test used, reused
    deliberately rather than re-invented. Naive even spacing hands the blinded
    RGB arm whatever frame happens to land on a bin boundary -- often a blank
    wall or a motion-blurred pan -- which would quietly disadvantage the one
    arm that can only see pixels. The score is class-free and answer-free: it
    rewards contrast and edge density and penalises clipping, and it cannot
    know what any question is about.
    """
    files = sorted(frames_dir.glob("*.png"))
    if not files:
        raise ValueError(f"no frames under {frames_dir}")
    if len(files) <= n:
        return files
    bounds = np.linspace(0, len(files), n + 1, dtype=int)
    chosen = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        # A wide bin is subsampled before scoring; scoring 500 frames per bin
        # costs minutes and changes nothing about which frame wins.
        candidates = list(range(int(start), max(int(start) + 1, int(end))))
        if len(candidates) > 24:
            step = len(candidates) / 24
            candidates = [candidates[int(i * step)] for i in range(24)]
        best = max(candidates,
                   key=lambda i: (image_information_score(files[i]), -i))
        chosen.append(best)
    return [files[i] for i in sorted(chosen)]


def frame_id(path: Path, rank: int) -> str:
    return f"frame_{rank:02d}_{path.stem.split('_')[-1]}"


def jpeg_data_uri(image: Image.Image, quality: int = 82) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def recode_png_uri(uri: str, px: int, quality: int = 80) -> str:
    """Re-encode a renderer PNG data URI as a smaller JPEG data URI.

    The shared renderer emits PNG. For stippled point-splat images PNG is a
    poor fit -- at 520 px, three renders per region across 35 regions pushed
    the 41069025 page to 13 MB, past what the viewer would open at all. The
    renders are secondary evidence behind a collapsed toggle, so they can pay
    the lossy cost; the capture photographs, which are what the owner actually
    identifies objects from, keep their size and quality.
    """
    raw = base64.b64decode(uri.split(",", 1)[1])
    image = Image.open(io.BytesIO(raw))
    image.thumbnail((px, px), Image.Resampling.LANCZOS)
    return jpeg_data_uri(image, quality=quality)


def context_frames(frames_dir: Path, n: int) -> list[dict]:
    rows = []
    for rank, path in enumerate(survey_frames(frames_dir, n)):
        image = Image.open(path)
        image.thumbnail((FRAME_PX, FRAME_PX), Image.Resampling.LANCZOS)
        rows.append({"id": frame_id(path, rank), "file": path.name,
                     "uri": jpeg_data_uri(image)})
    return rows


def delivered_regions(scene_id: str, paths: dict) -> list[dict]:
    """Geometry and real capture photographs. Never touches a label field.

    The photographs matter more than the renders. `extractors/
    arkitscenes_rgb_crops` exists because classifying an instance from
    texture-free point splats is the input pathology that produced this
    scene's label errors -- a sofa read as "projector", two cushions read as
    "rug". Asking a human to identify regions from those same splats repeats
    the mistake one level up, so each region leads with up to three real
    frames from the capture, cropped around where the instance actually
    appears and with the instance marked. The renders stay as secondary
    evidence for placement and extent.
    """
    mesh = read_mesh(paths["mesh"])
    instance_ids = np.load(paths["ids"])
    if len(instance_ids) != len(mesh.xyz):
        raise ValueError(f"{scene_id}: instance-id sidecar does not match the mesh")
    manifest = json.loads(paths["entities"].read_text())
    floor_z = min(e["bbox_aabb"][0][2] for e in manifest["entities"])
    renderer = SceneRenderer(mesh.xyz, mesh.rgb)

    # Same canonical geometry + rotation the labeler used, so a crop shows the
    # region the uid actually denotes rather than an approximately similar one.
    canon_mesh, rotation, _ = load_canonical_geometry(paths["scene_dir"])
    crops = RgbCropSource(paths["scene_dir"], canon_mesh.xyz, rotation,
                          stride=6, n_views=N_RGB_CROPS, mark_target=True)

    regions = []
    for entity in manifest["entities"]:
        uid = entity["identity"]["object_uid"]
        index = int(entity["geometry_handle"].rsplit("#", 1)[1])
        vertices = np.flatnonzero(instance_ids == index)
        (x0, y0, z0), (x1, y1, z1) = entity["bbox_aabb"]
        photos = []
        for image in crops.crops_for(vertices):
            image = image.copy()
            image.thumbnail((420, 420), Image.Resampling.LANCZOS)
            photos.append(jpeg_data_uri(image, quality=88))
        coverage = crops.coverage(vertices)
        regions.append({
            "uid": uid,
            "n_vertices": int(len(vertices)),
            "width_m": round(x1 - x0, 2), "depth_m": round(y1 - y0, 2),
            "height_m": round(z1 - z0, 2),
            "footprint_m2": round((x1 - x0) * (y1 - y0), 2),
            "underside_above_floor_m": round(z0 - floor_z, 2),
            "photos": photos,
            "best_visible_fraction": coverage.get("best_visible_fraction"),
            "ctx": [recode_png_uri(u, RENDER_PX) for u in renderer.context(vertices)],
            "iso": recode_png_uri(renderer.isolated(vertices), RENDER_PX),
        })
    regions.sort(key=lambda r: -r["n_vertices"])
    return regions


def anchors_for(scene_id: str, doc: dict) -> list[str]:
    """Every object name the scene's questions refer to, in stable order."""
    names: list[str] = []
    for question in doc["questions"]:
        if question["scene_id"] != scene_id:
            continue
        for field in ("subject", "object", "reference_a", "reference_b"):
            value = question.get(field)
            if value and value not in names:
                names.append(value)
        for value in question.get("candidate_objects", []):
            if value not in names:
                names.append(value)
    return names


def esc(value: object) -> str:
    import html
    return html.escape(str(value), quote=True)


def answer_control(question: dict) -> str:
    qid = question["id"]
    if question["form"] == "binary_near":
        options = [("yes", "yes — they are near"), ("no", "no — they are not near")]
    elif question["form"] == "comparative_near":
        options = [(question["reference_a"], f'closer to the {question["reference_a"]}'),
                   (question["reference_b"], f'closer to the {question["reference_b"]}'),
                   ("tie", "genuinely the same distance")]
    else:
        boxes = "".join(
            f'<label class="chk"><input type="checkbox" class="setitem" '
            f'value="{esc(name)}"> {esc(name)}</label>'
            for name in question["candidate_objects"])
        return f'<div class="setbox">{boxes}</div>'
    return "".join(
        f'<label class="chk"><input type="radio" name="{esc(qid)}" '
        f'class="pick" value="{esc(value)}"> {esc(label)}</label>'
        for value, label in options)


def question_card(question: dict) -> str:
    kind = {"binary_near": "binary NEAR", "comparative_near": "comparative (no threshold)",
            "near_set": "exhaustive set"}[question["form"]]
    cross = ('<span class="tag cross">cross-view: the objects may not appear '
             'together in any single frame</span>' if question.get("cross_view") else "")
    evidence = "".join(
        f'<label class="chk"><input type="radio" name="ev_{esc(question["id"])}" '
        f'class="ev" value="{esc(value)}"> {esc(label)}</label>'
        for value, label in EVIDENCE_CHOICES)
    return f"""
    <section class="qcard" data-qid="{esc(question['id'])}"
             data-form="{esc(question['form'])}">
      <h3>{esc(question['question'])}</h3>
      <div class="qmeta"><code>{esc(question['id'])}</code>
        <span class="tag">{esc(kind)}</span>{cross}</div>
      <div class="field"><b>Your answer</b>{answer_control(question)}
        <label class="chk amb"><input type="checkbox" class="ambiguous">
          ambiguous / cannot answer — exclude this item</label></div>
      <div class="field"><b>Evidence visibility</b>
        <span class="hint">judged independently of your answer: in how many
        separate RGB views can this be verified?</span>{evidence}</div>
      <div class="field"><b>Notes</b>
        <textarea class="notes" rows="2"
          placeholder="rationale, caveats, anything that made this hard"></textarea></div>
    </section>"""


def mapping_rows(anchors: list[str], regions: list[dict]) -> str:
    options = "".join(f'<option value="{r["uid"]}">{r["uid"]}</option>'
                      for r in regions)
    rows = []
    for name in anchors:
        rows.append(f"""
      <tr data-object="{esc(name)}">
        <td class="obj">{esc(name)}</td>
        <td><select class="pick">
          <option value="">— choose —</option>
          <option value="none_missing">none / missing (not delivered)</option>
          <option value="ambiguous">ambiguous (cannot tell)</option>
          {options}
        </select></td>
        <td><label class="om"><input type="checkbox" class="overmerged">
          overmerged into the chosen region</label></td>
      </tr>""")
    return "".join(rows)


def region_cards(regions: list[dict]) -> str:
    cards = []
    for r in regions:
        if r.get("photos"):
            # The best view gets the full card width. Three equal thumbnails in
            # one row is ~140 px each, which is the same illegibility the splat
            # renders had; the alternates are for disambiguation, not for
            # first identification.
            best, *rest = r["photos"]
            alternates = "".join(
                f'<div><img src="{uri}" alt="capture photo {i + 2}">'
                f'<span>view {i + 2}</span></div>' for i, uri in enumerate(rest))
            visible = r.get("best_visible_fraction")
            caveat = ("" if visible is None else
                      f'<span class="dim">best view has {visible:.0%} of the '
                      f'region unoccluded</span>')
            photo_block = (
                f'<div class="hero"><img src="{best}" alt="capture photo 1">'
                f'<span>best capture view — the region is outlined</span></div>'
                + (f'<div class="photos">{alternates}</div>' if alternates else "")
                + caveat)
        else:
            photo_block = ('<p class="dim nophoto">No usable capture photo: '
                           'this region is never sufficiently visible in a '
                           'posed frame. Judge it from the renders below, or '
                           'mark it ambiguous.</p>')
        cards.append(f"""
    <figure class="card" id="{r['uid']}">
      <figcaption><b>{r['uid']}</b>
        <span class="dim">{r['n_vertices']} verts ·
        {r['width_m']}×{r['depth_m']}×{r['height_m']} m ·
        {r['footprint_m2']} m² · underside {r['underside_above_floor_m']} m</span>
      </figcaption>
      {photo_block}
      <details>
        <summary>3D renders — where it sits and what it is on its own</summary>
        <div class="imgs">
          <div><img src="{r['ctx'][0]}" alt="context A"><span>context A</span></div>
          <div><img src="{r['ctx'][1]}" alt="context B"><span>context B</span></div>
          <div><img src="{r['iso']}" alt="isolated"><span>isolated</span></div>
        </div>
      </details>
    </figure>""")
    return "".join(cards)


STYLE = """
:root { --bg:#fff; --fg:#15171a; --mut:#666; --line:#e3e5e8; --hi:#c0158f;
  --warn:#8a5a00; --warnbg:#fff6e0; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#14161a; --fg:#e9ebee; --mut:#9aa1aa; --line:#2b2f36; --hi:#ff5cc8;
  --warn:#f0c060; --warnbg:#2a2214; } }
:root[data-theme="dark"] { --bg:#14161a; --fg:#e9ebee; --mut:#9aa1aa;
  --line:#2b2f36; --hi:#ff5cc8; --warn:#f0c060; --warnbg:#2a2214; }
* { box-sizing:border-box; }
body { background:var(--bg); color:var(--fg); margin:0 auto; padding:24px 20px 80px;
  max-width:1040px; font:15px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif; }
h1 { font-size:21px; margin:0 0 4px; }
h2 { font-size:16px; margin:34px 0 10px; padding-top:14px;
  border-top:1px solid var(--line); }
h3 { font-size:15px; margin:0 0 6px; }
.sub { color:var(--mut); font-size:13px; margin:0 0 16px; }
code { font:12px ui-monospace,SFMono-Regular,Menlo,monospace;
  background:rgba(128,128,128,.13); padding:.1em .35em; border-radius:3px; }
table { border-collapse:collapse; width:100%; font-size:14px; }
th,td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line);
  vertical-align:middle; }
th { color:var(--mut); font-weight:600; }
td.obj { font-weight:600; }
select,textarea { font:inherit; padding:5px; background:var(--bg); color:var(--fg);
  border:1px solid var(--line); border-radius:5px; }
select { max-width:280px; }
textarea { width:100%; }
.om { color:var(--mut); font-size:13px; }
.note { border-left:3px solid var(--hi); padding:6px 12px; color:var(--mut);
  font-size:13.5px; margin:12px 0; }
.warn { border-left:3px solid var(--warn); background:var(--warnbg);
  padding:10px 14px; font-size:13.5px; margin:14px 0; border-radius:0 6px 6px 0; }
.convention { border:1px solid var(--line); border-radius:8px; padding:12px 16px;
  margin:14px 0; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(430px,1fr));
  gap:16px; }
.hero { margin-bottom:6px; text-align:center; }
.hero img { width:100%; image-rendering:auto; border:1px solid var(--line);
  border-radius:6px; display:block; }
.hero span { font-size:11px; color:var(--mut); }
.photos { display:flex; gap:6px; margin-bottom:6px; }
.photos div { flex:1; text-align:center; }
.photos img { width:100%; border:1px solid var(--line); border-radius:5px;
  display:block; }
.photos span { font-size:10.5px; color:var(--mut); }
.nophoto { border-left:3px solid var(--warn); padding:6px 10px; margin:0 0 6px; }
details { margin-top:4px; }
summary { cursor:pointer; font-size:12.5px; color:var(--mut); }
.card { margin:0; border:1px solid var(--line); border-radius:8px; padding:8px; }
.card figcaption { font-size:13px; margin-bottom:6px; }
.dim { color:var(--mut); display:block; font-size:11.5px; }
.imgs { display:flex; gap:5px; }
.imgs div { flex:1; text-align:center; }
.imgs img { width:100%; border:1px solid var(--line); border-radius:4px; display:block; }
.imgs span { font-size:10.5px; color:var(--mut); }
.frames { display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr));
  gap:8px; }
.frames figure { margin:0; }
.frames img { width:100%; border:1px solid var(--line); border-radius:5px; display:block; }
.frames figcaption { font-size:10.5px; color:var(--mut); }
.qcard { border:1px solid var(--line); border-radius:8px; padding:14px 16px;
  margin:0 0 12px; }
.qmeta { color:var(--mut); font-size:12px; margin-bottom:10px;
  display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
.tag { background:rgba(128,128,128,.14); padding:1px 7px; border-radius:10px; }
.tag.cross { background:rgba(192,21,143,.14); color:var(--hi); }
.field { margin:10px 0; }
.field > b { display:block; font-size:12.5px; margin-bottom:4px; }
.hint { display:block; color:var(--mut); font-size:12px; margin-bottom:5px; }
.chk { display:inline-flex; align-items:center; gap:5px; margin:0 14px 5px 0;
  font-size:13.5px; }
.chk.amb { color:var(--warn); display:flex; margin-top:8px; }
.setbox { display:flex; flex-direction:column; gap:2px; }
button { font:inherit; padding:8px 16px; border-radius:6px; cursor:pointer;
  border:1px solid var(--line); background:var(--hi); color:#fff; }
pre { background:rgba(128,128,128,.10); padding:12px; border-radius:6px;
  overflow-x:auto; font-size:12px; max-height:420px; }
a { color:var(--hi); }
"""


def build_scene_page(scene_id: str, doc: dict, regions: list[dict],
                     frames: list[dict], provenance: dict) -> str:
    scene_meta = doc["scenes"][scene_id]
    questions = [q for q in doc["questions"] if q["scene_id"] == scene_id]
    anchors = anchors_for(scene_id, doc)
    convention = doc["near_convention"]

    discrepancy = ""
    if scene_meta.get("owner_review_discrepancy"):
        discrepancy = (f'<div class="warn"><b>Needs your adjudication.</b> '
                       f'{esc(scene_meta["owner_review_discrepancy"])}</div>')

    frame_figs = "".join(
        f'<figure><img src="{f["uri"]}" alt="{esc(f["id"])}">'
        f'<figcaption>{esc(f["id"])}</figcaption></figure>' for f in frames)

    return f"""<!DOCTYPE html>
<meta charset="utf-8">
<title>Relation review — {esc(scene_id)}</title>
<style>{STYLE}</style>

<h1>NEAR relation review — {esc(scene_id)}</h1>
<p class="sub">{len(questions)} questions · {len(anchors)} objects to map ·
{len(regions)} delivered regions · <a href="index.html">both scenes</a></p>

<p class="note"><b>No predicted class label, graph answer, edge, confidence or
expected result appears anywhere on this page.</b> Regions are ordered by size,
never by any semantic score. Pink is the region; grey is the rest of the
captured mesh from above with the ceiling clipped. The pink is drawn
<em>over</em> the scene rather than hidden behind it, so a region behind a wall
still shows — use the two azimuths to judge depth.</p>

{discrepancy}

<div class="convention">
  <b>The NEAR convention for this review</b>
  <p style="margin:6px 0">{esc(convention['statement'])}</p>
  <p class="sub" style="margin:0">Applies to the yes/no and set questions.
  {esc(convention['not_applied_to'])}.</p>
</div>

<h2>1 · Raw capture context</h2>
<p class="sub">{len(frames)} frames spread evenly across the capture, for
orientation. Your answers may use the whole capture, not only these.</p>
<div class="frames">{frame_figs}</div>

<h2>2 · Which delivered region is which object?</h2>
<p class="sub">Scroll to section 4 to see every region rendered. All four
outcomes are real answers — <i>none / missing</i> and <i>ambiguous</i> are as
informative as a uid, and the scorer abstains rather than guessing where
identity is unresolved.</p>
<table>
  <tr><th>object</th><th>choose one</th><th>or mark</th></tr>
  {mapping_rows(anchors, regions)}
</table>

<h2>3 · The questions</h2>
<p class="sub">Answer from the room as you know it. Record evidence visibility
independently of the answer — it is a separate judgement about the capture, and
it is what makes the sufficiency subtest honest rather than manufactured.</p>
{"".join(question_card(q) for q in questions)}

<p><button onclick="emit()">Build JSON for this scene</button></p>
<pre id="out">(fill in above, then press Build JSON)</pre>

<h2>4 · Delivered regions</h2>
<div class="grid">{region_cards(regions)}</div>

<script>
function emit() {{
  const uid_mappings = [];
  document.querySelectorAll('tr[data-object]').forEach(tr => {{
    const pick = tr.querySelector('.pick').value;
    const over = tr.querySelector('.overmerged').checked;
    const row = {{object: tr.dataset.object, uid: null, none_missing: null,
                 overmerged_into: null, ambiguous: null}};
    if (pick === 'none_missing') row.none_missing = true;
    else if (pick === 'ambiguous') row.ambiguous = true;
    else if (pick && over) row.overmerged_into = pick;
    else if (pick) row.uid = pick;
    uid_mappings.push(row);
  }});
  const truth = [];
  document.querySelectorAll('.qcard').forEach(card => {{
    const form = card.dataset.form;
    let answer = null;
    if (form === 'near_set') {{
      answer = Array.from(card.querySelectorAll('.setitem:checked')).map(i => i.value);
    }} else {{
      const hit = card.querySelector('.pick:checked');
      if (hit) answer = (form === 'binary_near')
        ? (hit.value === 'yes') : hit.value;
    }}
    const ev = card.querySelector('.ev:checked');
    const ambiguous = card.querySelector('.ambiguous').checked;
    truth.push({{
      id: card.dataset.qid,
      answer: ambiguous ? null : answer,
      ambiguous: ambiguous,
      evidence_views: ev ? ev.value : null,
      notes: card.querySelector('.notes').value || null
    }});
  }});
  const out = {{
    schema: '{KEY_SCHEMA}',
    scene_id: '{scene_id}',
    status: 'OWNER_CONFIRMED',
    questions_sha256: '{provenance["questions_sha256"]}',
    uid_mappings: uid_mappings,
    human_relation_truth: truth
  }};
  document.getElementById('out').textContent = JSON.stringify(out, null, 1);
}}
</script>
"""


def build_index(doc: dict, pages: dict, provenance: dict) -> str:
    rows = "".join(
        f'<tr><td><a href="{esc(name)}">{esc(scene_id)}</a></td>'
        f'<td>{sum(1 for q in doc["questions"] if q["scene_id"] == scene_id)}</td>'
        f'<td>{len(anchors_for(scene_id, doc))}</td>'
        f'<td>{esc(doc["scenes"][scene_id]["description"])}</td></tr>'
        for scene_id, name in pages.items())
    excluded = "".join(
        f'<li><b>{esc(x["class"])}</b> — {esc(x["why"])}</li>'
        for x in doc["excluded_question_classes"])
    todo = "".join(f"<li>{esc(item)}</li>" for item in doc["owner_review_required"])
    return f"""<!DOCTYPE html>
<meta charset="utf-8">
<title>NEAR relation challenge — review kit</title>
<style>{STYLE}</style>

<h1>NEAR relation challenge — owner review kit</h1>
<p class="sub">status <code>{esc(doc['status'])}</code> ·
{len(doc['questions'])} questions across {len(pages)} scenes ·
question manifest <code>{provenance['questions_sha256'][:16]}…</code></p>

<p class="note">Nothing here has been scored. The four evaluation layers are
built and synthetically tested, but no real answer exists until this review is
returned. The pages below expose no predicted label, no graph edge and no
expected answer.</p>

<h2>Scenes</h2>
<table>
  <tr><th>scene</th><th>questions</th><th>objects to map</th><th>what it is</th></tr>
  {rows}
</table>

<h2>What you need to record</h2>
<ol>{todo}</ol>

<h2>Why the relation is NEAR</h2>
<p>{esc(doc['purpose'])}</p>
<div class="convention">
  <b>Convention</b>
  <p style="margin:6px 0">{esc(doc['near_convention']['statement'])}</p>
  <p class="sub" style="margin:0"><b>Declared confound.</b>
  {esc(doc['near_convention']['declared_confound'])}</p>
</div>

<h2>Question classes deliberately excluded</h2>
<ul>{excluded}</ul>

<h2>How questions were chosen</h2>
<p class="sub">{esc(doc['selection_method']['not_consulted'])}</p>
<p class="sub">{esc(doc['selection_method']['anchor_rule'])}</p>

<p class="note">{esc(doc['interpretation_limit'])}</p>
"""


# --------------------------------------------------------------------------
# blinded RGB packet -- answer-free, no uid, no label, no expected value
# --------------------------------------------------------------------------
def prompt_text(packet: dict) -> str:
    lines = []
    for question in packet["questions"]:
        if question["form"] == "binary_near":
            expect = "true or false"
        elif question["form"] == "comparative_near":
            expect = (f'exactly one of "{question["reference_a"]}" or '
                      f'"{question["reference_b"]}"')
        else:
            expect = ("a JSON array containing any of: "
                      + ", ".join(f'"{c}"' for c in question["candidate_objects"]))
        lines.append(f'- {question["id"]} [{expect}]: {question["question"]}')
    frames = ", ".join(f["id"] for f in packet["frames"])
    return f"""You are evaluating a captured indoor scene from multiple RGB views.
Use ONLY visible evidence in the supplied frames. Do not fill gaps with common
room expectations. If an answer cannot be verified, return outcome "unknown"
instead of guessing.

NEAR means: the nearest points of the two objects' surfaces are within about one
metre of each other. Judge the gap between the objects themselves, not between
their centres. Comparative questions ask only which is closer and need no
threshold.

Some questions ask about two objects that may never appear together in a single
frame. Answer those from the room's layout as a whole if you can, or return
"unknown".

Valid evidence frame ids: {frames}

Questions:
{chr(10).join(lines)}

Return one JSON object and no prose. Note that "outcome" is a flag with only
two legal values, "answer" or "unknown"; the answer itself goes in "answer".
{{
  "schema": "{RESPONSE_SCHEMA}",
  "scene_id": "{packet['scene_id']}",
  "packet_sha256": {{"{packet['scene_id']}": "{packet['packet_sha256']}"}},
  "model": {{"provider": "...", "name": "...", "version": "..."}},
  "answers": [
    {{
      "id": "question id",
      "outcome": "the literal string \"answer\", or the literal string \"unknown\" -- not the answer itself",
      "answer": "boolean, object-name string, or array of object names; null if unknown",
      "confidence": 0.0,
      "evidence_frame_ids": ["at least two valid frame ids when answering"]
    }}
  ]
}}
"""


def build_packet(scene_id: str, doc: dict, paths: dict, out_dir: Path) -> dict:
    """Answer-free. Carries question text and frames; no uid, no expected value."""
    questions = []
    for question in doc["questions"]:
        if question["scene_id"] != scene_id:
            continue
        row = {"id": question["id"], "form": question["form"],
               "question": question["question"]}
        for field in ("subject", "object", "reference_a", "reference_b",
                      "candidate_objects"):
            if field in question:
                row[field] = question[field]
        questions.append(row)

    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    rows, thumbs = [], []
    for rank, src in enumerate(survey_frames(paths["frames_dir"], N_PACKET_FRAMES)):
        fid = frame_id(src, rank)
        dst = frame_dir / f"{fid}.png"
        dst.write_bytes(src.read_bytes())
        rows.append({"id": fid, "file": str(dst.relative_to(out_dir)),
                     "sha256": sha256(dst)})
        thumbs.append((fid, dst))

    packet = {
        "schema": PACKET_SCHEMA,
        "stage": "prepared_no_answers",
        "scene_id": scene_id,
        "relation_under_test": doc["relation_under_test"],
        "near_convention": doc["near_convention"]["statement"],
        "questions_sha256": json_sha256(questions),
        "questions": questions,
        "frames": rows,
        "selection": {"method": "even_temporal_spread_v1",
                      "n_selected": len(rows),
                      "source_frames_dir": str(paths["frames_dir"])},
        "disclosures": [
            "No expected answer, human key, uid or predicted label is included.",
            "The selected views may miss an object even when it exists.",
            "This arm sees RGB only; it receives no mesh and no graph state.",
            "Some question pairs are not co-visible in any single supplied frame.",
        ],
    }
    packet["packet_sha256"] = json_sha256(packet)
    (out_dir / "packet.json").write_text(
        json.dumps(packet, indent=1, sort_keys=True) + "\n")
    (out_dir / "prompt.txt").write_text(prompt_text(packet))

    cols = 3
    cell, label_h = 400, 24
    images = []
    for fid, path in thumbs:
        image = Image.open(path).convert("RGB")
        image.thumbnail((384, 288), Image.Resampling.LANCZOS)
        images.append((fid, image))
    rows_n = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell, rows_n * (288 + label_h)), "white")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)
    for i, (fid, image) in enumerate(images):
        x, y = (i % cols) * cell, (i // cols) * (288 + label_h)
        canvas.paste(image, (x + (cell - image.width) // 2, y + label_h))
        draw.text((x + 8, y + 5), fid, fill="black")
    canvas.save(out_dir / "contact_sheet.jpg", quality=88, optimize=True)
    return packet


# --------------------------------------------------------------------------
# validation of a returned key -- schema only, never a score
# --------------------------------------------------------------------------
def validate_returned(doc: dict, questions_path: Path, returned: dict) -> list[str]:
    """Structural check. Deliberately reports problems instead of raising once,
    so the owner sees every issue in one pass rather than one per round trip."""
    problems = []
    if returned.get("schema") != KEY_SCHEMA:
        problems.append(f"schema must be {KEY_SCHEMA!r}, got {returned.get('schema')!r}")
    if returned.get("status") != "OWNER_CONFIRMED":
        problems.append("status must be OWNER_CONFIRMED")
    # A sheet export pins the file it was generated from. Confirming the
    # manifest edits that file, so the hash the owner reviewed is also
    # accepted, recorded in the manifest as reviewed_as_sha256.
    accepted = {sha256(questions_path), doc.get("reviewed_as_sha256"),
                json_sha256(doc["questions"])}
    if returned.get("questions_sha256") not in accepted:
        problems.append("questions_sha256 pins neither the current manifest "
                        "nor the revision the owner reviewed")

    scene_id = returned.get("scene_id")
    if scene_id not in doc["scenes"]:
        problems.append(f"unknown scene_id {scene_id!r}")
        return problems

    asked = [q for q in doc["questions"] if q["scene_id"] == scene_id]
    by_id = {q["id"]: q for q in asked}
    truth = returned.get("human_relation_truth") or []
    got = [t.get("id") for t in truth]
    if sorted(got) != sorted(by_id):
        problems.append(f"expected answers for {sorted(by_id)}, got {sorted(got)}")

    for item in truth:
        qid = item.get("id")
        question = by_id.get(qid)
        if question is None:
            continue
        views = item.get("evidence_views")
        if item.get("ambiguous"):
            # An excluded item is not in any tally and contributes nothing to
            # the thin-evidence slice, so a visibility judgement is optional
            # here. Demanding one would force the owner to invent a number for
            # a question they have just said they cannot answer.
            if item.get("answer") is not None:
                problems.append(f"{qid}: ambiguous items must carry a null answer")
            if views is not None and views not in {c[0] for c in EVIDENCE_CHOICES}:
                problems.append(f"{qid}: evidence_views must be 0 / 1 / 2+ or null")
            continue
        if views not in {c[0] for c in EVIDENCE_CHOICES}:
            problems.append(f"{qid}: evidence_views must be one of 0 / 1 / 2+")
        answer = item.get("answer")
        if answer is None:
            problems.append(f"{qid}: needs an answer or ambiguous=true")
        elif question["form"] == "binary_near" and not isinstance(answer, bool):
            problems.append(f"{qid}: binary_near answer must be true or false")
        elif question["form"] == "comparative_near" and answer not in (
                question["reference_a"], question["reference_b"], "tie"):
            problems.append(f"{qid}: comparative answer must name one reference or 'tie'")
        elif question["form"] == "near_set":
            if not isinstance(answer, list):
                problems.append(f"{qid}: near_set answer must be a list")
            elif set(answer) - set(question["candidate_objects"]):
                problems.append(f"{qid}: set answer contains non-candidates "
                                f"{sorted(set(answer) - set(question['candidate_objects']))}")

    mapped = {m.get("object") for m in returned.get("uid_mappings") or []}
    missing = [a for a in anchors_for(scene_id, doc) if a not in mapped]
    if missing:
        problems.append(f"uid_mappings missing objects: {missing}")
    return problems


# --------------------------------------------------------------------------
def cmd_sheet(args) -> int:
    doc = json.loads(args.questions.read_text())
    provenance = {"questions_sha256": sha256(args.questions)}
    args.out.mkdir(parents=True, exist_ok=True)
    pages = {}
    for scene_id in doc["scenes"]:
        paths = scene_paths(scene_id, args.data_root)
        regions = delivered_regions(scene_id, paths)
        frames = context_frames(paths["frames_dir"], N_CONTEXT_FRAMES)
        page = build_scene_page(scene_id, doc, regions, frames, provenance)
        name = f"{scene_id}_relation_review.html"
        (args.out / name).write_text(page)
        pages[scene_id] = name
        size = (args.out / name).stat().st_size / 1e6
        flag = "  ** OVER LIMIT **" if size > MAX_PAGE_MB else ""
        print(f"    {scene_id}: {len(regions)} regions, {len(frames)} frames, "
              f"{size:.1f} MB -> {name}{flag}")
        if size > MAX_PAGE_MB:
            print(f"      a page over ~{MAX_PAGE_MB:.0f} MB may not open in a "
                  f"viewer at all; reduce RENDER_PX or N_RGB_CROPS")
    (args.out / "index.html").write_text(build_index(doc, pages, provenance))
    print(f"    index -> {args.out / 'index.html'}")
    return 0


def cmd_packets(args) -> int:
    doc = json.loads(args.questions.read_text())
    for scene_id in doc["scenes"]:
        paths = scene_paths(scene_id, args.data_root)
        out_dir = args.out / "blinded_rgb" / SCENE_DIRS[scene_id]
        out_dir.mkdir(parents=True, exist_ok=True)
        packet = build_packet(scene_id, doc, paths, out_dir)
        print(f"    {scene_id}: {len(packet['frames'])} frames, "
              f"{len(packet['questions'])} questions -> {out_dir}")
        print(f"      packet_sha256 {packet['packet_sha256']}")
    return 0


def merge_returned(doc: dict, questions_path: Path,
                   returned: list[dict]) -> dict:
    """Combine the per-scene sheets into the one key the scorer consumes.

    Each review page emits its own scene's block, because one 58-region page
    would not open. The scorer wants a single key spanning both scenes, so the
    join happens here -- once, validated, and never by hand.

    Only mappings the owner resolved to a uid are carried into the scored key.
    `none / missing` and `ambiguous` are deliberately NOT converted into a
    guess: they are dropped from the mapping table so the ceiling abstains on
    anything that needs them, and they are preserved verbatim under
    `unresolved_mappings` so the reason stays visible in the report.
    """
    problems = []
    for block in returned:
        problems += [f"{block.get('scene_id')}: {p}"
                     for p in validate_returned(doc, questions_path, block)]
    if problems:
        raise ValueError("cannot merge invalid returns:\n  " + "\n  ".join(problems))

    scenes = [b["scene_id"] for b in returned]
    if sorted(scenes) != sorted(doc["scenes"]):
        raise ValueError(f"expected one return per scene {sorted(doc['scenes'])}, "
                         f"got {sorted(scenes)}")

    uid_mappings, unresolved, truth = {}, {}, []
    for block in returned:
        scene_id = block["scene_id"]
        resolved, missing = {}, []
        for row in block["uid_mappings"]:
            if row.get("uid") and not row.get("ambiguous") and not row.get("none_missing"):
                resolved[row["object"]] = row["uid"]
            else:
                missing.append({
                    "object": row["object"],
                    "outcome": ("none_missing" if row.get("none_missing")
                                else "ambiguous" if row.get("ambiguous")
                                else "overmerged_into" if row.get("overmerged_into")
                                else "unset"),
                    "overmerged_into": row.get("overmerged_into"),
                })
        uid_mappings[scene_id] = resolved
        unresolved[scene_id] = missing
        truth += block["human_relation_truth"]

    return {
        "schema": KEY_SCHEMA,
        "status": "OWNER_CONFIRMED",
        "questions_sha256": sha256(questions_path),
        "questions_content_sha256": json_sha256(doc["questions"]),
        "reviewed_as_sha256": doc.get("reviewed_as_sha256"),
        "scene_ids": sorted(scenes),
        "merged_from": [f"{b['scene_id']} sheet export" for b in returned],
        "uid_mappings": uid_mappings,
        "unresolved_mappings": unresolved,
        "human_relation_truth": truth,
    }


def cmd_merge(args) -> int:
    doc = json.loads(args.questions.read_text())
    returned = [json.loads(p.read_text()) for p in args.returned]
    merged = merge_returned(doc, args.questions, returned)
    args.merged_out.parent.mkdir(parents=True, exist_ok=True)
    args.merged_out.write_text(json.dumps(merged, indent=1, sort_keys=True) + "\n")
    n_unresolved = sum(len(v) for v in merged["unresolved_mappings"].values())
    print(f"merged {len(returned)} scene returns -> {args.merged_out}")
    print(f"    questions answered : {len(merged['human_relation_truth'])}")
    print(f"    uid mappings kept  : "
          f"{sum(len(v) for v in merged['uid_mappings'].values())}")
    print(f"    unresolved, dropped: {n_unresolved} "
          f"(the ceiling abstains rather than guessing on these)")
    print("    nothing was scored")
    return 0


def cmd_validate(args) -> int:
    doc = json.loads(args.questions.read_text())
    returned = json.loads(args.returned.read_text())
    problems = validate_returned(doc, args.questions, returned)
    if problems:
        print(f"INVALID ({len(problems)} problems)")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"VALID: {args.returned} conforms to {KEY_SCHEMA} "
          f"and pins the current question manifest")
    print("       nothing was scored; run the scorer separately")
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("sheet")
    sub.add_parser("packets")
    v = sub.add_parser("validate")
    v.add_argument("--returned", type=Path, required=True)
    m = sub.add_parser("merge")
    m.add_argument("--returned", type=Path, nargs="+", required=True)
    m.add_argument("--merged-out", type=Path, required=True)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return {"sheet": cmd_sheet, "packets": cmd_packets,
            "validate": cmd_validate, "merge": cmd_merge}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
