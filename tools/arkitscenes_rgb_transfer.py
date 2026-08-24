"""Direct-RGB transfer test on one untouched scene.

Protocol: docs/arkitscenes_rgb_transfer_test.md, frozen before the scene was
downloaded or inspected.

  anchors   merge the three independent blind enumeration passes
  questions apply the fixed template allocation to the ordered anchors
  packet    answer-free RGB packet + blinded prompt
  sheet     one self-contained owner review sheet

This tool runs no segmenter, no labeler, no graph stage and no model. It reads
RGB frames and the enumeration passes, and it writes question and review
artifacts. No existing result, key, packet, report or demo artifact is touched.
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

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

QUESTIONS_SCHEMA = "arkitscenes_rgb_transfer_questions_v1"
KEY_SCHEMA = "arkitscenes_rgb_transfer_key_v1"
PACKET_SCHEMA = "arkitscenes_rgb_transfer_packet_v1"
RESPONSE_SCHEMA = "arkitscenes_rgb_transfer_responses_v1"

MIN_PASSES = 2          # an object is an anchor only on 2-of-3 agreement
MIN_ANCHORS = 6         # below this the protocol says the test does not run
MIN_SCORED = 6          # amendment 1: below this the RUN is void, not scored


def has_unique_referent(anchor: dict) -> bool:
    """All three passes agreed there is exactly one of it.

    Amendment 2. A comparative or near question names its objects with a
    singular definite description -- "the cushion" -- which picks out nothing
    when the passes counted four cushions. That is ill-posed in the same way
    "in this room" was ill-posed on a multi-room floor, and it is detectable
    without any model output. Presence and cardinality slots are unaffected:
    "how many framed pictures" is a fine question precisely because there are
    several.
    """
    counts = [c["count"] for c in anchor["counts"]]
    return bool(counts) and len(set(counts)) == 1 and counts[0] == 1

SCOPE = "the captured space"
SCOPE_DEFINITION = (
    "\"The captured space\" means everywhere visible anywhere in the supplied "
    "views. This capture covers a multi-room floor, not a single room, so "
    "questions are scoped to the whole captured space rather than to \"this "
    "room\", which would not pick out a definite area.")
COUNTING_CONVENTION = (
    "An instance is a physically separate object of the named class. Two "
    "objects of the same class in different parts of the captured space count "
    "as two even if they never appear in one frame together. " + SCOPE_DEFINITION)
NEAR_CONVENTION = (
    "Two objects are NEAR when the nearest points of their surfaces are "
    "within about one metre of each other. Judge the gap between the objects "
    "themselves, not between their centres. Comparative questions ask only "
    "which is closer and need no threshold.")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


# --------------------------------------------------------------------------
# anchor merging -- matching rule fixed BEFORE the passes returned
# --------------------------------------------------------------------------
def normalize(name: str) -> str:
    """Lowercase, de-punctuate, collapse whitespace, drop a plural 's'."""
    text = name.strip().lower().replace("_", " ").replace("-", " ")
    words = []
    for word in text.split():
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        words.append(word)
    return " ".join(words)


def names_match(a: str, b: str) -> bool:
    """Equal, or one is a more specific form of the other with the same head.

    Object-agnostic on purpose. "kitchen counter" and "counter" share the head
    noun and one word set contains the other, so they are one object; "coffee
    table" and "dining table" share the head but neither contains the other, so
    they stay two. No per-object synonym list is written anywhere, because
    writing one after seeing the passes is where an author's thumb goes on the
    scale.
    """
    if a == b:
        return True
    wa, wb = a.split(), b.split()
    if not wa or not wb or wa[-1] != wb[-1]:
        return False
    return set(wa) <= set(wb) or set(wb) <= set(wa)


def merge_passes(passes: list[dict], frame_order: list[str]) -> dict:
    """2-of-3 agreement, then mechanical ordering by first appearance."""
    index = {fid: i for i, fid in enumerate(frame_order)}
    groups: list[dict] = []
    for pass_i, block in enumerate(passes):
        for entry in block.get("objects", []):
            norm = normalize(entry["name"])
            hit = next((g for g in groups if names_match(g["norm"], norm)), None)
            if hit is None:
                hit = {"norm": norm, "variants": [], "passes": set(),
                       "frames": set(), "counts": []}
                groups.append(hit)
            # keep the most specific surface form seen for this object
            if len(norm) > len(hit["norm"]):
                hit["norm"] = norm
            hit["variants"].append(entry["name"])
            hit["passes"].add(pass_i)
            hit["frames"].update(f for f in entry.get("frame_ids", [])
                                 if f in index)
            hit["counts"].append({"pass": pass_i, "count": entry.get("count"),
                                  "confidence": entry.get("count_confidence")})

    admitted, rejected = [], []
    for group in groups:
        row = {
            "anchor": group["norm"],
            "surface_forms": sorted(set(group["variants"])),
            "n_passes": len(group["passes"]),
            "passes": sorted(group["passes"]),
            "frame_ids": sorted(group["frames"], key=lambda f: index[f]),
            "first_frame_rank": (min(index[f] for f in group["frames"])
                                 if group["frames"] else len(frame_order)),
            "counts": group["counts"],
            "counts_agree": len({c["count"] for c in group["counts"]}) == 1,
        }
        (admitted if len(group["passes"]) >= MIN_PASSES else rejected).append(row)

    # Mechanical: best-attested first. Amendment 1 replaced plain
    # first-appearance ordering, which put a small wall print seen in three
    # frames by two passes at rank 0, where it carried three of eight
    # questions. Ordering by agreement then observation count stops a
    # briefly-glimpsed object dominating, and still has nothing to do with
    # whether any system can answer.
    admitted.sort(key=lambda r: (-r["n_passes"], -len(r["frame_ids"]),
                                 r["first_frame_rank"], r["anchor"]))
    rejected.sort(key=lambda r: r["anchor"])
    return {"admitted": admitted, "rejected_single_pass": rejected,
            "min_passes_required": MIN_PASSES}


def non_covisible_pairs(anchors: list[dict]) -> list[tuple[int, int]]:
    """Anchor index pairs sharing no frame, by combined rank ascending.

    Frame sets are the UNION across agreeing passes, so a pair counts as
    non-co-visible only when NO pass saw them together -- the conservative
    direction for a cross-view claim.
    """
    pairs = []
    for i in range(len(anchors)):
        for j in range(i + 1, len(anchors)):
            if not (set(anchors[i]["frame_ids"]) & set(anchors[j]["frame_ids"])):
                pairs.append((i + j, i, j))
    pairs.sort()
    return [(i, j) for _, i, j in pairs]


# --------------------------------------------------------------------------
# fixed template allocation
# --------------------------------------------------------------------------
def build_questions(anchors: list[dict]) -> tuple[list[dict], list[str]]:
    """3 presence/cardinality, 4 comparative, 3 cross-view. No discretion.

    Ten rather than eight since amendment 1, for headroom against exclusions.
    Cross-view gains the extra slot because it lost both of its items in run 1
    and it is the category this work turns on.
    """
    notes: list[str] = []
    if len(anchors) < MIN_ANCHORS:
        raise ValueError(f"only {len(anchors)} anchors survived 2-of-3 "
                         f"agreement; the protocol requires {MIN_ANCHORS}")
    questions = []

    for rank in range(3):
        a = anchors[rank]
        certain = all(c["confidence"] == "certain" for c in a["counts"])
        if certain and a["counts_agree"]:
            questions.append({
                "id": f"t_card_{rank + 1}", "form": "cardinality",
                "answer_type": "integer",
                "question": (f"How many {a['anchor']}s are in "
                             f"{SCOPE}?"),
                "subject": a["anchor"], "template_rank": rank + 1})
        else:
            # The passes disagreed or hedged on the count, so asking for a
            # number would key an ambiguity rather than a fact.
            questions.append({
                "id": f"t_pres_{rank + 1}", "form": "presence",
                "answer_type": "boolean",
                "question": f"Is there a {a['anchor']} in {SCOPE}?",
                "subject": a["anchor"], "template_rank": rank + 1})
            notes.append(f"{a['anchor']}: passes did not agree on a count, so "
                         f"the slot became a presence question")

    # Amendment 2: relational slots draw only from unique-referent anchors.
    singular = [a for a in anchors if has_unique_referent(a)]
    if len(singular) < 6:
        raise ValueError(f"only {len(singular)} unique-referent anchors; "
                         "comparative slots need six")
    pool = singular[:6]
    triples = [(0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5)]
    for n, (s, x, y) in enumerate(triples, start=1):
        if max(s, x, y) >= len(pool):
            notes.append(f"comparative slot {n} dropped: fewer than "
                         f"{max(s, x, y) + 1} anchors available")
            continue
        questions.append({
            "id": f"t_cmp_{n}", "form": "comparative_near",
            "answer_type": "object_name",
            "question": (f"Is the {pool[s]['anchor']} closer to the "
                         f"{pool[x]['anchor']}, or closer to the "
                         f"{pool[y]['anchor']}?"),
            "subject": pool[s]["anchor"], "reference_a": pool[x]["anchor"],
            "reference_b": pool[y]["anchor"], "template_rank": n})

    pairs = non_covisible_pairs(singular)
    if not pairs:
        notes.append("no non-co-visible anchor pair exists in this capture, so "
                     "the cross-view slots are empty; the 'if naturally "
                     "present' condition of the protocol did not hold")
    for n, (i, j) in enumerate(pairs[:3], start=1):
        questions.append({
            "id": f"t_xview_{n}", "form": "binary_near",
            "answer_type": "boolean",
            "question": (f"Is the {singular[i]['anchor']} near the "
                         f"{singular[j]['anchor']}?"),
            "subject": singular[i]["anchor"], "object": singular[j]["anchor"],
            "cross_view": True, "template_rank": n,
            "why_cross_view": "no supplied frame contains both objects"})
    if len(pairs) < 3:
        notes.append(f"only {len(pairs)} non-co-visible pair(s) available; no "
                     f"substitute pair was invented")
    return questions, notes


def prompt_text(packet: dict) -> str:
    """Composed from the two existing prompts; no wording is rewritten."""
    lines = []
    for q in packet["questions"]:
        if q["answer_type"] == "integer":
            expect = "integer"
        elif q["answer_type"] == "boolean":
            expect = "true or false"
        else:
            expect = f'exactly one of "{q["reference_a"]}" or "{q["reference_b"]}"'
        lines.append(f'- {q["id"]} [{expect}]: {q["question"]}')
    frames = ", ".join(f["id"] for f in packet["frames"])
    return f"""You are evaluating a captured indoor scene from multiple RGB views.
Use ONLY visible evidence in the supplied frames. Do not fill gaps with common
room expectations. A repeated object across views counts once. If the answer
cannot be verified, return outcome "unknown" instead of guessing.

{COUNTING_CONVENTION}

{NEAR_CONVENTION}

Some questions ask about two objects that may never appear together in a single
frame. Answer those from the room's layout as a whole if you can, or return
"unknown".

Valid evidence frame ids: {frames}

Questions:
{chr(10).join(lines)}

Return one JSON object and no prose. Note that "outcome" is a flag with only
two legal values, "answer" or "unknown"; the answer itself goes in "answer":
{{
  "schema": "{RESPONSE_SCHEMA}",
  "scene_id": "{packet['scene_id']}",
  "packet_sha256": "{packet['packet_sha256']}",
  "model": {{"provider": "...", "name": "...", "version": "..."}},
  "answers": [
    {{
      "id": "question id",
      "outcome": "the literal string \\"answer\\", or the literal string \\"unknown\\" -- not the answer itself",
      "answer": "integer, boolean, or object-name string; null if unknown",
      "confidence": 0.0,
      "evidence_frame_ids": ["at least two valid frame ids when answering"]
    }}
  ]
}}
"""


# --------------------------------------------------------------------------
# owner review sheet
# --------------------------------------------------------------------------
def jpeg_uri(path: Path, px: int = 420, quality: int = 84) -> str:
    image = Image.open(path).convert("RGB")
    image.thumbnail((px, px), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


EVIDENCE_CHOICES = (("2+", "2 or more independent views"),
                    ("1", "exactly 1 view"),
                    ("0", "0 views (not visible in the capture)"))


def answer_control(question: dict) -> str:
    qid = question["id"]
    if question["answer_type"] == "integer":
        return (f'<input class="pick num" type="number" min="0" step="1" '
                f'inputmode="numeric" aria-label="count for {esc(qid)}">')
    if question["answer_type"] == "boolean":
        options = [("true", "yes"), ("false", "no")]
    else:
        options = [(question["reference_a"], f'closer to the {question["reference_a"]}'),
                   (question["reference_b"], f'closer to the {question["reference_b"]}'),
                   ("tie", "genuinely the same distance")]
    return "".join(
        f'<label class="chk"><input type="radio" name="{esc(qid)}" class="pick" '
        f'value="{esc(v)}"> {esc(label)}</label>' for v, label in options)


def build_sheet(scene: str, questions: list[dict], frames: list[dict],
                frames_dir: Path, anchors: dict, provenance: dict) -> str:
    figs = "".join(
        f'<figure><img src="{jpeg_uri(frames_dir / f["file"].split("/")[-1])}" '
        f'alt="{esc(f["id"])}"><figcaption>{esc(f["id"])}</figcaption></figure>'
        for f in frames)

    cards = []
    for q in questions:
        kind = {"cardinality": "count", "presence": "yes / no",
                "comparative_near": "which is closer (no threshold)",
                "binary_near": "near / not near"}[q["form"]]
        cross = ('<span class="tag cross">cross-view — no supplied frame '
                 'contains both objects</span>' if q.get("cross_view") else "")
        ev = "".join(
            f'<label class="chk"><input type="radio" name="ev_{esc(q["id"])}" '
            f'class="ev" value="{esc(v)}"> {esc(label)}</label>'
            for v, label in EVIDENCE_CHOICES)
        cards.append(f"""
    <section class="qcard" data-qid="{esc(q['id'])}"
             data-type="{esc(q['answer_type'])}">
      <h3>{esc(q['question'])}</h3>
      <div class="qmeta"><code>{esc(q['id'])}</code>
        <span class="tag">{esc(kind)}</span>{cross}</div>
      <div class="field"><b>Your answer</b>{answer_control(q)}
        <label class="chk amb"><input type="checkbox" class="ambiguous">
          ambiguous / cannot answer — exclude this item</label></div>
      <div class="field"><b>Evidence visibility</b>
        <span class="hint">judged separately from the answer: in how many
        distinct views can this be verified?</span>{ev}</div>
      <div class="field"><b>Notes</b><textarea class="notes" rows="2"
        placeholder="anything ambiguous, or a counting call you had to make"></textarea></div>
    </section>""")

    anchor_rows = "".join(
        f'<tr><td>{esc(a["anchor"])}</td><td class="n">{a["n_passes"]}/3</td>'
        f'<td class="n">{a["first_frame_rank"]}</td>'
        f'<td class="n">{len(a["frame_ids"])}</td></tr>'
        for a in anchors["admitted"])

    return f"""<!DOCTYPE html>
<meta charset="utf-8">
<title>RGB transfer review — {esc(scene)}</title>
<style>
:root {{ --bg:#fff; --fg:#15171a; --mut:#666; --line:#e3e5e8; --hi:#c0158f;
  --warn:#8a5a00; --warnbg:#fff6e0; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#14161a; --fg:#e9ebee; --mut:#9aa1aa; --line:#2b2f36; --hi:#ff5cc8;
  --warn:#f0c060; --warnbg:#2a2214; }} }}
* {{ box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--fg); margin:0 auto;
  padding:24px 20px 80px; max-width:1000px;
  font:15px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif; }}
h1 {{ font-size:21px; margin:0 0 4px; }}
h2 {{ font-size:16px; margin:32px 0 10px; padding-top:14px;
  border-top:1px solid var(--line); }}
h3 {{ font-size:15px; margin:0 0 6px; }}
.sub {{ color:var(--mut); font-size:13px; margin:0 0 14px; }}
code {{ font:12px ui-monospace,Menlo,monospace; background:rgba(128,128,128,.13);
  padding:.1em .35em; border-radius:3px; }}
table {{ border-collapse:collapse; width:100%; font-size:14px; }}
th,td {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); }}
th {{ color:var(--mut); font-weight:600; }}
td.n {{ font-variant-numeric:tabular-nums; }}
.convention {{ border:1px solid var(--line); border-radius:8px;
  padding:12px 16px; margin:14px 0; }}
.warn {{ border-left:3px solid var(--warn); background:var(--warnbg);
  padding:10px 14px; margin:14px 0; border-radius:0 6px 6px 0; font-size:13.5px; }}
.frames {{ display:grid; gap:8px;
  grid-template-columns:repeat(auto-fill,minmax(210px,1fr)); }}
figure {{ margin:0; }} figure img {{ width:100%; border:1px solid var(--line);
  border-radius:5px; display:block; }}
figcaption {{ font-size:10.5px; color:var(--mut); }}
.qcard {{ border:1px solid var(--line); border-radius:8px; padding:14px 16px;
  margin:0 0 12px; }}
.qmeta {{ color:var(--mut); font-size:12px; margin-bottom:10px; display:flex;
  gap:10px; flex-wrap:wrap; align-items:center; }}
.tag {{ background:rgba(128,128,128,.14); padding:1px 7px; border-radius:10px; }}
.tag.cross {{ background:rgba(192,21,143,.14); color:var(--hi); }}
.field {{ margin:10px 0; }}
.field > b {{ display:block; font-size:12.5px; margin-bottom:4px; }}
.hint {{ display:block; color:var(--mut); font-size:12px; margin-bottom:5px; }}
.chk {{ display:inline-flex; align-items:center; gap:5px; margin:0 14px 5px 0;
  font-size:13.5px; }}
.chk.amb {{ color:var(--warn); display:flex; margin-top:8px; }}
input.num {{ font:inherit; padding:5px; width:7rem; background:var(--bg);
  color:var(--fg); border:1px solid var(--line); border-radius:5px; }}
textarea {{ font:inherit; padding:5px; width:100%; background:var(--bg);
  color:var(--fg); border:1px solid var(--line); border-radius:5px; }}
button {{ font:inherit; padding:8px 16px; border-radius:6px; cursor:pointer;
  border:1px solid var(--line); background:var(--hi); color:#fff; }}
pre {{ background:rgba(128,128,128,.10); padding:12px; border-radius:6px;
  overflow-x:auto; font-size:12px; max-height:400px; }}
</style>

<h1>Direct-RGB transfer test — review sheet</h1>
<p class="sub">scene <code>{esc(scene)}</code> · {len(questions)} questions ·
{len(frames)} supplied views · about 20–30 minutes</p>

<div class="warn"><b>This scene is new.</b> It has no delivered instances, no
graph and no prior key, so there is nothing to map to <code>obj_N</code> and no
mapping panel here. Answer from the room as the frames show it. The same
questions and the same two conventions below were given to a blinded model in a
separate context; your answers are the key its answers will be scored against.</div>

<div class="convention"><b>Counting convention</b>
  <p style="margin:6px 0">{esc(COUNTING_CONVENTION)}</p>
  <b>NEAR convention</b>
  <p style="margin:6px 0 0">{esc(NEAR_CONVENTION)}</p>
</div>

<h2>1 · The 18 supplied views</h2>
<p class="sub">Selected by the frozen answer-free rule — 18 equal temporal bins,
best-information frame in each. You may answer from the room as a whole.</p>
<div class="frames">{figs}</div>

<h2>2 · The questions</h2>
<p class="sub">Record evidence visibility independently of your answer. If a
question is unanswerable or the wording does not fit the room, mark it
ambiguous rather than forcing an answer — excluded items cost nothing.</p>
{"".join(cards)}

<p><button onclick="emit()">Build JSON</button></p>
<pre id="out">(fill in above, then press Build JSON)</pre>

<h2>3 · How these questions were chosen</h2>
<p class="sub">Three independent passes listed the objects they could see in
the 18 views, with no access to each other, to any question, or to anything
else in the project. An object became an anchor only when at least two of the
three named it. Anchors were then ordered by first appearance and a fixed
template allocation decided which ones fill which question. No question was
chosen because a system could answer it.</p>
<table><thead><tr><th>anchor</th><th>passes</th><th>first frame</th>
  <th>frames seen in</th></tr></thead><tbody>{anchor_rows}</tbody></table>

<script>
function emit() {{
  var truth = [];
  document.querySelectorAll(".qcard").forEach(function (card) {{
    var type = card.dataset.type;
    var ambiguous = card.querySelector(".ambiguous").checked;
    var answer = null;
    if (type === "integer") {{
      var box = card.querySelector("input.num");
      if (box.value !== "") {{ answer = box.valueAsNumber; }}
    }} else {{
      var hit = card.querySelector(".pick:checked");
      if (hit) {{ answer = (type === "boolean") ? (hit.value === "true") : hit.value; }}
    }}
    var ev = card.querySelector(".ev:checked");
    truth.push({{
      id: card.dataset.qid,
      answer: ambiguous ? null : answer,
      ambiguous: ambiguous,
      evidence_views: ev ? ev.value : null,
      notes: card.querySelector(".notes").value || null
    }});
  }});
  var out = {{
    schema: "{KEY_SCHEMA}",
    scene_id: "{scene}",
    status: "OWNER_CONFIRMED",
    questions_content_sha256: "{provenance['questions_content_sha256']}",
    counting_convention_confirmed: true,
    human_truth: truth
  }};
  document.getElementById("out").textContent = JSON.stringify(out, null, 1);
}}
</script>
"""


# --------------------------------------------------------------------------
def cmd_build(args) -> int:
    """Merge passes, allocate questions, emit packet + prompt + review sheet."""
    selected = json.loads((args.run_dir / "selected_frames.json").read_text())
    frames = selected["frames"]
    frame_order = [f["id"] for f in frames]
    passes = [json.loads(p.read_text()) for p in args.passes]

    anchors = merge_passes(passes, frame_order)
    (args.run_dir / "anchors.json").write_text(
        json.dumps(anchors, indent=1, sort_keys=True) + "\n")
    print(f"    anchors admitted (>= {MIN_PASSES}/3): "
          f"{len(anchors['admitted'])} of "
          f"{len(anchors['admitted']) + len(anchors['rejected_single_pass'])}")

    questions, notes = build_questions(anchors["admitted"])
    doc = {
        "schema": QUESTIONS_SCHEMA,
        "status": "GENERATED_BY_FIXED_PROCEDURE",
        "scene_id": selected["scene"],
        "protocol": "docs/arkitscenes_rgb_transfer_test.md",
        "counting_convention": COUNTING_CONVENTION,
        "near_convention": NEAR_CONVENTION,
        "scope": SCOPE, "scope_definition": SCOPE_DEFINITION,
        "amendment": "1 -- run 1 void; scope wording, anchor ordering and "
                     "question count fixed. 2 -- relational slots restricted "
                     "to unique-referent anchors. Both applied before any "
                     "blinded response existed.",
        "generation": {
            "anchor_rule": f"admitted on >= {MIN_PASSES} of 3 independent "
                           "blind enumeration passes",
            "ordering": "passes desc, frames-seen desc, first appearance asc",
            "allocation": "3 presence/cardinality, 4 comparative, 3 cross-view",
            "relational_slots": "amendment 2: comparative and cross-view slots "
                                "draw only from anchors all three passes agreed "
                                "there is exactly one of, so every singular "
                                "definite description has a referent",
            "notes": notes,
        },
        "questions": questions,
    }
    doc["questions_content_sha256"] = json_sha256(questions)
    questions_path = args.run_dir / "questions.json"
    questions_path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"    questions        : {len(questions)}")
    for note in notes:
        print(f"      note: {note}")

    packet = {
        "schema": PACKET_SCHEMA,
        "stage": "prepared_no_answers",
        "scene_id": selected["scene"],
        "counting_convention": COUNTING_CONVENTION,
        "near_convention": NEAR_CONVENTION,
        "questions_content_sha256": doc["questions_content_sha256"],
        "questions": questions,
        "frames": frames,
        "selection": selected["selection"],
        "disclosures": [
            "No expected answer, human key or system output is included.",
            "The selected views may miss an object even when it exists.",
            "This arm sees RGB only; there is no mesh, graph or entity map "
            "for this scene and none was produced.",
        ],
    }
    packet["packet_sha256"] = json_sha256(packet)
    (args.run_dir / "packet.json").write_text(
        json.dumps(packet, indent=1, sort_keys=True) + "\n")
    (args.run_dir / "prompt.txt").write_text(prompt_text(packet))

    sheet = build_sheet(selected["scene"], questions, frames,
                        args.run_dir / "frames", anchors,
                        {"questions_content_sha256": doc["questions_content_sha256"]})
    sheet_path = args.run_dir / "review_sheet.html"
    sheet_path.write_text(sheet)

    print(f"    packet_sha256    : {packet['packet_sha256']}")
    print(f"    questions_sha256 : {doc['questions_content_sha256']}")
    print(f"    review sheet     : {sheet_path.stat().st_size / 1e6:.1f} MB")
    print(f"    -> {args.run_dir}")
    return 0


# --------------------------------------------------------------------------
# scoring -- exactly once, per the frozen protocol
# --------------------------------------------------------------------------
GATE_ACCURACY = 0.60
GATE_COVERAGE = 0.80


def _matches(question: dict, got: object, expected: object) -> bool:
    kind = question["answer_type"]
    if kind == "boolean":
        return isinstance(got, bool) and got == expected
    if kind == "integer":
        return isinstance(got, int) and not isinstance(got, bool) and got == expected
    return isinstance(got, str) and isinstance(expected, str) \
        and got.strip().lower() == expected.strip().lower()


def score_run(questions_doc: dict, key: dict, response: dict,
              packet: dict) -> dict:
    if key.get("questions_content_sha256") != questions_doc["questions_content_sha256"]:
        raise ValueError("key does not pin these questions")
    if response.get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError("response does not pin this packet")

    by = {q["id"]: q for q in questions_doc["questions"]}
    truth = {t["id"]: t for t in key["human_truth"]}
    given = {a["id"]: a for a in response["answers"]}
    if not (sorted(by) == sorted(truth) == sorted(given)):
        raise ValueError("questions, key and response do not cover the same items")

    rows = []
    for qid, question in by.items():
        t, a = truth[qid], given[qid]
        row = {"id": qid, "form": question["form"], "question": question["question"],
               "human_answer": t["answer"], "model_answer": a.get("answer"),
               "confidence": a.get("confidence"),
               "evidence_frame_ids": a.get("evidence_frame_ids", []),
               "evidence_views": t.get("evidence_views"),
               "cross_view": bool(question.get("cross_view"))}
        if t.get("ambiguous"):
            row["outcome"] = "excluded_owner_ambiguous"
        elif a.get("outcome") == "unknown":
            row["outcome"] = "unanswered"
        else:
            row["outcome"] = ("correct" if _matches(question, a.get("answer"),
                                                    t["answer"]) else "wrong")
        rows.append(row)

    scored = [r for r in rows if r["outcome"] != "excluded_owner_ambiguous"]
    correct = sum(1 for r in scored if r["outcome"] == "correct")
    wrong = sum(1 for r in scored if r["outcome"] == "wrong")
    unanswered = sum(1 for r in scored if r["outcome"] == "unanswered")
    n = len(scored)
    accuracy = round(correct / n, 4) if n else None
    coverage = round((correct + wrong) / n, 4) if n else None

    thin = [r for r in scored if r["evidence_views"] in {"0", "1"}]
    thin_answered = [r for r in thin if r["outcome"] in {"correct", "wrong"}]

    gates = {
        "exact_accuracy": {"required": GATE_ACCURACY, "measured": accuracy,
                           "pass": bool(accuracy is not None and accuracy >= GATE_ACCURACY)},
        "answer_coverage": {"required": GATE_COVERAGE, "measured": coverage,
                            "pass": bool(coverage is not None and coverage >= GATE_COVERAGE)},
        "scored_items": {"required": MIN_SCORED, "measured": n,
                         "pass": bool(n >= MIN_SCORED)},
    }
    passed = all(g["pass"] for g in gates.values())
    return {
        "schema": "arkitscenes_rgb_transfer_score_v1",
        "scene_id": questions_doc["scene_id"],
        "protocol": "docs/arkitscenes_rgb_transfer_test.md",
        "amendments_applied": questions_doc.get("amendment"),
        "model": response.get("model"),
        "tally": {"correct": correct, "wrong": wrong, "unanswered": unanswered,
                  "excluded_owner_ambiguous": len(rows) - n},
        "n_scored": n,
        "exact_accuracy": accuracy,
        "answer_coverage": coverage,
        "accuracy_when_answered": (round(correct / (correct + wrong), 4)
                                   if correct + wrong else None),
        "false_confident_rate": (round(wrong / (correct + wrong), 4)
                                 if correct + wrong else None),
        "gates": gates,
        "all_gates_pass": passed,
        "claim": ("Under a fixed procedure, direct multiview RGB transferred to "
                  "one previously untouched ARKitScenes handheld room."
                  if passed else
                  "NO TRANSFER CLAIM. At least one predeclared gate failed, so "
                  "the demo remains a fixed evaluation replay."),
        "thin_evidence_slice": {
            "n": len(thin),
            "ids": [r["id"] for r in thin],
            "answered": len(thin_answered),
            "correct": sum(1 for r in thin_answered if r["outcome"] == "correct"),
            "note": "owner-recorded 0- or 1-view items, judged independently "
                    "of the answer",
        },
        "by_form": {
            form: {
                "n": sum(1 for r in scored if r["form"] == form),
                "correct": sum(1 for r in scored
                               if r["form"] == form and r["outcome"] == "correct"),
                "unanswered": sum(1 for r in scored
                                  if r["form"] == form and r["outcome"] == "unanswered"),
            } for form in sorted({r["form"] for r in scored})
        },
        "rows": rows,
        "limits": [
            "One room, ten questions, one blinded response, one human reviewer.",
            "A pass would support the single sentence above and nothing wider.",
            "The reviewer is the project owner and is not blind to the "
            "hypothesis; the human key is itself a measurement with error.",
            "No graph, grounding or segmentation stage ran on this scene.",
        ],
    }


def cmd_score(args) -> int:
    d = args.run_dir
    report = score_run(json.loads((d / "questions.json").read_text()),
                       json.loads(args.key.read_text()),
                       json.loads(args.response.read_text()),
                       json.loads((d / "packet.json").read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"    scored items      : {report['n_scored']}")
    print(f"    correct / wrong / unanswered : "
          f"{report['tally']['correct']} / {report['tally']['wrong']} / "
          f"{report['tally']['unanswered']}")
    for name, gate in report["gates"].items():
        print(f"    {name:16s} required {gate['required']}  measured "
              f"{gate['measured']}  {'PASS' if gate['pass'] else 'FAIL'}")
    print(f"    accuracy when answered : {report['accuracy_when_answered']}")
    print(f"\n    {report['claim']}")
    print(f"    -> {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path,
                    default=REPO_ROOT / "runs" / "arkit_rgb_transfer" / "47331972")
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--passes", type=Path, nargs="+", required=True)
    s = sub.add_parser("score")
    s.add_argument("--key", type=Path, required=True)
    s.add_argument("--response", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    return {"build": cmd_build, "score": cmd_score}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
