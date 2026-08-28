#!/usr/bin/env python3
"""Blinded second-annotator review sheet for the ARKit relation challenge.

Three subcommands:

  generate     Build a single self-contained HTML sheet for the 12 relation
               questions, embedding the same frozen frames the blinded model
               packets declare (sha256-verified). The sheet shows NO system
               outputs, NO arm names, NO hypothesis, and never uses the word
               "graph". Question order is shuffled with a recorded seed.

  ingest       Normalize the JSON the annotator exports from the sheet into a
               schema-stamped committed record under eval/human_feedback/.

  agreement    Compare the ingested record against the v1 human key. Exact
               agreement is computed mechanically for ambiguity; answers,
               referents and relation labels get a tentative normalized
               auto-match that an --adjudications overlay can correct without
               overwriting the raw returned form.

The returned annotations are evaluation-key reproducibility evidence and
future calibration data. They must not be used to retune the evaluated
system tonight (CLAUDE.md overnight amendment 2026-08-27/28).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
QUESTIONS = REPO / "eval" / "questions" / "arkitscenes_relation_challenge_v1.json"
KEY_V1 = REPO / "eval" / "human_feedback" / "arkitscenes_relation_challenge_key_v1.json"
PACKET_DIRS = {
    "arkitscenes_41069025": REPO / "runs" / "arkit_relation_challenge" / "blinded_rgb" / "41069025",
    "arkitscenes_41069042": REPO / "runs" / "arkit_relation_challenge" / "blinded_rgb" / "41069042",
}
ROOM_LABELS = {"arkitscenes_41069025": "Room A", "arkitscenes_41069042": "Room B"}
SHEET_SEED = 20260828
OUT_SHEET = REPO / "runs" / "stagereach_review" / "annotator2_sheet.html"
OUT_RETURNED = REPO / "eval" / "human_feedback" / "arkitscenes_relation_challenge_annotator2_returned.json"
OUT_AGREEMENT = REPO / "eval" / "results" / "stagereach" / "annotator_agreement_v1.json"

CONFIDENCE_LEVELS = ["1 (guess)", "2", "3", "4", "5 (certain)"]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_questions() -> dict:
    return json.loads(QUESTIONS.read_text())


def embedded_frames(scene_id: str) -> list[dict]:
    """Load the packet-declared frames for a scene, verifying each pin."""
    pdir = PACKET_DIRS[scene_id]
    packet = json.loads((pdir / "packet.json").read_text())
    out = []
    for fr in packet["frames"]:
        raw = (pdir / fr["file"]).read_bytes()
        got = sha256_bytes(raw)
        if got != fr["sha256"]:
            raise SystemExit(
                f"frame pin mismatch for {scene_id} {fr['file']}: {got}")
        out.append({
            "id": fr["id"],
            "data_uri": "data:image/png;base64," + base64.b64encode(raw).decode(),
        })
    return out


def referent_phrases(q: dict) -> list[str]:
    phrases = []
    for field in ("subject", "reference_a", "reference_b"):
        v = q.get(field)
        if v:
            phrases.append(v)
    for v in q.get("candidate_objects") or []:
        phrases.append(v)
    # preserve order, drop duplicates
    seen, out = set(), []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def generate() -> int:
    doc = load_questions()
    questions = list(doc["questions"])
    rng = random.Random(SHEET_SEED)
    rng.shuffle(questions)
    convention = doc["near_convention"]["statement"]

    galleries = []
    for scene_id, label in ROOM_LABELS.items():
        frames = embedded_frames(scene_id)
        imgs = "\n".join(
            f'<img src="{fr["data_uri"]}" alt="{html.escape(fr["id"])}" '
            f'title="{html.escape(fr["id"])}">' for fr in frames)
        galleries.append(
            f'<details open id="gal-{html.escape(label[-1])}">'
            f"<summary><strong>{label}</strong> — {len(frames)} photos "
            f"(scroll; every question names its room)</summary>"
            f'<div class="gallery">{imgs}</div></details>')

    cards = []
    for i, q in enumerate(questions, 1):
        room = ROOM_LABELS[q["scene_id"]]
        refs = referent_phrases(q)
        ref_inputs = "\n".join(
            f'<label class="ref">&ldquo;{html.escape(p)}&rdquo; — which object '
            f"in the {room} photos did you take this to be? Describe it or say "
            f'&ldquo;not visible&rdquo;.<br>'
            f'<input type="text" data-q="{html.escape(q["id"])}" '
            f'data-field="referent:{html.escape(p)}" size="70"></label>'
            for p in refs)
        options = "\n".join(
            f'<option value="{html.escape(o)}">{html.escape(o)}</option>'
            for o in CONFIDENCE_LEVELS)
        cards.append(f"""
<section class="card" id="{html.escape(q['id'])}">
<h3>Question {i} of {len(questions)} <span class="room">({room})</span></h3>
<p class="qtext">{html.escape(q['question'])}</p>
<label>Your answer (answer only from what the photos show; if the photos do
not let you decide, write &ldquo;cannot determine&rdquo;):<br>
<input type="text" data-q="{html.escape(q['id'])}" data-field="answer" size="70"></label>
<label class="inline"><input type="checkbox" data-q="{html.escape(q['id'])}"
 data-field="ambiguous"> This question is ambiguous as written (say why in the notes)</label>
{ref_inputs}
<label>Spatial relation you judged (in your own words, e.g. &ldquo;near&rdquo;,
&ldquo;not near&rdquo;, &ldquo;closer to the first&rdquo;):<br>
<input type="text" data-q="{html.escape(q['id'])}" data-field="relation_label" size="40"></label>
<label>Confidence:
<select data-q="{html.escape(q['id'])}" data-field="confidence">{options}</select></label>
<label>Notes / rationale:<br>
<textarea data-q="{html.escape(q['id'])}" data-field="rationale" rows="2" cols="70"></textarea></label>
</section>""")

    sheet = f"""<!doctype html>
<meta charset="utf-8">
<title>Room photo questions — independent review</title>
<style>
 body {{ font: 15px/1.5 -apple-system, sans-serif; margin: 2rem auto; max-width: 60rem; }}
 .gallery {{ display: flex; flex-wrap: wrap; gap: 6px; }}
 .gallery img {{ width: 240px; border: 1px solid #ccc; }}
 .card {{ border: 1px solid #bbb; border-radius: 8px; padding: 1rem; margin: 1.2rem 0; }}
 .card label {{ display: block; margin: .5rem 0; }}
 .card .inline {{ display: inline-block; }}
 .qtext {{ font-weight: 600; }}
 .room {{ color: #666; font-weight: 400; }}
 #export {{ font-size: 1.1rem; padding: .5rem 1.2rem; }}
</style>
<h1>Room photo questions</h1>
<p>Please answer {len(questions)} questions about two rooms, using ONLY the
photos below. There are no trick questions and no right-answer sheet on your
side — answer what you can actually see. Work alone and do not discuss the
questions with anyone until you have exported your answers.</p>
<p><strong>What &ldquo;near&rdquo; means here:</strong> {html.escape(convention)}</p>
{''.join(galleries)}
{''.join(cards)}
<p><button id="export">Export my answers (downloads a small file — send it back)</button></p>
<input type="hidden" id="seed" value="{SHEET_SEED}">
<script>
document.getElementById('export').addEventListener('click', function () {{
  var answers = {{}};
  document.querySelectorAll('[data-q]').forEach(function (el) {{
    var q = el.getAttribute('data-q'), f = el.getAttribute('data-field');
    if (!answers[q]) answers[q] = {{}};
    answers[q][f] = el.type === 'checkbox' ? el.checked : el.value;
  }});
  var payload = {{
    schema: 'stagereach_annotator_sheet_return',
    schema_version: 1,
    sheet_seed: Number(document.getElementById('seed').value),
    exported_at: new Date().toISOString(),
    answers: answers
  }};
  var blob = new Blob([JSON.stringify(payload, null, 1)], {{type: 'application/json'}});
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'room_questions_answers.json';
  a.click();
}});
</script>
"""
    OUT_SHEET.parent.mkdir(parents=True, exist_ok=True)
    OUT_SHEET.write_text(sheet)
    size_mb = OUT_SHEET.stat().st_size / 1e6
    print(f"wrote {OUT_SHEET} ({size_mb:.1f} MB, seed {SHEET_SEED}, "
          f"{len(questions)} questions)")
    return 0


def ingest(source: Path) -> int:
    raw = source.read_bytes()
    doc = json.loads(raw)
    if doc.get("schema") != "stagereach_annotator_sheet_return":
        raise SystemExit(f"unexpected schema in {source}: {doc.get('schema')}")
    record = {
        "schema": "stagereach_annotator2_returned",
        "schema_version": 1,
        "source_sha256": sha256_bytes(raw),
        "sheet_seed": doc.get("sheet_seed"),
        "exported_at": doc.get("exported_at"),
        "answers": doc["answers"],
        "role": ("independent second annotator; blinded to system outputs, "
                 "arm names and hypothesis; key-reproducibility evidence and "
                 "future calibration data only"),
    }
    OUT_RETURNED.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT_RETURNED} (source sha256 {record['source_sha256'][:12]}...)")
    return 0


def norm(s: str) -> str:
    return " ".join(str(s).casefold().split())


def auto_answer_match(key_answer, returned_answer) -> bool:
    """Tentative normalized match; adjudication overlay has the final word."""
    if key_answer is None or returned_answer is None:
        return False
    ka, ra = norm(key_answer), norm(returned_answer)
    if not ka or not ra:
        return False
    return ka == ra or ka in ra or ra in ka


def kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    """Cohen's kappa for two binary raters; None when degenerate."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    pa = sum(1 for a, _ in pairs if a) / n
    pb = sum(1 for _, b in pairs if b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe == 1.0:
        return None  # no variation: kappa undefined
    return (po - pe) / (1 - pe)


def agreement(adjudications: Path | None) -> int:
    key = json.loads(KEY_V1.read_text())
    key_rows = {r["id"]: r for r in key["human_relation_truth"]}
    returned = json.loads(OUT_RETURNED.read_text())
    answers = returned["answers"]
    adj = json.loads(adjudications.read_text()) if adjudications else {}

    items, amb_pairs = [], []
    n_answer_match = n_adjudicated = 0
    for qid, krow in sorted(key_rows.items()):
        ret = answers.get(qid, {})
        auto = auto_answer_match(krow.get("answer"), ret.get("answer"))
        a = adj.get(qid, {})
        final = a.get("answer_match", auto)
        if "answer_match" in a:
            n_adjudicated += 1
        if final:
            n_answer_match += 1
        amb_pairs.append((bool(krow.get("ambiguous")), bool(ret.get("ambiguous"))))
        items.append({
            "id": qid,
            "key_answer": krow.get("answer"),
            "returned_answer": ret.get("answer"),
            "auto_answer_match": auto,
            "final_answer_match": final,
            "adjudicated": "answer_match" in a,
            "key_ambiguous": bool(krow.get("ambiguous")),
            "returned_ambiguous": bool(ret.get("ambiguous")),
            "returned_relation_label": ret.get("relation_label"),
            "returned_confidence": ret.get("confidence"),
            "returned_rationale": ret.get("rationale"),
            "adjudication_note": a.get("note"),
        })

    n = len(items)
    amb_exact = sum(1 for a, b in amb_pairs if a == b)
    report = {
        "schema": "stagereach_annotator_agreement",
        "schema_version": 1,
        "n_questions_reviewed": n,
        "n_eligible_primary": 10,
        "n_retained_for_exclusion_reproducibility": 2,
        "answer_exact_agreement": {"agree": n_answer_match, "of": n,
                                   "n_adjudicated": n_adjudicated},
        "ambiguity_exact_agreement": {"agree": amb_exact, "of": n},
        "ambiguity_kappa": kappa(amb_pairs),
        "kappa_note": ("kappa reported only for the binary ambiguity field; "
                       "free-text answers use exact agreement after "
                       "adjudication, preserved pre-adjudication below"),
        "items": items,
        "protocol": ("second annotator blinded to system outputs, arm names "
                     "and hypothesis; sheet order shuffled with recorded "
                     "seed; adjudication overlays never overwrite the "
                     "returned form"),
    }
    OUT_AGREEMENT.parent.mkdir(parents=True, exist_ok=True)
    OUT_AGREEMENT.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT_AGREEMENT}: answers {n_answer_match}/{n} "
          f"({n_adjudicated} adjudicated), ambiguity {amb_exact}/{n}, "
          f"ambiguity kappa {report['ambiguity_kappa']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate")
    p_in = sub.add_parser("ingest")
    p_in.add_argument("returned_json", type=Path)
    p_ag = sub.add_parser("agreement")
    p_ag.add_argument("--adjudications", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.cmd == "generate":
        return generate()
    if args.cmd == "ingest":
        return ingest(args.returned_json)
    return agreement(args.adjudications)


if __name__ == "__main__":
    raise SystemExit(main())
