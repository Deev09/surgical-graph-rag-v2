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

COUNTING_CONVENTION = (
    "An instance is a physically separate object of the named class. Two "
    "objects of the same class in different parts of the room count as two "
    "even if they never appear in one frame together.")
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

    # Mechanical: first appearance ascending, ties alphabetical. This ordering,
    # not judgement about answerability, decides which anchors get used.
    admitted.sort(key=lambda r: (r["first_frame_rank"], r["anchor"]))
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
    """3 presence/cardinality, 3 comparative, 2 cross-view. No discretion."""
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
                "question": f"How many {a['anchor']}s are in this room?",
                "subject": a["anchor"], "template_rank": rank + 1})
        else:
            # The passes disagreed or hedged on the count, so asking for a
            # number would key an ambiguity rather than a fact.
            questions.append({
                "id": f"t_pres_{rank + 1}", "form": "presence",
                "answer_type": "boolean",
                "question": f"Is there a {a['anchor']} in this room?",
                "subject": a["anchor"], "template_rank": rank + 1})
            notes.append(f"{a['anchor']}: passes did not agree on a count, so "
                         f"the slot became a presence question")

    pool = anchors[:6]
    triples = [(0, 1, 2), (1, 2, 3), (2, 4, 5)]
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

    pairs = non_covisible_pairs(anchors)
    if not pairs:
        notes.append("no non-co-visible anchor pair exists in this capture, so "
                     "the cross-view slots are empty; the 'if naturally "
                     "present' condition of the protocol did not hold")
    for n, (i, j) in enumerate(pairs[:2], start=1):
        questions.append({
            "id": f"t_xview_{n}", "form": "binary_near",
            "answer_type": "boolean",
            "question": (f"Is the {anchors[i]['anchor']} near the "
                         f"{anchors[j]['anchor']}?"),
            "subject": anchors[i]["anchor"], "object": anchors[j]["anchor"],
            "cross_view": True, "template_rank": n,
            "why_cross_view": "no supplied frame contains both objects"})
    if len(pairs) < 2:
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
