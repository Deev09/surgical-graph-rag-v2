"""Bounded representation kill test for human-grounded ARKit spatial QA.

This tool does not change perception.  It compares what can be answered from
the already-delivered object map, the already-delivered typed graph, a direct
vision-language response, and a small question-kind router.  The direct arm is
an explicit sidecar because the repository deliberately has no deployable VLM
runtime or credentials.

Two commands are intentionally separate:

  prepare --key KEY --scene-dir SCENE --out OUT
      Selects deterministic, diverse RGB frames and writes an answer-free VLM
      packet.  Human answers never enter the packet or prompt.

  score --key KEY --entities ENTITIES --graph GRAPH --packet PACKET \
        [--direct-responses RESPONSES] --out OUT
      Scores all available arms against the human key.  Without a response
      sidecar the direct and hybrid arms are visibly pending, never fabricated.

This is an evaluation-only screening experiment, not a benchmark-definition
change and not an end-to-end product claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extractors.arkitscenes_rgb_crops import Frame, load_frames
from tools.arkitscenes_spatial_qa_score import (
    load_entities,
    load_graph,
    score as score_object_system,
)

SCHEMA = "arkitscenes_representation_kill_test_v1"
RESPONSE_SCHEMA = "arkitscenes_direct_vlm_responses_v1"
N_VIEWS_DEFAULT = 18
PRIMARY_KINDS = {
    "class_cardinality",
    "class_presence",
    "support_relation",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def geometry_signature(manifest: dict) -> str:
    """Hash only UID and geometry fields; learned labels may legitimately differ."""
    rows = []
    for entity in manifest.get("entities", []):
        rows.append({
            "uid": entity.get("identity", {}).get("object_uid"),
            "bbox_aabb": entity.get("bbox_aabb"),
            "bbox_obb": entity.get("bbox_obb"),
            "centroid": entity.get("centroid"),
            "geometry_handle": entity.get("geometry_handle"),
        })
    return json_sha256({
        "scene_id": manifest.get("scene_id"),
        "frame": manifest.get("frame"),
        "representation_hash": manifest.get("representation_hash"),
        "entities": rows,
    })


def validate_graph_geometry(entities_path: Path, graph_path: Path) -> dict:
    """Prove reused label-independent edges reference the exact same partition."""
    current = json.loads(entities_path.read_text())
    graph_entities_path = graph_path.parent.parent / "entities" / "manifest.json"
    if not graph_entities_path.is_file():
        raise ValueError(f"graph source entity manifest missing: {graph_entities_path}")
    graph_entities = json.loads(graph_entities_path.read_text())
    current_sig = geometry_signature(current)
    graph_sig = geometry_signature(graph_entities)
    if current_sig != graph_sig:
        raise ValueError("graph and current entity geometry/UID partition differ")
    return {
        "status": "exact_geometry_match",
        "geometry_signature": current_sig,
        "graph_source_entity_manifest": str(graph_entities_path),
        "graph_source_entity_manifest_sha256": sha256(graph_entities_path),
        "note": "labels differ by arm; reused relation edges are geometry-only",
    }


def public_questions(key: dict) -> list[dict]:
    """Room-observable questions only; excludes system-output diagnostics."""
    rows = []
    for item in key["independent_questions"]:
        if item["kind"] not in PRIMARY_KINDS:
            continue
        rows.append({
            "id": item["id"],
            "kind": item["kind"],
            "question": item["question"],
            "answer_type": {
                "class_cardinality": "integer",
                "class_presence": "boolean",
                "support_relation": "object_class",
            }[item["kind"]],
        })
    return rows


def image_information_score(path: Path) -> float:
    """Cheap answer-free preference against blank, blurred, or clipped views."""
    image = Image.open(path).convert("L")
    image.thumbnail((160, 120), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.size == 0:
        return 0.0
    gx = np.abs(np.diff(array, axis=1)).mean() if array.shape[1] > 1 else 0.0
    gy = np.abs(np.diff(array, axis=0)).mean() if array.shape[0] > 1 else 0.0
    contrast = float(array.std())
    clipped = float(((array < 0.02) | (array > 0.98)).mean())
    return contrast + 2.0 * float(gx + gy) - 0.15 * clipped


def survey_frame_indices(frames: list[Frame], n_views: int) -> list[int]:
    """Cover the capture timeline while rejecting empty/low-information views.

    Pose-only farthest-point sampling can spend much of a small budget on blank
    walls because those camera directions are geometrically novel. ARKit video
    is a walkthrough, so equal temporal bins preserve the survey while an RGB
    information score chooses the most inspectable frame inside each bin. The
    score is class- and answer-free.
    """
    if n_views <= 0:
        raise ValueError("n_views must be positive")
    if not frames:
        raise ValueError("no frames")
    if len(frames) <= n_views:
        return list(range(len(frames)))
    boundaries = np.linspace(0, len(frames), n_views + 1, dtype=int)
    selected = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        candidates = list(range(int(start), max(int(start) + 1, int(end))))
        best = max(candidates,
                   key=lambda i: (image_information_score(frames[i].png), -i))
        selected.append(best)
    return sorted(selected)


def make_contact_sheet(images: list[tuple[str, Path]], path: Path) -> None:
    thumbs = []
    for frame_id, src in images:
        image = Image.open(src).convert("RGB")
        image.thumbnail((384, 288), Image.Resampling.LANCZOS)
        thumbs.append((frame_id, image.copy()))
    cols = 3
    cell_w, cell_h, label_h = 400, 320, 24
    rows = math.ceil(len(thumbs) / cols)
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (frame_id, image) in enumerate(thumbs):
        x = (i % cols) * cell_w
        y = (i // cols) * cell_h
        canvas.paste(image, (x + (cell_w - image.width) // 2, y + label_h))
        draw.text((x + 8, y + 4), frame_id, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def prompt_text(packet: dict) -> str:
    question_lines = "\n".join(
        f"- {q['id']} [{q['answer_type']}]: {q['question']}"
        for q in packet["questions"]
    )
    frame_ids = ", ".join(f["id"] for f in packet["frames"])
    return f"""You are evaluating a captured indoor scene from multiple RGB views.
Use ONLY visible evidence in the supplied frames. Do not fill gaps with common
room expectations. A repeated object across views counts once. If the answer
cannot be verified, return outcome \"unknown\" instead of guessing.

Valid evidence frame ids: {frame_ids}

Questions:
{question_lines}

Return one JSON object and no prose:
{{
  "schema": "{RESPONSE_SCHEMA}",
  "scene_id": "{packet['scene_id']}",
  "packet_sha256": "{packet['packet_sha256']}",
  "model": {{"provider": "...", "name": "...", "version": "..."}},
  "answers": [
    {{
      "id": "question id",
      "outcome": "answer or unknown",
      "answer": "integer, boolean, or object-class string; null if unknown",
      "confidence": 0.0,
      "evidence_frame_ids": ["at least two valid frame ids when answering"]
    }}
  ]
}}
"""


def prepare_packet(key_path: Path, scene_dir: Path, out: Path,
                   n_views: int = N_VIEWS_DEFAULT) -> Path:
    key = json.loads(key_path.read_text())
    questions = public_questions(key)
    if not questions:
        raise ValueError("key has no public questions")
    frames = load_frames(scene_dir, stride=6)
    chosen = survey_frame_indices(frames, n_views)

    frame_dir = out / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_rows = []
    contact_inputs = []
    for rank, index in enumerate(chosen):
        frame = frames[index]
        frame_id = f"frame_{rank:02d}_{frame.timestamp:.3f}"
        suffix = frame.png.suffix.lower() or ".png"
        dst = frame_dir / f"{frame_id}{suffix}"
        shutil.copyfile(frame.png, dst)
        frame_rows.append({
            "id": frame_id,
            "file": str(dst.relative_to(out)),
            "sha256": sha256(dst),
            "timestamp": frame.timestamp,
        })
        contact_inputs.append((frame_id, dst))

    packet = {
        "schema": SCHEMA,
        "stage": "prepared_no_answers",
        "scene_id": key["scene_id"],
        "key_sha256": sha256(key_path),
        "questions_sha256": json_sha256(questions),
        "questions": questions,
        "frames": frame_rows,
        "selection": {
            "method": "temporal_bins_rgb_information_v1",
            "n_usable_strided_frames": len(frames),
            "n_selected": len(frame_rows),
            "source_scene_dir": str(scene_dir),
        },
        "disclosures": [
            "No expected answer or review basis is included in this packet.",
            "The selected views may miss an object even when it exists.",
            "This arm sees RGB only; it does not receive mesh or graph state.",
        ],
    }
    # The packet hash is over answer-free content and pins the response input.
    packet["packet_sha256"] = json_sha256(packet)
    out.mkdir(parents=True, exist_ok=True)
    packet_path = out / "packet.json"
    packet_path.write_text(json.dumps(packet, indent=1, sort_keys=True) + "\n")
    prompt = prompt_text(packet)
    (out / "prompt.txt").write_text(prompt)
    make_contact_sheet(contact_inputs, out / "contact_sheet.jpg")
    return packet_path


def normalize_class(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip().lower().replace("_", "-").replace(" ", "-")


def direct_rows(key: dict, packet: dict, response: dict) -> list[dict]:
    if response.get("schema") != RESPONSE_SCHEMA:
        raise ValueError("wrong direct-response schema")
    if response.get("scene_id") != packet["scene_id"]:
        raise ValueError("direct response scene does not match packet")
    if response.get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError("direct response does not pin the prepared packet")
    model = response.get("model")
    if not isinstance(model, dict) or any(
            not isinstance(model.get(k), str) or not model[k].strip()
            for k in ("provider", "name", "version")):
        raise ValueError("direct response must record provider/name/version")
    valid_frames = {f["id"] for f in packet["frames"]}
    by_id = {a["id"]: a for a in response.get("answers", [])}
    if len(by_id) != len(response.get("answers", [])):
        raise ValueError("duplicate direct-response question id")

    rows = []
    public_ids = {q["id"] for q in packet["questions"]}
    key_items = {q["id"]: q for q in key["independent_questions"]}
    if set(by_id) != public_ids:
        raise ValueError("direct response must answer every packet question")
    for qid in [q["id"] for q in packet["questions"]]:
        item = key_items[qid]
        answer = by_id[qid]
        outcome = answer.get("outcome")
        cited = answer.get("evidence_frame_ids", [])
        confidence = answer.get("confidence")
        if (not isinstance(confidence, (int, float))
                or isinstance(confidence, bool)
                or not 0.0 <= float(confidence) <= 1.0):
            raise ValueError(f"{qid}: confidence must be in [0, 1]")
        if not isinstance(cited, list) or any(f not in valid_frames for f in cited):
            raise ValueError(f"{qid}: invalid evidence frame id")
        base = {
            "id": qid,
            "kind": item["kind"],
            "question": item["question"],
            "answer": answer.get("answer"),
            "confidence": float(confidence),
            "evidence_frame_ids": cited,
            "evidence_sufficient": len(set(cited)) >= 2,
            "source": "direct_vlm",
        }
        if outcome == "unknown":
            base.update(outcome="unanswered", expected=_expected(item))
        elif outcome != "answer":
            raise ValueError(f"{qid}: outcome must be answer or unknown")
        else:
            correct = _answer_matches(item, answer.get("answer"))
            base.update(outcome="correct" if correct else "wrong",
                        expected=_expected(item))
        rows.append(base)
    return rows


def _expected(item: dict) -> object:
    if item["kind"] == "support_relation":
        return item["expected_object_class"]
    return item["expected"]


def _answer_matches(item: dict, answer: object) -> bool:
    kind = item["kind"]
    if kind == "class_cardinality":
        return isinstance(answer, int) and not isinstance(answer, bool) \
            and answer == item["expected"]
    if kind in {"class_presence", "instance_extent"}:
        return isinstance(answer, bool) and answer == item["expected"]
    if kind == "support_relation":
        got = normalize_class(answer)
        wanted = normalize_class(item["expected_object_class"])
        return got == wanted or (wanted == "sofa" and got == "couch")
    return False


def summarize(rows: list[dict]) -> dict:
    counts = {"correct": 0, "wrong": 0, "unanswered": 0}
    for row in rows:
        counts[row["outcome"]] += 1
    answered = counts["correct"] + counts["wrong"]
    total = len(rows)
    return {
        "n_questions": total,
        "tally": counts,
        "coverage": round(answered / total, 4) if total else None,
        "exact_accuracy_all": round(counts["correct"] / total, 4) if total else None,
        "accuracy_when_answered": (
            round(counts["correct"] / answered, 4) if answered else None
        ),
        "false_confident_rate": (
            round(counts["wrong"] / answered, 4) if answered else None
        ),
    }


def hybrid_rows(packet: dict, object_rows: list[dict], graph_rows: list[dict],
                visual_rows: list[dict]) -> list[dict]:
    """Predeclared screening router; no answer-dependent route selection."""
    obj = {r["id"]: r for r in object_rows}
    graph = {r["id"]: r for r in graph_rows}
    visual = {r["id"]: r for r in visual_rows}
    valid_frames = {f["id"] for f in packet["frames"]}
    out = []
    for question in packet["questions"]:
        qid, kind = question["id"], question["kind"]
        if kind == "support_relation" and graph[qid].get("answer") is not None:
            row = dict(graph[qid])
            row.update(source="typed_graph", route="typed_relation")
        else:
            row = dict(visual[qid])
            cited = set(row.get("evidence_frame_ids", []))
            enough = (row["outcome"] in {"correct", "wrong"}
                      and len(cited) >= 2 and cited <= valid_frames)
            row.update(source="direct_vlm", route="visual_evidence")
            if not enough:
                row["outcome"] = "unanswered"
                row["answer"] = None
                row["insufficiency"] = "fewer than two valid cited views"
        out.append(row)
    return out


def decision(arms: dict[str, dict]) -> dict:
    if "hybrid" not in arms or "direct_vlm" not in arms:
        return {"status": "pending_direct_vlm", "proceed": None}
    singles = [arms[n]["summary"] for n in ("direct_vlm", "object_retrieval", "typed_graph")]
    best_accuracy = max(s["exact_accuracy_all"] for s in singles)
    hybrid = arms["hybrid"]["summary"]
    gain = hybrid["exact_accuracy_all"] - best_accuracy
    single_wrong = min(s["tally"]["wrong"] for s in singles)
    wrong_reduction = ((single_wrong - hybrid["tally"]["wrong"]) / single_wrong
                       if single_wrong else 0.0)
    pass_gain = gain >= 0.10
    pass_safety = wrong_reduction >= 0.30 and hybrid["coverage"] >= 0.60
    return {
        "status": "measured",
        "proceed": bool(pass_gain or pass_safety),
        "absolute_accuracy_gain_vs_best_single": round(gain, 4),
        "wrong_answer_reduction_vs_lowest_single": round(wrong_reduction, 4),
        "hybrid_coverage": hybrid["coverage"],
        "rules": {
            "accuracy": "gain >= 0.10",
            "safety": "wrong-answer reduction >= 0.30 at coverage >= 0.60",
            "interpretation": "screening decision only; one scene is not a claim",
        },
    }


def run_score(key_path: Path, entities_path: Path, graph_path: Path,
              packet_path: Path, direct_path: Path | None, out: Path) -> Path:
    key = json.loads(key_path.read_text())
    packet = json.loads(packet_path.read_text())
    if key["scene_id"] != packet["scene_id"]:
        raise ValueError("key and packet scene differ")
    if sha256(key_path) != packet["key_sha256"]:
        raise ValueError("key changed after packet preparation")
    graph_geometry = validate_graph_geometry(entities_path, graph_path)
    entities, entity_prov = load_entities(entities_path)
    edges, graph_prov = load_graph(graph_path)
    selected_ids = {q["id"] for q in packet["questions"]}
    filtered_key = dict(key)
    filtered_key["independent_questions"] = [
        q for q in key["independent_questions"] if q["id"] in selected_ids
    ]

    object_rows = score_object_system(filtered_key, entities, [])
    graph_rows = score_object_system(filtered_key, entities, edges)
    for row in object_rows:
        row["source"] = "object_retrieval"
    for row in graph_rows:
        row["source"] = "typed_graph"
    arms = {
        "object_retrieval": {"rows": object_rows, "summary": summarize(object_rows)},
        "typed_graph": {"rows": graph_rows, "summary": summarize(graph_rows)},
    }
    if direct_path is not None:
        direct = json.loads(direct_path.read_text())
        visual_rows = direct_rows(key, packet, direct)
        routed = hybrid_rows(packet, object_rows, graph_rows, visual_rows)
        arms["direct_vlm"] = {"rows": visual_rows, "summary": summarize(visual_rows)}
        arms["hybrid"] = {"rows": routed, "summary": summarize(routed)}

    report = {
        "schema": SCHEMA,
        "stage": "scored" if direct_path else "partial_pending_direct_vlm",
        "scene_id": key["scene_id"],
        "inputs": {
            "key": str(key_path), "key_sha256": sha256(key_path),
            "packet": str(packet_path), "packet_sha256": packet["packet_sha256"],
            "entities": entity_prov,
            "graph": graph_prov,
            "graph_geometry_compatibility": graph_geometry,
            "direct_responses": str(direct_path) if direct_path else None,
        },
        "question_scope": {
            "included": [q["id"] for q in packet["questions"]],
            "excluded": [
                q["id"] for q in key["independent_questions"]
                if q["id"] not in selected_ids
            ],
            "reason": ("room-observable questions only; delivered-UID and "
                       "system-output diagnostics excluded"),
        },
        "arms": arms,
        "decision": decision(arms),
        "limits": [
            "One human-reviewed scene is a screening test, not generalization.",
            "Object retrieval and graph arms share the same delivered instances and labels.",
            "System-output diagnostics such as reported-region extent are excluded from the common score.",
            "The direct arm sees selected RGB views and may miss objects outside them.",
            "No arm changes perception or the human key.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    return out


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--key", type=Path, required=True)
    p.add_argument("--scene-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-views", type=int, default=N_VIEWS_DEFAULT)
    s = sub.add_parser("score")
    s.add_argument("--key", type=Path, required=True)
    s.add_argument("--entities", type=Path, required=True)
    s.add_argument("--graph", type=Path, required=True)
    s.add_argument("--packet", type=Path, required=True)
    s.add_argument("--direct-responses", type=Path)
    s.add_argument("--out", type=Path, required=True)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "prepare":
        path = prepare_packet(args.key, args.scene_dir, args.out, args.n_views)
        print(f"prepared answer-free packet -> {path}")
        print(f"direct-VLM prompt             -> {args.out / 'prompt.txt'}")
        print(f"visual contact sheet          -> {args.out / 'contact_sheet.jpg'}")
        return 0
    path = run_score(args.key, args.entities, args.graph, args.packet,
                     args.direct_responses, args.out)
    report = json.loads(path.read_text())
    for name, arm in report["arms"].items():
        s = arm["summary"]
        print(f"{name:18s} correct={s['tally']['correct']} "
              f"wrong={s['tally']['wrong']} unknown={s['tally']['unanswered']} "
              f"coverage={s['coverage']:.3f} accuracy={s['exact_accuracy_all']:.3f}")
    print(f"decision: {report['decision']['status']} "
          f"proceed={report['decision']['proceed']}")
    print(f"report -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
