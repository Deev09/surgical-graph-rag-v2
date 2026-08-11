"""Score the delivered Mask3D + rgb_tight system against a human spatial-QA key.

  python3 tools/arkitscenes_spatial_qa_score.py --scene 41069025

Key: `eval/human_feedback/arkitscenes_41069025_spatial_qa_key_v1.json`, derived
solely from the owner's visual review.

EVALUATION ONLY. This tool loads finalized Lane A artifacts and reads them. It
runs no segmenter, no labeler and no relation stage, writes nothing back into
any bundle, and changes no perception. It also never opens an ARKitScenes
annotation: the ground truth here is the human review, which is a different and
independent source.

WHAT `unanswered` MEANS, AND WHY IT IS NOT `wrong`
--------------------------------------------------
The support-relation item asks what the cushions rest on. The delivered graph
emits `NEAR` edges only -- Checkpoint 2 recorded that the geometry-only floor
census is `unknown` in this scene, so no `ON_ENTITY_SURFACE` edge is promoted.
A system that cannot express an answer has not given a wrong one, and scoring
it as wrong would hide the difference between a mistake and a missing
capability. Both are reported, and the headline score reports them separately.

CARDINALITY IS COUNTED OVER ADMITTED DISPLAY LABELS, which is what the system
actually asserts about the room. Instances the labeler left anonymous are not
counted as anything; they are reported as unlabelled coverage so a low count
cannot be mistaken for a confident denial.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_KEY = (REPO_ROOT / "eval" / "human_feedback"
               / "arkitscenes_41069025_spatial_qa_key_v1.json")
DEFAULT_ENTITIES = (REPO_ROOT / "runs" / "arkit_label_image_ab_41069025"
                    / "rgb_tight" / "entities" / "manifest.json")
DEFAULT_GRAPH = (REPO_ROOT / "runs" / "arkit_vertical_slice" / "sealed_pair"
                 / "41069025" / "graph" / "manifest.json")
DEFAULT_OUT = REPO_ROOT / "runs" / "arkit_spatial_qa"


_ANONYMOUS = re.compile(r"^segment[_-]\d+$")


def is_anonymous(label: str | None) -> bool:
    """`segment_17` is the label stage's placeholder for "not admitted".

    It is a non-empty string, so a naive truthiness check counts it as a
    label -- which reported 35/35 labelled against a stage that admitted 27.
    """
    return not label or bool(_ANONYMOUS.match(label.strip().lower()))


def normalize(label: str | None) -> str | None:
    """Canonical hyphen form; the graph's own normalizer rule."""
    if is_anonymous(label):
        return None
    return label.strip().lower().replace("_", "-").replace(" ", "-")


# Classes the key asks about, and the delivered labels that count as them.
# `couch` is admitted for `sofa` because the review uses both words; no other
# synonymy is introduced.
SYNONYMS = {"sofa": {"sofa", "couch"}}


def matches(delivered: str | None, wanted: str) -> bool:
    return normalize(delivered) in SYNONYMS.get(wanted, {wanted})


def load_entities(path: Path) -> tuple[list[dict], dict]:
    manifest = json.loads(path.read_text())
    rows = []
    for entity in manifest["entities"]:
        identity = entity.get("identity", {})
        aabb = entity.get("bbox_aabb")
        rows.append({
            "uid": identity.get("object_uid"),
            "display_label": identity.get("display_label"),
            "aabb": aabb,
            "top_k": [h.get("label") for h in entity.get("semantic_hypotheses", [])],
        })
    provenance = {
        "entity_manifest": str(path),
        "entity_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bundle_hash": manifest.get("bundle_hash"),
        "n_entities": len(rows),
        "label_stage": manifest.get("notes", {}).get("label_stage", {}).get("name"),
        "image_source": manifest.get("notes", {}).get("label_stage", {})
                                .get("image_source"),
    }
    return rows, provenance


def load_graph(path: Path) -> tuple[list[dict], dict]:
    manifest = json.loads(path.read_text())
    edges = manifest.get("edges", [])
    kinds: dict[str, int] = {}
    for edge in edges:
        kinds[edge.get("type", "?")] = kinds.get(edge.get("type", "?"), 0) + 1
    return edges, {
        "graph_manifest": str(path),
        "graph_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bundle_hash": manifest.get("bundle_hash"),
        "entity_bundle_hash": manifest.get("entity_bundle_hash"),
        "n_edges": len(edges),
        "edge_types": kinds,
    }


def footprint(aabb) -> float:
    (x0, y0, _), (x1, y1, _) = aabb
    return abs(x1 - x0) * abs(y1 - y0)


def scene_footprint(entities: list[dict]) -> float:
    xs = [v for e in entities if e["aabb"] for v in (e["aabb"][0][0], e["aabb"][1][0])]
    ys = [v for e in entities if e["aabb"] for v in (e["aabb"][0][1], e["aabb"][1][1])]
    if not xs or not ys:
        return 0.0
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def score(key: dict, entities: list[dict], edges: list[dict]) -> list[dict]:
    labelled = [e for e in entities if not is_anonymous(e["display_label"])]
    results = []
    for item in key["independent_questions"]:
        kind = item["kind"]
        row = {"id": item["id"], "kind": kind, "question": item.get("question")}

        if kind == "class_cardinality":
            # Prefer the key's explicit class; the id-parsing fallback is for
            # v1/v2 items written before the field existed.
            wanted = item.get("class") or {
                "rug": "rug", "trash": "trash-can", "counter": "counter",
                "cushion": "cushion"}[item["id"].split("_")[1]]
            hits = [e for e in labelled if matches(e["display_label"], wanted)]
            row.update(expected=item["expected"], answer=len(hits),
                       matched_uids=[e["uid"] for e in hits],
                       outcome="correct" if len(hits) == item["expected"]
                       else "wrong")

        elif kind == "class_presence":
            wanted = "sofa" if "sofa" in item["id"] else "cushion"
            hits = [e for e in labelled if matches(e["display_label"], wanted)]
            row.update(expected=item["expected"], answer=bool(hits),
                       matched_uids=[e["uid"] for e in hits],
                       outcome="correct" if bool(hits) == item["expected"]
                       else "wrong")

        elif kind == "instance_identity":
            wanted = item["class"]
            got = sorted(e["uid"] for e in labelled
                         if matches(e["display_label"], wanted))
            expected_uids = sorted(item["expected_uids"])
            row.update(expected=expected_uids, answer=got,
                       spurious=sorted(set(got) - set(expected_uids)),
                       missed=sorted(set(expected_uids) - set(got)),
                       outcome="correct" if got == expected_uids else "wrong")

        elif kind == "support_relation":
            wanted_type = item["expected_relation"]
            present = [e for e in edges if e.get("type") == wanted_type]
            if not present:
                row.update(expected=f"{item['expected_subject_class']} "
                                    f"{wanted_type} {item['expected_object_class']}",
                           answer=None, outcome="unanswered",
                           reason=f"the delivered graph emits no {wanted_type} "
                                  "edge of any kind")
            else:
                by_uid = {e["uid"]: e["display_label"] for e in entities}
                hit = any(
                    matches(by_uid.get(e["source"]["uid"]),
                            item["expected_subject_class"])
                    and matches(by_uid.get(e["target"]["uid"]),
                                item["expected_object_class"])
                    for e in present)
                row.update(expected=f"{item['expected_subject_class']} "
                                    f"{wanted_type} {item['expected_object_class']}",
                           answer=hit,
                           outcome="correct" if hit else "wrong")

        elif kind == "instance_extent":
            limit = item["room_span_fraction"] * scene_footprint(entities)
            counters = [e for e in labelled
                        if matches(e["display_label"], "counter") and e["aabb"]]
            spanning = [{"uid": e["uid"],
                         "footprint_m2": round(footprint(e["aabb"]), 2)}
                        for e in counters if footprint(e["aabb"]) > limit]
            row.update(expected=item["expected"], answer=bool(spanning),
                       room_span_limit_m2=round(limit, 2),
                       counter_instances=[
                           {"uid": e["uid"],
                            "footprint_m2": round(footprint(e["aabb"]), 2)}
                           for e in counters],
                       room_spanning=spanning,
                       outcome="correct" if not spanning else "wrong")
        else:
            row.update(outcome="unanswered", reason=f"unknown item kind {kind}")
        results.append(row)
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", default="41069025")
    ap.add_argument("--key", type=Path, default=DEFAULT_KEY)
    ap.add_argument("--entities", type=Path, default=DEFAULT_ENTITIES)
    ap.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    key = json.loads(args.key.read_text())
    key_sha = hashlib.sha256(args.key.read_bytes()).hexdigest()
    entities, entity_prov = load_entities(args.entities)
    edges, graph_prov = load_graph(args.graph)
    results = score(key, entities, edges)

    diagnostics = []
    for item in key.get("delivery_diagnostics", []):
        hits = [e["uid"] for e in entities
                if not is_anonymous(e["display_label"])
                and matches(e["display_label"], item["class"])]
        diagnostics.append({
            "id": item["id"], "question": item["question"],
            "expected": item["expected"], "answer": len(hits),
            "matched_uids": hits, "scored": False,
            "agrees_with_expectation": len(hits) == item["expected"],
            "context": item.get("context"),
            "why_not_scored": item.get("why_not_scored"),
        })

    by_status: dict[str, int] = {}
    for mapping in key["unresolved_uid_mappings"]:
        by_status[mapping["status"]] = by_status.get(mapping["status"], 0) + 1

    tally = {"correct": 0, "wrong": 0, "unanswered": 0}
    for row in results:
        tally[row["outcome"]] += 1

    # Declared confounds: patterns where one underlying error scores as both a
    # pass and a fail, so the pass must not be read as a capability.
    outcomes = {row["id"]: row["outcome"] for row in results}
    triggered = []
    for confound in key.get("known_confounds", []):
        answers = {row["id"]: row.get("answer") for row in results}
        if confound["id"] == "sofa_cushion_coupling" and (
                outcomes.get("q4_sofa_present") == "wrong"
                and outcomes.get("q5_cushion_present") == "correct"):
            triggered.append(confound)
        if confound["id"] == "sofa_counted_as_cushion" and (
                outcomes.get("q4_sofa_present") == "wrong"
                and outcomes.get("q5_cushion_cardinality") == "wrong"):
            expected = next(q["expected"] for q in key["independent_questions"]
                            if q["id"] == "q5_cushion_cardinality")
            if answers.get("q5_cushion_cardinality") == expected + 1:
                triggered.append(confound)
    labelled = [e for e in entities if not is_anonymous(e["display_label"])]

    report = {
        "evaluation": "human_spatial_qa",
        "evaluation_only": True,
        "perception_changed": False,
        "scene_id": key["scene_id"],
        "system": "mask3d_delivered + rgb_tight openclip labels",
        "key": str(args.key),
        "key_sha256": key_sha,
        "key_status": key["status"],
        "entity_provenance": entity_prov,
        "graph_provenance": graph_prov,
        "coverage": {
            "n_entities": len(entities),
            "n_with_display_label": len(labelled),
            "n_anonymous": len(entities) - len(labelled),
            "note": "anonymous instances assert nothing and are counted as "
                    "nothing; a low class count is not a confident denial",
        },
        "tally": tally,
        "delivery_diagnostics": diagnostics,
        "confounds_triggered": triggered,
        "score_excluding_unanswered": (
            round(tally["correct"] / (tally["correct"] + tally["wrong"]), 4)
            if tally["correct"] + tally["wrong"] else None),
        "score_counting_unanswered_as_failure": round(
            tally["correct"] / len(results), 4) if results else None,
        "uid_mapping_status": by_status,
        "unresolved_uid_mappings": key["unresolved_uid_mappings"],
        "results": results,
        "interpretation_limit": key["interpretation_limit"],
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    path = args.out_root / f"{key['scene_id']}_human_spatial_qa.json"
    path.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")

    print(f"=== {key['scene_id']}   {args.key.stem}  ({key['status']})")
    print(f"    system   : {report['system']}")
    print(f"    entities : {len(entities)} delivered, {len(labelled)} labelled, "
          f"{len(entities) - len(labelled)} anonymous")
    print(f"    graph    : {graph_prov['n_edges']} edges {graph_prov['edge_types']}")
    print()
    for row in results:
        mark = {"correct": "ok  ", "wrong": "WRONG", "unanswered": "n/a "}[row["outcome"]]
        print(f"    [{mark}] {row['id']:32s} expected={row.get('expected')!s:>8s}  "
              f"answer={row.get('answer')!s}")
        if row.get("reason"):
            print(f"             {row['reason']}")
        if row.get("matched_uids"):
            print(f"             uids: {row['matched_uids']}")
        if row.get("room_spanning"):
            print(f"             room-spanning: {row['room_spanning']}")
    for d in diagnostics:
        mark = "agrees" if d["agrees_with_expectation"] else "disagrees"
        print(f"    [diag ] {d['id']:32s} expected={d['expected']}  "
              f"answer={d['answer']}  ({mark}; not tallied)")
        if d.get("context"):
            print(f"             {d['context']}")
    print()
    print(f"    correct {tally['correct']}, wrong {tally['wrong']}, "
          f"unanswered {tally['unanswered']}  of {len(results)}")
    print(f"    score excluding unanswered            : "
          f"{report['score_excluding_unanswered']}")
    print(f"    score counting unanswered as failure  : "
          f"{report['score_counting_unanswered_as_failure']}")
    for confound in triggered:
        print(f"    CONFOUND {confound['id']}: {confound['description']}")
    print("    UID mappings: " + ", ".join(
        f"{v} {k.lower()}" for k, v in sorted(by_status.items())))
    print(f"    report -> {path}   ({time.perf_counter() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
