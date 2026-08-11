"""Ingest owner support-relation judgements; flag, never silently fix.

  python3 tools/arkitscenes_support_truth_ingest.py --scene 41069025 \
      --truth eval/human_feedback/arkitscenes_41069025_support_truth_returned.json

Records what the owner said, reports coverage against the sheet that was sent,
and runs stated geometric consistency checks whose only output is a FLAG for
re-check. It changes no support logic and no threshold, and it never edits a
judgement.

WHY FLAG RATHER THAN DROP OR ACCEPT
-----------------------------------
A support key is calibration data. A wrong positive in it would be fitted to
later, so it cannot be waved through; but "the geometry looks odd to me" is not
grounds for deleting what a human reported, and quietly discarding judgements
would make the key a record of the code's opinion. Both failure modes are
avoided by keeping every judgement verbatim and attaching a flag with the
criterion that produced it.

THE TWO CHECKS, AND WHAT THEY DELIBERATELY DO NOT ASSUME
---------------------------------------------------------
Both are about the DIRECTION of the claim, not the size of a gap:

  floor_standing      the target's underside sits within FLOOR_MARGIN_M of the
                      lowest point in the scene. Something standing on the
                      floor is not resting on an elevated object.
  target_below_owner  the target's underside is below the owner's underside, so
                      the target extends beneath the thing it supposedly rests
                      on. Usually the pair is right and the direction is
                      reversed.

Explicitly NOT checked: whether the target sits above the owner's TOP. A book
on a middle shelf has its underside far below the bookcase's top and is a
perfectly good resting contact, so that test would flag exactly the cases the
support stage exists to find.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.arkitscenes_support_review_sheet import (
    DEFINITION, OWNER_CONFIRMED,
)

DEFAULT_SHEET = (REPO_ROOT / "eval" / "human_feedback"
                 / "arkitscenes_41069025_support_review.json")
DEFAULT_ENTITIES = (REPO_ROOT / "runs" / "arkit_label_image_ab_41069025"
                    / "rgb_tight" / "entities" / "manifest.json")
DEFAULT_OUT = REPO_ROOT / "eval" / "human_feedback"

FLOOR_MARGIN_M = 0.06


def geometry(entities_path: Path) -> tuple[dict, float]:
    manifest = json.loads(entities_path.read_text())
    boxes = {e["identity"]["object_uid"]: e["bbox_aabb"]
             for e in manifest["entities"]}
    floor_z = min(b[0][2] for b in boxes.values())
    return boxes, floor_z


def consistency_flags(target: str, owner: str, boxes: dict,
                      floor_z: float) -> list[dict]:
    flags = []
    t_low = boxes[target][0][2]
    o_low = boxes[owner][0][2]
    if t_low - floor_z <= FLOOR_MARGIN_M:
        flags.append({
            "flag": "floor_standing_target",
            "detail": f"target underside is {t_low - floor_z:.3f} m above the "
                      f"lowest scene point (<= {FLOOR_MARGIN_M} m), so it "
                      "appears to stand on the floor",
        })
    if t_low < o_low - 1e-6:
        flags.append({
            "flag": "target_below_owner",
            "detail": f"target underside {t_low - floor_z:.3f} m is below the "
                      f"owner underside {o_low - floor_z:.3f} m, so the target "
                      "extends beneath what it is said to rest on; the pair "
                      "may be right with the direction reversed",
        })
    return flags


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", default="41069025")
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--sheet", type=Path, default=DEFAULT_SHEET)
    ap.add_argument("--corrections", type=Path, default=None,
                    help="owner re-check that overrides returned judgements; "
                         "each correction is recorded with its rationale")
    ap.add_argument("--entities", type=Path, default=DEFAULT_ENTITIES)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    scene_id = f"arkitscenes_{args.scene}"
    returned = json.loads(args.truth.read_text())
    if returned.get("definition") != DEFINITION:
        raise ValueError(
            f"returned truth uses definition {returned.get('definition')!r}, "
            f"the sheet asked about {DEFINITION!r}")
    sheet = json.loads(args.sheet.read_text())
    evidence = {e["pair_id"]: e for e in sheet["system_candidate_evidence"]}
    boxes, floor_z = geometry(args.entities)

    judged = {row["pair_id"]: row["judgement"]
              for row in returned["human_relation_truth"]}
    corrections: dict[str, dict] = {}
    if args.corrections is not None:
        payload = json.loads(args.corrections.read_text())
        if payload.get("definition") != DEFINITION:
            raise ValueError("corrections use a different definition")
        for entry in payload["corrections"]:
            corrections[entry["pair_id"]] = entry
    unknown = sorted(set(judged) - set(evidence))
    if unknown:
        raise ValueError(f"judgements for pairs not on the sheet: {unknown}")

    prior = {f"{t}->{o}" for t, o in OWNER_CONFIRMED}
    omitted = sorted(set(evidence) - set(judged))

    records = []
    for pair_id, entry in evidence.items():
        if pair_id in corrections:
            # An explicit re-check outranks both the sheet and the prior step.
            # The superseded value is kept so the key shows what changed.
            judgement = corrections[pair_id]["judgement"]
            source = "owner_correction"
        elif pair_id in judged:
            judgement, source = judged[pair_id], "owner_review_sheet"
        elif pair_id in prior:
            # Pre-confirmed in the UID step and simply not re-ticked. The form
            # only emits checked rows, so silence here is the sheet's fault,
            # not a retraction.
            judgement, source = "supports", "owner_confirmed_prior_step"
        else:
            judgement, source = None, None
        record = {"pair_id": pair_id, "judgement": judgement, "source": source}
        if pair_id in corrections:
            record["rationale"] = corrections[pair_id].get("rationale")
            was = judged.get(pair_id, "supports" if pair_id in prior else None)
            if was is not None and was != judgement:
                record["superseded"] = was
        if judgement == "supports":
            record["flags"] = consistency_flags(
                entry["target_uid"], entry["owner_uid"], boxes, floor_z)
        records.append(record)

    positives = [r for r in records if r["judgement"] == "supports"]
    flagged = [r for r in positives if r.get("flags")]
    clean = [r for r in positives if not r.get("flags")]
    counts: dict[str, int] = {}
    for r in records:
        counts[str(r["judgement"])] = counts.get(str(r["judgement"]), 0) + 1

    # Algorithm agreement, reported only. No threshold is touched.
    candidates = {p for p, e in evidence.items() if e["relation_candidate"]}
    positive_ids = {r["pair_id"] for r in positives}
    agreement = {
        "n_algorithm_candidates": len(candidates),
        "n_owner_positives": len(positive_ids),
        "candidates_owner_confirms": sorted(candidates & positive_ids),
        "candidates_owner_rejects": sorted(
            c for c in candidates
            if judged.get(c) == "does_not_support"),
        "owner_positives_algorithm_misses": sorted(positive_ids - candidates),
        "note": "reported for the record; no logic or threshold changed here",
    }

    key = {
        "schema": "arkitscenes_support_relation_key_v1",
        "scene_id": scene_id,
        "definition": DEFINITION,
        # Once every flagged positive has been re-checked, the key is final.
        # Omissions no longer block it: they were resolved by correction.
        "status": ("PENDING_RECHECK"
                   if [r for r in flagged
                       if r["pair_id"] not in corrections] else "FINAL"),
        "coverage": {
            "rows_on_sheet": len(evidence),
            "rows_returned": len(judged),
            "rows_omitted": omitted,
            "omitted_reason": (
                "the sheet's Build JSON only emits rows with a checked radio, "
                "and the pre-confirmed pairs were highlighted with the note "
                "'leave or change them as you see fit'. Omission here is a "
                "form defect, not a retraction, so the prior confirmation is "
                "carried forward and marked with its own source."),
        },
        "judgement_counts": counts,
        "needs_owner_recheck": [
            {"pair_id": r["pair_id"], "flags": r["flags"]} for r in flagged],
        "human_relation_truth": records,
        "algorithm_agreement": agreement,
        "consistency_checks": {
            "floor_margin_m": FLOOR_MARGIN_M,
            "checks": ["floor_standing_target", "target_below_owner"],
            "not_checked": (
                "whether the target sits above the owner's TOP -- a book on a "
                "middle shelf legitimately does not"),
            "effect": "flag only; no judgement was altered or dropped",
        },
        "corrections_applied": [
            {"pair_id": p, "judgement": c["judgement"],
             "rationale": c.get("rationale")}
            for p, c in sorted(corrections.items())],
        "provenance": {
            "returned_truth_sha256": hashlib.sha256(
                args.truth.read_bytes()).hexdigest(),
            "corrections_sha256": (
                hashlib.sha256(args.corrections.read_bytes()).hexdigest()
                if args.corrections else None),
            "sheet_sha256": hashlib.sha256(args.sheet.read_bytes()).hexdigest(),
            "logic_changed": False,
            "thresholds_changed": False,
        },
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    path = args.out_root / f"{scene_id}_support_relation_key_v1.json"
    path.write_text(json.dumps(key, indent=1) + "\n")

    print(f"=== {scene_id} support-relation key  ({key['status']})")
    print(f"    returned          : {len(judged)}/{len(evidence)} rows"
          + (f"; omitted {omitted}" if omitted else ""))
    print("    judgements        : " + ", ".join(
        f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"    positives         : {len(positives)} "
          f"({len(clean)} clean, {len(flagged)} flagged for re-check)")
    for r in flagged:
        print(f"      ? {r['pair_id']:20s} "
              + "; ".join(f["flag"] for f in r["flags"]))
    print(f"    algorithm         : {agreement['n_algorithm_candidates']} "
          f"candidate(s); confirms {agreement['candidates_owner_confirms']}, "
          f"misses {len(agreement['owner_positives_algorithm_misses'])} owner "
          "positive(s)")
    print(f"    logic/thresholds  : unchanged")
    print(f"    -> {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
