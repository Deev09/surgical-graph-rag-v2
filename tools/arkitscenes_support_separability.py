"""Are the owner's support positives separable from the negatives?

  python3 tools/arkitscenes_support_separability.py --scene 41069025

Read-only. Reports precision/recall of the current support stage against the
finalized human key, classifies each positive by the failure that hides it, and
tests whether any axis-aligned threshold on (footprint overlap, |vertical gap|)
separates positives from negatives.

**It changes no threshold and no support logic.** That is the point: the answer
to "should the 0.50 gate move" is not "the owner found a positive at 0.4545",
it is "does a boundary exist that admits the reachable positives and admits no
negative, and is it supported by enough data to be worth trusting".

THREE FAILURE MODES, WHICH NEED DIFFERENT FIXES
------------------------------------------------
  found                 the stage already proposes it
  threshold_reachable   a patch exists at the right place -- small |gap| -- and
                        the pair fails only a numeric gate. A threshold change
                        could recover this one.
  evidence_missing      NO patch exists near the contact height at all. The
                        supporting surface was never extracted, so no threshold
                        on any of these features can recover it. Counting this
                        as a threshold failure would send the next change to the
                        wrong stage.

PATCH SELECTION IS SCORED SEPARATELY FROM THE GATES
----------------------------------------------------
The stage picks the patch with the highest footprint overlap. That is not
necessarily the patch the target rests ON: a shelf unit's top face can overlap a
target more than the shelf it actually sits on. Both selections are reported --
`by_overlap` (what the stage does) and `by_contact` (smallest |gap|) -- because
if they disagree on a positive, the bug is in selection, not in a gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_KEY = (REPO_ROOT / "eval" / "human_feedback"
               / "arkitscenes_41069025_support_relation_key_v1.json")
DEFAULT_REST = (REPO_ROOT / "runs" / "arkit_vertical_slice" / "sealed_pair_rest"
                / "41069025" / "entity_patch_rest.json")
DEFAULT_OUT = REPO_ROOT / "runs" / "arkit_support_calibration"

# A patch is "at the contact height" when the target's underside sits within
# this of it. Generous on purpose: the question is whether ANY usable patch
# exists, not whether it passes a gate.
CONTACT_BAND_M = 0.05


def patch_views(pair: dict) -> dict:
    """Best patch under two selection rules, plus the contact-band census."""
    overlapping = [p for p in pair["evaluated_patches"]
                   if p["overlap_ratio_target"] > 0
                   and p["vertical_gap_at_overlap_centroid_m"] is not None]
    if not overlapping:
        return {"by_overlap": None, "by_contact": None,
                "n_patches": len(pair["evaluated_patches"]),
                "n_in_contact_band": 0}
    by_overlap = max(overlapping, key=lambda p: p["overlap_ratio_target"])
    by_contact = min(overlapping,
                     key=lambda p: abs(p["vertical_gap_at_overlap_centroid_m"]))
    in_band = [p for p in overlapping
               if abs(p["vertical_gap_at_overlap_centroid_m"]) <= CONTACT_BAND_M]
    return {
        "by_overlap": {"patch_uid": by_overlap["patch_uid"],
                       "overlap": round(by_overlap["overlap_ratio_target"], 4),
                       "gap_m": round(
                           by_overlap["vertical_gap_at_overlap_centroid_m"], 4)},
        "by_contact": {"patch_uid": by_contact["patch_uid"],
                       "overlap": round(by_contact["overlap_ratio_target"], 4),
                       "gap_m": round(
                           by_contact["vertical_gap_at_overlap_centroid_m"], 4)},
        "n_patches": len(pair["evaluated_patches"]),
        "n_in_contact_band": len(in_band),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", default="41069025")
    ap.add_argument("--key", type=Path, default=DEFAULT_KEY)
    ap.add_argument("--rest", type=Path, default=DEFAULT_REST)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    key = json.loads(args.key.read_text())
    if key["status"] != "FINAL":
        raise ValueError(
            f"key status is {key['status']}; separability must not be computed "
            "against a key still under re-check")
    artifact = json.loads(args.rest.read_text())
    pairs = {f'{p["target_entity_uid"]}->{p["owner_entity_uid"]}': p
             for p in artifact["pairs"]}

    truth = {r["pair_id"]: r["judgement"] for r in key["human_relation_truth"]}
    positives = [p for p, j in truth.items() if j == "supports"]
    negatives = [p for p, j in truth.items() if j == "does_not_support"]

    rows = {}
    for pair_id in positives + negatives:
        rows[pair_id] = patch_views(pairs[pair_id])

    candidates = {p for p in truth if pairs[p]["relation_candidate"]}
    tp = sorted(candidates & set(positives))
    fp = sorted(candidates & set(negatives))
    fn = sorted(set(positives) - candidates)
    precision = len(tp) / len(candidates) if candidates else None
    recall = len(tp) / len(positives) if positives else None

    classified = []
    for pair_id in positives:
        view = rows[pair_id]
        if pair_id in candidates:
            mode = "found"
        elif view["n_in_contact_band"] > 0:
            mode = "threshold_reachable"
        else:
            mode = "evidence_missing"
        classified.append({"pair_id": pair_id, "mode": mode, **view})

    reachable = [c for c in classified if c["mode"] in
                 ("found", "threshold_reachable")]
    unreachable = [c for c in classified if c["mode"] == "evidence_missing"]

    # Separability on the CONTACT-selected patch, over reachable positives.
    def contact(pair_id):
        return rows[pair_id]["by_contact"]

    pos_pts = [(contact(c["pair_id"])["overlap"],
                abs(contact(c["pair_id"])["gap_m"])) for c in reachable]
    neg_pts = [(contact(n)["overlap"], abs(contact(n)["gap_m"]))
               for n in negatives if contact(n)]

    min_pos_overlap = min(o for o, _ in pos_pts) if pos_pts else None
    max_pos_gap = max(g for _, g in pos_pts) if pos_pts else None
    intruders = [n for n, (o, g) in zip(
        [n for n in negatives if contact(n)], neg_pts)
        if o >= min_pos_overlap and g <= max_pos_gap] if pos_pts else []

    # Overlap alone, for contrast with the two-feature rule.
    overlap_only_intruders = [
        n for n, (o, _) in zip([n for n in negatives if contact(n)], neg_pts)
        if o >= min_pos_overlap] if pos_pts else []

    report = {
        "analysis": "support_threshold_separability",
        "read_only": True,
        "logic_changed": False,
        "thresholds_changed": False,
        "scene_id": key["scene_id"],
        "key": str(args.key.relative_to(REPO_ROOT)),
        "key_status": key["status"],
        "current_stage": {
            "n_candidates": len(candidates),
            "true_positives": tp, "false_positives": fp,
            "false_negatives": fn,
            "precision": precision, "recall": recall,
        },
        "positives_by_failure_mode": classified,
        "separability": {
            "feature_space": "(footprint overlap, |vertical gap|) on the "
                             "contact-selected patch",
            "n_reachable_positives": len(reachable),
            "n_unreachable_positives": len(unreachable),
            "min_positive_overlap": min_pos_overlap,
            "max_positive_abs_gap": max_pos_gap,
            "negatives_inside_that_box": sorted(intruders),
            "separable_in_2d": bool(pos_pts) and not intruders,
            "negatives_above_min_positive_overlap": len(overlap_only_intruders),
            "separable_on_overlap_alone": bool(pos_pts)
                                          and not overlap_only_intruders,
        },
        "patch_selection": {
            "stage_rule": "highest footprint overlap",
            "disagrees_with_contact_on": sorted(
                c["pair_id"] for c in classified
                if c["by_overlap"] and c["by_contact"]
                and c["by_overlap"]["patch_uid"] != c["by_contact"]["patch_uid"]),
        },
        "caution": (
            f"{len(positives)} positives from one scene. A boundary that "
            "separates them is a hypothesis, not a calibration; a gate fitted "
            "to this would be fitted to two or three points."),
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    path = args.out_root / f"{key['scene_id']}_support_separability.json"
    path.write_text(json.dumps(report, indent=1) + "\n")

    print(f"=== {key['scene_id']} support separability   (read-only)")
    print(f"    current stage : {len(candidates)} candidate(s)  "
          f"precision {precision}  recall {recall:.3f}")
    print(f"                    TP {tp}  FP {fp}")
    print(f"                    FN {fn}")
    print("    positives by failure mode:")
    for c in classified:
        best = c["by_contact"]
        detail = (f"overlap {best['overlap']}, gap {best['gap_m']:+.4f}, "
                  f"{c['n_in_contact_band']}/{c['n_patches']} patches in the "
                  f"±{CONTACT_BAND_M} m contact band") if best else "no patch"
        print(f"      {c['pair_id']:18s} {c['mode']:20s} {detail}")
    sep = report["separability"]
    print(f"    reachable positives      : {sep['n_reachable_positives']} "
          f"(unreachable {sep['n_unreachable_positives']})")
    print(f"    overlap alone separates? : {sep['separable_on_overlap_alone']}"
          f"  ({sep['negatives_above_min_positive_overlap']} negatives at or "
          f"above overlap {sep['min_positive_overlap']})")
    print(f"    overlap + gap separates? : {sep['separable_in_2d']}"
          + (f"  intruders {sep['negatives_inside_that_box']}"
             if sep["negatives_inside_that_box"] else
             f"  (box: overlap >= {sep['min_positive_overlap']}, "
             f"|gap| <= {sep['max_positive_abs_gap']})"))
    if report["patch_selection"]["disagrees_with_contact_on"]:
        print("    patch selection disagrees with contact on: "
              + ", ".join(report["patch_selection"]["disagrees_with_contact_on"]))
    print(f"    CAUTION: {report['caution']}")
    print(f"    -> {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
