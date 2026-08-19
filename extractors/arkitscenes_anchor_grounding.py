"""Oracle-free language-to-entity grounding: anchor phrase -> delivered uid.

Protocol: docs/arkitscenes_grounding_bridge.md, whose rules were declared
before any prediction was made.

This module replaces exactly ONE stage of the deployable path -- how a
natural-language anchor becomes an object uid -- and changes nothing else. It
reads delivered geometry and real capture frames, and it emits a uid or an
abstention. Instances, segmentation, graph nodes, edges and relation
thresholds are untouched and unread.

ORACLE-FREE, ENFORCED
---------------------
It imports no human key, no uid mapping, no annotation box, no oracle label
and no evaluation module. Tests AST-check the module source and its transitive
first-party imports, because a docstring promise is not enforcement: the whole
value of the measurement is that this stage never saw the answer.

WHY IT RE-ENCODES INSTEAD OF READING A VECTOR
---------------------------------------------
`embedding_ref` is `None` on every delivered entity in both scenes; only the
top-3 semantic hypotheses persist from the label stage. There is no stored
image embedding to reuse, so crops are re-encoded here with the same pinned
weights, prompts and view-selection the label stage used.

THE ADMISSION RULE IS CROSS-VIEW AGREEMENT, NOT A THRESHOLD
------------------------------------------------------------
An anchor is admitted only when the entity with the best aggregate score is
ALSO the best entity in at least two independent view slots. No confidence
cutoff is applied and none is swept, so there is no constant here that could
be quietly fitted to the result.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from extractors.arkitscenes_rgb_crops import RgbCropSource
from segmenter.clip_labeler import MODEL_NAME, PRETRAINED, PROMPTS, ClipLabeler

SCHEMA = "arkitscenes_anchor_grounding_v1"

# The exact rgb_tight configuration the label stage used
# (`arkitscenes_rgb_crop_pad0.15_mark0`). Not tunable here.
CROP_STRIDE = 6
CROP_N_VIEWS = 3
CROP_CONTEXT_PAD = 0.15
CROP_MARK_TARGET = False

MIN_AGREEING_SLOTS = 2      # the admission rule; see the protocol


def _phrases(anchor: str, synonyms: dict) -> list[str]:
    """Anchor name plus its declared synonyms, deduplicated, order-stable."""
    out, seen = [], set()
    for phrase in [anchor, *synonyms.get(anchor, [])]:
        key = phrase.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(phrase)
    return out


def encode_entities(scene_dir: Path, xyz_canonical, rotation,
                    entities: list[dict], labeler: ClipLabeler) -> dict:
    """uid -> per-view L2-normalised image embeddings, best view first."""
    crops = RgbCropSource(scene_dir, xyz_canonical, rotation,
                          stride=CROP_STRIDE, n_views=CROP_N_VIEWS,
                          context_pad=CROP_CONTEXT_PAD,
                          mark_target=CROP_MARK_TARGET)
    by_uid = {}
    for entity in entities:
        images = crops.crops_for(entity["vertices"])
        if not images:
            by_uid[entity["uid"]] = None      # no usable view: cannot compete
            continue
        by_uid[entity["uid"]] = labeler.image_embeddings(images).numpy()
    return by_uid


def score_anchor(anchor: str, synonyms: dict, embeddings: dict,
                 labeler: ClipLabeler) -> dict:
    """Rank every entity for one anchor, and apply the cross-view rule.

    Returns the full evidence, admitted or not, so a human can inspect why an
    anchor resolved or abstained without rerunning anything.
    """
    phrases = _phrases(anchor, synonyms)
    text = labeler.text_embeddings(tuple(phrases)).numpy()

    per_slot: dict[int, dict[str, float]] = {}
    aggregate: dict[str, float] = {}
    per_uid_slots: dict[str, list[float]] = {}
    for uid, image in sorted(embeddings.items()):
        if image is None:
            continue
        # max over the anchor's phrases; synonyms are alternative names for one
        # object, so any single match is evidence for that object.
        slot_scores = (image @ text.T).max(axis=1)
        per_uid_slots[uid] = [round(float(v), 6) for v in slot_scores]
        for k, value in enumerate(slot_scores):
            per_slot.setdefault(k, {})[uid] = float(value)
        aggregate[uid] = float(slot_scores.mean())

    if not aggregate:
        return {"anchor": anchor, "phrases": phrases, "admitted": False,
                "uid": None, "reason": "no delivered entity has a usable crop",
                "ranking": [], "slot_winners": {}, "agreeing_slots": 0}

    # ties broken by uid ascending, so the output is fully deterministic
    ranking = sorted(aggregate.items(), key=lambda kv: (-kv[1], kv[0]))
    top_uid = ranking[0][0]
    slot_winners = {
        str(k): min((u for u, v in scores.items()
                     if v == max(scores.values())), default=None)
        for k, scores in sorted(per_slot.items())
    }
    agreeing = [k for k, winner in slot_winners.items() if winner == top_uid]

    admitted = len(agreeing) >= MIN_AGREEING_SLOTS
    return {
        "anchor": anchor,
        "phrases": phrases,
        "admitted": admitted,
        "uid": top_uid if admitted else None,
        "reason": (None if admitted else
                   f"top entity {top_uid} won only {len(agreeing)} view slot(s); "
                   f"the rule requires {MIN_AGREEING_SLOTS}"),
        "top_uid": top_uid,
        "agreeing_slots": len(agreeing),
        "agreeing_slot_ids": sorted(agreeing),
        "slot_winners": slot_winners,
        "ranking": [{"uid": u, "aggregate": round(v, 6),
                     "slot_scores": per_uid_slots.get(u, [])}
                    for u, v in ranking[:5]],
        "n_entities_ranked": len(ranking),
    }


def ground_scene(scene_id: str, anchors: list[str], synonyms: dict,
                 scene_dir: Path, xyz_canonical, rotation,
                 entities: list[dict], labeler: ClipLabeler) -> dict:
    embeddings = encode_entities(scene_dir, xyz_canonical, rotation,
                                 entities, labeler)
    rows = [score_anchor(a, synonyms, embeddings, labeler) for a in anchors]
    return {
        "scene_id": scene_id,
        "n_entities": len(entities),
        "n_entities_with_crops": sum(1 for v in embeddings.values() if v is not None),
        "anchors": rows,
        "admitted": {r["anchor"]: r["uid"] for r in rows if r["admitted"]},
    }


def sidecar(scenes: list[dict], provenance: dict) -> dict:
    body = {
        "schema": SCHEMA,
        "stage": "oracle_free_prediction",
        "rules": {
            "phrase_set": "anchor name plus its declared synonyms from the "
                          "frozen question manifest",
            "per_slot_score": "max over phrases of cosine(image, text)",
            "aggregate": "mean over the entity's available view slots",
            "ranking": "aggregate descending, ties by uid ascending",
            "admission": f"top aggregate entity must also win at least "
                         f"{MIN_AGREEING_SLOTS} independent view slots",
            "threshold": "none; no confidence cutoff is applied or swept",
        },
        "model": {"name": MODEL_NAME, "pretrained": PRETRAINED,
                  "prompts": list(PROMPTS)},
        "crop_config": {"stride": CROP_STRIDE, "n_views": CROP_N_VIEWS,
                        "context_pad": CROP_CONTEXT_PAD,
                        "mark_target": CROP_MARK_TARGET,
                        "equals_label_stage_source":
                            "arkitscenes_rgb_crop_pad0.15_mark0"},
        "provenance": provenance,
        "scenes": scenes,
        "disclosures": [
            "No human key, uid mapping, annotation box or oracle label was read.",
            "embedding_ref is None on every delivered entity; crops were "
            "re-encoded with the pinned weights rather than reusing a vector.",
            "Absent objects are not special-cased: if the cross-view rule "
            "admits a uid for an object that was never delivered, that is a "
            "precision error and is reported as one.",
        ],
    }
    # Hash the PREDICTION, not the run. Wall-clock and any other volatile
    # provenance are excluded: a hash that moves when nothing was predicted
    # differently cannot pin a prediction, which is the one thing it is for.
    volatile = {"runtime_s"}
    hashable = dict(body)
    hashable["provenance"] = {k: v for k, v in body["provenance"].items()
                              if k not in volatile}
    body["prediction_sha256"] = hashlib.sha256(
        json.dumps(hashable, sort_keys=True,
                   separators=(",", ":")).encode()).hexdigest()
    body["prediction_hash_excludes"] = sorted(volatile)
    return body
