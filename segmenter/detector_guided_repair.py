"""Detector-guided repair: Grounding DINO boxes prompt SAM 2.1, labels guide association.

Design note: `docs/repair_arm_design_note.md` (Checkpoint G). Pin:
`docs/grounding_dino_pin.json`.

WHY THIS ARM EXISTS
-------------------
The automatic-mask arm failed with a specific signature: SAM's class-agnostic
masks are object PARTS. Precision against annotated entities was 0.82-1.00 while
recall was 0.08-0.31, and the oracle-guided ceiling showed no union of parts
reaching IoU 0.50 for any missed entity (Checkpoint E), including after
topology-only splitting (Checkpoint F).

Nothing in that diagnosis says SAM cannot outline a whole object. It says
nothing *told SAM which object to outline*. A point grid seeds parts because a
point on a cabinet door is a valid prompt for the door. A detector box says
"the cabinet is here", which is the missing instruction.

So the single change under test is the PROMPT. Same SAM 2.1 checkpoint, same 32
frames, same lifting, same evaluator, same gates.

WHAT IS STRUCTURAL HERE
-----------------------
  * **No background or complement proposal, ever.** Only detected boxes produce
    masks. Pixels in no box contribute nothing; there is no "everything else"
    region, no inversion, no residual. A test asserts the module contains no
    complement operation.
  * **Every mask keeps its provenance** -- detector label, detector score, box,
    frame -- from the sidecar through association into the emitted proposal's
    notes, so any recovery can be attributed to a specific detection in a
    specific photograph.
  * **Association requires label compatibility AND 3D overlap.** Each detection
    carries the SET of vocabulary classes its phrase could name, and two masks
    join only if those sets intersect and they occupy the same surface. A
    `{armchair, chair}` detection joins a `{chair}` one; `{sofa}` and
    `{cushion}` never join. This is the one thing the class-agnostic arm could
    not do.

The vocabulary is `extractors.learned_labels.GLOBAL_INDOOR_VOCABULARY_V1`,
imported and unmodified.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from extractors.learned_labels import GLOBAL_INDOOR_VOCABULARY_V1
from segmenter.rgb_multiview_repair import RepairProposal, visible_vertices
from segmenter.sam_multiview_repair import (
    ASSOC_CONTAINMENT, ASSOC_CONTAINMENT_MIN_SIZE_RATIO, ASSOC_IOU,
    ASSOC_ANCHOR_STRIDE, LiftedMask, _anchor_matrix, _pairwise_intersections,
    classify_clusters, fuse_cluster, lift_masks,
)

VOCABULARY: tuple[str, ...] = GLOBAL_INDOOR_VOCABULARY_V1

# Association additionally requires the detector's candidate label SETS to
# intersect. The geometric thresholds are IMPORTED from the class-agnostic arm
# rather than re-declared, so the two arms differ only in the prompt and in
# this rule.
REQUIRE_SAME_LABEL = True

_VOCAB_INDEX = {name: i for i, name in enumerate(VOCABULARY)}


def _as_sets(labels) -> list[frozenset[str]]:
    """Accept a bare label or a candidate set; normalise to sets."""
    out = []
    for entry in labels:
        if isinstance(entry, str):
            out.append(frozenset({entry}))
        else:
            out.append(frozenset(entry))
    return out


def _label_bits(candidates: frozenset[str]) -> int:
    """Candidate set as a bitmask over the 41-class vocabulary."""
    bits = 0
    for name in candidates:
        index = _VOCAB_INDEX.get(name)
        if index is not None:
            bits |= 1 << index
    return bits


def prompt_text(vocabulary: tuple[str, ...] = VOCABULARY) -> str:
    """Grounding DINO text prompt for the fixed vocabulary.

    Hyphens become spaces in the PROMPT only -- 'trash-can' is not a phrase a
    grounding model has seen, 'trash can' is. The canonical hyphen form is what
    gets stored and matched, so this is formatting, not a vocabulary change.
    """
    return " . ".join(name.replace("-", " ") for name in vocabulary) + " ."


def labels_from_prompt_phrase(phrase: str,
                              vocabulary: tuple[str, ...] = VOCABULARY
                              ) -> frozenset[str]:
    """Map a detector phrase to the SET of vocabulary classes it could name.

    Grounding DINO returns the span of the prompt it matched. Three shapes
    occur, and all three are model evidence:

      exact        'sofa'            -> {sofa}
      fragment     'trash'           -> {trash-can}
      concatenated 'armchair chair'  -> {armchair, chair}

    The concatenated case is the detector matching a span that runs across two
    adjacent prompt entries. It is NOT failed inference: the model is saying
    "this is an armchair or a chair", which is a real and useful constraint.
    An earlier version of this function returned a single label or None, and
    so discarded 13 of 127 detections on the development scene -- 10 of them
    `armchair chair`. Returning a candidate set keeps that evidence and lets
    association decide, which is what a set intersection is for.

    A phrase naming nothing in the vocabulary still returns the empty set and
    is still rejected. Ambiguity between known classes is evidence; an unknown
    word is not.
    """
    cleaned = " ".join(phrase.lower().replace("-", " ").split())
    if not cleaned:
        return frozenset()
    canonical = {name.replace("-", " "): name for name in vocabulary}
    if cleaned in canonical:
        return frozenset({canonical[cleaned]})

    # Greedily consume the phrase as a sequence of vocabulary entries, longest
    # match first, so 'trash can table' resolves to {trash-can, table} rather
    # than to {trash-can} plus a dangling word.
    words = cleaned.split()
    by_length = sorted(canonical.items(), key=lambda kv: -len(kv[0].split()))
    found: set[str] = set()
    position = 0
    while position < len(words):
        for phrase_form, full in by_length:
            tokens = phrase_form.split()
            if words[position:position + len(tokens)] == tokens:
                found.add(full)
                position += len(tokens)
                break
        else:
            # A lone fragment of exactly one class is still usable
            # ('trash' -> trash-can). An unrecognised word contributes nothing
            # and is skipped; whatever classes the phrase DID name survive.
            prefix = [full for phrase_form, full in canonical.items()
                      if phrase_form.startswith(words[position])]
            if len(prefix) == 1 and len(words) == 1:
                found.add(prefix[0])
            position += 1
    return frozenset(found)


def label_from_prompt_phrase(phrase: str,
                             vocabulary: tuple[str, ...] = VOCABULARY
                             ) -> str | None:
    """Single-label view of `labels_from_prompt_phrase`, or None if not unique.

    Retained because it states the unambiguous case cleanly; the arm itself
    uses the set form so that ambiguity survives to association.
    """
    candidates = labels_from_prompt_phrase(phrase, vocabulary)
    return next(iter(candidates)) if len(candidates) == 1 else None


@dataclass(frozen=True)
class Detection:
    """One detector box and the SAM mask it prompted, with full provenance."""
    view: int
    mask_index: int
    label: str
    detector_score: float
    box_xyxy: tuple[float, float, float, float]
    sam_score: float
    notes: dict = field(default_factory=dict)


def associate_by_label_and_overlap(
        masks: list[LiftedMask], labels: list[str], n_vertices: int,
        *, anchor_stride: int = ASSOC_ANCHOR_STRIDE,
        require_same_label: bool = REQUIRE_SAME_LABEL) -> list[list[int]]:
    """Single-linkage clusters over masks that agree on BOTH class and surface.

    The geometric half is identical to the class-agnostic arm -- 3D IoU, or
    size-guarded containment -- so any difference in the result comes from the
    label constraint and from the boxes, not from a changed overlap rule.

    Masks from the same view are never linked, for the same reason as before:
    two detections in one frame are two hypotheses, and merging them here would
    quietly rebuild a partition.
    """
    if not masks:
        return []
    if len(labels) != len(masks):
        raise ValueError(
            f"{len(labels)} labels for {len(masks)} masks; provenance would be "
            "misaligned")
    indicator = _anchor_matrix(masks, n_vertices, anchor_stride)
    sizes = indicator.sum(axis=1)
    inter = _pairwise_intersections(indicator)
    union = sizes[:, None] + sizes[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        smaller = np.minimum(sizes[:, None], sizes[None, :])
        larger = np.maximum(sizes[:, None], sizes[None, :])
        containment = np.where(smaller > 0, inter / np.maximum(smaller, 1e-9), 0.0)
        ratio = np.where(larger > 0, smaller / np.maximum(larger, 1e-9), 0.0)

    views = np.array([m.view for m in masks])
    geometric = (
        (iou >= ASSOC_IOU)
        | ((containment >= ASSOC_CONTAINMENT)
           & (ratio >= ASSOC_CONTAINMENT_MIN_SIZE_RATIO)))
    link = (views[:, None] != views[None, :]) & geometric
    if require_same_label:
        # Candidate sets must INTERSECT, not match. {armchair, chair} joins
        # {chair} because both admit `chair`; {sofa} and {cushion} still never
        # join. Encoded as 41-bit masks so the pairwise test is one AND.
        bits = np.array([_label_bits(s) for s in _as_sets(labels)],
                        dtype=object)
        compatible = np.array(
            [[(a & b) != 0 for b in bits] for a in bits], dtype=bool)
        link &= compatible
    np.fill_diagonal(link, False)

    parent = list(range(len(masks)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(*np.nonzero(np.triu(link, 1))):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups: dict[int, list[int]] = {}
    for i in range(len(masks)):
        groups.setdefault(find(i), []).append(i)
    return [sorted(v) for _, v in sorted(groups.items())]


def _cluster_label(labels: list, group: list[int]) -> str:
    """Reporting label for a cluster of candidate sets.

    The intersection when it is non-empty -- every member admits that class.
    Single-linkage can chain {a,b}-{b,c}-{c,d} into a cluster with an empty
    intersection, so the fallback is the class the most members admit, with
    ties broken alphabetically. Reporting only; association already happened.
    """
    sets = _as_sets([labels[i] for i in group])
    common = set.intersection(*(set(s) for s in sets)) if sets else set()
    if common:
        return "|".join(sorted(common))
    tally: dict[str, int] = {}
    for candidate_set in sets:
        for name in candidate_set:
            tally[name] = tally.get(name, 0) + 1
    if not tally:
        return "?"
    return sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def config_record() -> dict:
    return {
        "detector_pin": "docs/grounding_dino_pin.json",
        "vocabulary_source": "extractors.learned_labels.GLOBAL_INDOOR_VOCABULARY_V1",
        "vocabulary_size": len(VOCABULARY),
        "vocabulary_modified": False,
        "label_rule": "candidate sets must intersect",
        "require_same_label": REQUIRE_SAME_LABEL,
        "assoc_iou": ASSOC_IOU,
        "assoc_containment": ASSOC_CONTAINMENT,
        "assoc_containment_min_size_ratio": ASSOC_CONTAINMENT_MIN_SIZE_RATIO,
        "assoc_anchor_stride": ASSOC_ANCHOR_STRIDE,
        "background_or_complement_proposals": "never created",
    }


def generate_from_sidecar(xyz_world: np.ndarray, frames: list,
                          selected: list[int], sidecar: dict,
                          baseline: list[np.ndarray], *,
                          progress=None) -> tuple[list[RepairProposal], dict]:
    """Full local pipeline: lift -> label-aware associate -> fuse -> emit.

    `sidecar[slot]` carries `masks`, `sam_scores`, `labels`, `detector_scores`
    and `boxes`, all aligned by mask index.
    """
    n_vertices = len(xyz_world)
    visibility: dict[int, np.ndarray] = {}
    masks: list[LiftedMask] = []
    labels: list[str] = []
    detections: list[Detection] = []
    per_view = []

    for slot, frame_index in enumerate(selected):
        frame = frames[frame_index]
        vertices, _ = visible_vertices(xyz_world, frame)
        visibility[frame_index] = vertices
        entry = sidecar[slot]
        raw = entry["masks"]
        lifted = lift_masks(xyz_world, frame, frame_index, raw,
                            entry["sam_scores"])
        for mask in lifted:
            label = entry["labels"][mask.mask_index]
            masks.append(mask)
            labels.append(label)
            detections.append(Detection(
                view=frame_index, mask_index=mask.mask_index, label=label,
                detector_score=float(entry["detector_scores"][mask.mask_index]),
                box_xyxy=tuple(float(v) for v in entry["boxes"][mask.mask_index]),
                sam_score=mask.predicted_iou))
        per_view.append({
            "slot": slot, "frame_index": frame_index,
            "timestamp": round(frame.timestamp, 3),
            "n_detections": int(len(raw)),
            "n_lifted": len(lifted),
            "n_visible_vertices": int(len(vertices)),
            "labels": sorted({name for m in lifted
                              for name in entry["labels"][m.mask_index]}),
        })
        if progress:
            progress(f"  [{slot + 1}/{len(selected)}] {len(raw)} detections -> "
                     f"{len(lifted)} lifted")

    groups = associate_by_label_and_overlap(masks, labels, n_vertices)
    if progress:
        progress(f"associated {len(masks)} lifted masks into {len(groups)} "
                 "label-consistent clusters")

    clusters, cluster_labels, unsupported = [], [], 0
    for group in groups:
        fused = fuse_cluster([masks[i] for i in group], frames, visibility,
                             n_vertices)
        if fused is None:
            unsupported += 1
            continue
        clusters.append(fused)
        cluster_labels.append(_cluster_label(labels, group))
    if progress:
        progress(f"{len(clusters)} clusters supported, {unsupported} dropped")

    proposals, diagnostics = classify_clusters(clusters, baseline, n_vertices)

    # Re-attach the label of the cluster each emitted proposal came from, by
    # INDEX. Matching on vertices does not work: a `split` proposal is clipped
    # to its Mask3D parent, so its first vertex need not be the cluster's, and
    # an earlier version of this code silently labelled every split "?".
    source_indices = diagnostics["source_cluster_indices"]
    if len(source_indices) != len(proposals):
        raise ValueError(
            f"{len(source_indices)} source indices for {len(proposals)} "
            "proposals; label provenance would be misaligned")
    labelled = [(proposal, cluster_labels[source])
                for proposal, source in zip(proposals, source_indices)]

    diagnostics["n_lifted_masks"] = len(masks)
    diagnostics["n_raw_clusters"] = len(groups)
    diagnostics["n_unsupported_clusters"] = unsupported
    diagnostics["per_view"] = per_view
    diagnostics["config"] = config_record()
    diagnostics["detections"] = [{
        "view": d.view, "mask_index": d.mask_index,
        # Candidate sets are serialised as sorted lists; a frozenset is not
        # JSON, and sorting keeps the diagnostics diffable run to run.
        "candidate_labels": sorted(d.label) if not isinstance(d.label, str)
        else [d.label],
        "detector_score": round(d.detector_score, 4),
        "box_xyxy": [round(v, 2) for v in d.box_xyxy],
        "sam_score": round(d.sam_score, 4),
    } for d in detections]
    diagnostics["emitted_labels"] = [label for _, label in labelled]
    return [p for p, _ in labelled], diagnostics
