"""Attach oracle-free learned label hypotheses to anonymous entities.

The stage consumes a canonical colored mesh, dense instance membership, and
an anonymous :class:`EntityArtifacts` bundle.  Each instance is rendered with
the existing gravity-aligned three-view point-splat renderer and classified by
the pinned CLIP labeler over one declared, scene-independent indoor vocabulary.

Labels remain hypotheses.  The stable object uid and source-instance reference
never change.  A learned top-1 label becomes the display label only when its
raw CLIP cosine score clears the declared admission threshold; otherwise the
anonymous ``segment_<id>`` display identity is retained.  The top-k ranking is
preserved in ``semantic_hypotheses`` in either case for inspection.

This module has no annotation/key/oracle input and imports no evaluation code.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np

from extractors.base import EntityArtifacts, SemanticHypothesis
from representations.base import SceneRepresentationBundle
from segmenter.base import load_segmentation_output
from segmenter.clip_labeler import MODEL_NAME, PRETRAINED, PROMPTS, ClipLabeler
from segmenter.instance_render import render_views
from segmenter.ply import parse_vertices_with_colors


LABEL_STAGE_NAME = "openclip_global_indoor_labels"
LABEL_STAGE_VERSION = "0.1"

# V1 is deliberately declared in code rather than assembled from any scene's
# annotation file.  Spellings use the graph's canonical hyphen form; its
# normalizer maps underscore/space variants to the same class family.
GLOBAL_INDOOR_VOCABULARY_V1: tuple[str, ...] = (
    "armchair",
    "bathtub",
    "bed",
    "bench",
    "blinds",
    "bookshelf",
    "bottle",
    "bowl",
    "box",
    "cabinet",
    "chair",
    "clock",
    "counter",
    "cushion",
    "desk",
    "door",
    "drawer",
    "indoor-plant",
    "lamp",
    "microwave",
    "mirror",
    "monitor",
    "nightstand",
    "oven",
    "picture",
    "plate",
    "plant-stand",
    "projector",
    "refrigerator",
    "rug",
    "shelf",
    "sink",
    "sofa",
    "stool",
    "table",
    "toilet",
    "trash-can",
    "tv-monitor",
    "vase",
    "whiteboard",
    "window",
)


@dataclass(frozen=True)
class LearnedLabelConfig:
    top_k: int = 3
    # Raw cosine similarity, not a calibrated probability.  This conservative
    # admission threshold is explicit so low-scoring predictions remain
    # anonymous rather than silently becoming object truth.
    min_top1_score: float = 0.28

    def __post_init__(self) -> None:
        if not 1 <= self.top_k <= len(GLOBAL_INDOOR_VOCABULARY_V1):
            raise ValueError(
                f"top_k must be in [1, {len(GLOBAL_INDOOR_VOCABULARY_V1)}], "
                f"got {self.top_k}")
        if not math.isfinite(self.min_top1_score):
            raise ValueError("min_top1_score must be finite")


class ImageLabeler(Protocol):
    weights_sha256: str | None

    def classify(self, images, vocabulary: list[str]) -> list[dict]: ...


class ImageSource(Protocol):
    """Produce the images an instance is classified from.

    Called with the instance's vertex indices and returns the views, best
    first. Returning an empty sequence is an error the caller must handle,
    not a licence to substitute a different image origin.
    """

    def __call__(self, vertex_indices): ...


def _instance_id(source_instance_ref: str) -> int:
    prefix = "segmenter:"
    if not source_instance_ref.startswith(prefix):
        raise ValueError(
            "learned-label stage requires source_instance_ref='segmenter:<id>'; "
            f"got {source_instance_ref!r}")
    try:
        instance_id = int(source_instance_ref[len(prefix):])
    except ValueError as exc:
        raise ValueError(
            f"invalid segmenter instance reference {source_instance_ref!r}") from exc
    if instance_id < 0:
        raise ValueError(f"instance id must be non-negative, got {instance_id}")
    return instance_id


def _validate_ranking(ranking: list[dict], vocabulary: tuple[str, ...]) -> None:
    if len(ranking) < 1:
        raise ValueError("labeler returned an empty ranking")
    allowed = set(vocabulary)
    seen: set[str] = set()
    previous = math.inf
    for row in ranking:
        if "label" not in row or "score" not in row:
            raise ValueError("labeler ranking rows require label and score")
        label = str(row["label"])
        score = float(row["score"])
        if label not in allowed:
            raise ValueError(f"labeler returned out-of-vocabulary label {label!r}")
        if label in seen:
            raise ValueError(f"labeler returned duplicate label {label!r}")
        if not math.isfinite(score):
            raise ValueError(f"labeler returned non-finite score for {label!r}")
        if score > previous:
            raise ValueError("labeler ranking is not sorted by descending score")
        seen.add(label)
        previous = score


def _content_hash(
    source_bundle_hash: str,
    segmentation_hash: str,
    config: LearnedLabelConfig,
    predictions: list[dict],
    weights_sha256: str | None,
    image_source_name: str,
) -> str:
    payload = json.dumps({
        "label_stage": LABEL_STAGE_NAME,
        "label_stage_version": LABEL_STAGE_VERSION,
        "source_bundle_hash": source_bundle_hash,
        "segmentation_output_sha256": segmentation_hash,
        "model": MODEL_NAME,
        "image_source": image_source_name,
        "pretrained": PRETRAINED,
        "prompts": list(PROMPTS),
        "weights_sha256": weights_sha256,
        "vocabulary": list(GLOBAL_INDOOR_VOCABULARY_V1),
        "top_k": config.top_k,
        "min_top1_score": config.min_top1_score,
        "predictions": predictions,
    }, sort_keys=True, separators=(",", ":"))
    return f"ent_{LABEL_STAGE_NAME}_{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def attach_learned_labels(
    representation: SceneRepresentationBundle,
    anonymous: EntityArtifacts,
    *,
    segmentation_dir: Path | None = None,
    config: LearnedLabelConfig = LearnedLabelConfig(),
    labeler: ImageLabeler | None = None,
    image_source: "ImageSource | None" = None,
    image_source_name: str = "instance_point_splat_3view",
) -> EntityArtifacts:
    """Return a copy of ``anonymous`` carrying learned top-k hypotheses.

    ``labeler`` is injectable for deterministic unit tests.  Production calls
    omit it and use the pinned CPU ``ClipLabeler``.

    ``image_source`` replaces WHAT the labeler looks at, leaving the model,
    vocabulary, top-k and admission threshold alone.  The default is the
    original three isolated point-splat renders.  Passing a source that
    returns real capture-RGB crops makes image origin the single variable in
    a paired comparison -- see ``extractors/arkitscenes_rgb_crops.py`` for
    why that is the interesting variable.  ``image_source_name`` is recorded
    on every hypothesis so two arms can never be mistaken for one another.
    """
    if anonymous.scene_id != representation.scene_id:
        raise ValueError(
            f"scene mismatch: entities={anonymous.scene_id!r}, "
            f"representation={representation.scene_id!r}")
    if anonymous.representation_hash != representation.representation_hash:
        raise ValueError(
            "representation hash mismatch between entities and geometry")
    if anonymous.frame != representation.frame:
        raise ValueError("entity and representation frames differ")
    if anonymous.structural_surfaces:
        raise ValueError(
            "learned-label stage expects the oracle-free anonymous bundle "
            "with no structural surfaces")
    if anonymous.notes.get("semantic_source") != "none":
        raise ValueError(
            "learned-label stage requires anonymous input with "
            "semantic_source='none'")
    if anonymous.notes.get("surface_source") != "none":
        raise ValueError(
            "learned-label stage requires anonymous input with "
            "surface_source='none'")
    if anonymous.notes.get("oracle_free") is not True:
        raise ValueError(
            "learned-label stage requires input explicitly marked oracle_free")
    if representation.geometry_handle.kind != "mesh_file":
        raise ValueError("learned-label stage requires mesh_file geometry")

    store_value = segmentation_dir or anonymous.geometry_store_path
    if store_value is None:
        raise ValueError(
            "segmentation_dir is required when geometry_store_path is absent")
    store = Path(store_value)
    seg = load_segmentation_output(store)
    mesh_path = Path(representation.geometry_handle.uri)
    xyz, rgb = parse_vertices_with_colors(mesh_path)
    if len(xyz) != seg.n_vertices:
        raise ValueError(
            f"mesh has {len(xyz)} vertices, segmentation has {seg.n_vertices}")
    from segmenter.base import sha256_file
    mesh_sha = sha256_file(mesh_path)
    if mesh_sha != seg.input_mesh_sha256:
        raise ValueError(
            "segmentation was not produced from the representation mesh: "
            f"{seg.input_mesh_sha256[:16]}... != {mesh_sha[:16]}...")

    active_labeler = labeler if labeler is not None else ClipLabeler()
    ids = seg.vertex_instance_ids
    source = f"openclip:{MODEL_NAME}/{PRETRAINED}"
    entities = []
    prediction_records: list[dict] = []
    # instances the image source could not supply any view for
    without_views: list[int] = []
    n_promoted = 0
    for entity in anonymous.entities:
        instance_id = _instance_id(entity.identity.source_instance_ref)
        if entity.semantic_hypotheses:
            raise ValueError(
                f"entity {entity.identity.object_uid} already has semantic "
                "hypotheses; label attachment requires anonymous input")
        expected_display = f"segment_{instance_id}"
        if entity.identity.display_label != expected_display:
            raise ValueError(
                f"entity {entity.identity.object_uid} is not anonymously "
                f"displayed: expected {expected_display!r}, got "
                f"{entity.identity.display_label!r}")
        points = ids == instance_id
        if not points.any():
            raise ValueError(
                f"entity {entity.identity.object_uid} has no assigned vertices")
        if image_source is None:
            images = list(render_views(xyz[points], rgb[points]).values())
        else:
            images = list(image_source(np.flatnonzero(points)))
            if not images:
                # Never fall back to splats: that would mix two image origins
                # inside one bundle and the comparison would be meaningless.
                # The instance stays ANONYMOUS and is recorded, so a coverage
                # gap is visible in the artifact rather than absorbed.
                without_views.append(instance_id)
                entities.append(entity)
                continue
        ranking = active_labeler.classify(
            list(images), list(GLOBAL_INDOOR_VOCABULARY_V1))
        _validate_ranking(ranking, GLOBAL_INDOOR_VOCABULARY_V1)
        if len(ranking) < config.top_k:
            raise ValueError(
                f"labeler returned {len(ranking)} rows, top_k={config.top_k}")
        top = ranking[:config.top_k]
        hypotheses = [
            SemanticHypothesis(
                label=str(row["label"]),
                confidence=float(row["score"]),
                source=source,
            )
            for row in top
        ]
        admitted = hypotheses[0].confidence >= config.min_top1_score
        identity = entity.identity
        if admitted:
            aliases = list(identity.aliases)
            if identity.display_label not in aliases:
                aliases.append(identity.display_label)
            identity = replace(
                identity,
                display_label=hypotheses[0].label,
                aliases=aliases,
            )
            n_promoted += 1
        diagnostics = dict(entity.extraction_diagnostics)
        diagnostics.update({
            "learned_label_admitted": admitted,
            "learned_label_top1_score": hypotheses[0].confidence,
            "learned_label_threshold": config.min_top1_score,
        })
        entities.append(replace(
            entity,
            identity=identity,
            semantic_hypotheses=hypotheses,
            extraction_diagnostics=diagnostics,
        ))
        prediction_records.append({
            "object_uid": identity.object_uid,
            "source_instance_ref": identity.source_instance_ref,
            "admitted": admitted,
            "display_label": identity.display_label,
            "top_k": [
                {"label": h.label, "score": h.confidence} for h in hypotheses
            ],
        })

    weights_sha256 = getattr(active_labeler, "weights_sha256", None)
    notes = dict(anonymous.notes)
    notes.update({
        "semantic_source": LABEL_STAGE_NAME,
        "label_stage": {
            "name": LABEL_STAGE_NAME,
            "version": LABEL_STAGE_VERSION,
            # Instances the image source could supply no view for. They stay
            # anonymous. Recorded so a coverage gap is a visible property of
            # the artifact instead of an invisible difference between arms.
            "instances_without_views": sorted(without_views),
            "model": MODEL_NAME,
            # Which images the model actually saw. Two arms differing only
            # here must never hash or read as the same bundle.
            "image_source": image_source_name,
            "pretrained": PRETRAINED,
            "weights_sha256": weights_sha256,
            "prompts": list(PROMPTS),
            "vocabulary_name": "global_indoor_v1",
            "vocabulary": list(GLOBAL_INDOOR_VOCABULARY_V1),
            "top_k": config.top_k,
            "min_top1_score": config.min_top1_score,
            "confidence_note": "raw CLIP cosine similarity; not calibrated",
            "n_promoted_display_labels": n_promoted,
            "n_retained_anonymous": len(entities) - n_promoted,
        },
        "oracle_free": True,
        "source_entity_bundle_hash": anonymous.bundle_hash,
    })

    return replace(
        anonymous,
        bundle_hash=_content_hash(
            anonymous.bundle_hash,
            seg.output_sha256,
            config,
            prediction_records,
            weights_sha256,
            image_source_name,
        ),
        extractor_name=f"{anonymous.extractor_name}+{LABEL_STAGE_NAME}",
        extractor_version=(
            f"{anonymous.extractor_version}+{LABEL_STAGE_VERSION}"
        ),
        entities=entities,
        diagnostics=replace(
            anonymous.diagnostics,
            notes=(
                f"{anonymous.diagnostics.notes}; learned labels: "
                f"{n_promoted}/{len(entities)} display labels admitted"
            ),
        ),
        notes=notes,
    )
