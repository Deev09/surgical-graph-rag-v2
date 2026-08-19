"""Run the oracle-free grounding bridge and write a hash-pinned sidecar.

  python3 tools/arkitscenes_grounding_run.py --out runs/arkit_relation_challenge/grounding.json

PREDICTION ONLY. This tool reads the frozen question manifest (for anchor
names and their declared synonyms), the delivered entity manifest (for
geometry handles), the mesh and the capture frames. It reads NO human key, NO
uid mapping, NO annotation and NO evaluation module, and a test AST-checks
that over this file and the bridge's transitive first-party imports.

The sidecar it writes is hashed before any human mapping is opened. Evaluation
is a separate tool that consumes the finished file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.arkitscenes import (
    ARKitScenesAdapter,
    build_arkitscenes_capture_bundle,
    read_mesh,
)
from adapters.base import ReconstructionConfig
from extractors.arkitscenes_anchor_grounding import SCHEMA, ground_scene, sidecar
from segmenter.clip_labeler import ClipLabeler

DEFAULT_QUESTIONS = (REPO_ROOT / "eval" / "questions"
                     / "arkitscenes_relation_challenge_v1.json")
DEFAULT_DATA_ROOT = Path.home() / "Desktop/datasets/arkitscenes/Validation"
SCENE_DIRS = {"arkitscenes_41069025": "41069025",
              "arkitscenes_41069042": "41069042"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_geometry(scene_dir: Path):
    """Canonical mesh and rotation, straight from the adapter.

    Inlined rather than imported from `tools.arkitscenes_eval`, which is an
    EVALUATION module and therefore off-limits to the prediction stage. The
    guard test caught that import; the three lines below are the whole of
    what was being borrowed. `adapters.arkitscenes` is annotation-free by
    design and has its own test asserting it never opens one.
    """
    bundle = ARKitScenesAdapter().reconstruct(
        build_arkitscenes_capture_bundle(scene_dir),
        ReconstructionConfig(name="arkitscenes_mesh", version="0.1"),
    )
    mesh = read_mesh(Path(bundle.geometry_handle.uri))
    rotation = np.asarray(
        bundle.geometry_handle.notes["rotation_row_major"], dtype=np.float64)
    return mesh, rotation


def anchors_for(scene_id: str, doc: dict) -> list[str]:
    """Every object name the scene's questions refer to, in stable order."""
    names: list[str] = []
    for question in doc["questions"]:
        if question["scene_id"] != scene_id:
            continue
        for field in ("subject", "object", "reference_a", "reference_b"):
            value = question.get(field)
            if value and value not in names:
                names.append(value)
        for value in question.get("candidate_objects", []):
            if value not in names:
                names.append(value)
    return names


def load_entities(scene_id: str, data_root: Path) -> tuple[list[dict], dict]:
    short = SCENE_DIRS[scene_id]
    manifest_path = (REPO_ROOT / "runs" / f"arkit_label_image_ab_{short}"
                     / "rgb_tight" / "entities" / "manifest.json")
    ids_path = (REPO_ROOT / "runs" / "arkitscenes_mask3d_transfer"
                / f"bundle_arkitscenes_{short}" / "vertex_instance_ids.npy")
    manifest = json.loads(manifest_path.read_text())
    instance_ids = np.load(ids_path)
    entities = []
    for entity in manifest["entities"]:
        index = int(entity["geometry_handle"].rsplit("#", 1)[1])
        entities.append({"uid": entity["identity"]["object_uid"],
                         "vertices": np.flatnonzero(instance_ids == index)})
    provenance = {
        "entity_manifest": str(manifest_path),
        "entity_manifest_sha256": sha256(manifest_path),
        "entity_bundle_hash": manifest.get("bundle_hash"),
        "n_entities": len(entities),
        "embedding_ref_present": any(
            e.get("embedding_ref") is not None for e in manifest["entities"]),
    }
    return entities, provenance


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    doc = json.loads(args.questions.read_text())
    synonyms = doc["object_synonyms"]
    labeler = ClipLabeler()

    t0 = time.perf_counter()
    scenes, per_scene_prov = [], {}
    for scene_id in doc["scenes"]:
        short = SCENE_DIRS[scene_id]
        scene_dir = args.data_root / short
        entities, prov = load_entities(scene_id, args.data_root)
        mesh, rotation = canonical_geometry(scene_dir)
        anchors = anchors_for(scene_id, doc)
        print(f"    {scene_id}: {len(anchors)} anchors over {len(entities)} entities")
        result = ground_scene(scene_id, anchors, synonyms, scene_dir,
                              mesh.xyz, rotation, entities, labeler)
        scenes.append(result)
        per_scene_prov[scene_id] = prov
        admitted = len(result["admitted"])
        print(f"      admitted {admitted}/{len(anchors)}, "
              f"abstained {len(anchors) - admitted}")

    provenance = {
        "questions": str(args.questions),
        "questions_sha256": sha256(args.questions),
        "weights_sha256": labeler.weights_sha256,
        "scenes": per_scene_prov,
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    body = sidecar(scenes, provenance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")

    print(f"    schema           : {SCHEMA}")
    print(f"    weights          : {labeler.weights_sha256[:16]}...")
    print(f"    prediction_sha256: {body['prediction_sha256']}")
    print(f"    file sha256      : {sha256(args.out)}")
    print(f"    -> {args.out}")
    print("    no human key, uid mapping or annotation was read")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
