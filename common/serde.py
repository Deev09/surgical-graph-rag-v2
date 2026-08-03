"""Serialization helpers for shared primitive types.

Per-bundle serde lives in each module (representations/serde.py,
extractors/serde.py, graph/serde.py, reasoner/serde.py). This module
provides the shared building blocks: primitive type encoders, schema
version checks, and numpy sidecar helpers.

Design rule: explicit per-bundle serde, no introspection magic. Readable
diffs over clever generics.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from common.types import (
    CameraPose, OrientedBBox, Plane, Quaternion, SceneFrame, Vec3,
)


class SchemaVersionError(ValueError):
    """Raised when a loaded artifact's schema_version does not match the
    current expected version. Loaders never coerce; they raise."""


def check_schema_version(loaded: int, expected: int, what: str) -> None:
    if loaded != expected:
        raise SchemaVersionError(
            f"{what}: schema_version {loaded} != expected {expected}. "
            "Migrate the artifact or rebuild the bundle."
        )


def vec3_to_list(v: Vec3) -> list[float]:
    return [float(v[0]), float(v[1]), float(v[2])]


def vec3_from_list(lst: list[float]) -> Vec3:
    return (float(lst[0]), float(lst[1]), float(lst[2]))


def quat_to_list(q: Quaternion) -> list[float]:
    return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def quat_from_list(lst: list[float]) -> Quaternion:
    return (float(lst[0]), float(lst[1]), float(lst[2]), float(lst[3]))


def plane_to_dict(p: Plane) -> dict[str, float]:
    return {"a": p.a, "b": p.b, "c": p.c, "d": p.d}


def plane_from_dict(d: dict[str, Any]) -> Plane:
    return Plane(a=float(d["a"]), b=float(d["b"]), c=float(d["c"]), d=float(d["d"]))


def obb_to_dict(b: OrientedBBox) -> dict[str, Any]:
    return {
        "center": vec3_to_list(b.center),
        "extents": vec3_to_list(b.extents),
        "rotation_quat": quat_to_list(b.rotation_quat),
    }


def obb_from_dict(d: dict[str, Any]) -> OrientedBBox:
    return OrientedBBox(
        center=vec3_from_list(d["center"]),
        extents=vec3_from_list(d["extents"]),
        rotation_quat=quat_from_list(d["rotation_quat"]),
    )


def camera_pose_to_dict(p: CameraPose) -> dict[str, Any]:
    return {
        "camera_id": p.camera_id,
        "position": vec3_to_list(p.position),
        "rotation_quat": quat_to_list(p.rotation_quat),
        "intrinsics": [float(x) for x in p.intrinsics],
        "width": p.width,
        "height": p.height,
    }


def camera_pose_from_dict(d: dict[str, Any]) -> CameraPose:
    intr = d["intrinsics"]
    return CameraPose(
        camera_id=str(d["camera_id"]),
        position=vec3_from_list(d["position"]),
        rotation_quat=quat_from_list(d["rotation_quat"]),
        intrinsics=(float(intr[0]), float(intr[1]), float(intr[2]), float(intr[3])),
        width=int(d["width"]),
        height=int(d["height"]),
    )


def scene_frame_to_dict(f: SceneFrame) -> dict[str, Any]:
    return {
        "gravity": vec3_to_list(f.gravity),
        "canonical_forward": vec3_to_list(f.canonical_forward) if f.canonical_forward is not None else None,
        "canonical_right": vec3_to_list(f.canonical_right) if f.canonical_right is not None else None,
        "units": f.units,
        "notes": f.notes,
        "kind": f.kind,
    }


def scene_frame_from_dict(d: dict[str, Any]) -> SceneFrame:
    cf = d.get("canonical_forward")
    cr = d.get("canonical_right")
    kind = d.get("kind", "world")   # payloads written before frame_kind existed
    if kind not in ("world", "viewpoint", "scene_canonical"):
        raise ValueError(f"unknown frame kind {kind!r}")
    return SceneFrame(
        gravity=vec3_from_list(d["gravity"]),
        canonical_forward=vec3_from_list(cf) if cf is not None else None,
        canonical_right=vec3_from_list(cr) if cr is not None else None,
        units=d["units"],
        notes=str(d.get("notes", "")),
        kind=kind,
    )


def write_npy_sidecar(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def read_npy_sidecar(path: Path) -> np.ndarray:
    return np.load(path)
