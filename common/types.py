"""Shared primitive types used across stages.

Pure data only. No business logic. Imported by everything; imports nothing
from sibling stage modules. Keeps the dependency graph acyclic between
adapters / representations / extractors / graph / reasoner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

JSON = Any  # values produced by json.loads — dict, list, str, int, float, bool, None
Vec3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]  # (x, y, z, w)


@dataclass(frozen=True)
class Plane:
    a: float
    b: float
    c: float
    d: float


@dataclass(frozen=True)
class OrientedBBox:
    center: Vec3
    extents: Vec3
    rotation_quat: Quaternion


@dataclass(frozen=True)
class CameraPose:
    camera_id: str
    position: Vec3
    rotation_quat: Quaternion
    intrinsics: tuple[float, float, float, float]
    width: int
    height: int


FrameKind = Literal["world", "viewpoint", "scene_canonical"]


@dataclass(frozen=True)
class SceneFrame:
    """The coordinate frame a bundle's geometry is expressed in.

    `kind` names that frame, and is the single source of truth every
    downstream `Edge.frame` label must agree with (graph.schema.Edge.frame
    has the same domain; tests/graph/test_edge_frame_label.py enforces the
    agreement). Meanings:

      "world"           the capture's raw axes, up to a pure translation.
                        Translation is included because every relation in
                        graph/relations/** is computed from coordinate
                        DIFFERENCES, so a global offset cannot change an
                        edge. A rotation can, and does — see
                        docs/frame_decision.md.
      "scene_canonical" the scene's own gravity-canonical frame: some
                        rotation has been applied so up is exactly +z, and
                        possibly a yaw de-rotation on top of it. NOT the
                        capture's axes.
      "viewpoint"       a camera-relative frame. Reserved; nothing emits it
                        yet, and "left of" in a viewpoint frame does not
                        mean what it means in the other two.

    Default is "world" so that any bundle which does not consciously
    canonicalize keeps claiming only what it has actually done.
    """
    gravity: Vec3
    canonical_forward: Vec3 | None
    canonical_right: Vec3 | None
    units: Literal["meters"]
    notes: str
    kind: FrameKind = "world"
