"""ARKitScenes ReconstructionAdapter — the first non-Replica capture path.

Consumes an ARKitScenes 3DOD raw capture directory and produces a
SceneRepresentationBundle in the `scene_canonical` frame.

BOUNDARY DISCIPLINE — the point of this module:

  * It reads ONLY `<video_id>_3dod_mesh.ply` (geometry + vertex colour).
  * It does NOT read `<video_id>_3dod_annotation.json`. Those are the
    dataset's ground-truth oriented boxes. Reading them here would rebuild
    the oracle entity path on a new dataset and silently make every
    downstream number an oracle number. Annotations belong behind the
    evaluation boundary, the way `tools/p1_selector_eval.py` keeps oracle
    IoU. `build_arkitscenes_capture_bundle` sets `semantic_export=None` and
    a test asserts this module never opens an annotation file.

WHY THE ROTATION IS ALWAYS APPLIED, EVEN WHEN IT IS NEARLY THE IDENTITY.
`docs/frame_decision.md` defines `scene_canonical` as: *some rotation was
applied so that up is exactly +z*. ARKitScenes 3DOD meshes happen to arrive
gravity-aligned already (measured: 0.065 deg - 0.261 deg off +z across the
three Validation scenes on disk), so the rotation here is very close to the
identity. It is still measured and applied rather than assumed, for two
reasons:

  1. A dataset that does NOT arrive aligned must go through the same code
     path and come out canonical. Assuming alignment because one dataset
     happens to have it is exactly how `frame="world"` became a lie on the
     Replica path.
  2. The declared frame must describe the coordinates the geometry is
     actually in. Declaring `scene_canonical` while shipping unrotated
     coordinates would reintroduce the bug `docs/frame_decision.md` fixed.

`gravity_align_matrix` is imported from `importers.replica` rather than
reimplemented, for the reason that module gives: the two importers must not
drift to numerically different rotations. It returns an exact identity when
the input is already aligned, so a no-op case costs nothing.

The canonical geometry is written to a derived `*_canonical.ply` next to the
source, and the GeometryHandle points at THAT file, not the original -- so
whatever loads the handle gets coordinates matching the declared frame.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from adapters.base import (
    CaptureBundle, ReconstructionCapabilities, ReconstructionConfig,
)
from common.types import SceneFrame
from geometry.frame import estimate_scene_frame
from importers.replica import gravity_align_matrix
from representations.base import (
    GeometryHandle, ReconstructionDiagnostics, RepresentationCapabilities,
    SceneRepresentationBundle,
)
from representations.serde import CURRENT_SCHEMA_VERSION as REPR_SCHEMA_VERSION

_ADAPTER_NAME = "arkitscenes"
_ADAPTER_VERSION = "0.1"

MESH_SUFFIX = "_3dod_mesh.ply"
CANONICAL_SUFFIX = "_3dod_mesh_canonical.ply"
ANNOTATION_SUFFIX = "_3dod_annotation.json"   # deliberately never read here

_PLY_DTYPES = {
    "char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2",
    "int": "i4", "uint": "u4", "float": "f4", "double": "f8",
    "int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2",
    "int32": "i4", "uint32": "u4", "float32": "f4", "float64": "f8",
}


@dataclass(frozen=True)
class ARKitScenesMesh:
    xyz: np.ndarray      # float64 [V,3]
    rgb: np.ndarray      # uint8   [V,3]
    faces: np.ndarray    # int64   [F,3]


def read_mesh(path: Path) -> ARKitScenesMesh:
    """Binary little-endian PLY reader that keeps vertex colour.

    `geometry.frame.load_mesh` returns xyz+faces only; the splat renderer in
    `segmenter/view_render.py` needs per-vertex rgb, so colour is read here.
    """
    path = Path(path)
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a PLY file")
        fmt = None
        elements: list[dict] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: PLY header never terminated")
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]
            if tag == b"format":
                fmt = parts[1].decode()
            elif tag == b"element":
                elements.append({"name": parts[1].decode(),
                                 "count": int(parts[2]), "props": []})
            elif tag == b"property":
                if parts[1] == b"list":
                    elements[-1]["props"].append(
                        {"list": True, "count_type": parts[2].decode(),
                         "item_type": parts[3].decode(),
                         "name": parts[4].decode()})
                else:
                    elements[-1]["props"].append(
                        {"list": False, "type": parts[1].decode(),
                         "name": parts[2].decode()})
            elif tag == b"end_header":
                break
        if fmt != "binary_little_endian":
            raise ValueError(
                f"{path}: only binary_little_endian is supported here, got {fmt!r}")

        vert = next(e for e in elements if e["name"] == "vertex")
        face = next(e for e in elements if e["name"] == "face")
        if any(p["list"] for p in vert["props"]):
            raise ValueError("vertex element must be all scalar properties")

        vdtype = np.dtype([(p["name"], _PLY_DTYPES[p["type"]])
                           for p in vert["props"]], align=False)
        verts = np.frombuffer(f.read(vert["count"] * vdtype.itemsize),
                              dtype=vdtype, count=vert["count"])
        xyz = np.column_stack(
            [verts["x"], verts["y"], verts["z"]]).astype(np.float64)
        names = {p["name"] for p in vert["props"]}
        if {"red", "green", "blue"} <= names:
            rgb = np.column_stack(
                [verts["red"], verts["green"], verts["blue"]]).astype(np.uint8)
        else:
            rgb = np.full((len(xyz), 3), 200, dtype=np.uint8)

        # Faces: `property list <count_type> <item_type> vertex_indices`.
        # Read the whole remaining block once, then slice: every ARKitScenes
        # face is a triangle, so the record stride is fixed.
        fp = next(p for p in face["props"] if p["list"])
        cnt_dt = np.dtype(_PLY_DTYPES[fp["count_type"]])
        item_dt = np.dtype(_PLY_DTYPES[fp["item_type"]])
        rec = np.dtype([("n", cnt_dt), ("v", item_dt, 3)], align=False)
        raw = f.read(face["count"] * rec.itemsize)
        if len(raw) != face["count"] * rec.itemsize:
            raise ValueError(
                f"{path}: face block is not uniformly triangular; "
                "this reader assumes 3-vertex faces")
        recs = np.frombuffer(raw, dtype=rec, count=face["count"])
        if not np.all(recs["n"] == 3):
            raise ValueError(f"{path}: non-triangular face encountered")
        faces = recs["v"].astype(np.int64)

    if not np.isfinite(xyz).all():
        raise ValueError(f"{path}: mesh contains non-finite vertices")
    if faces.size and (faces.min() < 0 or faces.max() >= len(xyz)):
        raise ValueError(f"{path}: face index out of range")
    return ARKitScenesMesh(xyz=xyz, rgb=rgb, faces=faces)


def write_mesh(path: Path, mesh: ARKitScenesMesh) -> None:
    """Binary little-endian PLY with xyz + rgb + triangles."""
    v, fc = len(mesh.xyz), len(mesh.faces)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"comment canonical frame, written by {_ADAPTER_NAME} v{_ADAPTER_VERSION}\n"
        f"element vertex {v}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element face {fc}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode()
    vdt = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"),
                    ("red", "u1"), ("green", "u1"), ("blue", "u1")], align=False)
    va = np.empty(v, dtype=vdt)
    va["x"], va["y"], va["z"] = mesh.xyz[:, 0], mesh.xyz[:, 1], mesh.xyz[:, 2]
    va["red"], va["green"], va["blue"] = (
        mesh.rgb[:, 0], mesh.rgb[:, 1], mesh.rgb[:, 2])
    fdt = np.dtype([("n", "u1"), ("v", "i4", 3)], align=False)
    fa = np.empty(fc, dtype=fdt)
    fa["n"] = 3
    fa["v"] = mesh.faces.astype(np.int32)
    tmp = Path(path).with_suffix(".ply.partial")
    with open(tmp, "wb") as f:
        f.write(header)
        f.write(va.tobytes())
        f.write(fa.tobytes())
    tmp.replace(path)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scene_id_for(scene_dir: Path) -> str:
    return f"arkitscenes_{Path(scene_dir).name}"


def build_arkitscenes_capture_bundle(scene_dir: Path) -> CaptureBundle:
    """CaptureBundle for one ARKitScenes raw scene directory.

    `semantic_export` is None BY DESIGN: the annotation JSON sitting in this
    directory is ground truth and must not enter the reconstruction path.
    """
    scene_dir = Path(scene_dir)
    vid = scene_dir.name
    mesh_path = scene_dir / f"{vid}{MESH_SUFFIX}"
    if not mesh_path.is_file():
        raise FileNotFoundError(f"no mesh at {mesh_path}")
    traj = scene_dir / "lowres_wide.traj"
    return CaptureBundle(
        bundle_hash=f"ark_{_sha256_file(mesh_path)[:16]}",
        scene_id=scene_id_for(scene_dir),
        images_dir=None,          # RGB frames not required for the mesh path
        poses=None,               # captured trajectory not consumed yet
        rgbd_dir=None,
        mesh_path=mesh_path,
        semantic_export=None,     # <- annotations deliberately excluded
        notes={
            "dataset": "arkitscenes-3dod-raw",
            "video_id": vid,
            "captured_trajectory_present": traj.is_file(),
            "annotation_present_but_unread": (
                scene_dir / f"{vid}{ANNOTATION_SUFFIX}").is_file(),
        },
    )


class ARKitScenesAdapter:
    name = _ADAPTER_NAME
    version = _ADAPTER_VERSION

    def capabilities(self) -> ReconstructionCapabilities:
        return ReconstructionCapabilities(
            produces_mesh=True,
            produces_pointcloud=True,
            produces_gaussian_splat=False,
            produces_nerf_field=False,
            estimates_poses=False,
            requires_gpu=False,
            typical_runtime_minutes=1,
        )

    def reconstruct(self, capture: CaptureBundle,
                    config: ReconstructionConfig) -> SceneRepresentationBundle:
        if capture.mesh_path is None:
            raise ValueError("ARKitScenesAdapter requires capture.mesh_path")
        if capture.semantic_export is not None:
            raise ValueError(
                "ARKitScenesAdapter refuses a semantic_export: ARKitScenes "
                "annotations are ground truth and belong behind the "
                "evaluation boundary, not in the reconstruction path")
        t0 = time.perf_counter()
        src = Path(capture.mesh_path)
        mesh = read_mesh(src)

        # measure, then align -- never assume the capture arrives level
        est = estimate_scene_frame(mesh.xyz, mesh.faces)
        up = np.asarray(est.up_axis, dtype=np.float64)
        up = up / np.linalg.norm(up)
        R = np.asarray(gravity_align_matrix((-up[0], -up[1], -up[2])),
                       dtype=np.float64)
        xyz_canon = mesh.xyz @ R.T
        residual_deg = float(np.degrees(np.arccos(
            max(-1.0, min(1.0, float((R @ up)[2]))))))
        if residual_deg > 1e-6:
            raise ValueError(
                f"alignment failed: up is still {residual_deg:.3e} deg off +z")

        canonical = src.with_name(src.name.replace(MESH_SUFFIX, CANONICAL_SUFFIX))
        write_mesh(canonical, ARKitScenesMesh(
            xyz=xyz_canon, rgb=mesh.rgb, faces=mesh.faces))

        rep_hash = hashlib.sha256(
            f"{capture.bundle_hash}|{_ADAPTER_NAME}|{_ADAPTER_VERSION}|"
            f"{np.round(R, 12).tobytes().hex()}".encode()).hexdigest()[:16]

        return SceneRepresentationBundle(
            schema_version=REPR_SCHEMA_VERSION,
            representation_hash=f"repr_{rep_hash}",
            scene_id=capture.scene_id,
            frame=SceneFrame(
                gravity=(0.0, 0.0, -1.0),      # true BY CONSTRUCTION post-rotation
                canonical_forward=None,        # no yaw de-rotation applied
                canonical_right=None,
                units="meters",
                notes=(
                    "ARKitScenes 3DOD mesh rotated so up is exactly +z. "
                    f"Source up axis measured at [{up[0]:+.6f} {up[1]:+.6f} "
                    f"{up[2]:+.6f}] in capture axes. Yaw NOT de-rotated."
                ),
                kind="scene_canonical",
            ),
            capabilities=RepresentationCapabilities(
                renderable_channels=frozenset({"rgb"}),
                supports_arbitrary_pose=True,
                deterministic=True,
                typical_render_ms=400,
            ),
            geometry_handle=GeometryHandle(
                kind="mesh_file",
                uri=str(canonical),
                notes={
                    "format": "binary_little_endian ply, xyz+rgb+tri",
                    "source_mesh": str(src),
                    "source_mesh_sha256_16": capture.bundle_hash,
                    "rotation_row_major": [list(r) for r in R.tolist()],
                },
            ),
            poses=[],
            diagnostics=ReconstructionDiagnostics(
                loss=None,
                coverage=None,
                pose_rmse=None,
                runtime_seconds=round(time.perf_counter() - t0, 3),
                notes=(
                    f"vertices={len(mesh.xyz)} faces={len(mesh.faces)} "
                    f"up_tilt_deg={est.diagnostics.get('up_tilt_deg', 'na')} "
                    f"room_diag_m={est.scale.room_diagonal_m:.3f} "
                    f"storey_h_m={est.scale.storey_height_m}"
                ),
            ),
            notes={
                "adapter": _ADAPTER_NAME,
                "adapter_version": _ADAPTER_VERSION,
                "config": config.name,
                "reads_annotations": False,
                "source_up_axis_capture_frame": [float(x) for x in up],
                "scale": {
                    "room_diagonal_m": est.scale.room_diagonal_m,
                    "floor_diagonal_m": est.scale.floor_diagonal_m,
                    "storey_height_m": est.scale.storey_height_m,
                    "floor_area_m2": est.scale.floor_area_m2,
                },
                "estimated_yaw_deg": est.yaw_deg,
                "floor_found": est.floor is not None,
                "ceiling_found": est.ceiling is not None,
            },
        )
