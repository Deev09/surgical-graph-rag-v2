"""Topology-only boundary cut: split each lifted SAM mask at mesh disconnections.

  python3 tools/arkitscenes_repair_topology_cut.py --scene 41069021 \
      --masks runs/arkitscenes_repair/arkitscenes_41069021/repair_sam_masks_arkitscenes_41069021.npz

Zero GPU. Reuses the existing pinned sidecar; changes no SAM parameter.

THE HYPOTHESIS UNDER TEST
-------------------------
Checkpoint E measured a precision wall: the union of every SAM part touching an
entity reaches recall 0.611 at precision 0.078, because the parts holding the
rest of each object are surface regions spanning several objects at once. One
explanation is that those masks bleed across object boundaries in 2D but are
DISCONNECTED on the mesh -- a countertop and the wall behind it are adjacent in
the image and separate in the geometry.

If that is right, cutting each lifted mask at its own mesh disconnections
recovers clean parts for free, with no new mask source and no new pin.

TOPOLOGY ONLY -- what this deliberately does not use
----------------------------------------------------
For each lifted mask: take its vertices, induce the mesh-adjacency subgraph on
exactly those vertices, split into connected components. That is all.

No distance threshold, no normal agreement, no depth discontinuity, no learned
cut, no curvature, no plane fitting. The only parameter is the EXISTING
`MIN_MASK_VERTICES` floor already used when lifting, and the mass it removes is
reported rather than absorbed. Two vertices end up in different components if
and only if the mesh contains no path between them through the mask.

NO ANNOTATION SELECTION. Every qualifying component is emitted. Components are
not ranked, filtered or chosen against the oracle, and the bank is finalized
and sha256-stamped here, before `tools/arkitscenes_repair_composition_ceiling.py`
opens a single annotation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.arkitscenes import scene_id_for
from eval.detection_repair import Proposal, ProposalArtifact
from extractors.arkitscenes_rgb_crops import load_frames
from geometry.mesh_surfaces import load_raw_triangle_mesh
from segmenter.base import sha256_file
from segmenter.proposal_fusion import mesh_edges
from segmenter.rgb_multiview_repair import FRAME_STRIDE, visible_vertices
from segmenter.sam_multiview_repair import MIN_MASK_VERTICES, lift_masks
from tools.arkitscenes_eval import DEFAULT_DATA_ROOT, load_canonical_geometry
from tools.arkitscenes_repair_eval import DEFAULT_OUT_ROOT
from tools.arkitscenes_repair_propose_sam import load_sidecar


def connected_components(vertices: np.ndarray, edges: np.ndarray,
                         member: np.ndarray, local: np.ndarray
                         ) -> list[np.ndarray]:
    """Components of the mesh-adjacency subgraph induced on `vertices`.

    `member` and `local` are scratch arrays sized to the mesh, reused across
    calls and left clean on exit -- allocating two million-element arrays per
    mask would dominate the runtime.

    A vertex with no mesh edge to any other mask vertex is its own component.
    That is correct and load-bearing: isolated speckle is exactly what the
    minimum-size rule should then remove, and counting it is how the mass
    report stays honest.
    """
    member[vertices] = True
    selected = edges[member[edges[:, 0]] & member[edges[:, 1]]]
    local[vertices] = np.arange(len(vertices), dtype=np.int64)

    parent = np.arange(len(vertices), dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    if len(selected):
        for u, v in zip(local[selected[:, 0]], local[selected[:, 1]]):
            ru, rv = find(int(u)), find(int(v))
            if ru != rv:
                parent[max(ru, rv)] = min(ru, rv)

    roots = np.array([find(i) for i in range(len(vertices))], dtype=np.int64)
    order = np.argsort(roots, kind="stable")
    sorted_roots = roots[order]
    bounds = np.flatnonzero(np.diff(sorted_roots)) + 1
    out = [np.sort(vertices[group]) for group in np.split(order, bounds)]

    member[vertices] = False
    local[vertices] = -1
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", default="41069021")
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = ap.parse_args(argv)

    scene_dir = args.data_root / args.scene
    scene_id = scene_id_for(scene_dir)
    out_dir = args.out_root / scene_id
    frames_json = out_dir / f"repair_frames_{scene_id}" / "frames.json"
    if not frames_json.is_file():
        print(f"missing {frames_json}")
        return 1
    t0 = time.perf_counter()

    manifest = json.loads(frames_json.read_text())
    mesh, rotation, bundle = load_canonical_geometry(scene_dir)
    mesh_sha = sha256_file(scene_dir / f"{args.scene}_3dod_mesh_canonical.ply")
    if manifest["canonical_mesh_sha256"] != mesh_sha:
        raise ValueError("frames were selected against a different mesh")
    xyz_world = mesh.xyz @ rotation
    n_vertices = len(xyz_world)

    raw = load_raw_triangle_mesh(
        scene_dir / f"{args.scene}_3dod_mesh_canonical.ply")
    edges = mesh_edges(raw.faces)
    sidecar_masks, env = load_sidecar(args.masks, manifest)
    frames = load_frames(scene_dir, stride=manifest.get("frame_stride",
                                                        FRAME_STRIDE))
    selected = [row["frame_index"] for row in manifest["frames"]]

    lifted = []
    for slot, frame_index in enumerate(selected):
        lifted.extend(lift_masks(xyz_world, frames[frame_index], frame_index,
                                 sidecar_masks[slot]["masks"],
                                 sidecar_masks[slot]["scores"]))
    print(f"=== {scene_id}   topology-only cut over {len(lifted)} lifted masks")
    print(f"    mesh {n_vertices} vertices, {len(edges)} adjacency edges")

    member = np.zeros(n_vertices, dtype=bool)
    local = np.full(n_vertices, -1, dtype=np.int64)
    records, per_mask_counts = [], []
    total_mass = kept_mass = 0
    n_components_all = 0
    for mask in lifted:
        components = connected_components(mask.vertices, edges, member, local)
        n_components_all += len(components)
        per_mask_counts.append(len(components))
        total_mass += int(len(mask.vertices))
        qualifying = [c for c in components if len(c) >= MIN_MASK_VERTICES]
        kept_mass += int(sum(len(c) for c in qualifying))
        for rank, component in enumerate(qualifying):
            records.append({
                "vertices": component,
                "view": mask.view,
                "mask_index": mask.mask_index,
                "predicted_iou": mask.predicted_iou,
                "n_components_in_parent": len(components),
                "parent_vertices": int(len(mask.vertices)),
                "component_rank": rank,
                "parent_was_split": len(components) > 1,
            })

    sizes = np.array([len(r["vertices"]) for r in records], dtype=np.int64)
    counts = np.array(per_mask_counts, dtype=np.int64)
    digests = {r["vertices"].tobytes() for r in records}

    artifact = ProposalArtifact.finalize(
        "repair_sam_topology_cut",
        [Proposal(r["vertices"], "repair", "topology_component",
                  r["predicted_iou"],
                  {"view": r["view"], "mask_index": r["mask_index"],
                   "n_components_in_parent": r["n_components_in_parent"],
                   "parent_was_split": r["parent_was_split"]})
         for r in records],
        n_vertices, out_dir / "topology_cut_bank.npz",
        {"mechanism": "tools/arkitscenes_repair_topology_cut.py",
         "rule": "mesh-adjacency connected components of each lifted SAM mask",
         "thresholds_used": {"min_component_vertices": MIN_MASK_VERTICES},
         "thresholds_deliberately_absent": [
             "distance", "normal agreement", "depth discontinuity",
             "curvature", "plane fitting", "learned cut"],
         "annotation_selection": False,
         "canonical_mesh_sha256": mesh_sha,
         "representation_hash": bundle.representation_hash,
         "selection_sha256": manifest["selection_sha256"],
         "sam_env": env})

    diagnostics = {
        "scene_id": scene_id,
        "n_lifted_masks": len(lifted),
        "n_components_before_min_size": n_components_all,
        "n_components_emitted": len(records),
        "n_unique_component_vertex_sets": len(digests),
        "min_component_vertices": MIN_MASK_VERTICES,
        "mass": {
            "lifted_mask_vertices_total": total_mass,
            "emitted_component_vertices_total": kept_mass,
            "mass_removed_by_min_size": total_mass - kept_mass,
            "mass_removed_fraction": round(1 - kept_mass / total_mass, 4),
        },
        "components_per_mask": {
            "median": int(np.median(counts)),
            "mean": round(float(counts.mean()), 2),
            "max": int(counts.max()),
            "masks_that_split": int((counts > 1).sum()),
            "masks_that_split_fraction": round(float((counts > 1).mean()), 4),
            "histogram": {str(k): int((counts == k).sum())
                          for k in range(1, 11)},
            "eleven_or_more": int((counts >= 11).sum()),
        },
        "component_size_distribution": {
            "min": int(sizes.min()), "p10": int(np.percentile(sizes, 10)),
            "median": int(np.median(sizes)), "p90": int(np.percentile(sizes, 90)),
            "max": int(sizes.max()),
        },
        "proposal_sha256": artifact.sha256,
        "canonical_mesh_sha256": mesh_sha,
        "runtime_seconds": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "topology_cut_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=1, sort_keys=True) + "\n")

    cpm = diagnostics["components_per_mask"]
    mass = diagnostics["mass"]
    size = diagnostics["component_size_distribution"]
    print(f"    components          : {n_components_all} before min-size, "
          f"{len(records)} emitted ({len(digests)} unique vertex sets)")
    print(f"    masks that split    : {cpm['masks_that_split']}"
          f"/{len(lifted)} ({cpm['masks_that_split_fraction']:.1%}), "
          f"median {cpm['median']}, mean {cpm['mean']}, max {cpm['max']}")
    print(f"    min-size removes    : {mass['mass_removed_fraction']:.1%} of "
          f"lifted mask mass ({mass['mass_removed_by_min_size']} vertices)")
    print(f"    component vertices  : min {size['min']}, p10 {size['p10']}, "
          f"median {size['median']}, p90 {size['p90']}, max {size['max']}")
    print(f"    proposal sha256     : {artifact.sha256[:16]}…")
    print(f"    bank  -> {out_dir / 'topology_cut_bank.npz'}")
    print(f"    diags -> {out_dir / 'topology_cut_diagnostics.json'}")
    print(f"    {diagnostics['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
