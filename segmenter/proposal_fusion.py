"""C1-P1 fixed 2D-to-3D fusion: masks + id buffers -> proposal bank.

Protocol: docs/c1_p1_multiview_proposals_protocol.md ("Fixed 2D-to-3D
fusion"). Deterministic, numpy-only, oracle-free by construction: inputs
are the per-view id buffers, the lifted 2D mask vertex sets, and the raw
mesh edges. Frozen values (protocol, not parameters): masks lifting to
<20 unique vertices are discarded; confidence cuts {0.25, 0.50, 0.75};
components kept at >=20 vertices and <=40% of scene vertices; dedupe at
vertex IoU >= 0.95 preferring higher cut, then larger vertex count, then
lexicographically smallest vertex-id digest.
"""
from __future__ import annotations

import hashlib

import numpy as np

MIN_LIFT_VERTICES = 20
CONFIDENCE_CUTS = (0.25, 0.50, 0.75)
MIN_COMPONENT_VERTICES = 20
MAX_COMPONENT_FRAC = 0.40
DEDUPE_IOU = 0.95


def lift_mask(mask: np.ndarray, id_buffer: np.ndarray) -> np.ndarray:
    """2D boolean mask -> sorted unique non-negative vertex ids under it."""
    ids = id_buffer[mask.astype(bool)]
    ids = np.unique(ids[ids >= 0])
    return ids if len(ids) >= MIN_LIFT_VERTICES else ids[:0]


def mesh_edges(faces: np.ndarray) -> np.ndarray:
    """Unique undirected raw-mesh edges [E,2], sorted rows, lexsorted."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e.sort(axis=1)
    e = np.unique(e, axis=0)
    return e.astype(np.int64)


EVIDENCE_DENOMINATORS = ("covisible", "masked")


def edge_confidence(edges: np.ndarray, n_vertices: int,
                    views: list[dict], *,
                    evidence_denominator: str = "covisible",
                    ) -> tuple[np.ndarray, np.ndarray]:
    """views: [{'visible': sorted int array, 'masks': [vertex-id arrays]}].

    Returns (co_visible_count, confidence) per edge. The numerator is always
    the number of co-visible views in which both endpoints share >=1
    accepted mask. `evidence_denominator` selects what that divides by:

    "covisible" (default, frozen behaviour)
        every co-visible view. A view where both endpoints are visible but
        UNMASKED therefore counts against the edge.

    "masked"
        only views that carry mask evidence for this edge -- both endpoints
        visible AND each inside at least one mask. An unmasked endpoint is
        absence of evidence, not evidence of separation.

    The two agree exactly whenever every visible vertex is masked in every
    view, so the choice is inert on fully-covered renders and matters in
    proportion to how much of the surface SAM leaves unmasked. Measured:
    46.5% of occupied pixels fall inside a mask on Replica room_2 against
    28.9% on ARKitScenes 41069021. See
    docs/arkitscenes_fusion_evidence_protocol.md.

    `co_visible_count` is unaffected by the mode -- it stays the true
    co-visible count so callers reporting coverage are not silently
    re-defined. Edges with an empty denominator score 0, matching the
    existing treatment of never-co-visible edges.
    """
    if evidence_denominator not in EVIDENCE_DENOMINATORS:
        raise ValueError(
            f"unknown evidence_denominator {evidence_denominator!r}; "
            f"supported: {EVIDENCE_DENOMINATORS}")
    masked_mode = evidence_denominator == "masked"
    co_vis = np.zeros(len(edges), dtype=np.int32)
    denom = np.zeros(len(edges), dtype=np.int32)
    co_mem = np.zeros(len(edges), dtype=np.int32)
    for view in views:
        visible = np.zeros(n_vertices, dtype=bool)
        visible[view["visible"]] = True
        both = visible[edges[:, 0]] & visible[edges[:, 1]]
        co_vis += both
        n_masks = len(view["masks"])
        if n_masks == 0:
            # no evidence at all from this view; under "covisible" it still
            # counts against every co-visible edge, which is the behaviour
            # under test.
            if not masked_mode:
                denom += both
            continue
        words = (n_masks + 63) // 64
        bits = np.zeros((n_vertices, words), dtype=np.uint64)
        for m, verts in enumerate(view["masks"]):
            bits[verts, m // 64] |= np.uint64(1 << (m % 64))
        share = np.zeros(len(edges), dtype=bool)
        for w in range(words):
            share |= (bits[edges[:, 0], w] & bits[edges[:, 1], w]) != 0
        if masked_mode:
            in_mask = np.zeros(n_vertices, dtype=bool)
            for w in range(words):
                in_mask |= bits[:, w] != 0
            denom += both & in_mask[edges[:, 0]] & in_mask[edges[:, 1]]
        else:
            denom += both
        co_mem += both & share
    conf = np.zeros(len(edges), dtype=np.float64)
    nz = denom > 0
    conf[nz] = co_mem[nz] / denom[nz]
    return co_vis, conf


def _components(edges: np.ndarray, keep: np.ndarray, n_vertices: int
                ) -> list[np.ndarray]:
    parent = np.arange(n_vertices, dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    for u, v in edges[keep]:
        ru, rv = find(int(u)), find(int(v))
        if ru != rv:
            parent[max(ru, rv)] = min(ru, rv)
    touched = np.unique(edges[keep].ravel())
    roots: dict[int, list[int]] = {}
    for t in touched:
        roots.setdefault(find(int(t)), []).append(int(t))
    lo, hi = MIN_COMPONENT_VERTICES, int(MAX_COMPONENT_FRAC * n_vertices)
    return [np.asarray(sorted(vs), dtype=np.int64)
            for _, vs in sorted(roots.items())
            if lo <= len(vs) <= hi]


def _digest(verts: np.ndarray) -> str:
    return hashlib.sha256(verts.tobytes()).hexdigest()


def build_bank(edges: np.ndarray, co_vis: np.ndarray, conf: np.ndarray,
               n_vertices: int) -> list[dict]:
    """Pool components across the three cuts with the frozen dedupe rule."""
    pool: list[dict] = []
    for cut in CONFIDENCE_CUTS:
        keep = (co_vis > 0) & (conf >= cut)
        for verts in _components(edges, keep, n_vertices):
            pool.append({"cut": cut, "vertices": verts,
                         "digest": _digest(verts)})
    # dedupe: prefer higher cut, then larger count, then smallest digest
    pool.sort(key=lambda p: (-p["cut"], -len(p["vertices"]), p["digest"]))
    kept: list[dict] = []
    for cand in pool:
        cv = cand["vertices"]
        dup = False
        for k in kept:
            kv = k["vertices"]
            inter = len(np.intersect1d(cv, kv, assume_unique=True))
            if inter and inter / (len(cv) + len(kv) - inter) >= DEDUPE_IOU:
                dup = True
                break
        if not dup:
            kept.append(cand)
    return kept
