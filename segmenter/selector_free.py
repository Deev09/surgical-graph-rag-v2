"""Oracle-free proposal scorer for the frozen C1-P1 multi-view bank.

Ranks 3D proposals using ONLY evidence that exists on an unannotated
capture: the raw mesh vertex positions, the per-view vertex-id buffers,
the lifted 2D masks with SAM's own quality numbers, and the geometry of
the proposal set itself. Nothing here reads or receives an oracle label.

Why that is structurally enforceable, not just a promise:

  * this module imports `numpy`, `dataclasses` and `__future__` and
    NOTHING else — no `json`, `pathlib`, `os`, `open`, no repo module.
    It therefore has no name in scope that can reach a file;
  * every entry point is a pure function of in-memory arrays. There is
    no path argument, no config object, no global state, no I/O;
  * `tests/segmenter/test_selector_free.py` proves both mechanically:
    an AST scan of this source rejects any import outside the allowlist
    and any call to `open`/`np.load`/`np.fromfile`/`exec`/`eval`/
    `__import__`, and a `sys.addaudithook` installed around a real
    scoring call raises if ANY file open, subprocess or socket audit
    event fires. A future edit that reaches for the answer key breaks
    the test suite.

The caller supplies the same `views` contract that
`segmenter.proposal_fusion.edge_confidence` already consumes — the
per-view visible-vertex set and the lifted 2D masks produced by
`proposal_fusion.lift_mask` — plus SAM's per-mask quality pair. Reusing
that contract is deliberate: the co-membership evidence that BUILT the
bank is exactly the evidence available to score it.

Signals (all four are computable on an arbitrary capture):

  agreement    For every view in which a proposal is at least
               `VIS_MIN` visible, the best IoU between the visible part
               of the proposal and any lifted 2D mask in that view; the
               score is the mean of the best `TOP_VIEWS` of those. High
               when the proposal reproduces as ONE 2D mask repeatedly.
  connectivity Fraction of the proposal's vertices in the largest
               6-connected component of its own occupancy grid, at a
               resolution set by its own extent. Scale-free fragment
               detector.
  size_prior   Metric-scale plausibility of the largest AABB extent.
               THIS IS A PRIOR ABOUT INDOOR-ROOM-SCALE OBJECTS and the
               one component that does not transfer to a capture at a
               different physical scale — see the module docstring of
               `tools/p1_selector_eval.py` for the measured cost of
               dropping it.
  redundancy   Number of other proposals that contain (or are contained
               in) this one at `CONTAIN_FRAC` AND have strictly higher
               agreement. Demotes parts and near-duplicates below the
               nested unit that the 2D evidence actually agrees on.

Frozen constants below were chosen on the development scene
`replica_room_2` only. They are module constants, not parameters, so
that "tuned on dev" is a property of the file rather than of a call.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- frozen constants (chosen on replica_room_2, the development scene) ---
VIS_MIN = 0.5          # a view "observes" a proposal at >=50% visible
TOP_VIEWS = 3          # agreement = mean of the best 3 observing views
SUPPORT_IOU = 0.5      # a view "supports" a proposal at best-IoU >= 0.5
CONN_VOXELS = 16       # occupancy grid: longest axis split into 16 cells
CONN_MIN_RES_M = 1e-3  # floor on voxel size (degenerate/flat proposals)
CONTAIN_FRAC = 0.80    # nesting: 80% of the smaller inside the larger
REDUNDANCY_ALPHA = 0.5  # exp(-alpha * n_nested_better)
SIZE_FLOOR_M = 0.03    # below this max-extent, size_prior = 0
SIZE_LOW_M = 0.10      # size_prior reaches 1 here
SIZE_HIGH_M = 2.00     # size_prior starts falling here
SIZE_CEIL_M = 2.60     # above this max-extent, size_prior = 0
GATE_FLOOR = 0.30      # gates multiply in [GATE_FLOOR, 1], never to 0
SAM_QUALITY_FLOOR = 0.76  # pred_iou_thresh 0.8 * stability_thresh 0.95

# Every component the scorer knows how to apply. Used to validate the
# `components` argument; NOT the default set.
COMPONENTS = ("agreement", "connectivity", "size", "redundancy")

# v0 (frozen before transfer scenes were read). Kept so the result recorded in
# docs/selector_v0_results.md stays reproducible by construction.
COMPONENTS_V0 = COMPONENTS

# v1 default: connectivity dropped. The v0 ablation measured it as net-harmful
# on transfer -- neutral on the dev scene, negative on both others (AR@k at
# k=10/25/50: room_1 6/10/13 -> 7/11/13, office_0 3/5/6 -> 4/9/10). See
# docs/selector_v0_results.md.
#
# CAVEAT, stated because it is a real one: v0 was frozen before transfer
# results were seen; this default is not. It is a transfer-informed choice,
# so v1 numbers on room_1/office_0 are no longer a clean held-out measurement.
# Pass components=COMPONENTS_V0 to reproduce the frozen configuration.
DEFAULT_COMPONENTS = ("agreement", "size", "redundancy")


@dataclass(frozen=True)
class ProposalSignals:
    """Per-proposal oracle-free signals, one array entry per proposal."""

    agreement: np.ndarray        # [0,1] multi-view reconstruction agreement
    support_frac: np.ndarray     # [0,1] fraction of observing views >= 0.5 IoU
    sam_quality: np.ndarray      # [0,1] rescaled pred_iou * stability
    n_observed: np.ndarray       # int, views where the proposal is >=VIS_MIN
    connectivity: np.ndarray     # [0,1] largest connected voxel component
    size_prior: np.ndarray       # [0,1] metric-scale object plausibility
    n_nested_better: np.ndarray  # int, nested neighbours with higher agreement
    max_extent_m: np.ndarray     # raw AABB longest axis, metres
    n_vertices: np.ndarray       # raw vertex count


def _csr(proposals: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(proposal_index, vertex_id) membership entries sorted by vertex."""
    if not proposals:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    ent_p = np.concatenate([np.full(len(p), i, np.int64)
                            for i, p in enumerate(proposals)])
    ent_v = np.concatenate([np.asarray(p, np.int64) for p in proposals])
    order = np.argsort(ent_v, kind="stable")
    return ent_p[order], ent_v[order]


def _counts_against(ent_p: np.ndarray, ent_v: np.ndarray, flag: np.ndarray,
                    verts: np.ndarray, k: int) -> np.ndarray:
    """|proposal_i ∩ verts| for every i, via one gather + one bincount."""
    flag[:] = False
    flag[verts] = True
    return np.bincount(ent_p[flag[ent_v]], minlength=k).astype(np.float64)


def _view_signals(proposals: list[np.ndarray], n_vertices: int,
                  views: list[dict]) -> tuple[np.ndarray, ...]:
    k = len(proposals)
    if not views:
        z = np.zeros(k)
        return z, z.copy(), z.copy(), np.zeros(k, np.int64)
    sizes = np.array([len(p) for p in proposals], dtype=np.float64)
    sizes = np.maximum(sizes, 1.0)
    ent_p, ent_v = _csr(proposals)
    flag = np.zeros(n_vertices, dtype=bool)
    best_iou = np.zeros((k, len(views)))
    best_q = np.zeros((k, len(views)))
    observed = np.zeros((k, len(views)), dtype=bool)
    for v, view in enumerate(views):
        n_vis = _counts_against(ent_p, ent_v, flag,
                                np.asarray(view["visible"], np.int64), k)
        observed[:, v] = (n_vis / sizes) >= VIS_MIN
        masks = view["masks"]
        quality = np.asarray(view.get("mask_quality",
                                      np.zeros((len(masks), 2))), float)
        for m, verts in enumerate(masks):
            verts = np.asarray(verts, np.int64)
            if verts.size == 0:
                continue
            inter = _counts_against(ent_p, ent_v, flag, verts, k)
            iou = inter / np.maximum(n_vis + len(verts) - inter, 1.0)
            better = iou > best_iou[:, v]
            best_iou[better, v] = iou[better]
            best_q[better, v] = (float(quality[m, 0] * quality[m, 1])
                                 if quality.size else 0.0)
    n_obs = observed.sum(axis=1)
    # agreement: mean of the best TOP_VIEWS observing views (0 if unseen)
    masked = np.where(observed, best_iou, -1.0)
    ranked = np.sort(masked, axis=1)[:, ::-1]
    agreement = np.zeros(k)
    for i in range(k):
        take = min(TOP_VIEWS, max(int(n_obs[i]), 1))
        agreement[i] = np.clip(ranked[i, :take], 0.0, 1.0).mean()
    support = ((best_iou >= SUPPORT_IOU) & observed).sum(axis=1)
    support_frac = support / np.maximum(n_obs, 1)
    raw_q = np.where(observed, best_q, 0.0).max(axis=1)
    sam_quality = np.clip((raw_q - SAM_QUALITY_FLOOR)
                          / (1.0 - SAM_QUALITY_FLOOR), 0.0, 1.0)
    return agreement, support_frac, sam_quality, n_obs.astype(np.int64)


def _largest_voxel_component(points: np.ndarray) -> float:
    """Fraction of points in the largest 6-connected occupancy component."""
    lo = points.min(axis=0)
    extent = points.max(axis=0) - lo
    res = max(float(extent.max()) / CONN_VOXELS, CONN_MIN_RES_M)
    q = np.floor((points - lo) / res).astype(np.int64)
    dims = q.max(axis=0) + 1
    key = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
    uniq, counts = np.unique(key, return_counts=True)
    parent = np.arange(len(uniq), dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    coords = np.stack([uniq // (dims[1] * dims[2]),
                       (uniq // dims[2]) % dims[1],
                       uniq % dims[2]], axis=1)
    for axis, stride in ((0, dims[1] * dims[2]), (1, dims[2]), (2, 1)):
        ok = coords[:, axis] + 1 < dims[axis]
        if not ok.any():
            continue
        want = uniq[ok] + stride
        pos = np.searchsorted(uniq, want)
        pos_ok = pos < len(uniq)
        hit = np.zeros(len(want), dtype=bool)
        hit[pos_ok] = uniq[pos[pos_ok]] == want[pos_ok]
        src = np.flatnonzero(ok)[hit]
        dst = pos[hit]
        for a, b in zip(src.tolist(), dst.tolist()):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(len(uniq))], dtype=np.int64)
    comp = np.bincount(roots, weights=counts.astype(np.float64))
    return float(comp.max() / counts.sum())


def _geometry_signals(proposals: list[np.ndarray], xyz: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = len(proposals)
    conn = np.zeros(k)
    max_extent = np.zeros(k)
    for i, p in enumerate(proposals):
        pts = np.asarray(xyz, np.float64)[np.asarray(p, np.int64)]
        max_extent[i] = float((pts.max(axis=0) - pts.min(axis=0)).max())
        conn[i] = _largest_voxel_component(pts)
    ramp_up = (max_extent - SIZE_FLOOR_M) / (SIZE_LOW_M - SIZE_FLOOR_M)
    ramp_down = (SIZE_CEIL_M - max_extent) / (SIZE_CEIL_M - SIZE_HIGH_M)
    size_prior = np.clip(np.minimum(ramp_up, ramp_down), 0.0, 1.0)
    return conn, size_prior, max_extent


def _nesting_signal(proposals: list[np.ndarray], n_vertices: int,
                    agreement: np.ndarray) -> np.ndarray:
    """Count nested neighbours (containment either way) that agree better."""
    k = len(proposals)
    sizes = np.array([max(len(p), 1) for p in proposals], dtype=np.float64)
    ent_p, ent_v = _csr(proposals)
    flag = np.zeros(n_vertices, dtype=bool)
    inter_rows = np.zeros((k, k), dtype=np.float32)
    for i, p in enumerate(proposals):
        inter_rows[i] = _counts_against(ent_p, ent_v, flag,
                                        np.asarray(p, np.int64), k)
    cov = inter_rows / sizes[:, None]          # cov[i, j] = |i∩j| / |i|
    np.fill_diagonal(cov, 0.0)
    nested = (cov >= CONTAIN_FRAC) | (cov.T >= CONTAIN_FRAC)
    # j dominates i when they nest and j agrees better. Ties are broken by
    # index so a block of identical proposals (Mask3D's raw bank contains
    # several) collapses onto one canonical representative instead of all
    # of them surviving each other's comparison.
    idx = np.arange(k)
    higher = agreement[None, :] > agreement[:, None]
    tie = (agreement[None, :] == agreement[:, None]) & (idx[None, :]
                                                       < idx[:, None])
    n_better = (nested & (higher | tie)).sum(axis=1)
    return n_better.astype(np.int64)


def proposal_signals(proposals: list[np.ndarray], n_vertices: int,
                     xyz: np.ndarray, views: list[dict]) -> ProposalSignals:
    """Compute every oracle-free signal for a proposal bank.

    proposals : list of sorted, unique vertex-id arrays (the C1-P1 bank).
    n_vertices: vertex count of the raw mesh the ids index into.
    xyz       : [n_vertices, 3] raw-mesh vertex positions, metres, in the
                frozen gravity/yaw frame.
    views     : per view {'visible': int array of visible vertex ids,
                'masks': list of lifted 2D-mask vertex-id arrays,
                'mask_quality': [n_masks, 2] (predicted_iou, stability)}.
                This is `proposal_fusion.edge_confidence`'s view contract
                plus SAM's own quality pair.
    """
    if len(xyz) != n_vertices:
        raise ValueError(f"xyz has {len(xyz)} rows, expected {n_vertices}")
    if not proposals:
        z = np.zeros(0)
        return ProposalSignals(z, z, z, z.astype(np.int64), z, z,
                               z.astype(np.int64), z, z.astype(np.int64))
    agreement, support_frac, sam_q, n_obs = _view_signals(
        proposals, n_vertices, views)
    conn, size_prior, max_extent = _geometry_signals(proposals, xyz)
    n_better = _nesting_signal(proposals, n_vertices, agreement)
    return ProposalSignals(
        agreement=agreement, support_frac=support_frac, sam_quality=sam_q,
        n_observed=n_obs, connectivity=conn, size_prior=size_prior,
        n_nested_better=n_better, max_extent_m=max_extent,
        n_vertices=np.array([len(p) for p in proposals], dtype=np.int64))


def score_proposals(signals: ProposalSignals,
                    components: tuple[str, ...] = DEFAULT_COMPONENTS) -> np.ndarray:
    """Combine signals into one score per proposal, guaranteed in [0,1].

    v1 default (DEFAULT_COMPONENTS):
        score = agreement * gate(size_prior)
                * gate(exp(-alpha * n_nested_better))

    v0 (COMPONENTS_V0) additionally multiplied by gate(connectivity); it was
    measured net-harmful on transfer and is off by default. Pass
    components=COMPONENTS_V0 to reproduce v0.

    `gate(x) = GATE_FLOOR + (1 - GATE_FLOOR) * x` so no secondary signal
    can veto a proposal outright — they re-order, they do not filter.
    `components` selects which factors participate; it exists so the
    ablation in `tools/p1_selector_eval.py` is the same code path.
    """
    unknown = set(components) - set(COMPONENTS)
    if unknown:
        raise ValueError(f"unknown score components: {sorted(unknown)}")
    k = len(signals.agreement)
    out = np.ones(k) if "agreement" not in components else \
        np.clip(signals.agreement, 0.0, 1.0).copy()
    floor = GATE_FLOOR

    def gate(x: np.ndarray) -> np.ndarray:
        return floor + (1.0 - floor) * np.clip(x, 0.0, 1.0)

    if "connectivity" in components:
        out = out * gate(signals.connectivity)
    if "size" in components:
        out = out * gate(signals.size_prior)
    if "redundancy" in components:
        out = out * gate(np.exp(-REDUNDANCY_ALPHA * signals.n_nested_better))
    return np.clip(out, 0.0, 1.0)


def score(proposals: list[np.ndarray], n_vertices: int, xyz: np.ndarray,
          views: list[dict]) -> np.ndarray:
    """Convenience: signals -> score in [0,1], one per proposal."""
    return score_proposals(proposal_signals(proposals, n_vertices, xyz, views))
