"""Detection-repair evaluator — does a repair bank recover entities Mask3D missed?

This module answers ONE question: when repair proposals are pooled BESIDE a
frozen Mask3D bank, does the pool match annotated entities the baseline bank
did not, without destroying what the baseline already had?

It is deliberately NOT a labelling, relation, or QA evaluator. Detection
metrics here say nothing about downstream answer quality, and this module
emits nothing that could be quoted as an end-to-end result.

WHY THIS IS A SEPARATE MODULE FROM `tools/arkitscenes_mask3d_eval.py`
---------------------------------------------------------------------
That tool scores banks INDEPENDENTLY against a ceiling. The repair question is
comparative and paired: which entities moved, which were preserved, and what
the pooled bank cost in junk. Bolting paired logic onto the bank reporter
would have made the baseline numbers depend on whether a repair arm was
present, and the whole point is that they must not.

THE ORDERING RULE, ENFORCED RATHER THAN DOCUMENTED
--------------------------------------------------
Proposal generation must never see annotations, and the proposal set must be
final before annotations are opened. Both halves are mechanical here:

  * `ProposalArtifact.finalize()` writes the proposal set to disk and stamps
    it with a sha256 over the vertex payload. Nothing in this half imports an
    oracle.
  * `score_bank()` and `compare_banks()` take an already-finalized artifact
    and RE-VERIFY its digest against the bytes on disk before scoring. An
    artifact that was regenerated, appended to, or filtered after the
    annotations were read fails the digest check and raises.

`OracleLike` is a structural type, not an import. This module never learns
the annotation filename, so the repo-level boundary sweep in
`tests/tools/test_arkitscenes_eval.py` needs no exemption for it: entities
are injected by the CLI, which owns the ordering.

DEFINITIONS, fixed here so two banks cannot be scored by two rules
------------------------------------------------------------------
  recovered entity      an annotated entity whose best vertex-IoU against ANY
                        proposal in the bank is >= t. Ranking-free, so it is
                        a ceiling: it says the bank CONTAINS the object, not
                        that a downstream resolver would deliver it.
  unique recovered      recovered by the pooled bank at t, not by baseline.
  preserved             recovered by baseline at t and still recovered by
                        pooled at t. Pooling is additive, so any loss here is
                        a bug in the pooling, not a modelling result.
  zero-overlap          proposals whose best IoU against every entity is
                        below ZERO_OVERLAP_IOU. NOT precision: ARKitScenes
                        boxes are not an exhaustive object inventory, so a
                        real object with no box counts as zero-overlap.
  giant mask            a proposal covering more than GIANT_FRAC of the mesh.
                        Room-sized blobs trivially raise IoU@0.10 against
                        large entities and are the classic way a repair arm
                        fakes a gain.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

# Reporting grid. Matches tools/arkitscenes_mask3d_eval.py so repair numbers
# stay comparable to the recorded Mask3D contract numbers.
IOUS = (0.25, 0.50)
# Prefix sizes for rank-sensitive junk reporting. The development gate is
# stated on 100.
TOP_KS = (100, None)
# Below this, a proposal is counted as overlapping nothing.
ZERO_OVERLAP_IOU = 0.10
# Share of mesh vertices above which a proposal is "giant".
GIANT_FRAC = 0.15

# Ranking rules available for top-k junk reporting. Both are annotation-free.
#   "confidence"  each source's own detector score, descending. Mask3D scores
#                 and repair consensus are NOT calibrated against each other;
#                 this is the ranking a naive pooled system would use, which
#                 is exactly why it is the one under gate.
#   "size"        vertex count descending. Confidence-free robustness check;
#                 reported alongside so the gate is not hostage to the
#                 cross-source calibration assumption above.
RANKINGS = ("confidence", "size")


class OracleLike(Protocol):
    """Structural view of `tools.arkitscenes_eval.OracleEntity`.

    Injected, never imported: this module must stay on the annotation-free
    side of the repo boundary sweep.
    """
    uid: str
    label: str
    vertices: np.ndarray


@dataclass(frozen=True)
class Proposal:
    """One candidate instance: a vertex set plus annotation-free provenance.

    `source` is the bank it came from ("mask3d", "repair"). `kind` is what the
    generator thought it was doing ("baseline", "additional", "split");
    `confidence` is that generator's own score. None of these are read by the
    scoring functions except to group and rank — they exist so a result can be
    attributed to a mechanism instead of to an anonymous blob.
    """
    vertices: np.ndarray
    source: str
    kind: str
    confidence: float
    notes: dict = field(default_factory=dict)


def _digest_proposals(proposals: Sequence[Proposal]) -> str:
    """Hash over vertex payload AND ordering.

    Ordering is included on purpose: top-k junk metrics depend on the rank
    order, so a reordered bank is a different artifact even when the vertex
    sets are identical.
    """
    h = hashlib.sha256()
    for p in proposals:
        v = np.asarray(p.vertices, dtype=np.int64)
        h.update(p.source.encode())
        h.update(b"\x00")
        h.update(p.kind.encode())
        h.update(b"\x00")
        h.update(np.float64(p.confidence).tobytes())
        h.update(v.tobytes())
        h.update(b"\x01")
    return h.hexdigest()


@dataclass(frozen=True)
class ProposalArtifact:
    """A finalized, hash-stamped proposal set.

    Construct with `finalize`, never by hand outside tests: the digest is what
    proves the set predates the annotations.
    """
    name: str
    n_vertices: int
    proposals: tuple[Proposal, ...]
    sha256: str
    path: Path | None
    provenance: dict

    @staticmethod
    def finalize(name: str, proposals: Sequence[Proposal], n_vertices: int,
                 out_path: Path | None = None,
                 provenance: dict | None = None) -> "ProposalArtifact":
        """Freeze a proposal set and write it beside its manifest.

        Called BEFORE anything opens an annotation. Vertex indices are
        range-checked here rather than at scoring time, because an
        out-of-range index would otherwise surface as a silently wrong IoU
        against a mesh this bank does not belong to.
        """
        frozen = tuple(
            Proposal(np.asarray(p.vertices, dtype=np.int64), p.source, p.kind,
                     float(p.confidence), dict(p.notes))
            for p in proposals)
        for p in frozen:
            if p.vertices.size and (p.vertices.min() < 0
                                    or p.vertices.max() >= n_vertices):
                raise ValueError(
                    f"{name}: proposal vertex index out of range for a mesh "
                    f"of {n_vertices} vertices")
        digest = _digest_proposals(frozen)
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            offsets = np.zeros(len(frozen) + 1, dtype=np.int64)
            for i, p in enumerate(frozen):
                offsets[i + 1] = offsets[i] + len(p.vertices)
            verts = (np.concatenate([p.vertices for p in frozen])
                     if frozen else np.zeros(0, dtype=np.int64))
            np.savez_compressed(
                out_path, vertices=verts, offsets=offsets,
                source=np.array([p.source for p in frozen]),
                kind=np.array([p.kind for p in frozen]),
                confidence=np.array([p.confidence for p in frozen],
                                    dtype=np.float64))
            out_path.with_suffix(".manifest.json").write_text(json.dumps({
                "name": name,
                "n_proposals": len(frozen),
                "n_vertices": n_vertices,
                "proposal_sha256": digest,
                "annotations_read": False,
                "provenance": provenance or {},
            }, indent=1, sort_keys=True) + "\n")
        return ProposalArtifact(name, int(n_vertices), frozen, digest,
                                out_path, dict(provenance or {}))

    def verify(self) -> None:
        """Re-derive the digest, and re-read the manifest if one was written.

        This is the gate that makes "finalized before the oracle opened"
        checkable instead of merely asserted.
        """
        actual = _digest_proposals(self.proposals)
        if actual != self.sha256:
            raise ValueError(
                f"{self.name}: proposal set changed after finalize "
                f"({self.sha256[:12]}… -> {actual[:12]}…); it was mutated "
                "after the artifact was frozen")
        if self.path is not None:
            manifest_path = self.path.with_suffix(".manifest.json")
            if not manifest_path.is_file():
                raise ValueError(
                    f"{self.name}: finalized artifact has no manifest at "
                    f"{manifest_path}")
            manifest = json.loads(manifest_path.read_text())
            if manifest["proposal_sha256"] != self.sha256:
                raise ValueError(
                    f"{self.name}: on-disk manifest records "
                    f"{manifest['proposal_sha256'][:12]}…, in-memory set "
                    f"hashes to {self.sha256[:12]}…")

    def pooled_with(self, other: "ProposalArtifact", name: str,
                    out_path: Path | None = None) -> "ProposalArtifact":
        """Concatenate two finalized artifacts into a third.

        Both inputs are verified first, so a pool can never launder a bank
        that was edited after freezing.
        """
        self.verify()
        other.verify()
        if self.n_vertices != other.n_vertices:
            raise ValueError(
                f"cannot pool {self.name} ({self.n_vertices} vertices) with "
                f"{other.name} ({other.n_vertices} vertices)")
        return ProposalArtifact.finalize(
            name, list(self.proposals) + list(other.proposals),
            self.n_vertices, out_path,
            {"pooled_from": {self.name: self.sha256, other.name: other.sha256}})


# --------------------------------------------------------------------
# ORACLE-CONSUMING HALF — entities are injected; nothing here opens a file
# --------------------------------------------------------------------
def iou_matrix(proposals: Sequence[Proposal],
               entities: Sequence[OracleLike]) -> np.ndarray:
    """[n_proposals, n_entities] vertex-set IoU.

    Same definition as `tools.arkitscenes_eval.iou_matrix`, recomputed here
    over `Proposal` objects rather than bare arrays. Vertex sets are sorted
    and unique by construction, so intersection is a merge, not a hash join.
    """
    out = np.zeros((len(proposals), len(entities)), dtype=np.float64)
    ent = [np.asarray(e.vertices, dtype=np.int64) for e in entities]
    for i, p in enumerate(proposals):
        ps = np.asarray(p.vertices, dtype=np.int64)
        if ps.size == 0:
            continue
        for j, es in enumerate(ent):
            if es.size == 0:
                continue
            inter = np.intersect1d(ps, es, assume_unique=False).size
            if inter:
                out[i, j] = inter / (ps.size + es.size - inter)
    return out


def rank_order(proposals: Sequence[Proposal], ranking: str) -> np.ndarray:
    """Annotation-free deterministic ordering. See RANKINGS."""
    if ranking not in RANKINGS:
        raise ValueError(f"unknown ranking {ranking!r}; supported: {RANKINGS}")
    sizes = np.array([len(p.vertices) for p in proposals], dtype=np.int64)
    if ranking == "confidence":
        keys = np.array([p.confidence for p in proposals], dtype=np.float64)
    else:
        keys = sizes.astype(np.float64)
    # Ties break by size then original index, so the order is total and
    # reproducible rather than dependent on numpy's sort stability alone.
    return np.array(sorted(range(len(proposals)),
                           key=lambda i: (-keys[i], -sizes[i], i)),
                    dtype=np.int64)


def _recovered_uids(ious: np.ndarray, entities: Sequence[OracleLike],
                    t: float, rows: np.ndarray | None = None) -> list[str]:
    if not len(entities):
        return []
    sub = ious if rows is None else ious[rows]
    if not len(sub):
        return []
    best = sub.max(axis=0)
    return [entities[j].uid for j in np.flatnonzero(best >= t)]


def score_bank(artifact: ProposalArtifact,
               entities: Sequence[OracleLike]) -> dict:
    """Per-bank metrics. Verifies the artifact digest before reading entities."""
    artifact.verify()
    proposals = artifact.proposals
    ious = iou_matrix(proposals, entities)
    sizes = np.array([len(p.vertices) for p in proposals], dtype=np.int64)
    n_ent = len(entities)
    best_per_proposal = (ious.max(axis=1) if n_ent and len(proposals)
                         else np.zeros(len(proposals)))

    recovered = {f"{t:.2f}": _recovered_uids(ious, entities, t) for t in IOUS}
    junk = {}
    for ranking in RANKINGS:
        order = rank_order(proposals, ranking)
        per_k = {}
        for k in TOP_KS:
            sel = order if k is None else order[:k]
            key = "all" if k is None else str(k)
            if not len(sel):
                per_k[key] = {"n": 0, "zero_overlap_rate": None}
                continue
            zero = float((best_per_proposal[sel] < ZERO_OVERLAP_IOU).mean())
            per_k[key] = {"n": int(len(sel)),
                          "zero_overlap_rate": round(zero, 4)}
        junk[ranking] = per_k

    by_source: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    for p in proposals:
        by_source[p.source] = by_source.get(p.source, 0) + 1
        by_kind[p.kind] = by_kind.get(p.kind, 0) + 1

    return {
        "bank": artifact.name,
        "proposal_sha256": artifact.sha256,
        "n_proposals": len(proposals),
        "n_entities": n_ent,
        "counts_by_source": dict(sorted(by_source.items())),
        "counts_by_kind": dict(sorted(by_kind.items())),
        "median_proposal_vertices": int(np.median(sizes)) if len(sizes) else 0,
        "recovered_entities": {t: sorted(uids) for t, uids in recovered.items()},
        "n_recovered": {t: len(uids) for t, uids in recovered.items()},
        "giant_mask_rate": (
            round(float((sizes > GIANT_FRAC * artifact.n_vertices).mean()), 4)
            if len(sizes) else 0.0),
        "n_giant_masks": int((sizes > GIANT_FRAC * artifact.n_vertices).sum())
                          if len(sizes) else 0,
        "zero_overlap": junk,
    }


def _entity_index(entities: Sequence[OracleLike]) -> dict[str, OracleLike]:
    return {e.uid: e for e in entities}


def compare_banks(baseline: ProposalArtifact, pooled: ProposalArtifact,
                  entities: Sequence[OracleLike]) -> dict:
    """Paired baseline-vs-pooled report: what moved, what was preserved, what it cost.

    `pooled` is expected to CONTAIN `baseline` — the repair arm is additive by
    construction. That is asserted rather than assumed, because a pool that
    silently replaced the baseline would produce a "preserved" number that
    means nothing.
    """
    baseline.verify()
    pooled.verify()
    base_digests = {_digest_proposals([p]) for p in baseline.proposals}
    pooled_digests = [_digest_proposals([p]) for p in pooled.proposals]
    missing = base_digests - set(pooled_digests)
    if missing:
        raise ValueError(
            f"{len(missing)} baseline proposals are absent from the pooled "
            "bank; the repair arm must add beside Mask3D, not replace it")

    base_report = score_bank(baseline, entities)
    pooled_report = score_bank(pooled, entities)
    index = _entity_index(entities)

    moved = {}
    for t in IOUS:
        key = f"{t:.2f}"
        b = set(base_report["recovered_entities"][key])
        p = set(pooled_report["recovered_entities"][key])
        unique = sorted(p - b)
        lost = sorted(b - p)
        moved[key] = {
            "baseline_recovered": len(b),
            "pooled_recovered": len(p),
            "unique_recovered": unique,
            "n_unique_recovered": len(unique),
            "unique_recovered_labels": [index[u].label for u in unique],
            "preserved_baseline_matches": len(b & p),
            "lost_baseline_matches": lost,
            "n_lost_baseline_matches": len(lost),
            "all_baseline_matches_preserved": not lost,
        }

    junk_delta = {
        ranking: {
            k: (None if (pooled_report["zero_overlap"][ranking][k]
                         ["zero_overlap_rate"] is None
                         or base_report["zero_overlap"][ranking][k]
                         ["zero_overlap_rate"] is None)
                else round(
                    pooled_report["zero_overlap"][ranking][k]["zero_overlap_rate"]
                    - base_report["zero_overlap"][ranking][k]["zero_overlap_rate"],
                    4))
            for k in base_report["zero_overlap"][ranking]
        }
        for ranking in RANKINGS
    }

    return {
        "baseline": base_report,
        "pooled": pooled_report,
        "entity_movement": moved,
        "zero_overlap_delta": junk_delta,
        "giant_mask_delta": round(pooled_report["giant_mask_rate"]
                                  - base_report["giant_mask_rate"], 4),
        "proposal_count_delta": (pooled_report["n_proposals"]
                                 - base_report["n_proposals"]),
        "interpretation_limit": (
            "Detection metrics only. These numbers do not measure labelling, "
            "relations, or end-to-end QA, and must not be quoted as such."),
    }


# --------------------------------------------------------------------
# Development gates, stated before the run
# --------------------------------------------------------------------
GATE_MIN_UNIQUE_AT_050 = 2
GATE_MAX_TOP100_ZERO_OVERLAP_WORSENING = 0.10
GATE_RANKING = "confidence"


def development_gates(comparison: dict,
                      audited_cases_hit: Sequence[str] | None = None) -> dict:
    """Evaluate the five pre-declared development conditions.

    `audited_cases_hit` is supplied by the caller from the human-feedback
    record; it is not derivable from annotation IoU, because some audited
    cases (curtains) have no annotation box at all.
    """
    moved = comparison["entity_movement"]["0.50"]
    pooled = comparison["pooled"]
    top100 = comparison["zero_overlap_delta"][GATE_RANKING]["100"]
    audited = list(audited_cases_hit or [])
    gates = {
        "adds_two_unique_matches_at_050": {
            "value": moved["n_unique_recovered"],
            "threshold": GATE_MIN_UNIQUE_AT_050,
            "pass": moved["n_unique_recovered"] >= GATE_MIN_UNIQUE_AT_050,
        },
        "covers_an_audited_case": {
            "value": audited,
            "threshold": 1,
            "pass": len(audited) >= 1,
        },
        "preserves_all_baseline_matches": {
            "value": moved["lost_baseline_matches"],
            "threshold": 0,
            "pass": moved["all_baseline_matches_preserved"],
        },
        "giant_mask_rate_zero": {
            "value": pooled["giant_mask_rate"],
            "threshold": 0.0,
            "pass": pooled["giant_mask_rate"] == 0.0,
        },
        "top100_zero_overlap_worsens_at_most_10pp": {
            "value": top100,
            "threshold": GATE_MAX_TOP100_ZERO_OVERLAP_WORSENING,
            "ranking": GATE_RANKING,
            "pass": (top100 is not None
                     and top100 <= GATE_MAX_TOP100_ZERO_OVERLAP_WORSENING),
        },
    }
    return {
        "gates": gates,
        "all_pass": all(g["pass"] for g in gates.values()),
        "failed": sorted(k for k, g in gates.items() if not g["pass"]),
    }
