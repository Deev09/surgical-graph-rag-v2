"""Synthetic detection-repair evaluator tests — no dataset, always runnable.

Every case is a hand-built toy mesh of numbered vertices with hand-built
"annotation" entities, so the expected metric is arithmetic rather than a
recorded number. The four scenarios the repair arm must be measurable on:

  missing-object recovery   a real entity the baseline bank does not contain,
                            recovered only after the repair proposal is pooled
  overmerge splitting       one baseline blob spanning two entities; splitting
                            it must recover the second WITHOUT dropping the
                            first, which is the failure mode a naive
                            replace-the-blob repair produces
  junk fragments            proposals overlapping nothing must show up in the
                            zero-overlap rate and must not silently improve
                            recovery
  giant masks               a room-sized proposal must be counted, because it
                            is the cheapest way to fake IoU@0.10 gains

Plus the ordering invariant that makes the whole evaluation trustworthy: a
proposal set mutated after finalize must fail its digest check.
"""
from __future__ import annotations

import ast
import json
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from eval.detection_repair import (
    GIANT_FRAC, Proposal, ProposalArtifact, compare_banks,
    development_gates, iou_matrix, rank_order, score_bank,
)

N_VERTICES = 1000


@dataclass(frozen=True)
class FakeEntity:
    """Structural stand-in for tools.arkitscenes_eval.OracleEntity."""
    uid: str
    label: str
    vertices: np.ndarray


def _ent(uid: str, label: str, lo: int, hi: int) -> FakeEntity:
    return FakeEntity(uid, label, np.arange(lo, hi, dtype=np.int64))


def _prop(lo: int, hi: int, source: str = "mask3d", kind: str = "baseline",
          confidence: float = 0.9) -> Proposal:
    return Proposal(np.arange(lo, hi, dtype=np.int64), source, kind, confidence)


def _finalize(name: str, proposals: list[Proposal]) -> ProposalArtifact:
    return ProposalArtifact.finalize(name, proposals, N_VERTICES)


def test_missing_object_is_recovered_only_after_pooling() -> None:
    """The core repair claim, in its simplest form."""
    entities = [_ent("e_sofa", "sofa", 0, 100), _ent("e_lamp", "lamp", 200, 300)]
    baseline = _finalize("mask3d", [_prop(0, 100)])
    repair = ProposalArtifact.finalize(
        "repair", [_prop(200, 300, "repair", "additional", 0.7)], N_VERTICES)
    pooled = baseline.pooled_with(repair, "pooled")

    cmp = compare_banks(baseline, pooled, entities)
    moved = cmp["entity_movement"]["0.50"]
    assert moved["baseline_recovered"] == 1, moved
    assert moved["pooled_recovered"] == 2, moved
    assert moved["unique_recovered"] == ["e_lamp"], moved
    assert moved["unique_recovered_labels"] == ["lamp"], moved
    assert moved["all_baseline_matches_preserved"], moved
    assert cmp["proposal_count_delta"] == 1, cmp["proposal_count_delta"]


def test_overmerge_split_recovers_second_entity_and_keeps_the_first() -> None:
    """One blob over two entities.

    The baseline blob spans 0..200 while the entities are 0..100 and 100..200,
    so its IoU against each is 0.5 — it matches BOTH at 0.50 and neither above.
    Splitting must therefore be scored on the pooled bank containing the split
    pieces, and the naive repair that emits only one piece must not be able to
    lose the other. Both pieces are emitted, and both entities must survive.
    """
    entities = [_ent("e_counter", "counter", 0, 100),
                _ent("e_cabinet", "cabinet", 100, 200)]
    baseline = _finalize("mask3d", [_prop(0, 200)])
    base_report = score_bank(baseline, entities)
    # A 0.5-IoU blob is exactly at threshold: both entities count as recovered.
    assert base_report["n_recovered"]["0.50"] == 2, base_report["n_recovered"]

    # Repair splits it. Pieces are near-perfect, so recovery survives with a
    # far larger margin; the metric that must move is the per-entity best IoU.
    repair = ProposalArtifact.finalize("repair", [
        _prop(0, 100, "repair", "split", 0.8),
        _prop(100, 200, "repair", "split", 0.8),
    ], N_VERTICES)
    pooled = baseline.pooled_with(repair, "pooled")
    cmp = compare_banks(baseline, pooled, entities)
    moved = cmp["entity_movement"]["0.50"]
    assert moved["all_baseline_matches_preserved"], moved
    assert moved["pooled_recovered"] == 2, moved

    pooled_ious = iou_matrix(pooled.proposals, entities)
    assert np.isclose(pooled_ious.max(axis=0), 1.0).all(), pooled_ious.max(axis=0)

    # And the stricter case the audited kitchen counter actually looks like:
    # a blob so much bigger than the entity that it recovers nothing.
    wide = _finalize("mask3d_wide", [_prop(0, 900)])
    wide_report = score_bank(wide, entities)
    assert wide_report["n_recovered"]["0.50"] == 0, wide_report["n_recovered"]
    wide_pooled = wide.pooled_with(
        ProposalArtifact.finalize(
            "repair", [_prop(0, 100, "repair", "split", 0.8)], N_VERTICES),
        "pooled_wide")
    wide_cmp = compare_banks(wide, wide_pooled, entities)
    assert wide_cmp["entity_movement"]["0.50"]["unique_recovered"] == \
        ["e_counter"], wide_cmp["entity_movement"]["0.50"]


def test_junk_fragments_raise_zero_overlap_without_helping_recovery() -> None:
    """Twenty proposals over empty space: recovery flat, junk rate up."""
    entities = [_ent("e_sofa", "sofa", 0, 100)]
    baseline = _finalize("mask3d", [_prop(0, 100)])
    junk = ProposalArtifact.finalize("repair", [
        _prop(500 + 10 * i, 505 + 10 * i, "repair", "additional", 0.5)
        for i in range(20)
    ], N_VERTICES)
    pooled = baseline.pooled_with(junk, "pooled")
    cmp = compare_banks(baseline, pooled, entities)

    moved = cmp["entity_movement"]["0.50"]
    assert moved["n_unique_recovered"] == 0, moved
    assert moved["all_baseline_matches_preserved"], moved
    base_zero = cmp["baseline"]["zero_overlap"]["confidence"]["all"]
    pooled_zero = cmp["pooled"]["zero_overlap"]["confidence"]["all"]
    assert base_zero["zero_overlap_rate"] == 0.0, base_zero
    # 20 junk of 21 proposals
    assert np.isclose(pooled_zero["zero_overlap_rate"], 20 / 21, atol=1e-4), \
        pooled_zero
    assert cmp["zero_overlap_delta"]["confidence"]["all"] > 0.9, \
        cmp["zero_overlap_delta"]

    # The declared gate must fail on this bank for both reasons.
    verdict = development_gates(cmp)
    assert not verdict["all_pass"], verdict
    assert "adds_two_unique_matches_at_050" in verdict["failed"], verdict
    assert "top100_zero_overlap_worsens_at_most_10pp" in verdict["failed"], \
        verdict


def test_giant_masks_are_counted_and_fail_the_gate() -> None:
    """A room-sized proposal is the cheapest way to fake a recovery gain."""
    entities = [_ent("e_sofa", "sofa", 0, 100), _ent("e_rug", "rug", 700, 800)]
    baseline = _finalize("mask3d", [_prop(0, 100)])
    threshold = int(GIANT_FRAC * N_VERTICES)
    giant = ProposalArtifact.finalize(
        "repair", [_prop(0, threshold + 50, "repair", "additional", 0.6)],
        N_VERTICES)
    pooled = baseline.pooled_with(giant, "pooled")
    report = score_bank(pooled, entities)
    assert report["n_giant_masks"] == 1, report
    assert np.isclose(report["giant_mask_rate"], 0.5), report["giant_mask_rate"]

    cmp = compare_banks(baseline, pooled, entities)
    verdict = development_gates(cmp, audited_cases_hit=["fake"])
    assert "giant_mask_rate_zero" in verdict["failed"], verdict

    # A bank with no giant mask must report exactly zero, not "small".
    clean = score_bank(baseline, entities)
    assert clean["giant_mask_rate"] == 0.0, clean


def test_pooling_must_be_additive() -> None:
    """A repair arm that replaces baseline proposals is rejected outright."""
    entities = [_ent("e_sofa", "sofa", 0, 100)]
    baseline = _finalize("mask3d", [_prop(0, 100)])
    replacement = _finalize(
        "replaced", [_prop(0, 90, "repair", "additional", 0.9)])
    try:
        compare_banks(baseline, replacement, entities)
    except ValueError as exc:
        assert "absent from the pooled bank" in str(exc), exc
    else:
        raise AssertionError("replacing the baseline should have raised")


def test_artifact_mutated_after_finalize_fails_verification() -> None:
    """The ordering guarantee: a set edited after freezing cannot be scored."""
    entities = [_ent("e_sofa", "sofa", 0, 100)]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "bank.npz"
        artifact = ProposalArtifact.finalize(
            "mask3d", [_prop(0, 100)], N_VERTICES, path)
        artifact.verify()
        score_bank(artifact, entities)

        manifest = json.loads(path.with_suffix(".manifest.json").read_text())
        assert manifest["proposal_sha256"] == artifact.sha256, manifest
        assert manifest["annotations_read"] is False, manifest

        # Same vertex sets, one extra proposal appended after the freeze.
        tampered = ProposalArtifact(
            artifact.name, artifact.n_vertices,
            artifact.proposals + (_prop(200, 300, "repair", "additional", 1.0),),
            artifact.sha256, artifact.path, artifact.provenance)
        try:
            score_bank(tampered, entities)
        except ValueError as exc:
            assert "changed after finalize" in str(exc), exc
        else:
            raise AssertionError("a mutated artifact should not score")

        # And a manifest that disagrees with the in-memory digest.
        path.with_suffix(".manifest.json").write_text(
            json.dumps({**manifest, "proposal_sha256": "0" * 64}) + "\n")
        try:
            artifact.verify()
        except ValueError as exc:
            assert "on-disk manifest records" in str(exc), exc
        else:
            raise AssertionError("a disagreeing manifest should not verify")


def test_out_of_range_vertices_are_rejected_at_finalize() -> None:
    """Catches a bank built against a different mesh before it scores 0.0."""
    try:
        _finalize("bad", [_prop(N_VERTICES - 10, N_VERTICES + 10)])
    except ValueError as exc:
        assert "out of range" in str(exc), exc
    else:
        raise AssertionError("an out-of-range proposal should not finalize")


def test_rankings_are_deterministic_and_annotation_free() -> None:
    """Top-k junk numbers depend on rank order, so the order must be total."""
    proposals = [
        _prop(0, 50, confidence=0.5),
        _prop(100, 300, confidence=0.5),     # same score, larger
        _prop(400, 410, confidence=0.9),
    ]
    artifact = _finalize("bank", proposals)
    by_conf = rank_order(artifact.proposals, "confidence")
    assert list(by_conf) == [2, 1, 0], by_conf     # 0.9 first, then size tiebreak
    by_size = rank_order(artifact.proposals, "size")
    assert list(by_size) == [1, 0, 2], by_size
    assert list(rank_order(artifact.proposals, "confidence")) == list(by_conf)


def test_full_gate_sheet_passes_on_a_clean_synthetic_repair() -> None:
    """A repair arm that does the right thing must actually clear the gates."""
    entities = [_ent("e_sofa", "sofa", 0, 100),
                _ent("e_lamp", "lamp", 200, 300),
                _ent("e_rug", "rug", 400, 500)]
    baseline = _finalize("mask3d", [_prop(0, 100), _prop(600, 650)])
    repair = ProposalArtifact.finalize("repair", [
        _prop(200, 300, "repair", "additional", 0.7),
        _prop(400, 500, "repair", "split", 0.7),
    ], N_VERTICES)
    pooled = baseline.pooled_with(repair, "pooled")
    cmp = compare_banks(baseline, pooled, entities)
    verdict = development_gates(cmp, audited_cases_hit=["e_rug"])
    assert verdict["all_pass"], verdict
    assert cmp["entity_movement"]["0.50"]["n_unique_recovered"] == 2, cmp
    # Baseline already carried one junk proposal, so pooling real repairs
    # IMPROVES the junk rate here; the gate is one-sided on purpose.
    assert cmp["zero_overlap_delta"]["confidence"]["100"] < 0, \
        cmp["zero_overlap_delta"]


def test_evaluator_never_names_or_imports_the_oracle() -> None:
    """The evaluator core stays on the annotation-free side of the boundary.

    Two checks, because they catch different mistakes. The substring sweep
    catches a hand-rolled annotation reader; the AST import scan catches the
    likelier one -- importing `tools.arkitscenes_eval` for convenience, which
    would let this module load annotations itself and quietly destroy the
    finalize-then-open ordering. Docstring cross-references to that module are
    fine and are deliberately not forbidden.
    """
    path = REPO_ROOT / "eval" / "detection_repair.py"
    text = path.read_text()
    for forbidden in ("3dod_annotation", "load_oracle_entities"):
        assert forbidden not in text, f"{path.name} references {forbidden!r}"

    imported: list[str] = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    offenders = [m for m in imported
                 if m.startswith(("tools", "adapters", "eval.")) ]
    assert not offenders, \
        f"{path.name} imports {offenders}; entities must be injected"


TESTS = [
    test_missing_object_is_recovered_only_after_pooling,
    test_overmerge_split_recovers_second_entity_and_keeps_the_first,
    test_junk_fragments_raise_zero_overlap_without_helping_recovery,
    test_giant_masks_are_counted_and_fail_the_gate,
    test_pooling_must_be_additive,
    test_artifact_mutated_after_finalize_fails_verification,
    test_out_of_range_vertices_are_rejected_at_finalize,
    test_rankings_are_deterministic_and_annotation_free,
    test_full_gate_sheet_passes_on_a_clean_synthetic_repair,
    test_evaluator_never_names_or_imports_the_oracle,
]


def main() -> int:
    failed = 0
    for test in TESTS:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
            print()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
