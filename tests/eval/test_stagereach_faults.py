#!/usr/bin/env python3
"""G-FAULT gate — evaluator-masked artifact-level fault localization.

The gate (freeze doc §9): 24/24 controlled injected faults localized
(8 fault classes x 3 relation types: NEAR, ON_ENTITY_SURFACE, ATTACHED_TO)
while the evaluator was masked to the injected class, plus zero failures on
clean artifacts.

Masking discipline verified structurally here: each injector mutates
EXACTLY ONE artifact of the committed chain, `evaluate_artifacts` receives
only the artifact chain (no fault label exists anywhere in its inputs), and
the expected label is compared against the attribution only AFTER
evaluation.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.stagereach import faults  # noqa: E402
from eval.stagereach.evaluator import attribute  # noqa: E402

FIXTURE_PATH = REPO_ROOT / faults.FIXTURE_RELPATH

_CACHE: dict = {}


def _fixture() -> dict:
    if "fx" not in _CACHE:
        _CACHE["fx"] = json.loads(FIXTURE_PATH.read_text())
    return _CACHE["fx"]


def test_committed_fixture_is_current_and_schema_stamped():
    """The committed fixture must be exactly what build_fixture() produces
    (deterministic chain), byte-for-byte under the house serialization."""
    fx = _fixture()
    assert fx["schema"] == faults.FIXTURE_SCHEMA
    assert fx["schema_version"] == faults.FIXTURE_SCHEMA_VERSION
    assert FIXTURE_PATH.read_bytes() == faults.fixture_bytes(
        faults.build_fixture())


def test_fixture_shape_nine_questions_three_relations_mixed_expectations():
    fx = _fixture()
    qs = fx["evaluation_key"]["questions"]
    assert len(qs) == 9
    by_rel: dict[str, list] = {}
    for q in qs:
        by_rel.setdefault(q["relation"], []).append(q["expected_outcome"])
    assert sorted(by_rel) == ["ATTACHED_TO", "NEAR", "ON_ENTITY_SURFACE"]
    for rel, outcomes in by_rel.items():
        assert sorted(outcomes) == ["answer", "answer", "empty"], rel
    # the chain is internally consistent: serialized edges mirror the
    # computed relations, and every grounded anchor resolves by name
    rels = {tuple(t) for t in fx["relation_artifact"]["relations"]}
    edges = {(e["relation"], e["subject"], e["object"])
             for e in fx["serialized_graph"]["edges"]}
    assert rels == edges
    names = {e["uid"]: e["name"] for e in fx["entity_artifact"]["entities"]}
    for qid, cand in fx["grounded_candidates"].items():
        assert names[cand["uid"]] == cand["anchor"], qid


def test_relations_are_computed_from_geometry_not_asserted():
    """compute_relations on the committed entity artifact reproduces the
    committed relation artifact — the chain is real, not hand-written."""
    fx = _fixture()
    assert faults.compute_relations(fx["entity_artifact"]["entities"]) == \
        fx["relation_artifact"]["relations"]
    # and all three relation types under test actually occur
    kinds = {t[0] for t in fx["relation_artifact"]["relations"]}
    assert {"NEAR", "ON_ENTITY_SURFACE", "ATTACHED_TO"} <= kinds


def test_clean_artifacts_produce_zero_failures():
    traces = faults.evaluate_artifacts(_fixture())
    assert len(traces) == 9
    attrs = faults.attributions(traces)
    assert all(a is None for a in attrs.values()), attrs
    for t in traces.values():
        assert t.result in ("correct",), t.question_id


def test_each_injection_mutates_exactly_one_artifact():
    fx = _fixture()
    for name, inject in faults.INJECTIONS.items():
        for rel in faults.RELATION_VOCAB:
            mutated = inject(fx, rel)
            changed = [k for k in faults.ARTIFACT_KEYS
                       if mutated[k] != fx[k]]
            assert len(changed) == 1, (name, rel, changed)
            # the frozen ground-truth block is never touched
            assert mutated["ground_truth"] == fx["ground_truth"], (name,
                                                                   rel)
    # ambiguous_key mutates the KEY artifact specifically
    mutated = faults.inject_ambiguous_key(fx, "NEAR")
    assert mutated["evaluation_key"] != fx["evaluation_key"]


def test_gate_fault_battery_localizes_twenty_four_of_twenty_four():
    """The go/no-go: 24/24 masked localizations, zero clean failures."""
    b = faults.run_battery(_fixture())
    assert b["n_total"] == 24
    assert b["n_fault_classes"] == 8
    assert b["n_relation_types"] == 3
    assert b["clean_failures"] == 0
    misses = [c for c in b["cells"] if not c["localized"]]
    assert b["n_localized"] == 24, f"unlocalized cells: {misses}"


def test_masked_attribution_matches_hidden_label_only_post_hoc():
    """Re-run one cell per class by hand: evaluate first (no label in any
    input), read the hidden expected label strictly afterwards."""
    fx = _fixture()
    for name, inject in sorted(faults.INJECTIONS.items()):
        mutated = inject(fx, "ON_ENTITY_SURFACE")
        # nothing in the evaluator's input names the fault
        assert name not in json.dumps(mutated)
        traces = faults.evaluate_artifacts(mutated)
        got = attribute(traces[faults.TARGET_QID["ON_ENTITY_SURFACE"]])
        expected_stage, expected_status = faults.EXPECTED_LOCALIZATION[name]
        assert got is not None, name
        assert (got["stage"], got["status"]) == (expected_stage,
                                                 expected_status), name


def test_pre_vs_at_serialization_faults_are_distinguished():
    """The pair the freeze doc singles out: a relation absent BEFORE
    serialization attributes to relation_correctness; a correct computed
    relation omitted (or predicate-corrupted) AT serialization attributes
    to serialization_consistency — for every relation type."""
    fx = _fixture()
    for rel in faults.RELATION_VOCAB:
        target = faults.TARGET_QID[rel]
        pre = faults.inject_relation_deletion_pre_serialization(fx, rel)
        a = attribute(faults.evaluate_artifacts(pre)[target])
        assert a["stage"] == "relation_correctness", (rel, a)
        for at_serialization in (faults.inject_serialization_corruption,
                                 faults.inject_predicate_change):
            m = at_serialization(fx, rel)
            a = attribute(faults.evaluate_artifacts(m)[target])
            assert a["stage"] == "serialization_consistency", (rel, a)
        # in the pre-serialization case the serialized graph ALSO disagrees
        # with the relation artifact, but attribution reports the FIRST
        # gating fail, and serialization does not gate on correctness
        t = faults.evaluate_artifacts(pre)[target]
        assert t.status("serialization_consistency") == "fail"
        assert t.status("relation_correctness") == "fail"


def test_ambiguous_key_localizes_to_eligibility_and_excludes():
    fx = _fixture()
    for rel in faults.RELATION_VOCAB:
        m = faults.inject_ambiguous_key(fx, rel)
        t = faults.evaluate_artifacts(m)[faults.TARGET_QID[rel]]
        assert t.result == "excluded"
        a = attribute(t)
        assert a["stage"] == "key_eligibility"
        # nothing downstream is reached, let alone attributed
        assert t.status("answer_generation") == "not_reached"


TESTS = [
    test_committed_fixture_is_current_and_schema_stamped,
    test_fixture_shape_nine_questions_three_relations_mixed_expectations,
    test_relations_are_computed_from_geometry_not_asserted,
    test_clean_artifacts_produce_zero_failures,
    test_each_injection_mutates_exactly_one_artifact,
    test_gate_fault_battery_localizes_twenty_four_of_twenty_four,
    test_masked_attribution_matches_hidden_label_only_post_hoc,
    test_pre_vs_at_serialization_faults_are_distinguished,
    test_ambiguous_key_localizes_to_eligibility_and_excludes,
]


def main() -> int:
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
            print()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
