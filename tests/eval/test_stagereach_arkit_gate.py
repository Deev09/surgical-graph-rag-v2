#!/usr/bin/env python3
"""G-ARKIT gate tests — the night's go/no-go for the ARKit track.

G-ARKIT-ARMS: the per-arm survival ladders, derived by the StageReach ARKit
adapter INDEPENDENTLY of tools/paper_statistics.py, must reproduce exactly:

    delivered_graph              10 -> 8 -> 8 -> 8 -> 0
    grounded_delivered_graph     10 -> 8 -> 8 -> 8 -> 3 -> 2
    stored_graph_human_identity  10 -> 7
    blinded_rgb_vlm (direct)     10 -> 7

G-ARKIT-LEGACY: a clearly-labeled compatibility function must reproduce the
legacy MIXED ladder 10 -> 8 -> 8 -> 8 -> 3 -> 0 and the held/reached
numbers, asserted field-by-field against the COMMITTED
eval/results/paper_statistics.json — never against paper_statistics.py's
code, which stays byte-for-byte untouched.

If any number here fails to reproduce, the fix is NEVER to weaken the
assertion.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.stagereach import metrics  # noqa: E402
from eval.stagereach.adapters import arkit  # noqa: E402
from eval.stagereach.evaluator import attribute  # noqa: E402
from eval.stagereach.schema import PATHS  # noqa: E402

COMMITTED_STATS = REPO_ROOT / "eval" / "results" / "paper_statistics.json"

_CACHE: dict = {}


def _traces() -> dict:
    if "traces" not in _CACHE:
        report = arkit.load_report(REPO_ROOT)
        _CACHE["report"] = report
        _CACHE["traces"] = arkit.derive_traces(report)
    return _CACHE["traces"]


def _ladder(arm: str, path_id: str) -> list[int]:
    return metrics.ladder_counts(_traces()[arm], PATHS[path_id])


# ------------------------------------------------------------ G-ARKIT-ARMS
def test_gate_arkit_arms_delivered_ladder():
    assert _ladder("delivered_graph", "graph_deployable_delivered") == \
        [10, 8, 8, 8, 0]


def test_gate_arkit_arms_grounded_ladder():
    assert _ladder("grounded_delivered_graph", "graph_deployable_grounded") \
        == [10, 8, 8, 8, 3, 2]


def test_gate_arkit_arms_stored_human_identity_ladder():
    assert _ladder("stored_graph_human_identity", "graph_identity_oracle") \
        == [10, 7]


def test_gate_arkit_arms_direct_rgb_ladder():
    assert _ladder("blinded_rgb_vlm", "direct_rgb") == [10, 7]


def test_gate_arkit_arm_ladder_stages_are_the_declared_ones():
    """The rungs must be the declared gating stages, in path order, with
    the delivered arm's referent_grounding absent because it is UNMEASURED
    there (unknown), not because it passed or was bypassed."""
    t = _traces()
    rungs = metrics.survival_ladder(t["delivered_graph"],
                                    PATHS["graph_deployable_delivered"])
    assert [r["stage"] for r in rungs] == [
        "key_eligibility", "object_delivery", "relation_applicability",
        "serialization_consistency", "answer_generation"]
    scored = [tr for tr in t["delivered_graph"]
              if tr.status("key_eligibility") == "pass"]
    delivered_ok = [tr for tr in scored
                    if tr.status("object_delivery") == "pass"]
    assert all(tr.status("referent_grounding") == "unknown"
               for tr in delivered_ok)
    rungs = metrics.survival_ladder(t["grounded_delivered_graph"],
                                    PATHS["graph_deployable_grounded"])
    assert [r["stage"] for r in rungs] == [
        "key_eligibility", "object_delivery", "relation_applicability",
        "serialization_consistency", "referent_grounding",
        "answer_generation"]


def test_gate_arkit_population_and_scopes():
    t = _traces()
    for arm, path_id, scope in arkit.ARMS:
        traces = t[arm]
        assert len(traces) == 12, arm
        scored = [tr for tr in traces
                  if tr.status("key_eligibility") == "pass"]
        assert len(scored) == 10, arm
        excluded = [tr for tr in traces if tr.result == "excluded"]
        assert len(excluded) == 2, arm
        assert all(tr.scope == scope and tr.path_id == path_id
                   for tr in traces), arm
    # deployable arms never carry an oracle scope
    assert all(tr.scope == "deployable" for tr in t["delivered_graph"])
    assert all(tr.scope == "identity_oracle"
               for tr in t["stored_graph_human_identity"])


def test_gate_arkit_relation_correctness_is_unknown_everywhere():
    """No independent semantic relation annotation exists on ARKit; the
    stage must be unknown (or unreachable), never pass/fail, on every arm
    whose path carries it."""
    for arm, path_id, _ in arkit.ARMS:
        if "relation_correctness" not in PATHS[path_id].stage_names():
            continue
        for tr in _traces()[arm]:
            status = tr.status("relation_correctness")
            assert status in ("unknown", "not_reached"), (arm,
                                                          tr.question_id)
            if status == "unknown":
                assert tr.record("relation_correctness").source == \
                    arkit.NO_RELATION_ANNOTATION


def test_gate_arkit_attribution_names_gating_stages_only():
    """Spot-check attribution against the report's own reading: the two
    set questions die at object_delivery; five grounded questions die at
    referent_grounding; unknown is never attributed."""
    t = _traces()
    by_id = {tr.question_id: tr for tr in t["grounded_delivered_graph"]}
    for qid in ("q25_set_near_sofa", "q42_set_near_window"):
        a = attribute(by_id[qid])
        assert a is not None and a["stage"] == "object_delivery", qid
    grounding_fails = [tr.question_id for tr in by_id.values()
                       if (a := attribute(tr)) is not None
                       and a["stage"] == "referent_grounding"]
    assert len(grounding_fails) == 5, grounding_fails
    for tr in t["delivered_graph"]:
        a = attribute(tr)
        assert a is None or a["stage"] != "relation_correctness"
        assert a is None or a["stage"] != "referent_grounding"


# ---------------------------------------------------------- G-ARKIT-LEGACY
def test_gate_arkit_legacy_ledger_compatibility():
    """LEGACY-LEDGER COMPATIBILITY (never the primary output): the mixed
    ladder and held/reached numbers, field-by-field against the committed
    paper_statistics.json reachability block."""
    committed = json.loads(COMMITTED_STATS.read_text())["reachability"]
    legacy = arkit.legacy_reachability_block(_traces())
    assert "legacy-ledger compatibility" in legacy["label"]

    for field in ("n_scored", "held_by_representation",
                  "reached_by_deployable_grounding",
                  "reached_by_delivered_graph", "reached_by_direct_rgb"):
        assert legacy[field] == committed[field], (
            f"{field}: stagereach {legacy[field]} != committed "
            f"{committed[field]}")

    ours = legacy["survivors_by_stage"]
    theirs = committed["survivors_by_stage"]
    assert len(ours) == len(theirs) == 6
    for o, c in zip(ours, theirs):
        assert o["stage"] == c["stage"], (o["stage"], c["stage"])
        assert o["surviving"] == c["surviving"], (
            f"{o['stage']}: stagereach {o['surviving']} != committed "
            f"{c['surviving']}")
        assert o["lost_here"] == c["lost_here"], (
            f"{o['stage']}: lost_here {o['lost_here']} != committed "
            f"{c['lost_here']}")

    # the exact frozen sequence, spelled out
    assert [s["surviving"] for s in ours] == [10, 8, 8, 8, 3, 0]
    assert (legacy["held_by_representation"],
            legacy["reached_by_deployable_grounding"],
            legacy["reached_by_delivered_graph"],
            legacy["reached_by_direct_rgb"],
            legacy["n_scored"]) == (7, 2, 0, 7, 10)


def test_committed_arkit_artifact_is_current():
    """The committed stagereach artifact must match what the tool produces
    now (byte-compare via the tool's own --check)."""
    import subprocess
    r = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "stagereach_eval.py"),
         "--track", "arkit", "--check"],
        capture_output=True, text=True, cwd=REPO_ROOT)
    assert r.returncode == 0, (
        f"arkit stagereach artifact is stale: {r.stdout}{r.stderr[-300:]}")


def test_generated_paper_numbers_are_current():
    """docs/3dv/sec/generated_numbers.tex must match what the numbers tool
    derives now from the committed artifacts (byte-compare via --check),
    and must never carry de-anonymising strings."""
    import re
    import subprocess
    r = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "stagereach_numbers.py"),
         "--check"],
        capture_output=True, text=True, cwd=REPO_ROOT)
    assert r.returncode == 0, (
        f"generated_numbers.tex is stale: {r.stdout}{r.stderr[-300:]}")
    tex = (REPO_ROOT / "docs" / "3dv" / "sec" /
           "generated_numbers.tex").read_text()
    assert not re.search(r"\bdocs/\w+", tex)
    assert not re.search(r"\btools/\w+\.py", tex)
    assert not re.search(r"\b[0-9a-f]{7,40}\b", tex)
    assert not re.search(r"surgical.graph.rag", tex, re.I)


TESTS = [
    test_gate_arkit_arms_delivered_ladder,
    test_gate_arkit_arms_grounded_ladder,
    test_gate_arkit_arms_stored_human_identity_ladder,
    test_gate_arkit_arms_direct_rgb_ladder,
    test_gate_arkit_arm_ladder_stages_are_the_declared_ones,
    test_gate_arkit_population_and_scopes,
    test_gate_arkit_relation_correctness_is_unknown_everywhere,
    test_gate_arkit_attribution_names_gating_stages_only,
    test_gate_arkit_legacy_ledger_compatibility,
    test_committed_arkit_artifact_is_current,
    test_generated_paper_numbers_are_current,
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
