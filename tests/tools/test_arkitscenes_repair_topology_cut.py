"""Topology-only mask cutting: it must cut on adjacency and nothing else.

The point of these tests is that the cut is decided by mesh connectivity alone.
Two vertices land in different components exactly when the mask contains no
path between them — never because they are far apart, face different ways, or
sit at different depths. A future "improvement" that adds any of those
thresholds should have to delete a test to do it.
"""
from __future__ import annotations

import ast
import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from tools.arkitscenes_repair_topology_cut import connected_components

N = 1000
RUN_ROOT = REPO_ROOT / "runs" / "arkitscenes_repair"


def _scratch():
    return np.zeros(N, dtype=bool), np.full(N, -1, dtype=np.int64)


def _chain(lo: int, hi: int) -> np.ndarray:
    """Edges linking lo..hi-1 in a path."""
    a = np.arange(lo, hi - 1, dtype=np.int64)
    return np.stack([a, a + 1], axis=1)


def test_a_connected_mask_stays_one_component() -> None:
    member, local = _scratch()
    edges = _chain(0, 50)
    out = connected_components(np.arange(0, 50, dtype=np.int64), edges,
                               member, local)
    assert len(out) == 1, [len(c) for c in out]
    assert len(out[0]) == 50


def test_a_mask_spanning_two_disconnected_surfaces_splits() -> None:
    member, local = _scratch()
    edges = np.concatenate([_chain(0, 30), _chain(100, 140)])
    mask = np.concatenate([np.arange(0, 30), np.arange(100, 140)]).astype(np.int64)
    out = connected_components(mask, edges, member, local)
    assert len(out) == 2, [len(c) for c in out]
    assert sorted(len(c) for c in out) == [30, 40]


def test_adjacency_outside_the_mask_does_not_join_components() -> None:
    """The subgraph is INDUCED: a path leaving the mask does not count.

    Vertices 0-9 and 20-29 are connected in the full mesh only through 10-19,
    which this mask excludes. They must come back as two components.
    """
    member, local = _scratch()
    edges = _chain(0, 30)
    mask = np.concatenate([np.arange(0, 10), np.arange(20, 30)]).astype(np.int64)
    out = connected_components(mask, edges, member, local)
    assert len(out) == 2, [len(c) for c in out]


def test_isolated_vertices_are_their_own_components() -> None:
    """Speckle must be counted, not silently absorbed — the mass report
    depends on it."""
    member, local = _scratch()
    edges = _chain(0, 10)
    mask = np.concatenate([np.arange(0, 10), [500, 700]]).astype(np.int64)
    out = connected_components(mask, edges, member, local)
    assert len(out) == 3, [len(c) for c in out]
    assert sorted(len(c) for c in out) == [1, 1, 10]


def test_geometry_is_never_consulted() -> None:
    """Distance cannot split, and cannot join.

    Two vertices adjacent in the mesh stay together no matter how far apart
    their coordinates would be, because coordinates are never passed in.
    `connected_components` takes no xyz argument at all, which is the
    structural version of this guarantee.
    """
    import inspect
    signature = inspect.signature(connected_components)
    assert "xyz" not in signature.parameters, signature
    assert set(signature.parameters) == {"vertices", "edges", "member", "local"}, \
        signature

    member, local = _scratch()
    far_pair = np.array([[0, 999]], dtype=np.int64)
    out = connected_components(np.array([0, 999], dtype=np.int64), far_pair,
                               member, local)
    assert len(out) == 1, "an adjacent pair was split; something used distance"


def test_scratch_arrays_are_left_clean() -> None:
    """They are reused across hundreds of masks; a leak silently merges masks."""
    member, local = _scratch()
    edges = _chain(0, 20)
    connected_components(np.arange(0, 20, dtype=np.int64), edges, member, local)
    assert not member.any(), "member array was left dirty"
    assert (local == -1).all(), "local index array was left dirty"


def test_components_are_sorted_and_disjoint() -> None:
    member, local = _scratch()
    edges = np.concatenate([_chain(0, 30), _chain(100, 140)])
    mask = np.concatenate([np.arange(0, 30), np.arange(100, 140)]).astype(np.int64)
    out = connected_components(mask, edges, member, local)
    seen: set[int] = set()
    for component in out:
        assert list(component) == sorted(component), "component is unsorted"
        assert not (set(component.tolist()) & seen), "components overlap"
        seen |= set(component.tolist())
    assert seen == set(mask.tolist()), "components do not partition the mask"


def test_the_tool_declares_no_geometric_thresholds() -> None:
    path = REPO_ROOT / "tools" / "arkitscenes_repair_topology_cut.py"
    tree = ast.parse(path.read_text())
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.ImportFrom):
            used.update(a.name for a in node.names)
    banned = {"SURFACE_DEPTH_TOLERANCE_M", "normals", "curvature",
              "plane_fit", "DEPTH_TOLERANCE_M"} & used
    assert not banned, f"topology cut uses geometric thresholds: {sorted(banned)}"


def test_diagnostics_report_the_mass_the_size_rule_removes() -> None:
    """Dataset-guarded. The size floor is the one threshold in play, so what
    it discards has to be visible rather than absorbed."""
    reports = sorted(RUN_ROOT.glob("*/topology_cut_diagnostics.json"))
    if not reports:
        print("  skip: no topology_cut_diagnostics.json on disk")
        return
    for path in reports:
        report = json.loads(path.read_text())
        mass = report["mass"]
        assert mass["lifted_mask_vertices_total"] > 0, path
        assert (mass["mass_removed_by_min_size"]
                == mass["lifted_mask_vertices_total"]
                - mass["emitted_component_vertices_total"]), path
        assert 0.0 <= mass["mass_removed_fraction"] <= 1.0, path
        assert report["n_components_before_min_size"] >= \
            report["n_components_emitted"], path


TESTS = [
    test_a_connected_mask_stays_one_component,
    test_a_mask_spanning_two_disconnected_surfaces_splits,
    test_adjacency_outside_the_mask_does_not_join_components,
    test_isolated_vertices_are_their_own_components,
    test_geometry_is_never_consulted,
    test_scratch_arrays_are_left_clean,
    test_components_are_sorted_and_disjoint,
    test_the_tool_declares_no_geometric_thresholds,
    test_diagnostics_report_the_mass_the_size_rule_removes,
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
