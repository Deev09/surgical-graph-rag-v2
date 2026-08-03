"""Tests for the oracle-free proposal scorer.

Run: python tests/segmenter/test_selector_free.py

Two kinds of test live here:

  1. ISOLATION — mechanical proof that `segmenter/selector_free.py`
     cannot read ground truth. An AST scan rejects any import outside a
     tiny allowlist and any call that could open a file or import at
     runtime; a `sys.addaudithook` around a real scoring call fails the
     test if ANY file open, subprocess or socket event fires. These are
     the tests that make "no oracle read anywhere in this module"
     structural rather than a comment.

  2. BEHAVIOUR — synthetic geometry only. A clean, multiply-corroborated
     object must outrank a fragment, a scattered proposal, an object at
     an implausible metric scale, and a nested duplicate; the score must
     stay in [0, 1] and be deterministic.

No Replica data is needed; this file runs anywhere.
"""
from __future__ import annotations

import ast
import sys
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from segmenter import selector_free
from segmenter.selector_free import (
    COMPONENTS, proposal_signals, score, score_proposals,
)

MODULE_PATH = Path(selector_free.__file__)
ALLOWED_IMPORTS = {"numpy", "dataclasses", "__future__"}
FORBIDDEN_CALLS = {"open", "load", "loadtxt", "fromfile", "genfromtxt",
                   "memmap", "exec", "eval", "__import__", "compile",
                   "input", "getattr"}


# ---------------------------------------------------------------- isolation
def test_module_imports_are_allowlisted():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            seen.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            seen.add((node.module or "").split(".")[0])
    bad = seen - ALLOWED_IMPORTS
    if bad:
        raise AssertionError(
            f"selector_free may only import {sorted(ALLOWED_IMPORTS)}; "
            f"found {sorted(bad)}. Anything that can reach the filesystem "
            f"breaks the oracle-free guarantee.")


def test_module_makes_no_io_or_dynamic_calls():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = (f.id if isinstance(f, ast.Name)
                else f.attr if isinstance(f, ast.Attribute) else None)
        if name in FORBIDDEN_CALLS:
            bad.append(f"{name} (line {node.lineno})")
    if bad:
        raise AssertionError(f"forbidden calls in selector_free: {bad}")


def test_scoring_performs_no_filesystem_access():
    """Audit-hook proof: a real scoring call opens nothing at all."""
    props, n, xyz, views = _synthetic_scene()
    violations: list[str] = []
    watched = ("open", "os.system", "subprocess.Popen", "socket.__new__",
               "os.listdir", "os.scandir", "import")

    def hook(event, args):
        if event in watched and not _is_shutdown():
            violations.append(f"{event}{args!r:.120}")

    sys.addaudithook(hook)          # cannot be removed; run this test last
    out = score(props, n, xyz, views)
    if violations:
        raise AssertionError(f"selector_free touched the outside world: "
                             f"{violations[:5]}")
    if not np.isfinite(out).all():
        raise AssertionError("scores must be finite")


def _is_shutdown() -> bool:
    return False


# ---------------------------------------------------------------- behaviour
def _grid(origin, size, n=6, jitter=0.0):
    """A small axis-aligned point cloud shell around `origin`."""
    lin = np.linspace(0.0, size, n)
    g = np.stack(np.meshgrid(lin, lin, lin, indexing="ij"), -1).reshape(-1, 3)
    return g + np.asarray(origin, float) + jitter


def _synthetic_scene():
    """Four proposals with known character, over 40 fake single-mask views.

    p0 clean 0.4 m object, seen as exactly one 2D mask in every view;
    p1 a fragment of p0 (nested, worse agreement);
    p2 two far-apart clumps (disconnected);
    p3 a 5 m object (implausible metric scale) that also reconstructs.
    """
    # sampling must be finer than the occupancy resolution (extent/16),
    # as it is on a real mesh: 24 samples per axis clears extent/16.
    a = _grid((0.0, 0.0, 0.0), 0.4, n=24)      # 13824 pts, 0.4 m
    b = _grid((5.0, 0.0, 0.0), 5.0, n=24)      # 13824 pts, 5 m (too big)
    c1 = _grid((30.0, 0.0, 0.0), 0.2, n=4)     # 64 pts
    c2 = _grid((30.0, 9.0, 0.0), 0.2, n=4)     # 64 pts, far away
    xyz = np.concatenate([a, b, c1, c2])
    n = len(xyz)
    ia = np.arange(0, len(a))
    ib = np.arange(len(a), len(a) + len(b))
    ic = np.arange(len(a) + len(b), n)
    p0 = ia
    p1 = ia[:60]                                # fragment of p0
    p2 = ic                                     # two disconnected clumps
    p3 = ib                                     # oversized
    props = [p0, p1, p2, p3]
    visible = np.arange(n)
    views = []
    for _ in range(40):
        views.append({
            "visible": visible,
            # one 2D mask per real object: p0 and p3 reconstruct exactly,
            # p1 is only ever seen inside p0's mask, p2 never as one mask
            "masks": [ia, ib, ic[:64], ic[64:]],
            "mask_quality": np.array([[0.99, 0.99], [0.99, 0.99],
                                      [0.9, 0.96], [0.9, 0.96]]),
        })
    return props, n, xyz, views


def test_score_in_unit_interval_and_deterministic():
    props, n, xyz, views = _synthetic_scene()
    s1 = score(props, n, xyz, views)
    s2 = score(props, n, xyz, views)
    if not np.array_equal(s1, s2):
        raise AssertionError("scoring must be deterministic")
    if s1.shape != (len(props),):
        raise AssertionError(f"one score per proposal, got {s1.shape}")
    if s1.min() < 0.0 or s1.max() > 1.0:
        raise AssertionError(f"scores must lie in [0,1]: {s1}")


def test_signals_have_expected_character():
    props, n, xyz, views = _synthetic_scene()
    sig = proposal_signals(props, n, xyz, views)
    if sig.agreement[0] < 0.95:
        raise AssertionError(
            f"an object reconstructed as one 2D mask in every view must "
            f"have near-1 agreement, got {sig.agreement[0]:.3f}")
    if sig.agreement[1] >= sig.agreement[0]:
        raise AssertionError("a fragment must agree worse than its whole")
    if sig.connectivity[2] > 0.75:
        raise AssertionError(
            f"two far-apart clumps must not look connected, got "
            f"{sig.connectivity[2]:.3f}")
    if sig.connectivity[0] < 0.99:
        raise AssertionError("a solid grid must be fully connected")
    if sig.size_prior[3] != 0.0:
        raise AssertionError("a 5 m 'object' must fail the metric-scale prior")
    if sig.size_prior[0] != 1.0:
        raise AssertionError("a 0.4 m object must pass the metric-scale prior")
    if sig.n_nested_better[1] < 1:
        raise AssertionError("the fragment must be dominated by its whole")


def test_clean_object_outranks_its_failure_modes():
    props, n, xyz, views = _synthetic_scene()
    s = score(props, n, xyz, views)
    if not (s[0] > s[1] and s[0] > s[2] and s[0] > s[3]):
        raise AssertionError(f"clean object must rank first, got {s}")


def test_nested_duplicates_collapse_to_one_representative():
    """Identical proposals must not all survive each other's comparison."""
    props, n, xyz, views = _synthetic_scene()
    dup = props + [props[0].copy(), props[0].copy()]
    sig = proposal_signals(dup, n, xyz, views)
    dominated = [int(sig.n_nested_better[i]) for i in (0, 4, 5)]
    if dominated[0] != 0 or dominated[1] < 1 or dominated[2] < 1:
        raise AssertionError(
            f"exactly one of three identical proposals may be undominated, "
            f"got n_nested_better {dominated}")


def test_ablation_components_change_the_ranking_path():
    props, n, xyz, views = _synthetic_scene()
    sig = proposal_signals(props, n, xyz, views)
    full = score_proposals(sig, COMPONENTS)
    no_size = score_proposals(sig, ("agreement", "connectivity",
                                    "redundancy"))
    if np.array_equal(full, no_size):
        raise AssertionError("dropping the size prior must change scores "
                             "on a scene containing an oversized proposal")
    for name in COMPONENTS:
        score_proposals(sig, (name,))          # each alone must be legal
    try:
        score_proposals(sig, ("not_a_component",))
    except ValueError:
        pass
    else:
        raise AssertionError("unknown components must raise")


def test_empty_bank_and_no_views_are_handled():
    props, n, xyz, views = _synthetic_scene()
    if score([], n, xyz, views).shape != (0,):
        raise AssertionError("an empty bank must score to an empty array")
    s = score(props, n, xyz, [])
    if s.shape != (len(props),) or not np.isfinite(s).all():
        raise AssertionError("a bank with zero views must still score")


TESTS = [
    test_module_imports_are_allowlisted,
    test_module_makes_no_io_or_dynamic_calls,
    test_score_in_unit_interval_and_deterministic,
    test_signals_have_expected_character,
    test_clean_object_outranks_its_failure_modes,
    test_nested_duplicates_collapse_to_one_representative,
    test_ablation_components_change_the_ranking_path,
    test_empty_bank_and_no_views_are_handled,
    # audit hook cannot be uninstalled -> keep last
    test_scoring_performs_no_filesystem_access,
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
