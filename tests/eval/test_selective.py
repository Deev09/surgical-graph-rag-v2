"""Tests for eval/selective.py -- selective-prediction metrics.

Run: python tests/eval/test_selective.py

Every AURC / E-AURC assertion below is against a value computed BY HAND from
the definitions in the module docstring, not against a golden file produced
by the code under test. The hand computation is written out in each test so
a reviewer can check it without running anything.

Covers:
  - AURC on perfect / worst / all-correct / all-wrong / mixed rankings
  - exact discrete AURC* and E-AURC == 0 iff the ranking is oracle-optimal
  - the discrete-vs-closed-form AURC* gap at small n (why we use discrete)
  - tie policy (pessimistic / optimistic / given) and the degeneracy flag
  - coverage@risk<=t maximises over operating points (non-monotone risk)
  - abstained items sink below every real prediction
  - deployable threshold curve collapses tie groups
  - input coercion for tuples / dicts / SelectiveItem
  - E-AURC >= 0 as an invariant over many random orderings
"""
from __future__ import annotations

import math
import random
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.selective import (  # noqa: E402
    SelectiveItem,
    aurc_from_points,
    aurc_optimal_closed_form,
    aurc_optimal_discrete,
    evaluate,
    report_to_dict,
)


TOL = 1e-12


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def _items(confs, corrects, abstained=None):
    ab = abstained or [False] * len(confs)
    return [
        SelectiveItem(item_id=f"q{i}", confidence=c, correct=k, abstained=a)
        for i, (c, k, a) in enumerate(zip(confs, corrects, ab))
    ]


# --------------------------------------------------------------------------
# AURC on hand-computed cases
# --------------------------------------------------------------------------

def test_aurc_perfect_ranking():
    """n=4, correct=[T,T,T,F] ranked by descending confidence.
    risk(k) = 0, 0, 0, 1/4  ->  AURC = (0+0+0+0.25)/4 = 0.0625
    AURC*(n=4, e=1) = (1/4) * (4-3)/4 = 0.0625  ->  E-AURC = 0."""
    r = evaluate(_items([0.9, 0.8, 0.7, 0.6], [True, True, True, False]))
    assert _close(r.aurc, 0.0625), r.aurc
    assert _close(r.aurc_optimal, 0.0625), r.aurc_optimal
    assert _close(r.e_aurc, 0.0), r.e_aurc


def test_aurc_worst_ranking():
    """Same accuracy, inverted ranking: correct=[F,T,T,T].
    risk(k) = 1, 1/2, 1/3, 1/4  ->  sum = 25/12
    AURC = (25/12)/4 = 25/48 = 0.520833...
    E-AURC = 25/48 - 1/16 = 22/48 = 0.458333..."""
    r = evaluate(_items([0.9, 0.8, 0.7, 0.6], [False, True, True, True]))
    assert _close(r.aurc, 25.0 / 48.0), r.aurc
    assert _close(r.e_aurc, 22.0 / 48.0), r.e_aurc


def test_aurc_all_correct_is_zero():
    r = evaluate(_items([0.9, 0.5, 0.1], [True, True, True]))
    assert _close(r.aurc, 0.0), r.aurc
    assert _close(r.aurc_optimal, 0.0), r.aurc_optimal
    assert _close(r.e_aurc, 0.0), r.e_aurc
    assert _close(r.base_risk, 0.0)


def test_aurc_all_wrong_is_one():
    """risk(k) = 1 at every coverage, and the oracle can do no better."""
    r = evaluate(_items([0.9, 0.5, 0.1], [False, False, False]))
    assert _close(r.aurc, 1.0), r.aurc
    assert _close(r.aurc_optimal, 1.0), r.aurc_optimal
    assert _close(r.e_aurc, 0.0), r.e_aurc
    assert _close(r.base_risk, 1.0)


def test_aurc_mixed_ranking_hand_computed():
    """n=5, correct by descending confidence = [T,F,T,T,F].
    risk(k) = 0, 1/2, 1/3, 1/4, 2/5     sum = 89/60
    AURC   = (89/60)/5 = 89/300 = 0.296666...
    AURC*(n=5, e=2) = (1/5) * (1/4 + 2/5) = 0.13
    E-AURC = 89/300 - 13/100 = 1/6 = 0.166666..."""
    r = evaluate(_items([0.9, 0.8, 0.7, 0.6, 0.5],
                        [True, False, True, True, False]))
    assert _close(r.aurc, 89.0 / 300.0), r.aurc
    assert _close(r.aurc_optimal, 0.13), r.aurc_optimal
    assert _close(r.e_aurc, 1.0 / 6.0), r.e_aurc
    assert _close(r.base_risk, 0.4)


def test_aurc_from_points_matches_report():
    r = evaluate(_items([0.9, 0.8, 0.7, 0.6, 0.5],
                        [True, False, True, True, False]))
    assert _close(aurc_from_points(r.points), r.aurc)


# --------------------------------------------------------------------------
# AURC* : discrete oracle vs continuous closed form
# --------------------------------------------------------------------------

def test_aurc_optimal_discrete_direct():
    # n=4, e=1  ->  only k=4 contributes: (4-3)/4 = 0.25, /4 = 0.0625
    assert _close(aurc_optimal_discrete(4, 1), 0.0625)
    # n=5, e=2  ->  (1/4 + 2/5)/5 = 0.13
    assert _close(aurc_optimal_discrete(5, 2), 0.13)
    assert _close(aurc_optimal_discrete(10, 0), 0.0)
    assert _close(aurc_optimal_discrete(3, 3), 1.0)


def test_closed_form_disagrees_with_discrete_at_small_n():
    """Documented reason the report uses the discrete oracle for E-AURC:
    at n=4 the continuous limit r + (1-r)ln(1-r) is ~45% smaller, which
    would inflate E-AURC. Both are reported; only discrete is subtracted."""
    disc = aurc_optimal_discrete(4, 1)
    closed = aurc_optimal_closed_form(4, 1)
    expected_closed = 0.25 + 0.75 * math.log(0.75)
    assert _close(closed, expected_closed), closed
    assert closed < disc, (closed, disc)
    assert _close(aurc_optimal_closed_form(4, 4), 1.0)
    assert _close(aurc_optimal_closed_form(4, 0), 0.0)


# --------------------------------------------------------------------------
# Tie policy
# --------------------------------------------------------------------------

def test_tie_policy_pessimistic_is_worst_case():
    """All confidences equal -> the score ranks nothing. Pessimistic must
    reproduce the worst-case AURC (25/48), not the input order."""
    items = _items([0.5] * 4, [True, True, True, False])
    r = evaluate(items, tie_policy="pessimistic")
    assert _close(r.aurc, 25.0 / 48.0), r.aurc


def test_tie_policy_optimistic_and_given():
    items = _items([0.5] * 4, [True, True, True, False])
    assert _close(evaluate(items, tie_policy="optimistic").aurc, 0.0625)
    # input order already has the error last, so 'given' == optimistic here
    assert _close(evaluate(items, tie_policy="given").aurc, 0.0625)
    # ...but with the error first, 'given' tracks input order, not the best case
    items2 = _items([0.5] * 4, [False, True, True, True])
    assert _close(evaluate(items2, tie_policy="given").aurc, 25.0 / 48.0)
    assert _close(evaluate(items2, tie_policy="optimistic").aurc, 0.0625)


def test_degenerate_confidence_flagged():
    r = evaluate(_items([0.0] * 6, [True, False, True, False, True, False]))
    assert r.n_distinct_confidences == 1
    d = report_to_dict(r)
    assert d["confidence_is_degenerate"] is True
    assert _close(r.max_tie_fraction, 1.0)
    r2 = evaluate(_items([0.9, 0.1], [True, False]))
    assert r2.n_distinct_confidences == 2
    assert report_to_dict(r2)["confidence_is_degenerate"] is False


def test_partial_tie_degeneracy_flagged():
    """n_distinct > 1 is not enough. 8 of 10 items share one confidence, so
    AURC is dominated by the tie-break, not by the score."""
    confs = [0.5] * 8 + [0.9, 0.1]
    corr = [True, False] * 5
    r = evaluate(_items(confs, corr))
    assert r.n_distinct_confidences == 3
    assert _close(r.max_tie_fraction, 0.8), r.max_tie_fraction
    d = report_to_dict(r)
    assert d["confidence_is_degenerate"] is False
    assert d["confidence_mostly_tied"] is True
    # the spread is exactly what the tie-break, not the score, contributes
    assert r.tie_policy_spread > 0.0
    assert _close(r.aurc - r.aurc_optimistic, r.tie_policy_spread)


def test_tie_spread_is_zero_when_all_confidences_distinct():
    r = evaluate(_items([0.9, 0.8, 0.7, 0.6], [True, False, True, True]))
    assert _close(r.tie_policy_spread, 0.0)
    assert _close(r.aurc, r.aurc_optimistic)
    assert _close(r.max_tie_fraction, 0.25)
    assert report_to_dict(r)["confidence_mostly_tied"] is False


def test_optimistic_never_worse_than_pessimistic():
    rng = random.Random(20260802)
    for _ in range(200):
        n = rng.randint(2, 12)
        confs = [rng.choice([0.0, 0.25, 0.5, 0.75, 1.0]) for _ in range(n)]
        corr = [rng.random() < 0.6 for _ in range(n)]
        items = _items(confs, corr)
        opt = evaluate(items, tie_policy="optimistic").aurc
        pes = evaluate(items, tie_policy="pessimistic").aurc
        assert opt <= pes + TOL, (confs, corr, opt, pes)


# --------------------------------------------------------------------------
# coverage @ risk
# --------------------------------------------------------------------------

def test_coverage_at_risk_basic():
    r = evaluate(_items([0.9, 0.8, 0.7, 0.6], [True, True, True, False]),
                 risk_targets=(0.05, 0.10, 0.30))
    c05 = r.coverage_at_risk["0.05"]
    assert _close(c05["coverage"], 0.75), c05
    assert _close(c05["risk"], 0.0)
    assert _close(c05["tau"], 0.7)
    assert _close(r.coverage_at_risk["0.10"]["coverage"], 0.75)
    assert _close(r.coverage_at_risk["0.30"]["coverage"], 1.0)


def test_coverage_at_risk_maximises_not_first_crossing():
    """risk is NOT monotone in coverage. correct=[T,T,T,T,F,T,T,T,T,T]:
    risk(k) = 0,0,0,0, 0.2, 1/6, 1/7, 1/8, 1/9, 0.1
    A first-crossing implementation stops at k=4 (coverage 0.4).
    The correct answer for target 0.10 is k=10 (coverage 1.0)."""
    confs = [1.0 - 0.05 * i for i in range(10)]
    corr = [True, True, True, True, False, True, True, True, True, True]
    r = evaluate(_items(confs, corr), risk_targets=(0.10,))
    got = r.coverage_at_risk["0.10"]
    assert _close(got["coverage"], 1.0), got
    assert _close(got["risk"], 0.1), got


def test_coverage_at_risk_unreachable():
    r = evaluate(_items([0.9, 0.5], [False, False]), risk_targets=(0.05,))
    got = r.coverage_at_risk["0.05"]
    assert got["coverage"] == 0.0
    assert got["k"] == 0
    assert math.isnan(got["risk"])
    assert report_to_dict(r)["coverage_at_risk"]["0.05"]["risk"] is None


# --------------------------------------------------------------------------
# Abstention
# --------------------------------------------------------------------------

def test_abstained_items_sink_below_every_prediction():
    """The abstained item carries the HIGHEST confidence (0.95) but must
    still rank last -- an abstention is a refusal, not a prediction."""
    items = _items(
        confs=[0.9, 0.95, 0.8, 0.7],
        corrects=[True, False, True, True],
        abstained=[False, True, False, False],
    )
    r = evaluate(items)
    assert r.n_abstained == 1
    assert [p.n_errors for p in r.points] == [0, 0, 0, 1]
    assert _close(r.aurc, 0.0625), r.aurc
    # sanity: without the abstain flag the same numbers give the worst case
    r_naive = evaluate(_items([0.9, 0.95, 0.8, 0.7],
                              [True, False, True, True]))
    assert _close(r_naive.aurc, 25.0 / 48.0), r_naive.aurc


# --------------------------------------------------------------------------
# Deployable threshold curve
# --------------------------------------------------------------------------

def test_threshold_curve_collapses_tie_groups():
    """confs [0.9,0.9,0.5,0.5]; only k=2 and k=4 are realisable by a
    'select confidence >= tau' rule."""
    items = _items([0.9, 0.9, 0.5, 0.5], [True, False, True, False])
    r = evaluate(items, tie_policy="pessimistic")
    assert len(r.points) == 4
    ks = [p.k for p in r.threshold_points]
    assert ks == [2, 4], ks
    assert _close(r.threshold_points[0].coverage, 0.5)
    assert _close(r.threshold_points[0].risk, 0.5)
    assert _close(r.threshold_points[0].tau, 0.9)
    assert _close(r.threshold_points[1].coverage, 1.0)
    assert _close(r.threshold_points[1].risk, 0.5)


def test_threshold_curve_all_distinct_keeps_every_point():
    r = evaluate(_items([0.9, 0.8, 0.7], [True, True, False]))
    assert [p.k for p in r.threshold_points] == [1, 2, 3]


# --------------------------------------------------------------------------
# Input coercion
# --------------------------------------------------------------------------

def test_accepts_bare_triples():
    """The documented minimal contract: (prediction, confidence, correct)."""
    r = evaluate([
        ("obj_1", 0.9, True),
        ("obj_2", 0.8, True),
        ("obj_3", 0.7, True),
        ("obj_9", 0.6, False),
    ])
    assert _close(r.aurc, 0.0625), r.aurc
    assert r.points[0].tau == 0.9


def test_accepts_dicts():
    r = evaluate([
        {"id": "a", "confidence": 0.9, "correct": True},
        {"id": "b", "confidence": 0.1, "correct": False, "abstained": True},
    ])
    assert r.n == 2 and r.n_abstained == 1
    assert _close(r.aurc, (0.0 + 0.5) / 2)


def test_empty_stream_raises():
    try:
        evaluate([])
    except ValueError:
        return
    raise AssertionError("empty stream must raise, not return an empty curve")


def test_missing_fields_raise():
    for bad in ([{"id": "a", "correct": True}], [{"id": "a", "confidence": 0.5}]):
        try:
            evaluate(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


# --------------------------------------------------------------------------
# Invariants
# --------------------------------------------------------------------------

def test_e_aurc_non_negative_over_random_streams():
    """AURC* is the minimum AURC achievable at a fixed (n, e), so E-AURC
    can never be negative under any tie policy."""
    rng = random.Random(11235)
    for _ in range(400):
        n = rng.randint(1, 25)
        items = _items(
            [rng.random() for _ in range(n)],
            [rng.random() < 0.5 for _ in range(n)],
        )
        for policy in ("pessimistic", "optimistic", "given"):
            r = evaluate(items, tie_policy=policy)
            assert r.e_aurc >= -TOL, (policy, r.e_aurc, n)
            assert 0.0 - TOL <= r.aurc <= 1.0 + TOL, r.aurc


def test_curve_shape_and_serialisation():
    r = evaluate(_items([0.9, 0.8, 0.7, 0.6], [True, True, True, False]))
    assert [p.coverage for p in r.points] == [0.25, 0.5, 0.75, 1.0]
    assert all(_close(p.precision, 1.0 - p.risk) for p in r.points)
    d = report_to_dict(r)
    assert d["schema"] == "selective_prediction_report"
    assert len(d["points"]) == 4
    assert d["points"][0]["tau"] == 0.9
    # abstained tau is -inf in memory and must serialise as null, not crash
    r2 = evaluate(_items([0.5, 0.4], [True, False], abstained=[False, True]))
    assert report_to_dict(r2)["points"][-1]["tau"] is None


TESTS = [
    test_aurc_perfect_ranking,
    test_aurc_worst_ranking,
    test_aurc_all_correct_is_zero,
    test_aurc_all_wrong_is_one,
    test_aurc_mixed_ranking_hand_computed,
    test_aurc_from_points_matches_report,
    test_aurc_optimal_discrete_direct,
    test_closed_form_disagrees_with_discrete_at_small_n,
    test_tie_policy_pessimistic_is_worst_case,
    test_tie_policy_optimistic_and_given,
    test_degenerate_confidence_flagged,
    test_partial_tie_degeneracy_flagged,
    test_tie_spread_is_zero_when_all_confidences_distinct,
    test_optimistic_never_worse_than_pessimistic,
    test_coverage_at_risk_basic,
    test_coverage_at_risk_maximises_not_first_crossing,
    test_coverage_at_risk_unreachable,
    test_abstained_items_sink_below_every_prediction,
    test_threshold_curve_collapses_tie_groups,
    test_threshold_curve_all_distinct_keeps_every_point,
    test_accepts_bare_triples,
    test_accepts_dicts,
    test_empty_stream_raises,
    test_missing_fields_raise,
    test_e_aurc_non_negative_over_random_streams,
    test_curve_shape_and_serialisation,
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
