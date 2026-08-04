"""Stage-0 validity gates for the fusion evidence denominator.

Protocol: `docs/arkitscenes_fusion_evidence_protocol.md`, gates V3 and V4.

V3 — where every visible vertex is masked in every view, `"masked"` and
     `"covisible"` must agree EXACTLY. The change has to be inert on
     fully-covered renders, or it is not the change the protocol describes.
V4 — with a known unmasked view, the two must differ in the predicted
     direction and by the hand-computed amount.

Every expected value below is derived from the definition and written out in
the assertion, not captured from a run.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from segmenter.proposal_fusion import EVIDENCE_DENOMINATORS, edge_confidence

N = 4
EDGES = np.array([[0, 1]], dtype=np.int64)
ALL_VISIBLE = np.arange(N, dtype=np.int64)


def _view(masks):
    return {"visible": ALL_VISIBLE,
            "masks": [np.asarray(m, dtype=np.int64) for m in masks]}


def _conf(views, mode):
    _co_vis, conf = edge_confidence(EDGES, N, views, evidence_denominator=mode)
    return float(conf[0])


def test_rejects_an_unknown_mode() -> None:
    try:
        edge_confidence(EDGES, N, [_view([[0, 1]])], evidence_denominator="nope")
    except ValueError:
        return
    raise AssertionError("accepted an unknown evidence_denominator")


def test_v3_inert_when_every_visible_vertex_is_masked() -> None:
    """Both endpoints masked in every view; one view agrees, one splits.

    numerator = 1 (only view 0 shares a mask)
    covisible denominator = 2 (both views co-visible)
    masked    denominator = 2 (both endpoints masked in both views)
    => both modes 1/2, exactly.
    """
    views = [
        _view([[0, 1, 2, 3]]),           # both in mask 0 -> share
        _view([[0, 2], [1, 3]]),         # both masked, different masks
    ]
    a = _conf(views, "covisible")
    b = _conf(views, "masked")
    if a != b:
        raise AssertionError(f"modes disagree on a fully-masked fixture: {a} vs {b}")
    if abs(a - 0.5) > 1e-12:
        raise AssertionError(f"expected exactly 1/2, got {a!r}")


def test_v4_unmasked_view_is_excluded_by_the_masked_mode() -> None:
    """Three co-visible views: agree / split / NEITHER endpoint masked.

    numerator = 1
    covisible denominator = 3  -> 1/3
    masked    denominator = 2  -> 1/2   (view 2 carries no evidence)
    """
    views = [
        _view([[0, 1]]),                 # share
        _view([[0], [1]]),               # both masked, no share
        _view([[2, 3]]),                 # neither endpoint masked
    ]
    a = _conf(views, "covisible")
    b = _conf(views, "masked")
    if abs(a - 1.0 / 3.0) > 1e-12:
        raise AssertionError(f"covisible expected 1/3, got {a!r}")
    if abs(b - 0.5) > 1e-12:
        raise AssertionError(f"masked expected 1/2, got {b!r}")
    if not b > a:
        raise AssertionError("masked mode must not lower confidence here")


def test_v4_view_with_no_masks_at_all_is_excluded() -> None:
    """A view SAM returned nothing for is absence of evidence too.

    numerator = 1; covisible denominator = 2 -> 1/2; masked -> 1/1 = 1.0.
    """
    views = [_view([[0, 1]]), _view([])]
    if abs(_conf(views, "covisible") - 0.5) > 1e-12:
        raise AssertionError("covisible expected 1/2")
    if abs(_conf(views, "masked") - 1.0) > 1e-12:
        raise AssertionError("masked expected 1/1")


def test_no_masked_evidence_scores_zero_not_nan() -> None:
    """An edge with no evidence-bearing view must score 0 — the same
    treatment never-co-visible edges already get — never NaN."""
    views = [_view([[2, 3]]), _view([[2], [3]])]
    b = _conf(views, "masked")
    if not np.isfinite(b) or b != 0.0:
        raise AssertionError(f"expected exactly 0.0, got {b!r}")


def test_one_endpoint_masked_is_still_not_evidence() -> None:
    """Asymmetric case: vertex 0 masked, vertex 1 not. The protocol says a
    view carries evidence only when BOTH endpoints are assigned.

    numerator = 1; covisible = 2 -> 1/2; masked = 1 -> 1.0.
    """
    views = [_view([[0, 1]]), _view([[0, 2]])]
    if abs(_conf(views, "covisible") - 0.5) > 1e-12:
        raise AssertionError("covisible expected 1/2")
    if abs(_conf(views, "masked") - 1.0) > 1e-12:
        raise AssertionError("masked expected 1/1 — a lone masked endpoint "
                             "must not count as evidence")


def test_co_visible_count_is_mode_independent() -> None:
    """The returned co-visible count reports coverage and must not be
    silently redefined by the mode."""
    views = [_view([[0, 1]]), _view([]), _view([[2, 3]])]
    a, _ = edge_confidence(EDGES, N, views, evidence_denominator="covisible")
    b, _ = edge_confidence(EDGES, N, views, evidence_denominator="masked")
    if not np.array_equal(a, b) or int(a[0]) != 3:
        raise AssertionError(f"co_visible changed with mode: {a} vs {b}")


def test_default_is_the_frozen_mode() -> None:
    views = [_view([[0, 1]]), _view([[2, 3]])]
    _cv, explicit = edge_confidence(EDGES, N, views,
                                    evidence_denominator="covisible")
    _cv2, implied = edge_confidence(EDGES, N, views)
    if not np.array_equal(explicit, implied):
        raise AssertionError("the default is no longer 'covisible'")
    if EVIDENCE_DENOMINATORS[0] != "covisible":
        raise AssertionError("frozen mode must remain first/default")


TESTS = [
    test_rejects_an_unknown_mode,
    test_v3_inert_when_every_visible_vertex_is_masked,
    test_v4_unmasked_view_is_excluded_by_the_masked_mode,
    test_v4_view_with_no_masks_at_all_is_excluded,
    test_no_masked_evidence_scores_zero_not_nan,
    test_one_endpoint_masked_is_still_not_evidence,
    test_co_visible_count_is_mode_independent,
    test_default_is_the_frozen_mode,
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
