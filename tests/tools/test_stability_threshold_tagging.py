"""Stage-0 gate V3 for the mask-coverage protocol.

Protocol: `docs/arkitscenes_mask_coverage_protocol.md`.

M1 breaks a frozen SAM pin, so from here on two different SAM
parameterisations exist for the same scene. The whole experiment is void if
they can share a filename or be fused into each other. That has already
happened twice in this line of work under different guises — `with_suffix`
collapsing render variants onto the baseline tar, and `fuse_one` pairing new
views with old masks — so it is asserted rather than assumed.

Runs with or without the dataset present.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.arkitscenes_fuse import FROZEN_STABILITY, bank_paths, stability_tag

NB = REPO_ROOT / "notebooks" / "c1p1_sam2_colab.ipynb"


def test_frozen_threshold_produces_no_tag() -> None:
    """The pin must keep the original filenames, so every artifact committed
    before this protocol stays addressable and downstream defaults are
    untouched."""
    if FROZEN_STABILITY != 0.95:
        raise AssertionError(f"the C1-P1 pin is 0.95, not {FROZEN_STABILITY}")
    if stability_tag(0.95) != "":
        raise AssertionError(f"pin must be untagged, got {stability_tag(0.95)!r}")


def test_non_frozen_thresholds_get_distinct_tags() -> None:
    seen = {}
    for v in (0.85, 0.90, 0.80, 0.75):
        t = stability_tag(v)
        if t == "":
            raise AssertionError(f"{v} produced the frozen (empty) tag")
        if t in seen:
            raise AssertionError(
                f"{v} and {seen[t]} collide on tag {t!r}")
        seen[t] = v
    if stability_tag(0.85) != ".stab085":
        raise AssertionError(stability_tag(0.85))


def test_bank_paths_separate_by_threshold() -> None:
    root = Path("/tmp/_v3")
    a, aj = bank_paths(root, "s", "covisible", stability_tag(0.95))
    b, bj = bank_paths(root, "s", "covisible", stability_tag(0.85))
    if a == b or aj == bj:
        raise AssertionError(f"two thresholds share a bank path: {a} / {b}")
    if a.name != "bank_s.npz":
        raise AssertionError(f"pin path changed: {a.name}")


def test_threshold_tag_composes_with_the_other_variants() -> None:
    """Three independent dimensions now exist (denominator, render kernels,
    SAM threshold). No two combinations may collide."""
    combos, paths = [], {}
    for denom in ("covisible", "masked"):
        for render in ("", ".rgb5x5_id5x5"):
            for stab in (0.95, 0.85):
                v = render + stability_tag(stab)
                p, _ = bank_paths(Path("/tmp/_v3"), "s", denom, v)
                combos.append((denom, render, stab))
                if p in paths:
                    raise AssertionError(
                        f"{(denom, render, stab)} collides with {paths[p]} "
                        f"on {p.name}")
                paths[p] = (denom, render, stab)
    if len(paths) != 8:
        raise AssertionError(f"expected 8 distinct paths, got {len(paths)}")


def test_notebook_tag_matches_the_local_tag() -> None:
    """The notebook builds the same suffix independently, in Colab, where it
    cannot import this module. If the two formulas drift, masks land under a
    name the fuse tool will not find — or worse, one it will."""
    nb = json.loads(NB.read_text())
    src = "".join("".join(c["source"]) for c in nb["cells"])
    if "STABILITY_SCORE_THRESH" not in src:
        raise AssertionError("notebook no longer declares the threshold")
    if "stability_score_thresh=STABILITY_SCORE_THRESH" not in src:
        raise AssertionError("notebook does not consume the variable")
    if "c1p1_masks_{SCENE}{TAG}.npz" not in src:
        raise AssertionError("notebook output filename is not tagged")
    # the formula, mirrored: int(round(v*100)) zero-padded to 3
    if 'int(round(STABILITY_SCORE_THRESH * 100)):03d' not in src:
        raise AssertionError(
            "notebook tag formula changed; it must match "
            "tools.arkitscenes_fuse.stability_tag")
    for v in (0.85, 0.90, 0.75):
        local = stability_tag(v)
        colab = f".stab{int(round(v * 100)):03d}"
        if local != colab:
            raise AssertionError(f"{v}: local {local!r} vs notebook {colab!r}")


def test_notebook_default_is_still_the_pin() -> None:
    nb = json.loads(NB.read_text())
    src = "".join("".join(c["source"]) for c in nb["cells"])
    if "STABILITY_SCORE_THRESH = 0.95" not in src:
        raise AssertionError(
            "the notebook's default threshold is no longer the frozen pin; "
            "a fresh run would silently produce a different parameterisation")


def test_only_the_threshold_is_unpinned_in_the_notebook() -> None:
    """Every other SAM parameter must remain a literal."""
    nb = json.loads(NB.read_text())
    src = "".join("".join(c["source"]) for c in nb["cells"])
    for frozen in ("points_per_side=32", "pred_iou_thresh=0.8",
                   "box_nms_thresh=0.7", "crop_n_layers=0",
                   "min_mask_region_area=0", "multimask_output=True"):
        if frozen not in src:
            raise AssertionError(
                f"{frozen!r} is no longer pinned as a literal in the notebook "
                "— M1 permits exactly one variable")


TESTS = [
    test_frozen_threshold_produces_no_tag,
    test_non_frozen_thresholds_get_distinct_tags,
    test_bank_paths_separate_by_threshold,
    test_threshold_tag_composes_with_the_other_variants,
    test_notebook_tag_matches_the_local_tag,
    test_notebook_default_is_still_the_pin,
    test_only_the_threshold_is_unpinned_in_the_notebook,
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
