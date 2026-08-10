"""SAM sidecar validation: the joins that protect an off-machine GPU run.

Synthetic and fast. The failure these guard against is specific and silent: a
mask sidecar produced from a DIFFERENT frame selection still unpacks, still
has the right slot count, and still lifts — onto the wrong geometry, with no
other symptom. Same for a sidecar from a different model pin.

The full post-GPU path (propose -> finalize -> pool -> evaluate) was exercised
on a synthetic sidecar over the real development scene; see the Checkpoint C
section of `docs/repair_arm_design_note.md`. This file pins the cheap checks so
they cannot rot between GPU sessions.
"""
from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from tools.arkitscenes_repair_propose_sam import (
    PINNED_CHECKPOINT_SHA256, PINNED_SAM2_COMMIT, load_sidecar,
)

HEIGHT, WIDTH = 4, 8
SELECTION = "a" * 64


def _manifest(n_frames: int = 2, selection: str = SELECTION) -> dict:
    return {
        "selection_sha256": selection,
        "frames": [{"slot": i, "frame_index": i, "height": HEIGHT,
                    "width": WIDTH, "source_png": f"src_{i}.png"}
                   for i in range(n_frames)],
    }


def _sidecar(path: Path, *, n_frames: int = 2, selection: str = SELECTION,
             commit: str = PINNED_SAM2_COMMIT,
             checkpoint: str = PINNED_CHECKPOINT_SHA256,
             n_masks: int = 2, shape=(HEIGHT, WIDTH)) -> Path:
    payload = {}
    for slot in range(n_frames):
        masks = np.zeros((n_masks, shape[0] * shape[1]), dtype=bool)
        masks[:, : shape[1]] = True
        payload[f"masks_{slot:02d}"] = np.packbits(masks, axis=1)
        payload[f"scores_{slot:02d}"] = np.full((n_masks, 2), 0.9,
                                                dtype=np.float32)
        payload[f"shape_{slot:02d}"] = np.array(shape, dtype=np.int32)
    env = {"selection_sha256": selection, "sam2_commit": commit,
           "checkpoint_sha256": checkpoint, "n_frames": n_frames,
           "device": "test", "elapsed_seconds": 0.0}
    np.savez_compressed(path, env=json.dumps(env), **payload)
    return path


def _expect(message: str, fn) -> None:
    try:
        fn()
    except ValueError as exc:
        assert message in str(exc), f"wrong error: {exc}"
    else:
        raise AssertionError(f"expected a ValueError containing {message!r}")


def test_a_matching_sidecar_unpacks_to_boolean_masks() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = _sidecar(Path(td) / "s.npz")
        masks, env = load_sidecar(path, _manifest())
        assert set(masks) == {0, 1}, masks.keys()
        assert masks[0]["masks"].shape == (2, HEIGHT, WIDTH), \
            masks[0]["masks"].shape
        assert masks[0]["masks"].dtype == bool
        # The fixture sets the first WIDTH flat entries, i.e. the top row.
        assert masks[0]["masks"][0].sum() == WIDTH, masks[0]["masks"][0].sum()
        assert masks[0]["masks"][0][0].all(), "the top row did not survive unpacking"
        assert not masks[0]["masks"][0][1:].any(), "unpacking bled past one row"
        assert env["selection_sha256"] == SELECTION


def test_a_sidecar_from_another_selection_is_refused() -> None:
    """The load-bearing check: same slots, different photographs."""
    with tempfile.TemporaryDirectory() as td:
        path = _sidecar(Path(td) / "s.npz", selection="b" * 64)
        _expect("do not belong to this frame set",
                lambda: load_sidecar(path, _manifest()))


def test_an_off_pin_model_is_refused() -> None:
    with tempfile.TemporaryDirectory() as td:
        wrong_commit = _sidecar(Path(td) / "a.npz", commit="0" * 40)
        _expect("pin is", lambda: load_sidecar(wrong_commit, _manifest()))
        wrong_ckpt = _sidecar(Path(td) / "b.npz", checkpoint="0" * 64)
        _expect("not the pinned SAM 2.1 Hiera-L weight",
                lambda: load_sidecar(wrong_ckpt, _manifest()))


def test_a_frame_count_or_shape_mismatch_is_refused() -> None:
    with tempfile.TemporaryDirectory() as td:
        short = _sidecar(Path(td) / "a.npz", n_frames=1)
        _expect("frames, the selection has",
                lambda: load_sidecar(short, _manifest(n_frames=2)))
        reshaped = _sidecar(Path(td) / "b.npz", shape=(HEIGHT + 1, WIDTH))
        _expect("the frame is",
                lambda: load_sidecar(reshaped, _manifest()))


def test_a_frame_with_no_masks_is_allowed() -> None:
    """SAM legitimately returns nothing on some frames; that is not an error."""
    with tempfile.TemporaryDirectory() as td:
        path = _sidecar(Path(td) / "s.npz", n_masks=0)
        masks, _ = load_sidecar(path, _manifest())
        assert masks[0]["masks"].shape == (0, HEIGHT, WIDTH), \
            masks[0]["masks"].shape


TESTS = [
    test_a_matching_sidecar_unpacks_to_boolean_masks,
    test_a_sidecar_from_another_selection_is_refused,
    test_an_off_pin_model_is_refused,
    test_a_frame_count_or_shape_mismatch_is_refused,
    test_a_frame_with_no_masks_is_allowed,
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
