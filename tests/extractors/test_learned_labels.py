"""Synthetic tests for oracle-free learned-label attachment."""
from __future__ import annotations

import ast
import sys
import tempfile
import traceback
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.equality import array_aware_equal
from extractors.base import SemanticHypothesis
from extractors.arkitscenes_segments import build_arkitscenes_segment_artifacts
from extractors.learned_labels import (
    GLOBAL_INDOOR_VOCABULARY_V1,
    LearnedLabelConfig,
    attach_learned_labels,
)
from extractors.serde import dump_entity_artifacts, load_entity_artifacts
from tests.extractors.test_arkitscenes_segments import _fixture


MODULE = REPO_ROOT / "extractors" / "learned_labels.py"


class FakeLabeler:
    weights_sha256 = "fake-weights-sha"

    def __init__(self, rankings: list[list[dict]]):
        self.rankings = list(rankings)
        self.calls: list[tuple[int, tuple[str, ...]]] = []

    def classify(self, images, vocabulary: list[str]) -> list[dict]:
        self.calls.append((len(images), tuple(vocabulary)))
        return self.rankings.pop(0)


def _ranking(first: str, score: float) -> list[dict]:
    rest = [label for label in GLOBAL_INDOOR_VOCABULARY_V1 if label != first]
    return ([{"label": first, "score": score}]
            + [{"label": label, "score": score - (i + 1) * 0.01}
               for i, label in enumerate(rest)])


def _anonymous(root: Path):
    rep, seg_dir, _ = _fixture(root)
    artifacts = build_arkitscenes_segment_artifacts(
        rep, seg_dir, min_vertices=4)
    return rep, seg_dir, artifacts


def test_topk_and_low_confidence_identity_policy() -> None:
    with tempfile.TemporaryDirectory() as td:
        rep, seg_dir, anonymous = _anonymous(Path(td))
        fake = FakeLabeler([
            _ranking("table", 0.41),
            _ranking("chair", 0.22),
        ])
        labeled = attach_learned_labels(
            rep, anonymous, segmentation_dir=seg_dir,
            config=LearnedLabelConfig(top_k=3, min_top1_score=0.28),
            labeler=fake,
        )
        entities = {e.identity.object_uid: e for e in labeled.entities}
        high, low = entities["obj_10"], entities["obj_42"]

        if high.identity.display_label != "table":
            raise AssertionError("high-confidence label was not admitted")
        if "segment_10" not in high.identity.aliases:
            raise AssertionError("promoted entity lost its anonymous alias")
        if low.identity.display_label != "segment_42":
            raise AssertionError("low-confidence entity lost anonymous identity")
        if len(high.semantic_hypotheses) != 3 or len(low.semantic_hypotheses) != 3:
            raise AssertionError("top-k hypotheses were not preserved")
        if [h.label for h in high.semantic_hypotheses] != [
                "table", "armchair", "bathtub"]:
            raise AssertionError("semantic hypothesis ranking drifted")
        if low.semantic_hypotheses[0].label != "chair":
            raise AssertionError("low-confidence ranking should remain inspectable")
        if any(n != 3 for n, _ in fake.calls):
            raise AssertionError("each entity must use all three rendered views")
        if any(v != GLOBAL_INDOOR_VOCABULARY_V1 for _, v in fake.calls):
            raise AssertionError("labeler did not receive the global vocabulary")
        if labeled.structural_surfaces:
            raise AssertionError("label attachment invented structural surfaces")
        if labeled.notes["oracle_free"] is not True:
            raise AssertionError("oracle-free provenance was not preserved")
        if labeled.notes["label_stage"]["n_promoted_display_labels"] != 1:
            raise AssertionError("admission summary is incorrect")


def test_bundle_hash_and_serde_capture_predictions() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        rep, seg_dir, anonymous = _anonymous(root)

        def run(second_score: float):
            return attach_learned_labels(
                rep, anonymous, segmentation_dir=seg_dir,
                config=LearnedLabelConfig(top_k=3, min_top1_score=0.28),
                labeler=FakeLabeler([
                    _ranking("table", 0.41),
                    _ranking("chair", second_score),
                ]),
            )

        first = run(0.22)
        same = run(0.22)
        changed = run(0.21)
        if first.bundle_hash != same.bundle_hash:
            raise AssertionError("identical label output changed the bundle hash")
        if first.bundle_hash == changed.bundle_hash:
            raise AssertionError("changed label evidence did not change the hash")

        out = root / "labeled_entities"
        dump_entity_artifacts(first, out)
        loaded = load_entity_artifacts(out)
        if not array_aware_equal(first, loaded):
            raise AssertionError("labeled EntityArtifacts did not round-trip")


def test_rejects_mismatched_or_semantic_input() -> None:
    with tempfile.TemporaryDirectory() as td:
        rep, seg_dir, anonymous = _anonymous(Path(td))
        fake_rows = [_ranking("table", 0.4), _ranking("chair", 0.4)]
        invalid = (
            (replace(rep, scene_id="other"), anonymous),
            (replace(rep, representation_hash="other"), anonymous),
            (rep, replace(anonymous, notes={
                **anonymous.notes, "semantic_source": "oracle"})),
            (rep, replace(anonymous, notes={
                **anonymous.notes, "oracle_free": False})),
            (rep, replace(anonymous, entities=[
                replace(anonymous.entities[0], semantic_hypotheses=[
                    SemanticHypothesis(
                        label="table", confidence=1.0, source="preexisting")]),
                *anonymous.entities[1:],
            ])),
        )
        for bad_rep, bad_artifacts in invalid:
            try:
                attach_learned_labels(
                    bad_rep, bad_artifacts, segmentation_dir=seg_dir,
                    labeler=FakeLabeler(fake_rows.copy()))
            except ValueError:
                continue
            raise AssertionError("invalid label-stage input was accepted")


def test_rejects_invalid_labeler_output() -> None:
    with tempfile.TemporaryDirectory() as td:
        rep, seg_dir, anonymous = _anonymous(Path(td))
        invalid_first_rankings = (
            [],
            [{"label": "table", "score": 0.5}],
            [{"label": "not-in-global-vocabulary", "score": 0.5}],
            [{"label": "table", "score": float("nan")}],
            [{"label": "table", "score": 0.2},
             {"label": "chair", "score": 0.3}],
            [{"label": "table", "score": 0.3},
             {"label": "table", "score": 0.2}],
        )
        for ranking in invalid_first_rankings:
            try:
                attach_learned_labels(
                    rep, anonymous, segmentation_dir=seg_dir,
                    labeler=FakeLabeler([ranking, _ranking("chair", 0.4)]))
            except ValueError:
                continue
            raise AssertionError(f"invalid labeler ranking was accepted: {ranking}")


def test_module_has_no_evaluation_or_oracle_import() -> None:
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    forbidden = [name for name in imported
                 if name.startswith(("tools", "eval", "adapters.oracle_replica",
                                     "extractors.oracle_replica"))]
    if forbidden:
        raise AssertionError(f"deployable label stage imports oracle/eval code: {forbidden}")


TESTS = [
    test_topk_and_low_confidence_identity_policy,
    test_bundle_hash_and_serde_capture_predictions,
    test_rejects_mismatched_or_semantic_input,
    test_rejects_invalid_labeler_output,
    test_module_has_no_evaluation_or_oracle_import,
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
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
