#!/usr/bin/env python3
"""Synthetic tests for the oracle-free grounding bridge.

No torch, no weights, no real scene: a stub labeler supplies embeddings so the
RULES can be tested in isolation from the model. What matters here is that the
admission rule is cross-view agreement with no numeric threshold, and that the
prediction stage cannot reach the human key.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extractors import arkitscenes_anchor_grounding as mod
from extractors.arkitscenes_anchor_grounding import (
    MIN_AGREEING_SLOTS,
    _phrases,
    score_anchor,
)

# Tokens that name ANSWER-KEY material specifically. Bare "expected" was tried
# and removed: it is an ordinary English word, and it fired on
# `check_schema_version(loaded, expected, ...)` in common/serde.py, which has
# nothing to do with an answer key. A guard that cries wolf gets disabled.
FORBIDDEN = {
    "human_relation_truth", "uid_mappings", "answer_key", "expected_answer",
    "human_feedback", "human_key", "arkitscenes_relation_challenge_score",
    "arkitscenes_spatial_qa_score", "3dod_annotation", "oracle_label",
    "oracle_box",
}


class StubLabeler:
    """Deterministic stand-in: phrase -> unit vector on its own axis."""

    def __init__(self, axes: dict[str, int], dim: int = 4):
        self.axes, self.dim = axes, dim

    def text_embeddings(self, vocabulary):
        out = np.zeros((len(vocabulary), self.dim))
        for i, phrase in enumerate(vocabulary):
            out[i, self.axes[phrase]] = 1.0
        return _Torchish(out)


class _Torchish:
    def __init__(self, a): self._a = a
    def numpy(self): return self._a


def labeler():
    return StubLabeler({"sofa": 0, "couch": 0, "table": 1})


def emb(*rows):
    return np.array(rows, dtype=float)


# ------------------------------------------------------------------ guards
def _first_party_sources() -> list[Path]:
    """The bridge plus its transitive first-party imports, and the runner."""
    seen, stack = set(), [Path(mod.__file__),
                          REPO_ROOT / "tools" / "arkitscenes_grounding_run.py"]
    out = []
    while stack:
        path = stack.pop()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        out.append(path)
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            elif isinstance(node, ast.Import):
                names += [a.name for a in node.names]
            for name in names:
                candidate = REPO_ROOT / (name.replace(".", "/") + ".py")
                if candidate.is_file():
                    stack.append(candidate)
    return out


# `adapters/arkitscenes.py` names ANNOTATION_SUFFIX as a constant it
# deliberately never opens, and carries its own test asserting that. An
# allowlist of one, with the reason written down, is honest; silently dropping
# the token from the ban would weaken the guard everywhere else.
ANNOTATION_FREE_BY_ITS_OWN_TEST = {"arkitscenes.py"}


def test_prediction_stage_cannot_reach_the_human_key():
    """AST over the bridge, the runner AND their first-party import graph.

    Checking only the top file would miss a key read one import away, which is
    exactly how an oracle leak survives review.
    """
    checked = _first_party_sources()
    assert len(checked) >= 3, f"import graph looks too small: {checked}"
    for path in checked:
        tree = ast.parse(path.read_text())
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        literals = {n.value for n in ast.walk(tree)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        # clean=False so the raw constant matches; the cleaned form does not,
        # which silently defeated the subtraction on the first attempt.
        docstrings = {ast.get_docstring(n, clean=False) for n in ast.walk(tree)
                      if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))}
        literals -= {d for d in docstrings if d}
        banned = FORBIDDEN - ({"3dod_annotation"}
                              if path.name in ANNOTATION_FREE_BY_ITS_OWN_TEST
                              else set())
        leaked = {f for f in banned
                  if f in names or any(f in lit for lit in literals)}
        assert not leaked, f"{path.name} reaches oracle material: {leaked}"


def test_prediction_stage_imports_no_evaluation_module():
    """`tools/arkitscenes_eval` was imported for three lines of geometry.

    An evaluation module is off-limits to the prediction stage regardless of
    which function is called from it, because the import is what makes a leak
    possible later.
    """
    for path in _first_party_sources():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.ImportFrom) and node.module:
                mods.append(node.module)
            elif isinstance(node, ast.Import):
                mods += [a.name for a in node.names]
            for name in mods:
                assert "_eval" not in name and not name.endswith(".eval"), \
                    f"{path.name} imports evaluation module {name}"


def test_bridge_pins_the_label_stage_crop_configuration():
    assert (mod.CROP_STRIDE, mod.CROP_N_VIEWS, mod.CROP_CONTEXT_PAD,
            mod.CROP_MARK_TARGET) == (6, 3, 0.15, False)


def test_no_confidence_threshold_constant_exists():
    """The only gate is cross-view agreement; a cutoff could be fitted."""
    source = Path(mod.__file__).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and \
                        isinstance(comparator.value, float):
                    raise AssertionError(
                        f"float comparison at line {node.lineno} looks like a "
                        "confidence threshold")


# -------------------------------------------------------------- the rules
def test_admits_when_the_top_entity_wins_two_view_slots():
    embeddings = {
        "obj_0": emb([1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]),   # wins 2 of 3
        "obj_1": emb([0, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]),
    }
    row = score_anchor("sofa", {"sofa": ["couch"]}, embeddings, labeler())
    assert row["admitted"] is True
    assert row["uid"] == "obj_0"
    assert row["agreeing_slots"] >= MIN_AGREEING_SLOTS


def test_abstains_when_the_top_entity_wins_only_one_slot():
    embeddings = {
        "obj_0": emb([1, 0, 0, 0], [0.4, 0, 0, 0], [0.4, 0, 0, 0]),
        "obj_1": emb([0.0, 1, 0, 0], [0.5, 0, 0, 0], [0.5, 0, 0, 0]),
    }
    row = score_anchor("sofa", {}, embeddings, labeler())
    assert row["admitted"] is False
    assert row["uid"] is None
    assert "view slot" in row["reason"]
    assert row["top_uid"] is not None, "the evidence is kept even when abstaining"


def test_synonyms_are_combined_by_max_not_average():
    """A rare synonym must not dilute the anchor's own name."""
    embeddings = {"obj_0": emb([1, 0, 0, 0], [1, 0, 0, 0])}
    with_syn = score_anchor("sofa", {"sofa": ["couch", "table"]},
                            embeddings, labeler())
    without = score_anchor("sofa", {}, embeddings, labeler())
    assert with_syn["ranking"][0]["aggregate"] == without["ranking"][0]["aggregate"]


def test_phrase_set_is_deduplicated_and_order_stable():
    assert _phrases("sofa", {"sofa": ["couch", "sofa", "COUCH"]}) == ["sofa", "couch"]


def test_entities_without_a_usable_crop_do_not_compete():
    embeddings = {"obj_0": None,
                  "obj_1": emb([1, 0, 0, 0], [1, 0, 0, 0])}
    row = score_anchor("sofa", {}, embeddings, labeler())
    assert row["uid"] == "obj_1"
    assert row["n_entities_ranked"] == 1


def test_ranking_is_deterministic_under_ties():
    embeddings = {"obj_9": emb([1, 0, 0, 0], [1, 0, 0, 0]),
                  "obj_1": emb([1, 0, 0, 0], [1, 0, 0, 0])}
    rows = [score_anchor("sofa", {}, embeddings, labeler()) for _ in range(3)]
    assert {r["top_uid"] for r in rows} == {"obj_1"}, "ties break by uid ascending"
    assert rows[0] == rows[1] == rows[2]


def test_absent_objects_are_not_special_cased():
    """No branch may suppress an anchor for being undeliverable.

    Knowing an object is absent would require the human key. If the rule
    admits a uid for one, that is a precision error to report, not to hide.
    """
    source = Path(mod.__file__).read_text().lower()
    for token in ("rug", "radiator", "none_missing", "absent_object"):
        assert f'== "{token}"' not in source and f"== '{token}'" not in source


def main() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
