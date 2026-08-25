#!/usr/bin/env python3
"""The paper's numbers must trace to the registry, and the figures must be reproducible.

These are integrity tests over committed artifacts, not experiments. They fail if
the paper drifts from `docs/project_results_registry.csv`, if a scope claim is
overstated, or if the figures stop being byte-reproducible.
"""
from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

REGISTRY = REPO / "docs" / "project_results_registry.csv"
AUDIT = REPO / "docs" / "paper_claim_audit.csv"
PAPER = REPO / "docs" / "paper_draft.md"
FIGS = REPO / "docs" / "figures"


def registry() -> dict:
    with REGISTRY.open(newline="") as fh:
        return {r["result_id"]: r for r in csv.DictReader(fh)}


def audit() -> list[dict]:
    with AUDIT.open(newline="") as fh:
        return list(csv.DictReader(fh))


def paper_ids() -> set[str]:
    """result_ids the paper cites, which it writes as `[C01]` or `[C01, C02]`."""
    groups = re.findall(r"`\[([A-F]\d{2}(?:,\s*[A-F]\d{2})*)\]`", PAPER.read_text())
    return {i for g in groups for i in re.findall(r"[A-F]\d{2}", g)}


def test_every_paper_number_traces_to_the_registry():
    reg, cited = registry(), paper_ids()
    assert cited, "the paper cites no result_ids at all"
    unknown = sorted(cited - set(reg))
    assert not unknown, f"paper cites result_ids absent from the registry: {unknown}"


def test_every_cited_id_appears_in_the_claim_audit():
    """A number in the paper with no audit row is an unaudited claim."""
    audited = {i for r in audit() for i in r["result_ids"].split()}
    missing = sorted(paper_ids() - audited)
    assert not missing, f"cited but not audited: {missing}"


def test_audit_ids_all_exist_and_scopes_match_the_registry():
    reg = registry()
    for row in audit():
        for rid in row["result_ids"].split():
            assert rid in reg, f"{row['claim_id']}: unknown result_id {rid}"
        expected = sorted({reg[rid]["scope"] for rid in row["result_ids"].split()})
        assert row["scopes"] == ", ".join(expected), (
            f"{row['claim_id']}: scopes column {row['scopes']!r} disagrees with the "
            f"registry {expected}")


def test_component_result_is_never_called_deployable_or_end_to_end():
    """The single most dangerous overstatement in this paper."""
    reg = registry()
    component = {rid for rid, r in reg.items()
                 if r["scope"] == "oracle_free_component_eval"}
    for row in audit():
        ids = set(row["result_ids"].split())
        if ids & component:
            blob = (row["claim"] + " " + row["scope_note"]).lower()
            assert "not end-to-end" in blob or "component result only" in blob, (
                f"{row['claim_id']} cites a component result without saying so")
            assert not re.search(r"\bdeployable performance\b", row["claim"].lower())

    text = PAPER.read_text()
    for phrase in ("end-to-end top-1", "deployable top-1", "end-to-end matched-instance"):
        assert phrase not in text.lower(), f"paper calls a component result {phrase!r}"


def test_identity_oracle_rows_are_marked_as_bounds():
    reg = registry()
    oracle = {rid for rid, r in reg.items() if r["scope"] == "identity_oracle"}
    for row in audit():
        if set(row["result_ids"].split()) & oracle:
            assert "bound" in row["scope_note"].lower() or "diagnostic" in row["scope_note"].lower(), (
                f"{row['claim_id']} cites an identity_oracle row without calling it a bound")


def test_the_exploratory_support_result_is_not_headlined():
    text = PAPER.read_text()
    abstract = text.split("## 1. Introduction")[0]
    for rid in ("E27", "E28", "E29"):
        assert rid not in abstract, f"{rid} (exploratory) appears in the abstract"
    conclusion = text.split("## 7. Conclusion")[1]
    for rid in ("E27", "E28", "E29"):
        assert rid not in conclusion, f"{rid} (exploratory) appears in the conclusion"
    assert "deliberately not a headline" in text


def test_the_context_control_supports_rather_than_proves():
    text = PAPER.read_text().lower()
    assert "supports** the interpretation" in text or "supports the interpretation" in text
    for overclaim in ("proves the interpretation", "confirms that the gain",
                      "proves that the gain"):
        assert overclaim not in text, f"paper overclaims the control: {overclaim!r}"


def test_figures_exist_and_regenerate_byte_identically():
    names = ["fig1_evaluation_ladder.svg", "fig2_component_result.svg",
             "fig3_held_but_unreachable.svg"]
    before = {}
    for n in names:
        p = FIGS / n
        assert p.is_file(), f"missing figure {n}"
        before[n] = p.read_bytes()
    r = subprocess.run([sys.executable, str(REPO / "tools" / "paper_figures.py")],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, f"figure generation failed: {r.stderr[-400:]}"
    for n in names:
        assert (FIGS / n).read_bytes() == before[n], f"{n} is not byte-reproducible"


def test_figures_carry_their_result_ids_and_scope_warnings():
    f2 = (FIGS / "fig2_component_result.svg").read_text()
    assert "oracle_free_component_eval" in f2 and "NOT end-to-end" in f2
    f3 = (FIGS / "fig3_held_but_unreachable.svg").read_text()
    assert "NOT DEPLOYABLE" in f3
    for rid in ("F35", "F40", "F45", "F50"):
        assert rid in f3, f"figure 3 omits {rid}"


def main() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
