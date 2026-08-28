#!/usr/bin/env python3
"""Integrity tests for the supplementary material.

The supplement is a submitted artifact like the main paper: it must be
anonymised by the same rules, its generated code index must be current, and
the index must cover every result id the main paper cites. Building the PDF
is not required for a green run (mirroring the main paper's page-limit
test): the build assertions skip when no compiled PDF is present.
"""
from __future__ import annotations

import csv
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEX = REPO / "docs" / "3dv"
SEC = TEX / "sec"
REGISTRY = REPO / "docs" / "project_results_registry.csv"


def _supp_files() -> list:
    files = [TEX / "supp.tex"] + sorted(SEC.glob("supp_*.tex"))
    assert (TEX / "supp.tex").is_file(), "docs/3dv/supp.tex is missing"
    assert len(files) > 1, "no sec/supp_*.tex sections found"
    return files


def _supp_source() -> str:
    return "\n".join(f.read_text() for f in _supp_files())


def _main_section_codes() -> set:
    codes = set()
    for f in sorted(SEC.glob("*.tex")):
        if f.name.startswith("supp_"):
            continue
        for group in re.findall(r"\\rid\{([^}]*)\}", f.read_text()):
            codes.update(re.findall(r"[A-Z]\d{2}", group))
    return codes


def test_supplement_is_anonymised():
    """The supplement must satisfy the exact regexes of test_latex_is_anonymised.

    That test scans the main paper's sources; this one applies the same four
    checks, verbatim, to supp.tex and every sec/supp_*.tex (the generated
    code index included).
    """
    src = _supp_source()
    bad = []
    if re.search(r"\bdocs/\w+", src):
        bad.append("a docs/ path")
    if re.search(r"\btools/\w+\.py", src):
        bad.append("a tools/ script name")
    if re.search(r"\b[0-9a-f]{7,40}\b", src):
        bad.append("what looks like a commit hash")
    if re.search(r"surgical.graph.rag", src, re.I):
        bad.append("the repository name")
    assert not bad, f"the supplement carries de-anonymising material: {bad}"


def test_supp_code_index_is_current():
    r = subprocess.run(
        [sys.executable, str(REPO / "tools" / "paper_supp_index.py"), "--check"],
        capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, f"supp code index is stale: {r.stdout}{r.stderr[-200:]}"


def test_index_covers_every_main_paper_citation():
    """Every \\rid code the main sections cite must have a row in the index.

    Reserved codes without registry rows appear as "pending integration"
    rows, so coverage holds even mid-integration.
    """
    cited = _main_section_codes()
    assert cited, "the main sections cite no result ids"
    index = (SEC / "supp_code_index.tex").read_text()
    indexed = set(re.findall(r"^\\texttt\{([A-Z]\d{2})\}", index, re.M))
    missing = sorted(cited - indexed)
    assert not missing, f"main-paper result ids absent from the supp index: {missing}"


def test_scope_table_matches_the_registry():
    """S1's row counts are computed, not transcribed; this keeps them that way."""
    with REGISTRY.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    expected = Counter(r["scope"] for r in rows)

    text = (SEC / "supp_1_scopes.tex").read_text()
    text = text.replace("\\allowbreak", "").replace("\n", " ")
    stated = {}
    for m in re.finditer(r"\\texttt\{([^}]*)\}\s*&[^&]*&\s*(\d+)\s*\\\\", text):
        name = m.group(1).replace("\\_", "_").replace(" ", "").strip()
        stated[name] = int(m.group(2))
    assert stated == dict(expected), (
        f"S1 scope counts disagree with the registry: table {stated}, "
        f"registry {dict(expected)}")

    total = re.search(r"total\s*&\s*&\s*(\d+)\s*\\\\", text)
    assert total, "S1 has no total row"
    assert int(total.group(1)) == len(rows), (
        f"S1 total {total.group(1)} != registry row count {len(rows)}")


def test_the_supplement_pdf_if_built():
    """Building the supplement is not required for a green run.

    Mirrors the main paper's page-limit test: skip when no compiled PDF is
    present, assert basic sanity when one is.
    """
    pdf = TEX / "out" / "supp.pdf"
    if not pdf.is_file():
        print("  SKIP (no compiled PDF; run tectonic on supp.tex in docs/3dv)")
        return
    assert pdf.stat().st_size > 0, "out/supp.pdf is empty"


def main() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
