#!/usr/bin/env python3
"""The paper's numbers must trace to the registry, and the figures must be reproducible.

These are integrity tests over committed artifacts, not experiments. They fail if
the paper drifts from `docs/project_results_registry.csv`, if a scope claim is
overstated, or if the figures stop being byte-reproducible.
"""
from __future__ import annotations

import csv
import json
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
    # A-Z, not A-F: the hard-coded range silently stopped seeing citations the
    # moment derived G-rows were added, and every test that depends on this
    # function passed vacuously as a result.
    groups = re.findall(r"`\[([A-Z]\d{2}(?:,\s*[A-Z]\d{2})*)\]`", PAPER.read_text())
    return {i for g in groups for i in re.findall(r"[A-Z]\d{2}", g)}


def test_every_paper_number_traces_to_the_registry():
    reg, cited = registry(), paper_ids()
    assert cited, "the paper cites no result_ids at all"
    unknown = sorted(cited - set(reg))
    assert not unknown, f"paper cites result_ids absent from the registry: {unknown}"


# Deliberately empty. Every quantity in the paper traces to a registry row, and
# section cross-references (which are not quantities) are stripped before the scan
# rather than excused here. If a number needs an entry in this set, the honest fix
# is almost always an audited registry row instead.
PROSE_CONSTANTS: set[str] = set()


def _registry_numbers(reg: dict) -> set[str]:
    """Every number the registry actually asserts, in the forms prose may use."""
    out: set[str] = set()
    for row in reg.values():
        num = (row.get("numerator") or "").strip()
        den = (row.get("denominator") or "").strip()
        for text in ((row.get("value") or "").strip(), num, den):
            for frac in re.findall(r"\d+\s*/\s*\d+", text):
                a, b = re.split(r"\s*/\s*", frac)
                out.add(f"{a}/{b}")
            for dec in re.findall(r"\d+\.\d+", text):
                out.add(dec)
                out.add(dec.rstrip("0").rstrip("."))
        if num.isdigit() and den.isdigit():
            out.add(f"{num}/{den}")
            out.update({num, den})
    return out


def test_no_paper_number_is_derived_outside_the_registry():
    """A quantity the prose states must be a quantity the registry asserts.

    This exists because reconstructed numbers passed review twice: "21 of 44
    annotated entities" was summed across three scenes, and a false-confident
    rate of "0.222" was recomputed from two wrong among nine answered. Both were
    arithmetically right and neither was auditable, which is exactly what the
    registry exists to prevent.
    """
    reg = registry()
    known = _registry_numbers(reg) | PROSE_CONSTANTS
    text = PAPER.read_text()
    text = re.sub(r"`\[[^\]]*\]`", " ", text)   # citation brackets
    text = re.sub(r"`[^`]*`", " ", text)         # code spans: ids, paths, settings
    text = re.sub(r"§\s*\d+(\.\d+)?", " ", text)  # section cross-references
    text = re.sub(r"^#+ *\d+(\.\d+)?", " ", text, flags=re.M)  # section headings

    found: list[str] = []
    for a, b in re.findall(r"(?<![\d./])(\d+)\s*/\s*(\d+)(?![\d./])", text):
        if f"{a}/{b}" not in known:
            found.append(f"{a}/{b}")
    for dec in re.findall(r"(?<![\d.])(\d+\.\d+)(?!\d)", text):
        if dec not in known and dec.rstrip("0").rstrip(".") not in known:
            found.append(dec)
    for a, b in re.findall(r"(?<![\d.])(\d+)\s+of\s+(\d+)(?![\d.])", text):
        if f"{a}/{b}" not in known:
            found.append(f"{a} of {b}")

    assert not found, (
        "quantities stated in the paper that no registry row asserts "
        f"(add an audited row, or report the registered value): {sorted(set(found))}"
    )


def test_the_blinding_claim_is_not_overstated():
    """Commit order does not prove a response predates a key.

    For the transfer run the key commit (45f8ec9) precedes the response hash-pin
    (e193e6f) by eight minutes. The defensible claim is isolation plus a hash pin
    before scoring, so the paper must not reassert the ordering version.
    """
    # normalise: the prose wraps lines and uses ** emphasis mid-phrase
    text = re.sub(r"\s+", " ", PAPER.read_text().replace("*", "")).lower()
    banned = ["committed before the human key is opened",
              "auditable in version history",
              "hash-pinned and committed before the human key"]
    hit = [b for b in banned if b in text]
    assert not hit, f"the paper restates the disproved commit-ordering claim: {hit}"
    assert "no access to the key" in text, (
        "the paper must state the defensible blinding claim: the response was "
        "generated in an isolated context without key access")


# result_ids previously cited for claims they do not support.
MISCITATION_GUARDS = [
    ("F76", "was voided"),    # F76 is run 2's "correct 5/10", not the voiding
    ("F70", "label stage"),   # F70 is the grounding bridge's admissions
]


def test_known_miscitations_do_not_reappear():
    text = PAPER.read_text()
    bad = []
    for rid, phrase in MISCITATION_GUARDS:
        for m in re.finditer(re.escape(f"[{rid}"), text):
            if phrase.lower() in text[max(0, m.start() - 240):m.start()].lower():
                bad.append(f"{rid} cited near '{phrase}'")
    assert not bad, f"a corrected miscitation has returned: {bad}"


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
    conclusion = re.split(r"## \d+\. Conclusion", text)[1]
    for rid in ("E27", "E28", "E29"):
        assert rid not in conclusion, f"{rid} (exploratory) appears in the conclusion"
    assert "deliberately not a headline" in text


def test_the_context_control_supports_rather_than_proves():
    text = PAPER.read_text().lower()
    assert "supports** the interpretation" in text or "supports the interpretation" in text
    for overclaim in ("proves the interpretation", "confirms that the gain",
                      "proves that the gain"):
        assert overclaim not in text, f"paper overclaims the control: {overclaim!r}"


MAIN_PAPER_FIGURES = ("fig1_evaluation_ladder.svg", "fig4_reachability.svg")


def test_figures_exist_and_regenerate_byte_identically():
    names = sorted(p.name for p in FIGS.glob("*.svg"))
    assert len(names) == 4, f"expected 4 figures, found {names}"
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
    """Every generated figure keeps its scope warning, used in the paper or not."""
    f2 = (FIGS / "fig2_component_result.svg").read_text()
    assert "oracle_free_component_eval" in f2 and "NOT end-to-end" in f2
    f3 = (FIGS / "fig3_held_but_unreachable.svg").read_text()
    assert "NOT DEPLOYABLE" in f3
    for rid in ("F35", "F40", "F45", "F50"):
        assert rid in f3, f"figure 3 omits {rid}"
    # figure 4 carries the same four ids and must, since it replaced figure 3
    # in the main paper and is now the only place a reader sees the four arms
    f4 = (FIGS / "fig4_reachability.svg").read_text()
    for rid in ("F35", "F40", "F45", "F50"):
        assert rid in f4, f"figure 4 omits {rid}"
    assert "identity_oracle" in f4, "figure 4 lost its scope banner"


def test_main_paper_embeds_only_the_chosen_figures():
    """The author cut figures 2 and 3 for the page limit; keep it that way.

    Figure 2's table was kept instead of the figure, and figure 4 subsumes
    figure 3. Both still generate, for the supplement.
    """
    embedded = set(re.findall(r"!\[[^\]]*\]\(figures/([^)]+)\)", PAPER.read_text()))
    assert embedded == set(MAIN_PAPER_FIGURES), (
        f"main paper embeds {sorted(embedded)}, expected {sorted(MAIN_PAPER_FIGURES)}")


BIB = REPO / "docs" / "paper_references.bib"


def _section_2() -> str:
    text = PAPER.read_text()
    return text[text.index("## 2. Related work"):text.index("## 3. Method")]


def test_every_citation_resolves_to_the_bibliography():
    """A citation key in the prose must exist in the .bib, and vice versa.

    Related work is the one section whose claims are about OTHER papers, so the
    failure mode is a key that looks right and resolves to nothing. This makes a
    typo a build failure rather than a dangling reference in a submission.
    """
    assert BIB.is_file(), "docs/paper_references.bib is missing"
    defined = set(re.findall(r"@\w+\{([^,]+),", BIB.read_text()))
    assert defined, "the bibliography defines no entries"

    used: set[str] = set()
    for group in re.findall(r"\[([a-z][a-z0-9]*\d{4}[a-z0-9]*(?:,\s*[a-z][a-z0-9]*\d{4}[a-z0-9]*)*)\]",
                            _section_2()):
        used.update(k.strip() for k in group.split(","))
    assert used, "section 2 cites nothing"

    dangling = sorted(used - defined)
    assert not dangling, f"cited in section 2 but absent from the bibliography: {dangling}"
    orphan = sorted(defined - used)
    assert not orphan, f"in the bibliography but never cited: {orphan}"


def test_related_work_is_written():
    """Guards against the section reverting to its placeholder."""
    sec = _section_2()
    assert "Left to the author" not in sec, "section 2 is back to its placeholder"
    assert len(sec.split()) > 600, f"section 2 is only {len(sec.split())} words"


def test_no_citation_is_marked_unverified():
    """Every bibliography entry carries a fact-check verdict, and none failed."""
    text = BIB.read_text()
    entries = re.findall(r"@\w+\{([^,]+),(.*?)\n\}", text, flags=re.S)
    assert entries, "no parseable bibliography entries"
    bad = [k for k, body in entries
           if "verification: exists_as_stated" not in body]
    assert not bad, (
        "bibliography entries whose fact-check did not come back clean "
        f"(re-check or remove them before submission): {bad}")


STATS = REPO / "eval" / "results" / "paper_statistics.json"
LEDGER = REPO / "docs" / "paper_reachability_ledger.csv"


def test_statistics_and_ledger_are_current():
    """The committed statistics must match what the tool produces now.

    Both the ledger and figure 4 are generated from the same report, so a stale
    report would silently desynchronise the paper, the CSV and the figure.
    """
    r = subprocess.run([sys.executable, str(REPO / "tools" / "paper_statistics.py"), "--check"],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, f"statistics are stale: {r.stdout}{r.stderr[-300:]}"


def test_paired_tests_reproduce_the_registry_marginals():
    """The paired analysis must agree with the rows it claims to strengthen.

    A paired table that did not reproduce C01-C04's marginals would mean the
    per-instance files and the summary rows disagree, which is a data problem,
    not a statistics one.
    """
    reg, stats = registry(), json.loads(STATS.read_text())
    t1, t3 = stats["paired_label_tests"]

    def num(rid: int | str) -> int:
        return int(reg[rid]["numerator"])

    expected = {"top1_correct": (num("C01"), num("C02")),
                "top3_correct": (num("C03"), num("C04"))}
    for t in (t1, t3):
        splat, rgb = expected[t["field"]]
        assert t["splat_correct"] == splat, (
            f"{t['field']}: paired table says splat {t['splat_correct']}, registry says {splat}")
        assert t["rgb_tight_correct"] == rgb, (
            f"{t['field']}: paired table says rgb {t['rgb_tight_correct']}, registry says {rgb}")
        assert t["n_instances"] == int(reg["C01"]["denominator"]) == 21
        # b + c must equal the discordant count the p-value was computed from
        assert t["rgb_only_correct"] + t["splat_only_correct"] == t["discordant_pairs"]


def test_the_clustering_limitation_is_recorded():
    """A p-value over 21 instances from 3 rooms must carry its caveat."""
    stats = json.loads(STATS.read_text())
    for t in stats["paired_label_tests"]:
        note = t.get("clustering_limitation", "")
        assert "clustered" in note and "not" in note.lower(), (
            f"{t['field']} has no clustering caveat")
    for rid in ("G01", "G02"):
        assert "INSTANCE-LEVEL" in registry()[rid]["notes"], (
            f"{rid} does not record that its p-value is instance-level only")


def test_ledger_does_not_conflate_multi_view_with_cross_view():
    """The ledger's multi-view flag is not the transfer test's cross-view.

    The transfer test's cross-view items are non-co-visible by construction;
    the ledger's flag only records that an answer drew on more than one view.
    Pooling them would repeat exactly the denominator merge this project has
    already had to correct once.
    """
    header = LEDGER.read_text().splitlines()[0]
    assert "cross_view" not in header, (
        "the ledger has a bare cross_view column again; it must be named "
        "evidence_spans_multiple_views to keep it distinct from the transfer test")
    assert "evidence_spans_multiple_views" in header
    stats = json.loads(STATS.read_text())
    note = stats["reachability"]["multi_view_evidence"]["note"]
    assert "non-co-visible" in note and "0/3" in note


def test_derived_registry_rows_are_marked_as_derived():
    """G-rows are re-readings of scored outcomes and must say so."""
    reg = registry()
    g = {k: v for k, v in reg.items() if k.startswith("G0")}
    assert g, "no derived rows found"
    for rid, row in g.items():
        assert "DERIVED ROW" in row["notes"], f"{rid} is not marked as derived"
        # the point is that it is re-read in-repo from committed evidence, not that
        # it came from one particular file
        assert "derived in-repo" in row["source_commit_or_tag"], (
            f"{rid} does not record that it was derived in-repo from committed evidence")
        src = row["primary_source_artifact"]
        assert src.startswith("eval/results/"), f"{rid} points outside the evidence tree: {src}"


TEX = REPO / "docs" / "3dv"


def _tex_source() -> str:
    return "\n".join(f.read_text() for f in sorted((TEX / "sec").glob("*.tex")))


def test_latex_result_ids_all_exist_and_are_audited():
    """The submission PDF is built from the LaTeX, not the markdown.

    Every \\rid{...} in the LaTeX must resolve to a registry row and appear in
    the claim audit, exactly as the markdown's citations must. Without this the
    two can drift and the artifact that actually gets submitted is unchecked.
    """
    reg, audited = registry(), set()
    for row in audit():
        audited.update(row["result_ids"].split())
    ids = set()
    for group in re.findall(r"\\rid\{([^}]*)\}", _tex_source()):
        ids.update(re.findall(r"[A-Z]\d{2}", group))
    assert ids, "the LaTeX cites no result ids"
    unknown = sorted(ids - set(reg))
    assert not unknown, f"LaTeX cites ids absent from the registry: {unknown}"
    unaudited = sorted(ids - audited)
    assert not unaudited, f"LaTeX cites ids absent from the claim audit: {unaudited}"


def test_latex_cites_no_number_the_markdown_does_not():
    """The LaTeX is a port, not a new set of claims."""
    md = paper_ids()
    tex = set()
    for group in re.findall(r"\\rid\{([^}]*)\}", _tex_source()):
        tex.update(re.findall(r"[A-Z]\d{2}", group))
    extra = sorted(tex - md)
    assert not extra, (
        f"the LaTeX introduces result ids the markdown does not carry: {extra}")


def test_latex_bibliography_is_generated_from_the_working_file():
    r = subprocess.run([sys.executable, str(REPO / "tools" / "paper_bib_latex.py"), "--check"],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, f"refs.bib is stale: {r.stdout}{r.stderr[-200:]}"


def test_latex_is_anonymised():
    """3DV rejects non-anonymous submissions without review.

    The working draft is full of repository paths and commit hashes; those are
    de-anonymising if the repository is findable, so the LaTeX must not carry
    them.
    """
    src = _tex_source() + (TEX / "main.tex").read_text()
    bad = []
    if re.search(r"\bdocs/\w+", src):
        bad.append("a docs/ path")
    if re.search(r"\btools/\w+\.py", src):
        bad.append("a tools/ script name")
    if re.search(r"\b[0-9a-f]{7,40}\b", src):
        bad.append("what looks like a commit hash")
    if re.search(r"surgical.graph.rag", src, re.I):
        bad.append("the repository name")
    assert not bad, f"the LaTeX carries de-anonymising material: {bad}"


def test_figures_carry_no_baked_in_number():
    """Numbering belongs to the document, not the graphic.

    LaTeX renumbers figures on every edit; a title reading "Figure 4" inside the
    graphic then contradicts the caption beneath it, which is exactly what
    happened once the paper dropped two figures.
    """
    offenders = []
    for f in sorted(FIGS.glob("*.svg")):
        if re.search(r"Figure\s*\d", f.read_text()):
            offenders.append(f.name)
    qual = REPO / "tools" / "paper_qualitative_figure.py"
    if qual.is_file() and re.search(r"<h1>Figure\s*\d", qual.read_text()):
        offenders.append("fig5_qualitative")
    assert not offenders, f"figures bake in their own number: {offenders}"


def test_the_submission_pdf_is_built_and_within_the_page_limit():
    """The compiled PDF is the page-limit authority, not word count.

    3DV allows eight pages excluding references. The main text must therefore
    end on page 8 or earlier; reference pages beyond that do not count.
    """
    pdf = TEX / "out" / "main.pdf"
    aux = TEX / "out" / "main.aux"
    if not (pdf.is_file() and aux.is_file()):
        print("  SKIP (no compiled PDF; run tectonic in docs/3dv)")
        return
    m = re.search(r"\\newlabel\{endofmaintext\}\{\{[^}]*\}\{(\d+)\}", aux.read_text())
    assert m, "main.aux has no endofmaintext label; the marker was removed"
    last = int(m.group(1))
    assert last <= 8, f"main text ends on page {last}; the limit is 8 excluding references"


def main() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
