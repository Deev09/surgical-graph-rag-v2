#!/usr/bin/env python3
"""Generate the supplement's index of result identifiers cited by the main paper.

Scans the MAIN paper sections (docs/3dv/sec/*.tex, excluding the supp_*
files) for \\rid{...} citations, joins each distinct code against
docs/project_results_registry.csv, and emits docs/3dv/sec/supp_code_index.tex:
a longtable of (code, metric, value, scope, source artifact), which supp.tex
inputs as its final section.

The emitted table is anonymisation-safe by construction: source artifacts
are reduced to basenames, long hex/digit runs (commit hashes, scene ids) are
elided, and over-long decimals are truncated, so the generated file passes
the same regexes as `test_latex_is_anonymised`.

Codes G07--G12 are reserved for results still being integrated and may be
absent from the registry; any cited code with no registry row is emitted
with a "pending integration" marker rather than failing.

    tools/paper_supp_index.py [--check]

`--check` recomputes and fails if the committed output would change
(byte-compare, mirroring tools/paper_statistics.py --check).
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SEC = REPO / "docs" / "3dv" / "sec"
REGISTRY = REPO / "docs" / "project_results_registry.csv"
OUT = SEC / "supp_code_index.tex"

# Reserved for results landing at integration time; these may legitimately be
# cited before their registry rows exist.
RESERVED = {f"G{n:02d}" for n in range(7, 13)}

_ESCAPES = [
    ("\\", r"\textbackslash{}"),
    ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
    ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
    ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
    ("<", r"\textless{}"), (">", r"\textgreater{}"),
]


def latex_escape(text: str) -> str:
    for char, rep in _ESCAPES:
        text = text.replace(char, rep)
    return text


def sanitize(text: str) -> str:
    """Make a registry field safe for the anonymised LaTeX.

    Truncates decimals longer than six places (a seven-digit run matches the
    commit-hash regex), then elides any remaining 7+ character hex/digit run
    (commit hashes, scene ids, bundle hashes).
    """
    text = re.sub(r"(\d\.\d{6})\d+", r"\1", text)
    text = re.sub(r"\b[0-9a-f]{7,40}\b", "(id)", text)
    text = text.replace(" \u2014 ", ": ").replace("\u2014", ":")
    # Digit-bearing hex runs are elided even inside underscore-joined names
    # (scene ids and hashes embedded in filenames have no word boundary).
    text = re.sub(r"(?=[0-9a-f]*\d)[0-9a-f]{7,40}", "(id)", text)
    return text


def sanitize_artifact(path: str) -> str:
    """Reduce a source-artifact path to an anonymisation-safe basename."""
    text = path.replace("eval/results/project_census_v1/", "evidence pack: ")
    text = re.sub(r"(?:[\w.~-]+/)+", "", text)
    text = re.sub(r"  +", " ", text)
    return sanitize(text).strip() or "(unstated)"


def main_section_files() -> list:
    return sorted(f for f in SEC.glob("*.tex") if not f.name.startswith("supp_"))


def cited_codes() -> list:
    codes = set()
    for f in main_section_files():
        for group in re.findall(r"\\rid\{([^}]*)\}", f.read_text()):
            codes.update(re.findall(r"[A-Z]\d{2}", group))
    return sorted(codes)


def registry() -> dict:
    with REGISTRY.open(newline="") as fh:
        return {r["result_id"]: r for r in csv.DictReader(fh)}


def build() -> str:
    reg = registry()
    codes = cited_codes()
    lines = [
        "% Auto-generated index of the result identifiers cited by the main",
        "% paper, joined against the results registry. Do not edit by hand;",
        "% regenerate with the supplementary index generator (--check compares",
        "% bytes). Source artifacts are given by basename only.",
        r"\section{Index of cited result identifiers}",
        r"\label{ssec:codeindex}",
        "",
        "Every result identifier cited in the main paper, joined against the",
        "results registry that accompanies the submission. Source artifacts",
        "are given by filename only; full paths, producing commits and per-row",
        "notes are in the registry and the evidence-pack manifest.",
        "",
        r"{\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{longtable}{@{}l p{0.34\textwidth} p{0.17\textwidth} p{0.15\textwidth} p{0.17\textwidth}@{}}",
        r"\toprule",
        r"code & metric & value & scope & source artifact\\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"code & metric & value & scope & source artifact\\",
        r"\midrule",
        r"\endhead",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for code in codes:
        row = reg.get(code)
        if row is None:
            marker = "pending integration --- reserved identifier, no registry row yet"
            if code not in RESERVED:
                marker = "pending integration --- no registry row yet"
            lines.append(
                r"\texttt{%s} & \multicolumn{4}{l}{\emph{%s}}\\" % (code, marker))
            continue
        metric = latex_escape(sanitize(row["metric_name"].strip()))
        value = latex_escape(sanitize(row["value"].strip() or "(unstated)"))
        # \allowbreak after each escaped underscore lets long scope names wrap
        # inside their narrow column.
        scope = latex_escape(sanitize(row["scope"].strip())).replace(
            r"\_", r"\_\allowbreak{}")
        artifact = latex_escape(sanitize_artifact(row["primary_source_artifact"].strip())).replace(
            r"\_", r"\_\allowbreak{}")
        lines.append(
            r"\texttt{%s} & %s & %s & \texttt{%s} & %s\\"
            % (code, metric, value, scope, artifact))
    lines += [r"\end{longtable}", "}", ""]
    return "\n".join(lines)


def main(argv: list) -> int:
    content = build()
    if "--check" in argv:
        before = OUT.read_bytes() if OUT.is_file() else b""
        OUT.write_text(content)
        after = OUT.read_bytes()
        if before != after:
            print("committed supp code index is stale; re-run the generator")
            return 1
        print("supp code index is current")
        return 0
    OUT.write_text(content)
    pending = [c for c in cited_codes() if c not in registry()]
    print(f"wrote {OUT.relative_to(REPO)}: {len(cited_codes())} codes"
          + (f", {len(pending)} pending integration ({', '.join(pending)})"
             if pending else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
