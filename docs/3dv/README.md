# 3DV 2027 submission build

Anonymous review version. Paper deadline 2026-08-28 23:59 AoE
(= 2026-08-29 06:59 CDT); supplementary 2026-09-02 11:00 PDT. No extension.
Eight pages excluding references; reference pages do not count.

## Build

    mkdir -p out && tectonic -X compile main.tex --outdir out

`tectonic` is self-contained and fetches its own packages, so no TeX
installation is needed. There is no pdflatex on this machine.

## Regenerating inputs

    tools/paper_figures.py       # the SVG figures, from the results registry
    tools/paper_figures_pdf.py   # the two the paper embeds, SVG -> cropped PDF
    tools/paper_bib_latex.py     # refs.bib, from the working bibliography

All three have a `--check` mode that fails if the committed output is stale;
the test suite runs them.

## Anonymisation

This version carries no repository paths, no commit hashes, no tool names and
no repository name. `test_latex_is_anonymised` enforces that. The working draft
in `docs/paper_draft.md` keeps them, and they go back in for the camera-ready.

## Page count

The compiled PDF is the authority, not word count. Read it with:

    grep -o '\newlabel{endofmaintext}{{[^}]*}{[0-9]*}' out/main.aux

which reports the last page of the main text.
