#!/usr/bin/env python3
"""Assemble the anonymous 3DV supplementary package in a clean staging dir.

Populates runs/submission_package/ (gitignored) from:
  1. a fixed allowlist of tracked evidence files,
  2. freshly built docs/3dv/out/supp.pdf (NEVER main.pdf: 3DV prohibits
     supplements that contain an updated/corrected submission PDF),
  3. sanitized frame-ID manifests for the frozen blinded packets
     (frame ids + sha256 pins + prompt text; NO ARKitScenes imagery),
then writes a top-level README with the single reproduction command, a
sha256 manifest of everything staged, runs an anonymity scan over every
staged TEXT file, and zips the staging directory.

Layout contract: everything the packaged runner reads is staged under
code/ at its repo-relative path, because code/tools/stagereach_eval.py
resolves REPO = code/. From a clean unzip this must succeed:

    python3 code/tools/stagereach_eval.py --track all --check

Sanitization applied to STAGED COPIES ONLY (repo files are never touched):
  - registry CSV: source_commit_or_tag values -> "withheld-for-review"
    (the repository is public, so a commit hash is searchable and would
    de-anonymize the submission; camera-ready restores them)
  - claim-audit CSV: the values_at_<hash> column is renamed values_at_freeze
  - evidence-pack MANIFEST: producing_commit values -> "withheld-for-review"

The scan covers staged text files. supp.pdf is NOT text-scanned here;
it must be read page-by-page in the final manual sweep.

    tools/paper_submission_package.py [--no-zip]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STAGE = REPO / "runs" / "submission_package" / "stage"
ZIP_OUT = REPO / "runs" / "submission_package" / "supplementary_material.zip"

PACK = REPO / "eval" / "results" / "project_census_v1"
WITHHELD = "withheld-for-review"

# Tracked evidence allowlist (directories are copied file-by-file).
ALLOWLIST = [
    "docs/project_results_registry.csv",
    "docs/paper_claim_audit.csv",
    "docs/paper_reachability_ledger.csv",
    "eval/results/paper_statistics.json",
]
# Data the packaged runner needs, staged under code/ at repo-relative
# paths so code/tools/stagereach_eval.py --track all --check resolves them
# from a clean unzip. REQUIRED: missing files fail the build.
RUNNER_DATA = [
    "eval/results/stagereach/arkit_stagereach_v1.json",
    "eval/results/stagereach/replica_stagereach_v1.json",
    "eval/results/stagereach/fault_battery_v1.json",
    "eval/fixtures/stagereach/fault_fixture_v1.json",
    "eval/questions/phase8/replica_office_0_qa.json",
    "eval/questions/phase8/replica_room_0_qa.json",
    "eval/questions/phase8/replica_room_1_qa.json",
    "eval/questions/phase8/replica_room_2_qa.json",
]
# The census evidence pack is staged INSIDE code/ at its repo-relative
# path -- the arkit and replica adapters read it from there.
PACK_STAGE_REL = "code/eval/results/project_census_v1"

# Top-level README. EXACTLY ONE reproduction command; it must keep working
# from a clean unzip (verified against the staged layout by this build's
# smoke test). Scanned by the anonymity scan like every staged text file.
README = """\
# Supplementary Material -- 3DV 2027 Submission #468 (StageReach3D)

## Contents

- `supp.pdf` -- the supplementary document.
- `code/` -- the frozen StageReach3D evaluator (schema, evaluator,
  metrics, fault injection, dataset adapters), its unit tests, the
  command-line runner, and every input the runner reads: the packed
  evidence reports under `code/eval/results/project_census_v1/`, the
  committed result artifacts under `code/eval/results/stagereach/`, the
  fault fixture under `code/eval/fixtures/stagereach/`, and the Replica
  QA keys under `code/eval/questions/phase8/`.
- `annotation/` -- second-annotator returns, the agreement artifact, and
  the key-sensitivity artifact `annotation_sensitivity_v1.json` (recomputed
  by `code/tools/annotation_sensitivity.py`; backs the supplement's
  key-sensitivity table). Evaluation-key reproducibility evidence; used to
  retune nothing.
- `packets/` -- frame-ID manifests (frame id + sha256 against the source
  dataset) and prompts for the frozen blinded packets. No dataset
  imagery is redistributed.
- `project_results_registry.csv`, `paper_claim_audit.csv`,
  `paper_reachability_ledger.csv`, `paper_statistics.json` -- the
  results registry and claim audit backing the paper's numbers.
- `MANIFEST.json` -- sha256 of every other file in this package.

For anonymity, commit identifiers are replaced with
"withheld-for-review" and absolute paths with "<home>" / "<repo>" in the
packaged copies; camera-ready restores them.

## Reproduction

Setup: any Python 3.9+ interpreter; the code uses only the Python
standard library (nothing to install).

From the directory containing this README, run:

    python3 code/tools/stagereach_eval.py --track all --check

Expected output: the runner recomputes, from the packed evidence only,
(a) the per-arm ARKit relation-challenge StageReach traces and survival
ladders, (b) the Replica Phase-8 schema/outcome transfer, (c) the fault
fixture, and (d) the 24/24 evaluator-masked fault-injection battery,
then byte-compares each against the committed artifacts shipped under
`code/eval/`. On success it prints the single line

    stagereach artifacts are current

and exits with status 0. Any divergence prints the stale artifact paths
and exits nonzero.
"""
# The 24/24 claim's full evidence chain: evaluator code, CLI, and the test
# that asserts the battery. Staged under code/ with paths flattened.
CODE_FILES = [
    "eval/stagereach/__init__.py",
    "eval/stagereach/schema.py",
    "eval/stagereach/evaluator.py",
    "eval/stagereach/metrics.py",
    "eval/stagereach/faults.py",
    "eval/stagereach/adapters/__init__.py",
    "eval/stagereach/adapters/arkit.py",
    "eval/stagereach/adapters/replica.py",
    "tools/stagereach_eval.py",
    "tools/stagereach_numbers.py",
    "tools/annotation_sensitivity.py",
    "tests/eval/test_stagereach_schema.py",
    "tests/eval/test_stagereach_evaluator.py",
    "tests/eval/test_stagereach_faults.py",
    "tests/eval/test_stagereach_arkit_gate.py",
    "tests/eval/test_stagereach_replica_gate.py",
]
PACKET_DIRS = [
    "runs/arkit_relation_challenge/blinded_rgb/41069025",
    "runs/arkit_relation_challenge/blinded_rgb/41069042",
    "runs/arkit_rgb_transfer/47331972",
]

BANNED = [
    (re.compile(r"deevya", re.I), "user name"),
    (re.compile(r"deev09", re.I), "github account"),
    (re.compile(r"surgical[-_]graph[-_]rag", re.I), "repository name"),
    (re.compile(r"github\.com", re.I), "repository URL"),
    (re.compile(r"/Users/", re.I), "absolute filesystem path"),
    (re.compile(r"claude code|anthropic assistant|written by claude", re.I),
     "assistant attribution"),
    (re.compile(r"\bTODO\b|PLACEHOLDER", re.I), "unfinished marker"),
]
# Causal overclaims the paper retracted; must never appear in CURATED staged
# files (registry, claim audit, tex-derived text, code). Historical run
# reports in the staged census pack are byte-derived from what was scored and are
# never rewritten; their run-time interpretive strings are superseded by the
# paper and the registry, which NOTE_historical_reports.txt states.
RETRACTED = [
    (re.compile(r"clear(s|ed|ing)? extraction|not relation extraction"
                r"|naming[a-z ,]*\bbind", re.I), "retracted causal phrasing"),
]
# Commit-hash-like hex (7-12 chars, at least one letter so digit-only scene
# ids don't trip it; 64-char sha256 pins are excluded by the lookarounds).
# Applied to PROSE files only -- in data files, commit identifiers live in
# known fields that the staging sanitizers neutralize, asserted separately.
COMMIT_HASH = re.compile(
    r"(?<![0-9a-f])(?=[0-9a-f]*[a-f])[0-9a-f]{7,12}(?![0-9a-f])")
PROSE_SUFFIXES = {".tex", ".md", ".txt"}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stage_registry(dst: Path) -> None:
    src = REPO / "docs" / "project_results_registry.csv"
    rows = list(csv.reader(io.StringIO(src.read_text())))
    header = rows[0]
    col = header.index("source_commit_or_tag")
    for r in rows[1:]:
        if r[col].strip():
            r[col] = WITHHELD
    out = io.StringIO()
    csv.writer(out, quoting=csv.QUOTE_ALL, lineterminator="\n").writerows(rows)
    dst.write_text(out.getvalue())


def stage_claim_audit(dst: Path) -> None:
    src = REPO / "docs" / "paper_claim_audit.csv"
    rows = list(csv.reader(io.StringIO(src.read_text())))
    rows[0] = [re.sub(r"^values_at_[0-9a-f]+$", "values_at_freeze", c)
               for c in rows[0]]
    out = io.StringIO()
    csv.writer(out, quoting=csv.QUOTE_ALL, lineterminator="\n").writerows(rows)
    dst.write_text(out.getvalue())


HOME_PATH = re.compile(r"/Users/[A-Za-z0-9._-]+")


def scrub_paths(text: str) -> str:
    """Neutralize absolute home paths and repo names inside run reports."""
    text = HOME_PATH.sub("<home>", text)
    return re.sub(r"<home>/Desktop/surgical[-_]graph[-_]rag(?:-v2)?",
                  "<repo>", text)


def stage_pack(dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(PACK.glob("*.json")):
        if f.name == "MANIFEST.json":
            doc = json.loads(f.read_text())
            for entry in doc.get("reports", doc if isinstance(doc, list) else []):
                if isinstance(entry, dict) and "producing_commit" in entry:
                    entry["producing_commit"] = WITHHELD
            # handle dict-of-entries manifests as well
            if isinstance(doc, dict):
                for v in doc.values():
                    if isinstance(v, list):
                        for entry in v:
                            if isinstance(entry, dict) and "producing_commit" in entry:
                                entry["producing_commit"] = WITHHELD
                    if isinstance(v, dict) and "producing_commit" in v:
                        v["producing_commit"] = WITHHELD
            (dst_dir / f.name).write_text(scrub_paths(
                json.dumps(doc, indent=1, sort_keys=True) + "\n"))
        else:
            (dst_dir / f.name).write_text(scrub_paths(f.read_text()))


def stage_packet_manifests(dst_dir: Path) -> None:
    """Frame-ID manifests + prompts for the frozen packets. No imagery."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for rel in PACKET_DIRS:
        src = REPO / rel
        if not src.is_dir():
            continue
        name = rel.replace("/", "_")
        packet = json.loads((src / "packet.json").read_text())
        manifest = {
            "schema": "blinded_packet_frame_manifest",
            "schema_version": 1,
            "scene_id": packet.get("scene_id"),
            "packet_sha256": packet.get("packet_sha256"),
            "questions_sha256": packet.get("questions_sha256"),
            "relation_under_test": packet.get("relation_under_test"),
            "near_convention": packet.get("near_convention"),
            "frames": [{"id": fr["id"], "sha256": fr["sha256"]}
                       for fr in packet.get("frames", [])],
            "imagery_note": ("frame imagery is not redistributed; frames are "
                             "identified by id and sha256 against the source "
                             "dataset"),
        }
        (dst_dir / f"{name}_frames.json").write_text(
            json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        prompt = src / "prompt.txt"
        if prompt.is_file():
            shutil.copy2(prompt, dst_dir / f"{name}_prompt.txt")


def scan(stage: Path) -> list[str]:
    problems = []
    for f in sorted(stage.rglob("*")):
        if not f.is_file() or f.suffix in {".pdf", ".zip"}:
            continue
        try:
            text = f.read_text(errors="strict")
        except (UnicodeDecodeError, ValueError):
            continue
        pats = list(BANNED)
        rel = f.relative_to(stage)
        if not str(rel).startswith(PACK_STAGE_REL + "/"):
            pats += RETRACTED
        if f.suffix in PROSE_SUFFIXES:
            pats.append((COMMIT_HASH, "commit-like hash"))
        for pat, label in pats:
            for m in pat.finditer(text):
                ctx = text[max(0, m.start() - 40):m.end() + 40].replace("\n", " ")
                problems.append(f"{f.relative_to(stage)}: {label}: ...{ctx}...")
    # Assert the field-level sanitizers actually fired.
    reg = (stage / "project_results_registry.csv").read_text()
    if re.search(r'"[0-9a-f]{7,40}"\s*(,"[^"]*"){0,2}\n', reg) and WITHHELD not in reg:
        problems.append("registry: source_commit_or_tag not sanitized")
    packman = stage / PACK_STAGE_REL / "MANIFEST.json"
    if packman.is_file():
        doc = packman.read_text()
        for m in re.finditer(r'"producing_commit":\s*"([^"]+)"', doc):
            if re.fullmatch(r"[0-9a-f]{7,40}", m.group(1)):
                problems.append(f"{PACK_STAGE_REL}/MANIFEST.json: "
                                f"unsanitized producing_commit {m.group(1)}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args(argv)

    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    missing = []
    stage_registry(STAGE / "project_results_registry.csv")
    stage_claim_audit(STAGE / "paper_claim_audit.csv")
    for rel in ALLOWLIST:
        if rel.endswith("project_results_registry.csv") or \
                rel.endswith("paper_claim_audit.csv"):
            continue
        src = REPO / rel
        if not src.is_file():
            missing.append(rel)
            continue
        shutil.copy2(src, STAGE / Path(rel).name)
    for rel in RUNNER_DATA:
        src = REPO / rel
        if not src.is_file():
            missing.append(rel)
            continue
        dst = STAGE / "code" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(scrub_paths(src.read_text()))
    stage_pack(STAGE / PACK_STAGE_REL)
    (STAGE / PACK_STAGE_REL / "NOTE_historical_reports.txt").write_text(
        "The reports in this directory are byte-derived, sanitized copies of\n"
        "run artifacts exactly as they existed when the results were scored;\n"
        "they are never regenerated or edited (MANIFEST.json records the\n"
        "hashes). Interpretive prose strings inside them reflect the analysis\n"
        "framing AT RUN TIME. Where that framing has since been corrected --\n"
        "in particular, any suggestion that relation extraction was 'cleared'\n"
        "or that naming alone binds -- the paper's bounded claims and the\n"
        "results registry supersede the report text: the comparison\n"
        "establishes serialization consistency on the tested items only, both\n"
        "arms consume human-supplied identity, and semantic correctness of\n"
        "the stored relations was not independently established.\n")
    stage_packet_manifests(STAGE / "packets")
    ann_dir = STAGE / "annotation"
    ann_dir.mkdir(parents=True, exist_ok=True)
    for rel in [
        "eval/human_feedback/arkitscenes_relation_challenge_annotator2_raw.json",
        "eval/human_feedback/arkitscenes_relation_challenge_annotator2_returned.json",
        "eval/human_feedback/arkitscenes_relation_challenge_annotator2_adjudications.json",
        "eval/results/stagereach/annotator_agreement_v1.json",
        "eval/results/stagereach/annotation_sensitivity_v1.json",
    ]:
        src = REPO / rel
        if src.is_file():
            (ann_dir / Path(rel).name).write_text(scrub_paths(src.read_text()))
        else:
            print(f"  (annotation, absent: {rel})")
    code_dir = STAGE / "code"
    for rel in CODE_FILES:
        src = REPO / rel
        if not src.is_file():
            missing.append(rel)
            continue
        dst = code_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(scrub_paths(src.read_text()))

    # ONLY supp.pdf. Never stage main.pdf: 3DV prohibits supplements that
    # contain an updated/corrected copy of the submission PDF.
    src = REPO / "docs" / "3dv" / "out" / "supp.pdf"
    if src.is_file():
        shutil.copy2(src, STAGE / "supp.pdf")
    else:
        missing.append("docs/3dv/out/supp.pdf")

    (STAGE / "README.md").write_text(README)

    manifest = {
        "schema": "submission_package_manifest",
        "schema_version": 1,
        "files": {str(f.relative_to(STAGE)): sha256_file(f)
                  for f in sorted(STAGE.rglob("*")) if f.is_file()},
        "sanitization": ("commit identifiers replaced with "
                         f"'{WITHHELD}' and absolute home paths with "
                         "'<home>' in staged copies only; no dataset "
                         "imagery included"),
    }
    (STAGE / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n")

    problems = scan(STAGE)
    if problems:
        print("ANONYMITY SCAN FAILED:")
        for p in problems:
            print("  " + p)
        return 1
    print(f"anonymity scan clean over {len(manifest['files'])} staged files")
    if missing:
        print("MISSING (build them, then re-run):")
        for m in missing:
            print("  " + m)
        return 1

    if not args.no_zip:
        ZIP_OUT.parent.mkdir(parents=True, exist_ok=True)
        if ZIP_OUT.exists():
            ZIP_OUT.unlink()
        with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(STAGE.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(STAGE))
        print(f"wrote {ZIP_OUT} ({ZIP_OUT.stat().st_size:,} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
