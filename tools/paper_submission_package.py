#!/usr/bin/env python3
"""Assemble the anonymous 3DV supplementary package in a clean staging dir.

Populates runs/submission_package/ (gitignored) from:
  1. a fixed allowlist of tracked evidence files,
  2. freshly built docs/3dv/out/main.pdf and out/supp.pdf,
  3. sanitized frame-ID manifests for the frozen blinded packets
     (frame ids + sha256 pins + prompt text; NO ARKitScenes imagery),
then writes a sha256 manifest of everything staged, runs an anonymity scan
over every staged TEXT file, and zips the staging directory.

Sanitization applied to STAGED COPIES ONLY (repo files are never touched):
  - registry CSV: source_commit_or_tag values -> "withheld-for-review"
    (the repository is public, so a commit hash is searchable and would
    de-anonymize the submission; camera-ready restores them)
  - claim-audit CSV: the values_at_<hash> column is renamed values_at_freeze
  - evidence-pack MANIFEST: producing_commit values -> "withheld-for-review"

The scan covers staged text files. The two PDFs are NOT text-scanned here;
they must be read page-by-page in the final manual sweep.

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
OPTIONAL = [
    "eval/results/stagereach/arkit_stagereach_v1.json",
    "eval/results/stagereach/replica_stagereach_v1.json",
    "eval/results/stagereach/annotator_agreement_v1.json",
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
    packman = stage / "evidence_pack" / "MANIFEST.json"
    if packman.is_file():
        doc = packman.read_text()
        for m in re.finditer(r'"producing_commit":\s*"([^"]+)"', doc):
            if re.fullmatch(r"[0-9a-f]{7,40}", m.group(1)):
                problems.append(f"evidence_pack/MANIFEST.json: unsanitized "
                                f"producing_commit {m.group(1)}")
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
    for rel in OPTIONAL:
        src = REPO / rel
        if src.is_file():
            shutil.copy2(src, STAGE / Path(rel).name)
        else:
            print(f"  (optional, absent: {rel})")
    stage_pack(STAGE / "evidence_pack")
    stage_packet_manifests(STAGE / "packets")

    for pdf in ("main.pdf", "supp.pdf"):
        src = REPO / "docs" / "3dv" / "out" / pdf
        if src.is_file():
            shutil.copy2(src, STAGE / pdf)
        else:
            missing.append(f"docs/3dv/out/{pdf}")

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
