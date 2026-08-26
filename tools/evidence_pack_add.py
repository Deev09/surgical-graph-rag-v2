#!/usr/bin/env python3
"""Copy and sanitize an existing run report into the census evidence pack.

The pack exists because most run artifacts live under the gitignored `runs/`
tree, so a published number can otherwise only be checked by someone who has
that tree. This tool copies a report in, applies the pack's documented
sanitization policy, and records provenance in MANIFEST.json.

It NEVER regenerates anything. It reads a file that already exists and writes a
sanitized copy. If the source is missing it fails rather than inventing one.

    tools/evidence_pack_add.py <source> <pack_name> "<why included>"

Idempotent: re-adding a file that is already packed with identical content is a
no-op, so it is safe to re-run.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACK = REPO / "eval" / "results" / "project_census_v1"
MANIFEST = PACK / "MANIFEST.json"

# The pack's stated policy, kept verbatim so manifest entries stay comparable.
DROP_KEYS = ["embedding", "embeddings", "image", "images", "masks",
             "point_indices", "rows_raw", "vertex_ids", "vertices"]
MAX_LIST = 4000
POLICY = (f"keys dropped if present: {DROP_KEYS}; "
          f"lists over {MAX_LIST} elements elided")


def sanitize(node):
    """Drop bulk payload keys and elide oversized lists, recursively."""
    if isinstance(node, dict):
        return {k: sanitize(v) for k, v in node.items() if k not in DROP_KEYS}
    if isinstance(node, list):
        if len(node) > MAX_LIST:
            return [f"<elided: {len(node)} elements exceeds pack limit {MAX_LIST}>"]
        return [sanitize(v) for v in node]
    return node


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def add(source: Path, pack_name: str, why: str) -> int:
    if not source.is_file():
        print(f"source does not exist: {source}", file=sys.stderr)
        return 1
    raw = source.read_bytes()
    doc = json.loads(raw)
    clean = sanitize(doc)
    # sort_keys matches the existing pack files so diffs stay readable
    out = json.dumps(clean, indent=1, sort_keys=True).encode() + b"\n"

    target = PACK / pack_name
    if target.is_file() and target.read_bytes() == out:
        print(f"  unchanged  {pack_name}")
        return 0
    target.write_bytes(out)

    manifest = json.loads(MANIFEST.read_text())
    entry = {
        "original_bytes": len(raw),
        "original_path": str(source.relative_to(REPO)),
        "original_sha256": sha256(raw),
        "pack_file": pack_name,
        "producing_commit": "untracked (runs/ is gitignored)",
        "sanitization": POLICY,
        "sanitized_sha256": sha256(out),
        "why_included": why,
    }
    files = [f for f in manifest["files"] if f["pack_file"] != pack_name]
    files.append(entry)
    manifest["files"] = sorted(files, key=lambda f: f["pack_file"])
    MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"  packed     {pack_name}  ({len(raw)} B -> {len(out)} B)")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__)
        return 2
    return add(Path(argv[1]) if Path(argv[1]).is_absolute() else REPO / argv[1],
               argv[2], argv[3])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
