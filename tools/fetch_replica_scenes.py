"""Phase 8 E1 — fetch + pin the expanded Replica scene corpus.

Phase 8 evaluation needs more raw scenes than the two on disk (room_0,
apartment_0). This tool fetches three files per Phase 8 scene and pins them
in a lock file, reusing the P2.01 verifier's init_lock/verify unchanged:
`habitat/info_semantic.json` (boxes/labels/surfaces), `habitat/
mesh_semantic.ply` (instance mesh -> A/B variant B), and `mesh.ply` (raw
mesh, no object attribution -> the C-variant segmentation input). See
wanted_members() for sizes.

Scene choice (diversity, see docs/phase8 plan):
  - room_1, room_2       same family as the only ground-truth scene (room_0)
                         -> near-distribution generalization
  - office_0             different environment type (desk/shelf-dense)
  - frl_apartment_0      different capture campaign, multi-room layout
                         -> stresses importer nearest-floor wall orientation

Subcommands:
  --download   stream the release tar parts and extract just the wanted
               members. The Replica v1.0 archive is a 34 GB gzip stream split
               into 17 x 2 GB GitHub release parts; BSD `tar -q` stops after
               the first match of EACH pattern, so one pass extracts all four
               files. NOTE: room_1/room_2 sit near the END of the alphabetical
               archive, so this pass streams most of the 34 GB once (disk cost
               < 2 MB). Run it in the background.
  --init       pin sha256 + sizes of the fetched files into
               tools/replica_scenes.lock.json (checked in; data gitignored).
  (default)    verify on-disk files against the lock. Exit 0 on full match.

An importer crash on a new scene is a Phase 8 *finding*, not a fetch failure;
record it and (temporarily) exclude the scene from the lock.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.verify_replica_inputs import init_lock, verify

PHASE8_SCENES: tuple[str, ...] = ("room_1", "room_2", "office_0", "frl_apartment_0")

DATA_ROOT = Path.home() / "Desktop/datasets/replica"
LOCK_PATH = REPO_ROOT / "tools" / "replica_scenes.lock.json"

BASE_URL = (
    "https://github.com/facebookresearch/Replica-Dataset/releases/download/"
    "v1.0/replica_v1_0.tar.gz"
)
N_PARTS = 17  # partaa .. partaq


def part_suffixes(n_parts: int = N_PARTS) -> tuple[str, ...]:
    """GitHub release part suffixes: 'aa', 'ab', ... (split -a2 order)."""
    if not 1 <= n_parts <= 26 * 26:
        raise ValueError(f"n_parts out of range: {n_parts}")
    out = []
    for i in range(n_parts):
        out.append("a" + chr(ord("a") + i) if i < 26 else chr(ord("a") + i // 26) + chr(ord("a") + i % 26))
    return tuple(out)


def wanted_members(scenes: tuple[str, ...] = PHASE8_SCENES) -> tuple[str, ...]:
    """Three pinned files per scene:
      habitat/info_semantic.json   boxes + labels + surfaces (~100-400 KB)
      habitat/mesh_semantic.ply    instance mesh, per-face object_id (~30-80 MB)
                                   -> A/B variant B (mesh-derived boxes)
      mesh.ply                     raw visual mesh, NO object attribution
                                   (~25-80 MB) -> C variants (segmentation
                                   front-end input)
    """
    out: list[str] = []
    for scene in scenes:
        out.append(f"{scene}/habitat/info_semantic.json")
        out.append(f"{scene}/habitat/mesh_semantic.ply")
        out.append(f"{scene}/mesh.ply")
    return tuple(out)


def download_command(
    scenes: tuple[str, ...] = PHASE8_SCENES,
    data_root: Path = DATA_ROOT,
    n_parts: int = N_PARTS,
) -> str:
    """One streaming pass: curl all parts | gunzip | tar -xq <members...>.

    `tar -q` (--fast-read) exits once every pattern has matched; curl/gunzip
    then die on SIGPIPE, which is the intended early stop.
    """
    urls = " ".join(f'"{BASE_URL}.part{s}"' for s in part_suffixes(n_parts))
    members = " ".join(f"'{m}'" for m in wanted_members(scenes))
    return (
        f"curl -sL {urls} | gunzip -c | "
        f"tar -xq -f - -C '{data_root}' {members}"
    )


def download(
    scenes: tuple[str, ...] = PHASE8_SCENES,
    data_root: Path = DATA_ROOT,
    n_parts: int = N_PARTS,
) -> int:
    data_root.mkdir(parents=True, exist_ok=True)
    cmd = download_command(scenes, data_root, n_parts)
    print(f"[--download] streaming up to {n_parts} x 2 GB parts for "
          f"{len(scenes)} member(s); this can take a while.")
    print(f"[--download] {cmd}")
    proc = subprocess.run(["/bin/sh", "-c", cmd])
    missing = [m for m in wanted_members(scenes) if not (data_root / m).is_file()]
    if missing:
        print("[--download] FAILED: members not extracted:")
        for m in missing:
            print("   -", m)
        return 1
    print("[--download] all members extracted:")
    for m in wanted_members(scenes):
        print(f"   {m}  size={(data_root / m).stat().st_size}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--download", action="store_true",
                        help="Stream the release parts and extract the Phase 8 members.")
    parser.add_argument("--init", action="store_true",
                        help="Pin sha256 + sizes of the fetched files into the lock file.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT,
                        help=f"Dataset root (default: {DATA_ROOT}).")
    parser.add_argument("--lock", type=Path, default=LOCK_PATH,
                        help=f"Lock-file path (default: {LOCK_PATH.relative_to(REPO_ROOT)}).")
    args = parser.parse_args(argv)

    if args.download:
        return download(data_root=args.data_root)
    if args.init:
        return init_lock(args.data_root, args.lock, wanted_members())
    return verify(args.data_root, args.lock)


if __name__ == "__main__":
    sys.exit(main())
