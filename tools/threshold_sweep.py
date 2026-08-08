"""Phase 8 E5 — contact-band threshold sensitivity sweep.

The structural relations hinge on centimeter-scale contact bands (floor
0.02/0.03, wall 0.02/0.02). demo/README.md documents that borderline objects
flip in/out with ~1 cm box-source differences — this tool makes that fragility
MEASURABLE per scene instead of anecdotal.

For each relation family it sweeps ONE config parameter at a time (CLAUDE.md
one-variable-at-a-time rule) over multipliers of the default, builds a
single-family bundle per grid point, and reports:

  - edge diff vs the x1.0 baseline (added / removed stored edges),
  - answer flips: the family's battery questions re-asked per variant
    (gained / lost cited UIDs) — the directly interpretable output,
  - fragile edges: edges that flip within +/-25% of the default band.

Interpretation limits (also stamped into the report):
  - fragility measures SENSITIVITY, not correctness — AABB-derived contact
    gaps carry systematic geometry error, so an edge flipping at 1.25x may be
    a true positive the default band just misses, or noise;
  - defaults are NEVER modified; every variant is a throwaway bundle;
  - numbers are per-scene and form a NEW Phase 8 track — not comparable to
    v1 or any phase scorecard.

Usage:
  python3 tools/threshold_sweep.py [room_dir scene_id] [--families f1 f2] [--out DIR]
Defaults to room_0. Output: runs/phase8_threshold_sweep/<scene_id>.json.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph.builder import ExtractorRun, build_graph
from graph.relations.attached_to import AttachedToConfig, AttachedToExtractor
from graph.relations.contacts_surface import ContactsSurfaceConfig, ContactsSurfaceExtractor
from graph.relations.on_entity_surface import OnEntitySurfaceConfig, OnEntitySurfaceExtractor
from graph.relations.on_surface import OnSurfaceConfig, OnSurfaceExtractor
from reasoner.base import CompletenessProfile, ExecutionContext
from reasoner.compiler_rules import RulesCompiler
from reasoner.executor import RulesExecutor
from reasoner.router import Router
from reasoner.verbalizer import StandardVerbalizer

DEFAULT_ROOM = Path.home() / "Desktop/datasets/replica/room_0"
DEFAULT_OUT_DIR = REPO_ROOT / "runs" / "phase8_threshold_sweep"

GRID: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5)
FRAGILE_MULTIPLIERS: tuple[float, ...] = (0.75, 1.25)

_SUPPORT_CLASSES = ("table", "desk", "shelf", "counter", "stool",
                    "bench", "sofa", "chair", "plant-stand")


@dataclass(frozen=True)
class FamilySpec:
    extractor_factory: type
    config_cls: type
    params: tuple[str, ...]
    questions: tuple[str, ...]


FAMILIES: dict[str, FamilySpec] = {
    "on_surface": FamilySpec(
        OnSurfaceExtractor, OnSurfaceConfig,
        ("contact_threshold_m", "penetration_tolerance_m"),
        ("what is on the floor?",),
    ),
    "contacts_surface": FamilySpec(
        ContactsSurfaceExtractor, ContactsSurfaceConfig,
        ("contact_threshold_m", "penetration_tolerance_m"),
        ("what is against the wall?",),
    ),
    "on_entity_surface": FamilySpec(
        OnEntitySurfaceExtractor, OnEntitySurfaceConfig,
        ("contact_threshold_m", "penetration_tolerance_m"),
        tuple(f"what is on the {c}?" for c in _SUPPORT_CLASSES),
    ),
    "attached_to": FamilySpec(
        AttachedToExtractor, AttachedToConfig,
        # wall-contact gate pair, then floor-rest disqualifier pair
        ("contact_threshold_m", "penetration_tolerance_m",
         "floor_contact_threshold_m", "floor_penetration_tolerance_m"),
        ("what is attached to the wall?",),
    ),
}


def variant_run(family: str, overrides: dict[str, float] | None = None) -> ExtractorRun:
    spec = FAMILIES[family]
    config = spec.config_cls(**(overrides or {}))
    return ExtractorRun(spec.extractor_factory(), config)


def _router() -> Router:
    return Router(compiler=RulesCompiler(), executor=RulesExecutor(),
                  verbalizer=StandardVerbalizer())


def _ctx() -> ExecutionContext:
    return ExecutionContext(completeness=CompletenessProfile(
        source="oracle", entity_recall_by_class={}, edge_recall_by_type={}))


def evaluate_variant(arts, run: ExtractorRun, questions: tuple[str, ...]):
    """Build a single-family bundle; return (edge_set, answers-by-question)."""
    bundle, _ = build_graph(arts, [run], density_policy="phase2_telemetry_only")
    edges = {(e.type, e.source.uid, e.target.uid) for e in bundle.edges}
    router, ctx = _router(), _ctx()
    answers = {q: sorted(router.answer(q, bundle, ctx).cited_uids) for q in questions}
    return edges, answers


def _edge_list(edges: set[tuple[str, str, str]]) -> list[list[str]]:
    return [list(t) for t in sorted(edges)]


def sweep_family(arts, family: str) -> dict:
    spec = FAMILIES[family]
    base_edges, base_answers = evaluate_variant(arts, variant_run(family), spec.questions)
    defaults = spec.config_cls()
    out: dict = {
        "baseline": {
            "edge_count": len(base_edges),
            "answers": base_answers,
        },
        "params": {},
        "fragile_edges": [],
    }
    fragile: set[tuple[str, str, str]] = set()
    for param in spec.params:
        base_value = getattr(defaults, param)
        points: dict = {}
        for mult in GRID:
            value = base_value * mult
            edges, answers = evaluate_variant(
                arts, variant_run(family, {param: value}), spec.questions)
            added = edges - base_edges
            removed = base_edges - edges
            flips = {}
            for q in spec.questions:
                gained = sorted(set(answers[q]) - set(base_answers[q]))
                lost = sorted(set(base_answers[q]) - set(answers[q]))
                if gained or lost:
                    flips[q] = {"gained": gained, "lost": lost}
            if mult == 1.0 and (added or removed):
                raise AssertionError(
                    f"x1.0 point diverged from default build for "
                    f"{family}.{param} — determinism violated")
            points[f"{mult}"] = {
                "value": value,
                "edges_added": _edge_list(added),
                "edges_removed": _edge_list(removed),
                "answer_flips": flips,
            }
            if mult in FRAGILE_MULTIPLIERS:
                fragile |= added | removed
        out["params"][param] = {"base_value": base_value, "points": points}
    out["fragile_edges"] = _edge_list(fragile)
    return out


def sweep_scene(arts, scene_id: str, families: list[str]) -> dict:
    return {
        "schema": "phase8_threshold_sweep",
        "schema_version": 1,
        "scene_id": scene_id,
        "defaults_unchanged": True,
        "grid_multipliers": list(GRID),
        "fragile_definition": (
            f"edge flips at a +/-25% band multiplier "
            f"({', '.join(str(m) for m in FRAGILE_MULTIPLIERS)})"
        ),
        "interpretation": (
            "Sensitivity, not correctness: AABB contact gaps carry systematic "
            "geometry error. One parameter varied at a time; single-family "
            "bundles; defaults never modified. New Phase 8 track — not "
            "comparable to v1 or phase scorecards."
        ),
        "families": {f: sweep_family(arts, f) for f in families},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("room_dir", nargs="?", type=Path, default=DEFAULT_ROOM)
    parser.add_argument("scene_id", nargs="?", default="replica_room_0")
    parser.add_argument("--families", nargs="+", choices=sorted(FAMILIES),
                        default=sorted(FAMILIES))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    if not (args.room_dir / "habitat" / "info_semantic.json").exists():
        print(f"Refusing: {args.room_dir}/habitat/info_semantic.json not found.")
        return 1

    from demo.replica_habitat_import import import_habitat_room
    arts = import_habitat_room(args.room_dir, args.scene_id)
    report = sweep_scene(arts, args.scene_id, list(args.families))

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{args.scene_id}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"THRESHOLD SWEEP: {args.scene_id}")
    for fam, data in report["families"].items():
        n_frag = len(data["fragile_edges"])
        print(f"  {fam:18} baseline_edges={data['baseline']['edge_count']:5}  "
              f"fragile(+/-25%)={n_frag}")
        for edge in data["fragile_edges"][:8]:
            print(f"     fragile: {edge[0]} {edge[1]} -> {edge[2]}")
        if n_frag > 8:
            print(f"     ... +{n_frag - 8} more")
    print(f"report -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
