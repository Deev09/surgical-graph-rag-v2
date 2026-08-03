"""Selective-prediction sweep for the rules reasoner (v2-calibration).

Runs the 56 human-verified Phase 8 questions through the real Router with
answer-level confidence turned on (reasoner/confidence.py), sweeps the
abstention threshold tau, and scores the result with eval/selective.py --
the SAME metrics code the VLM baseline uses, so the curves are comparable.

WHAT THIS IS FOR, AND THE TRAP IT IS BUILT TO AVOID
---------------------------------------------------
The cached rules scorecard emits no per-question confidence, so its
risk-coverage "curve" is a single point: AURC 0.7405 with
tie_policy_spread == E-AURC == 0.617521 exactly. Every bit of that AURC is
tie-break artifact. Beating it is therefore trivial and MEANINGLESS on its
own: eval/selective.py's default tie policy is pessimistic (errors ranked
first inside a tie group), so ANY score that merely breaks ties -- including
a random one -- scores better than the all-tied baseline. Reporting
"AURC 0.74 -> 0.23" without saying that is a lie by omission.

So this tool refuses to report a bare improvement and always computes four
controls on the identical items:

  RANDOM                    seeded uniform noise. No information at all.
                            This, not the tied baseline, is the honest
                            floor.
  RANDOM within outcome     noise, but every `empty` ranked above every
                            `bindings` above every `abstain`. Isolates how
                            much of the score is just "which outcome did
                            the executor produce".
  RANDOM within relation    noise, but grouped by relation family in
                            descending observed accuracy. Isolates the
                            relation-type confound.
  ORACLE                    correct-above-incorrect; equals AURC*.

A candidate score is only interesting if it lands outside the p05-p95 band
of the relevant control. Set --trials to change the control resolution.

RELATION-TYPE CONFOUND
----------------------
Directional margins are saturated while entity-surface margins are not, so a
cross-relation curve can be "which relation did this question use" wearing a
confidence costume. The per-relation section reports each family's accuracy
and, for the one family big enough to score on its own
(ON_ENTITY_SURFACE, 40 of 56), a full within-family curve with the relation
held constant.

Usage:
  .venv/bin/python3 tools/rules_selective_eval.py
  .venv/bin/python3 tools/rules_selective_eval.py --aggregation mean --trials 4000
  .venv/bin/python3 tools/rules_selective_eval.py --no-margins   # frozen default

Outputs: runs/rules_selective/{triples,selective,tau_sweep,summary}.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.router_qa import SUCCESS_CATEGORIES, score_questions
from eval.selective import evaluate, format_report, report_to_dict
from graph.builder import ExtractorRun, build_graph
from graph.relations.attached_to import AttachedToConfig, AttachedToExtractor
from graph.relations.contacts_surface import (
    ContactsSurfaceConfig, ContactsSurfaceExtractor,
)
from graph.relations.directional import DirectionalConfig, DirectionalExtractor
from graph.relations.on_entity_surface import (
    OnEntitySurfaceConfig, OnEntitySurfaceExtractor,
)
from graph.relations.on_surface import OnSurfaceConfig, OnSurfaceExtractor
from graph.relations.surface import (
    SurfaceProximityConfig, SurfaceProximityExtractor,
)
from reasoner.base import CompletenessProfile, ExecutionContext
from reasoner.compiler_rules import RulesCompiler
from reasoner.confidence import AGGREGATIONS
from reasoner.executor import RulesExecutor
from reasoner.router import Router
from reasoner.verbalizer import StandardVerbalizer

MANIFEST_PATH = REPO_ROOT / "eval" / "questions" / "phase8" / "scene_manifest.json"
DEFAULT_OUT_DIR = REPO_ROOT / "runs" / "rules_selective"
HUMAN_VERIFIED = "human_verified"

# The cached all-tied rules curve, for the honest side-by-side.
BASELINE = {
    "source": "runs/selective_v0/rules_phase8_human_verified.selective.json",
    "aurc": 0.7405393328208909,
    "e_aurc": 0.61752056388782,
    "tie_policy_spread": 0.61752056388782,
    "n_distinct_confidences": 2,
    "note": "no per-question confidence; spread == E-AURC exactly",
}

DEFAULT_TAUS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 0.999, 1.0)


def extractor_runs(emit_margins: bool) -> list[ExtractorRun]:
    """The battery-path extractor set (demo/question_battery.py::_runs), with
    `emit_margins` threaded through. Identical edge SET either way -- the flag
    only replaces the literal 1.0 confidence with the real margin score, and
    `hash_omit_if_default` keeps the emit_margins=False bundle_hash unchanged."""
    return [
        ExtractorRun(DirectionalExtractor(),
                     DirectionalConfig(mode="sparse", emit_margins=emit_margins)),
        ExtractorRun(SurfaceProximityExtractor(),
                     SurfaceProximityConfig(use_polygon_clip=True,
                                            exclude_room_scale_flat=True,
                                            emit_margins=emit_margins)),
        ExtractorRun(OnSurfaceExtractor(), OnSurfaceConfig(emit_margins=emit_margins)),
        ExtractorRun(ContactsSurfaceExtractor(),
                     ContactsSurfaceConfig(exclude_room_scale_flat=True,
                                           emit_margins=emit_margins)),
        ExtractorRun(OnEntitySurfaceExtractor(),
                     OnEntitySurfaceConfig(emit_margins=emit_margins)),
        ExtractorRun(AttachedToExtractor(), AttachedToConfig(emit_margins=emit_margins)),
    ]


def _ctx(rejections, tau: float) -> ExecutionContext:
    return ExecutionContext(
        completeness=CompletenessProfile(
            source="oracle", entity_recall_by_class={}, edge_recall_by_type={}),
        answer_tau=tau,
        rejections=tuple(rejections),
    )


def load_scenes(manifest_path: Path, emit_margins: bool) -> list[dict]:
    """Build every scene that has a human_verified answer key. Skips scenes
    whose Replica data is not on this machine (recorded, not silently)."""
    from demo.replica_habitat_import import import_habitat_room

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenes: list[dict] = []
    skipped: list[str] = []
    for entry in manifest["scenes"]:
        key_path = REPO_ROOT / entry["key_path"]
        room_dir = Path(entry["room_dir"])
        if not key_path.exists():
            skipped.append(f"{entry['scene_id']} (no human_verified key)")
            continue
        if not (room_dir / "habitat" / "info_semantic.json").exists():
            skipped.append(f"{entry['scene_id']} (no data at {room_dir})")
            continue
        key = json.loads(key_path.read_text(encoding="utf-8"))
        if key.get("answer_key_type") != HUMAN_VERIFIED:
            skipped.append(f"{entry['scene_id']} (key is {key.get('answer_key_type')!r})")
            continue
        arts = import_habitat_room(room_dir, entry["scene_id"])
        bundle, diag = build_graph(arts, extractor_runs(emit_margins),
                                   density_policy="phase2_telemetry_only")
        scenes.append({
            "scene_id": entry["scene_id"], "key": key,
            "bundle": bundle, "diagnostics": diag,
            "n_rejections_sampled": len(diag.rejection_samples),
            "n_rejections_total": sum(diag.rejections_per_type.values()),
        })
    return scenes, skipped


def collect_items(scenes: list[dict], aggregation: str, tau: float = 0.0) -> list[dict]:
    """One row per question: confidence, correctness, and the decomposition.

    Correctness is eval/router_qa.py's category in SUCCESS_CATEGORIES --
    the same definition the cached baseline used, so the two curves are
    comparable.
    """
    router = Router(compiler=RulesCompiler(), executor=RulesExecutor(),
                    verbalizer=StandardVerbalizer(),
                    confidence_aggregation=aggregation)
    items: list[dict] = []
    for scene in scenes:
        ctx = _ctx(scene["diagnostics"].rejection_samples, tau)
        bundle = scene["bundle"]
        questions = scene["key"]["questions"]
        card = score_questions(questions, bundle, router, ctx)
        by_id = {q["question_id"]: q for q in questions}
        for row in card["per_question"]:
            ans = router.answer(by_id[row["question_id"]]["question"], bundle, ctx)
            parts = dict(ans.confidence_parts)
            items.append({
                "id": f"{scene['scene_id']}:{row['question_id']}",
                "scene_id": scene["scene_id"],
                "relation": by_id[row["question_id"]].get("relation", "UNSPECIFIED"),
                "confidence": float(ans.confidence if ans.confidence is not None else 0.0),
                "correct": row["category"] in SUCCESS_CATEGORIES,
                # An outcome that is not a claim is a refusal, not a
                # low-confidence prediction. Same convention as the cached
                # baseline triples, so the two curves stay comparable.
                "abstained": row["actual_outcome"] not in ("bindings", "empty"),
                "category": row["category"],
                "expected_outcome": row["expected_outcome"],
                "actual_outcome": row["actual_outcome"],
                "n_cited": len(row["cited_uids"]),
                "confidence_parts": parts,
            })
    items.sort(key=lambda r: r["id"])
    return items


# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------

def _aurc(items: list[dict]) -> float:
    return evaluate(items).aurc


def random_control(
    items: list[dict], *, group_key=None, trials: int, seed: int,
) -> dict:
    """Expected AURC of a score carrying NO information beyond `group_key`.

    group_key(item) -> a numeric rank; larger sorts first. Within a group the
    ordering is uniform noise. This is the reference the candidate has to
    beat: the all-tied baseline is not, because a pessimistic tie policy
    charges it the worst case.
    """
    rng = random.Random(seed)
    vals: list[float] = []
    for _ in range(trials):
        shuffled = [
            {"id": it["id"],
             "confidence": (0.0 if group_key is None else float(group_key(it)))
                           + rng.random() * 1e-3,
             "correct": it["correct"], "abstained": it.get("abstained", False)}
            for it in items
        ]
        vals.append(_aurc(shuffled))
    vals.sort()
    return {
        "mean": statistics.fmean(vals),
        "stdev": statistics.pstdev(vals),
        "p05": vals[int(0.05 * len(vals))],
        "p95": vals[int(0.95 * len(vals))],
        "trials": trials,
    }


_OUTCOME_RANK = {"empty": 2.0, "bindings": 1.0, "abstain": 0.0,
                 "unknown": 0.0, "parser_failure": 0.0}


def relation_rank_map(items: list[dict]) -> dict[str, float]:
    """Rank relation families by their OBSERVED accuracy on this very set.
    This is an oracle control -- it cannot be deployed, and that is the
    point: it upper-bounds how much a pure relation-type signal could give."""
    acc: dict[str, list[bool]] = {}
    for it in items:
        acc.setdefault(it["relation"], []).append(it["correct"])
    return {k: sum(v) / len(v) for k, v in acc.items()}


def auroc(items: list[dict]) -> float | None:
    """P(confidence of a correct item > confidence of an incorrect one),
    ties counted half. 0.5 is chance; BELOW 0.5 means the score is
    anti-correlated with correctness, which a bare AURC can hide."""
    pos = [i["confidence"] for i in items if i["correct"]]
    neg = [i["confidence"] for i in items if not i["correct"]]
    if not pos or not neg:
        return None
    total = 0.0
    for p in pos:
        for q in neg:
            total += 1.0 if p > q else (0.5 if p == q else 0.0)
    return total / (len(pos) * len(neg))


# --------------------------------------------------------------------------
# tau sweep (end-to-end through the Router)
# --------------------------------------------------------------------------

def tau_sweep(scenes: list[dict], aggregation: str, taus, base_items: list[dict]) -> list[dict]:
    """Re-run every question through the Router at each tau and record what
    actually happens, then check it against the arithmetic the
    risk-coverage curve assumes.

    The consistency check is the point: eval/selective.py derives coverage
    and risk from the confidence column alone. If the deployed gate does not
    reproduce that, the curve describes a system nobody can ship.
    """
    correct = {i["id"]: i["correct"] for i in base_items}
    n = len(base_items)
    out: list[dict] = []
    for tau in taus:
        items = collect_items(scenes, aggregation, tau=tau)
        # Deployed coverage: how many answers the system still stands behind.
        # An item that was already a refusal at tau=0 was never covered.
        n_claims = sum(1 for i in items if i["actual_outcome"] in ("bindings", "empty"))
        # Curve coverage: what eval/selective.py's threshold_points assume.
        covered_ids = [
            i["id"] for i in base_items
            if not i["abstained"] and i["confidence"] >= tau
        ]
        n_cov = len(covered_ids)
        errs = sum(1 for k in covered_ids if not correct[k])
        counts: dict[str, int] = {}
        for i in items:
            counts[i["category"]] = counts.get(i["category"], 0) + 1
        out.append({
            "tau": tau,
            "n_gated_to_unknown": sum(
                1 for i in items if i["confidence_parts"].get("gated_by_tau")),
            "n_claims": n_claims,
            "coverage_observed": n_claims / n,
            "coverage_from_curve": n_cov / n,
            "selective_risk_from_curve": (errs / n_cov) if n_cov else None,
            "gate_matches_curve": n_claims == n_cov,
            "category_counts": dict(sorted(counts.items())),
        })
    return out


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def _band(value: float, control: dict) -> str:
    if value < control["p05"]:
        return "BETTER than control (outside p05)"
    if value > control["p95"]:
        return "WORSE than control (outside p95)"
    return "INSIDE the control band -> no signal beyond the control"


def subset_report(items: list[dict], title: str, *, trials: int, seed: int) -> dict:
    rep = evaluate(items)
    rnd = random_control(items, trials=trials, seed=seed)
    frac = (rep.tie_policy_spread / rep.e_aurc) if rep.e_aurc > 0 else float("nan")
    return {
        "title": title,
        "n": rep.n, "n_errors": rep.n_errors,
        "aurc": rep.aurc, "aurc_optimal": rep.aurc_optimal, "e_aurc": rep.e_aurc,
        "tie_policy_spread": rep.tie_policy_spread,
        "spread_over_e_aurc": frac,
        "n_distinct_confidences": rep.n_distinct_confidences,
        "max_tie_fraction": rep.max_tie_fraction,
        "coverage_at_risk": {
            k: {kk: vv for kk, vv in v.items()} for k, v in rep.coverage_at_risk.items()
        },
        "auroc": auroc(items),
        "control_random": rnd,
        "verdict_vs_random": _band(rep.aurc, rnd),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--aggregation", default="min", choices=list(AGGREGATIONS),
                    help="headline aggregation; all three are reported anyway")
    ap.add_argument("--trials", type=int, default=4000,
                    help="random-control permutations")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-margins", action="store_true",
                    help="build with emit_margins=False (frozen default): every "
                         "edge confidence is literally 1.0. Shows what the gate "
                         "does with no calibration underneath it.")
    args = ap.parse_args(argv)

    emit_margins = not args.no_margins
    scenes, skipped = load_scenes(args.manifest, emit_margins)
    if not scenes:
        print("no human_verified scenes available; nothing to score")
        for s in skipped:
            print(f"  [skipped] {s}")
        return 1

    headline_items = collect_items(scenes, args.aggregation)
    n = len(headline_items)

    print("=" * 78)
    print(f"RULES SELECTIVE EVAL  n={n}  aggregation={args.aggregation}  "
          f"emit_margins={emit_margins}")
    print("=" * 78)
    print(f"  scenes: {', '.join(s['scene_id'] for s in scenes)}")
    for s in skipped:
        print(f"  [skipped] {s}")
    rej_sampled = sum(s["n_rejections_sampled"] for s in scenes)
    rej_total = sum(s["n_rejections_total"] for s in scenes)
    print(f"  rejection samples retained: {rej_sampled} of {rej_total} actual "
          f"({rej_sampled / rej_total:.1%}) -- graph/relations/* caps each "
          f"extractor at 64, in iteration order, NOT by margin.")

    print()
    print("--- BASELINE (cached, no per-question confidence) ---")
    print(f"  AURC={BASELINE['aurc']:.6f}  E-AURC={BASELINE['e_aurc']:.6f}  "
          f"spread={BASELINE['tie_policy_spread']:.6f}  "
          f"distinct={BASELINE['n_distinct_confidences']}")
    print("  spread == E-AURC exactly: 100% of that AURC is tie-break artifact.")

    print()
    print("--- CANDIDATE, all three aggregations (n=%d) ---" % n)
    per_agg: dict[str, dict] = {}
    for agg in AGGREGATIONS:
        its = headline_items if agg == args.aggregation else collect_items(scenes, agg)
        r = subset_report(its, f"answer confidence [{agg}]",
                          trials=args.trials, seed=args.seed)
        per_agg[agg] = r
        print(f"  {agg:8} AURC={r['aurc']:.4f}  E-AURC={r['e_aurc']:.4f}  "
              f"spread={r['tie_policy_spread']:.4f}  "
              f"spread/E-AURC={r['spread_over_e_aurc']:.2f}  "
              f"distinct={r['n_distinct_confidences']}  AUROC={r['auroc']:.4f}")

    head = per_agg[args.aggregation]
    print()
    print("--- CONTROLS on the identical items (lower AURC = better) ---")
    controls = {
        "random": random_control(headline_items, trials=args.trials, seed=args.seed),
        "random_within_outcome": random_control(
            headline_items, group_key=lambda i: _OUTCOME_RANK.get(i["actual_outcome"], 0.0),
            trials=args.trials, seed=args.seed),
    }
    relacc = relation_rank_map(headline_items)
    controls["random_within_relation"] = random_control(
        headline_items, group_key=lambda i: relacc[i["relation"]],
        trials=args.trials, seed=args.seed)
    oracle = evaluate([
        {"id": i["id"], "confidence": 1.0 if i["correct"] else 0.0,
         "correct": i["correct"], "abstained": i["abstained"]}
        for i in headline_items]).aurc
    for name, c in controls.items():
        print(f"  {name:24} AURC={c['mean']:.4f} +-{c['stdev']:.4f}  "
              f"[p05={c['p05']:.4f}, p95={c['p95']:.4f}]")
    print(f"  {'oracle (= AURC*)':24} AURC={oracle:.4f}")
    print(f"  {'CANDIDATE ' + args.aggregation:24} AURC={head['aurc']:.4f}   "
          f"vs random: {_band(head['aurc'], controls['random'])}")
    print(f"  {'':24} vs random_within_outcome: "
          f"{_band(head['aurc'], controls['random_within_outcome'])}")

    print()
    print("--- SUBSETS ---")
    subsets = {
        "bindings_only": [i for i in headline_items if i["actual_outcome"] == "bindings"],
        "empty_only": [i for i in headline_items if i["actual_outcome"] == "empty"],
    }
    subset_reports: dict[str, dict] = {}
    for name, its in subsets.items():
        if len(its) < 2 or len({i["correct"] for i in its}) < 2:
            continue
        r = subset_report(its, name, trials=args.trials, seed=args.seed)
        subset_reports[name] = r
        print(f"  {name:16} n={r['n']:3} err={r['n_errors']:3} AURC={r['aurc']:.4f}  "
              f"AUROC={r['auroc']:.4f}  spread/E-AURC={r['spread_over_e_aurc']:.2f}  "
              f"-> {r['verdict_vs_random']}")

    print()
    print("--- RELATION-TYPE BREAKDOWN (confound check) ---")
    print("  relation             n  correct  acc     mean_conf  median_conf")
    by_rel: dict[str, list[dict]] = {}
    for i in headline_items:
        by_rel.setdefault(i["relation"], []).append(i)
    rel_rows: dict[str, dict] = {}
    for rel in sorted(by_rel):
        its = by_rel[rel]
        confs = [i["confidence"] for i in its]
        acc = sum(1 for i in its if i["correct"]) / len(its)
        rel_rows[rel] = {
            "n": len(its), "n_correct": sum(1 for i in its if i["correct"]),
            "accuracy": acc, "mean_confidence": statistics.fmean(confs),
            "median_confidence": statistics.median(confs),
        }
        print(f"  {rel:20} {len(its):3}  {rel_rows[rel]['n_correct']:6}  {acc:.3f}   "
              f"{statistics.fmean(confs):.4f}     {statistics.median(confs):.4f}")

    # Held-constant test: the one family large enough to score alone.
    biggest = max(by_rel, key=lambda k: len(by_rel[k]))
    held = by_rel[biggest]
    held_report = None
    if len({i["correct"] for i in held}) > 1:
        held_report = subset_report(held, f"{biggest} only", trials=args.trials,
                                    seed=args.seed)
        held_ctrl = random_control(
            held, group_key=lambda i: _OUTCOME_RANK.get(i["actual_outcome"], 0.0),
            trials=args.trials, seed=args.seed)
        held_report["control_random_within_outcome"] = held_ctrl
        held_report["verdict_vs_random_within_outcome"] = _band(
            held_report["aurc"], held_ctrl)
        print()
        print(f"  RELATION HELD CONSTANT ({biggest}, n={held_report['n']}):")
        print(f"    AURC={held_report['aurc']:.4f}  E-AURC={held_report['e_aurc']:.4f}  "
              f"spread={held_report['tie_policy_spread']:.4f}  "
              f"spread/E-AURC={held_report['spread_over_e_aurc']:.2f}")
        print(f"    vs random:                {held_report['verdict_vs_random']}")
        print(f"    vs random_within_outcome: "
              f"{held_report['verdict_vs_random_within_outcome']}")

    print()
    print("--- TAU SWEEP (end-to-end through the Router) ---")
    sweep = tau_sweep(scenes, args.aggregation, DEFAULT_TAUS, headline_items)
    print("    tau     gated  coverage  sel.risk  gate==curve  categories")
    for row in sweep:
        risk = ("%.4f" % row["selective_risk_from_curve"]
                if row["selective_risk_from_curve"] is not None else "   --  ")
        print(f"  {row['tau']:6.3f}  {row['n_gated_to_unknown']:5}  "
              f"{row['coverage_observed']:8.4f}  {risk}  "
              f"{str(row['gate_matches_curve']):>11}  {row['category_counts']}")
    all_consistent = all(r["gate_matches_curve"] for r in sweep)
    print(f"  deployed gate reproduces the risk-coverage curve at every tau: "
          f"{all_consistent}")

    print()
    print(format_report(evaluate(headline_items),
                        title=f"rules answer-confidence [{args.aggregation}]"))

    # ---------------- verdict ----------------
    inside_outcome_band = (
        controls["random_within_outcome"]["p05"]
        <= head["aurc"]
        <= controls["random_within_outcome"]["p95"]
    )
    spread_frac = head["spread_over_e_aurc"]
    verdict_lines = []
    if inside_outcome_band:
        verdict_lines.append(
            "NEGATIVE RESULT: the candidate AURC sits inside the p05-p95 band of a "
            "score that knows ONLY which outcome the executor produced (empty > "
            "bindings > abstain) and is otherwise random. The margin content adds "
            "nothing measurable.")
    if spread_frac == spread_frac and spread_frac >= 0.25:
        verdict_lines.append(
            f"INERT RANKING: tie_policy_spread is {spread_frac:.0%} of E-AURC. That "
            "fraction of the reported gain is tie-break policy, not confidence.")
    b = subset_reports.get("bindings_only")
    if b is not None and b["auroc"] is not None and b["auroc"] < 0.5:
        verdict_lines.append(
            f"ANTI-CORRELATED ON NON-EMPTY ANSWERS: AUROC {b['auroc']:.4f} < 0.5 on "
            "the bindings subset. Aggregating with min() over cited edges is a "
            "cardinality detector, and on this set cardinality runs the wrong way "
            "(correct answers cite many edges, misses cite one or two).")
    if not verdict_lines:
        verdict_lines.append(
            "The candidate beat every control band. Check the per-relation table "
            "before calling it calibration rather than relation identity.")
    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    for line in verdict_lines:
        print("  * " + line)

    args.out.mkdir(parents=True, exist_ok=True)
    triples = {
        "schema": "selective_triples", "schema_version": 1,
        "source": "tools/rules_selective_eval.py",
        "answer_key": "eval/questions/phase8/*_qa.json (human_verified)",
        "runner_name": "rules+answer_confidence",
        "correctness_field": "router_qa category in SUCCESS_CATEGORIES",
        "confidence_source": f"reasoner/confidence.py aggregation={args.aggregation}",
        "emit_margins": emit_margins,
        "items": headline_items,
    }
    (args.out / "triples.json").write_text(
        json.dumps(triples, indent=2) + "\n", encoding="utf-8")
    (args.out / "selective.json").write_text(
        json.dumps(report_to_dict(evaluate(headline_items)), indent=2) + "\n",
        encoding="utf-8")
    (args.out / "tau_sweep.json").write_text(
        json.dumps({"schema": "rules_tau_sweep", "schema_version": 1,
                    "aggregation": args.aggregation,
                    "gate_reproduces_curve": all_consistent,
                    "rows": sweep}, indent=2) + "\n", encoding="utf-8")
    (args.out / "summary.json").write_text(
        json.dumps({
            "schema": "rules_selective_summary", "schema_version": 1,
            "n": n, "aggregation": args.aggregation,
            "emit_margins": emit_margins,
            "trials": args.trials, "seed": args.seed,
            "rejection_samples_retained": rej_sampled,
            "rejections_actual": rej_total,
            "baseline": BASELINE,
            "per_aggregation": per_agg,
            "controls": controls,
            "oracle_aurc": oracle,
            "subsets": subset_reports,
            "per_relation": rel_rows,
            "relation_held_constant": held_report,
            "verdict": verdict_lines,
        }, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nreports -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
