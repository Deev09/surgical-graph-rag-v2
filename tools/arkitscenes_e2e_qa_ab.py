"""End-to-end GRAPH-CONSISTENCY QA: does naming make the graph askable?

  python3 tools/arkitscenes_e2e_qa_ab.py

Matched-instance label accuracy improved sharply with real capture-RGB crops
(`docs/arkitscenes_rgb_label_results.md`). That is a COMPONENT result. This
asks the integration question: given the same geometry and the same relation
graph, does the Router answer more questions when the entities are named from
photographs instead of point splats?

SCOPE OF THE CLAIM. This is graph-consistency QA, NOT human-ground-truth
spatial QA. Citations are scored against the NEAR-neighbour set computed on
the SAME graph both arms use, so the metric measures anchor resolution and
faithful neighbour return -- nothing more. If the graph is wrong about what
is near what (missing instance, overmerged plane, misplaced geometry) both
arms are scored against the same wrong neighbourhood and this cannot detect
it. A human-ground-truth answer key, built from annotations independently of
the graph, does not exist yet and is what the repair arm will need.

Only the label stage differs. Both arms load already-finalized
`EntityArtifacts` produced by `tools/arkitscenes_label_image_ab.py`, build the
graph with the same frozen proximity extractor, and answer the SAME questions.

QUESTION SET -- deliberately not label-derived. `tools/arkit_vertical_slice.py`
generates "what is near <first node's display label>?", which is a DIFFERENT
question per arm and therefore uncomparable. Here the questions come from the
scene's annotation classes, so both arms are asked the same thing and an arm
that names nothing simply fails to answer. Choosing questions from the oracle
is an evaluation-side decision, stated rather than hidden; the deployable lane
never sees it.

Reported separately, because they fail differently:
  * outcome mix, including abstentions -- what the Router did
  * UID citation P/R/F1 -- did it cite the right entities
  * semantic citation accuracy -- of what it cited, how much was the class asked for
  * answer changes -- how many questions moved between arms, and which way

No tuning: nothing here reads or adjusts a threshold, vocabulary or crop
parameter.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extractors.serde import load_entity_artifacts
from graph.builder import ExtractorRun, build_graph
from graph.relations.proximity import ProximityConfig, ProximityExtractor
from reasoner.base import CompletenessProfile, ExecutionContext
from reasoner.compiler_rules import RulesCompiler
from reasoner.executor import RulesExecutor
from reasoner.router import Router
from reasoner.verbalizer import StandardVerbalizer

NEAR_THRESHOLD_M = 1.0          # same constant the vertical slice uses
ARMS = ("splat", "rgb_tight")
SCENES = {
    "41069021": REPO_ROOT / "runs/arkit_label_image_ab",
    "41069025": REPO_ROOT / "runs/arkit_label_image_ab_41069025",
    "41069042": REPO_ROOT / "runs/arkit_label_image_ab_41069042",
}
OUT = REPO_ROOT / "runs/arkit_e2e_qa_ab"


def build_graph_for(entity_dir: Path):
    arts = load_entity_artifacts(entity_dir)
    graph, diagnostics = build_graph(
        arts,
        [ExtractorRun(
            ProximityExtractor(),
            ProximityConfig(mode="sparse", sparse_version=2,
                            sparse_near_threshold=NEAR_THRESHOLD_M),
        )],
        density_policy="phase2_telemetry_only",
    )
    return arts, graph, diagnostics


def oracle_class_by_uid(arm_dir: Path) -> dict[str, str]:
    """uid -> annotation class, from the committed label evaluation.

    Geometry is identical across arms, so the matching is too; either arm's
    report gives the same map.
    """
    rep = json.loads((arm_dir / "label_eval.json").read_text())
    return {r["object_uid"]: r["oracle_class_normalized"]
            for r in rep["metrics"]["per_match"]}


def run_arm(entity_dir: Path, questions: list[tuple[str, str]],
            truth: dict[str, str]) -> dict:
    _arts, graph, _diag = build_graph_for(entity_dir)
    router = Router(compiler=RulesCompiler(), executor=RulesExecutor(),
                    verbalizer=StandardVerbalizer())
    ctx = ExecutionContext(completeness=CompletenessProfile(
        source="unknown", entity_recall_by_class={}, edge_recall_by_type={}))

    rows, outcomes = [], {}
    tp = fp = fn = 0
    sem_hit = sem_total = 0
    for cls, question in questions:
        a = router.answer(question, graph, ctx)
        outcomes[a.outcome] = outcomes.get(a.outcome, 0) + 1
        cited = set(a.cited_uids)
        # "what is near the X?" should cite entities NEAR an X, not the Xs
        # themselves. Ground truth is therefore the NEAR-neighbours, in the
        # SHARED graph, of every entity the annotation calls class X. The
        # graph is identical across arms, so any difference here is anchor
        # resolution -- i.e. naming -- which is the only stage that varies.
        anchors = {u for u, c in truth.items() if c == cls}
        expected = set()
        for e in graph.edges:
            if e.source.uid in anchors:
                expected.add(e.target.uid)
            if e.target.uid in anchors:
                expected.add(e.source.uid)
        expected -= anchors
        tp += len(cited & expected)
        fp += len(cited - expected)
        fn += len(expected - cited)
        # Semantic citation: of the cited entities we have an annotation
        # class for, how many are genuinely NEAR-neighbours of an X.
        for u in cited:
            if u in truth:
                sem_total += 1
                sem_hit += int(u in expected)
        rows.append({"cls": cls, "question": question, "outcome": a.outcome,
                     "text": a.text, "cited": sorted(cited),
                     "expected": sorted(expected)})
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else 0.0
    return {
        "n_questions": len(questions),
        "n_entities": len(graph.nodes),
        "n_edges": len(graph.edges),
        "outcomes": outcomes,
        "n_abstained": sum(v for k, v in outcomes.items()
                           if k in ("abstain", "empty", "unknown",
                                    "parser_failure")),
        "uid_citation": {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4) if prec is not None else None,
            "recall": round(rec, 4) if rec is not None else None,
            "f1": round(f1, 4),
        },
        "semantic_citation": {
            "n_cited_matched": sem_total,
            "n_correct_class": sem_hit,
            "accuracy": round(sem_hit / sem_total, 4) if sem_total else None,
        },
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    summary = {}
    for vid, root in SCENES.items():
        arm_dirs = {a: root / a for a in ARMS}
        if not all((d / "label_eval.json").is_file() for d in arm_dirs.values()):
            print(f"{vid}: SKIP (run tools/arkitscenes_label_image_ab.py first)")
            continue
        truth = oracle_class_by_uid(arm_dirs["splat"])
        classes = sorted(set(truth.values()))
        questions = [(c, f"what is near the {c}?") for c in classes]

        print(f"\n=== {vid}   {len(questions)} questions "
              f"({', '.join(classes)})")
        per_arm = {}
        for arm in ARMS:
            per_arm[arm] = run_arm(arm_dirs[arm] / "entities", questions, truth)
            r = per_arm[arm]
            u, s = r["uid_citation"], r["semantic_citation"]
            print(f"  {arm:10s} entities={r['n_entities']:3d} "
                  f"edges={r['n_edges']:4d}  "
                  f"abstained={r['n_abstained']}/{r['n_questions']}  "
                  f"uid P/R/F1="
                  f"{(u['precision'] if u['precision'] is not None else 0):.2f}/"
                  f"{(u['recall'] if u['recall'] is not None else 0):.2f}/"
                  f"{u['f1']:.2f}  "
                  f"semantic={s['accuracy'] if s['accuracy'] is not None else '—'}")
        # answer changes, reported as its own quantity
        changed_outcome = changed_cites = 0
        for a, b in zip(per_arm["splat"]["rows"], per_arm["rgb_tight"]["rows"]):
            changed_outcome += int(a["outcome"] != b["outcome"])
            changed_cites += int(a["cited"] != b["cited"])
        print(f"  answer changes: outcome {changed_outcome}/{len(questions)}, "
              f"citations {changed_cites}/{len(questions)}")
        summary[vid] = {"classes": classes, "arms": per_arm,
                        "changed_outcome": changed_outcome,
                        "changed_citations": changed_cites}

    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(f"\n-> {args.out/'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
