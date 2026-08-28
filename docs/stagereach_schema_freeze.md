# StageReach3D schema freeze — 2026-08-28 (overnight, owner-authorized)

This document freezes the StageReach3D design for the overnight parallel
build. Agents implement against THIS document. Any change to it goes through
the integrator only. The approved plan of record is the owner-approved night
plan; this file is the machine-facing subset.

## 1. Stage vocabulary

Ordered universe of stages (a path uses an ordered subset):

    key_eligibility            # is the question scored at all (human key exists, not excluded)
    object_delivery            # referenced objects present in the delivered partition
    relation_applicability     # the relation vocabulary can express the question
    relation_correctness       # semantic truth of the computed relation vs independent ground truth
    serialization_consistency  # serialized edge agrees with recomputation under the same convention
    identity_injection         # human-supplied identity binding (oracle arms only)
    referent_grounding         # deployable language->entity binding
    answer_generation          # final answer / abstention

`relation_correctness` and `serialization_consistency` are DISTINCT stages.
On ARKit NEAR, `relation_correctness` is `unknown` (no independent semantic
annotation exists); `serialization_consistency` is measured
(`geometry_vs_stored_graph.agree`).

## 2. Statuses

    pass | fail | unknown | not_applicable | abstain | not_reached

- `unknown` is never counted as `pass`, never attributed as a failure, and
  never zeroes downstream survival.
- `not_applicable` is legal ONLY on stages the trace's path declares bypassed.
- `not_reached` is required downstream of a `fail` on a depended-on stage.

## 3. Scopes (mirror the registry exactly)

    deployable | identity_oracle | delivered | oracle_free_component_eval |
    proposal_ceiling | definition_change | bug_diagnostic

## 4. Paths (per-arm DAGs — there is NO shared mixed ladder)

Declared as data in `eval/stagereach/schema.py`:

| path_id | stages (ordered) | bypassed | allowed scopes |
|---|---|---|---|
| `graph_deployable_delivered` | key_eligibility, object_delivery, relation_applicability, relation_correctness, serialization_consistency, referent_grounding, answer_generation | — | deployable, delivered |
| `graph_deployable_grounded` | same as above | — | deployable, delivered |
| `graph_identity_oracle` | key_eligibility, object_delivery, relation_applicability, relation_correctness, serialization_consistency, identity_injection, answer_generation | referent_grounding | identity_oracle |
| `geometry_ceiling` | key_eligibility, object_delivery, relation_applicability, identity_injection, answer_generation | relation_correctness, serialization_consistency, referent_grounding | proposal_ceiling, identity_oracle |
| `direct_rgb` | key_eligibility, answer_generation | object_delivery, relation_applicability, relation_correctness, serialization_consistency, referent_grounding | deployable |

Stage attributes: `answer_path: bool`, `oracle_fed: bool`
(`key_eligibility` is `answer_path=False, oracle_fed=True` on every path;
`identity_injection` is `answer_path=True, oracle_fed=True`).

**Gating dependencies** are declared per stage per path (topological
predecessors that gate reachability). `serialization_consistency` does NOT
depend on `relation_correctness`. `relation_correctness` is a NON-GATING
audit stage on ARKit paths.

## 5. Survival & attribution semantics (amendment 3, verbatim)

Report reached, pass, fail, and unknown separately at every stage. Causal
survival is computed over declared gating dependencies only; non-gating audit
stages remain visible but do not reduce downstream reachability. Attribution
reports the first `fail` on a gating stage; `unknown` stages are reported as
unmeasured, never attributed.

## 6. Invariants (each is a test; violation raises)

1. unknown != pass anywhere in survival counting.
2. A `deployable`-scope trace may not pass-consume any `oracle_fed`
   `answer_path` stage (CLAUDE.md rule 3 as code).
3. No silent bypass: any reached status downstream of a `fail` on a
   depended-on stage is an error; `not_applicable` only on declared bypasses.
4. No pooled accuracy: metrics are keyed by `(expected_outcome, result)`;
   positives are never pooled with true-empties; no single "accuracy" is
   exposed by the API.
5. Non-exhaustive keys (e.g. Replica NEAR_SURFACE): the evaluator REFUSES to
   emit precision/recall/true-negative metrics.
6. `definition_change`-scope results cannot be pooled with frozen-track
   results (pooling raises).

## 7. Trace schema (`stagereach_trace`, schema_version 1)

```json
{
 "schema": "stagereach_trace", "schema_version": 1,
 "question_id": "...", "scene_id": "...", "arm": "...",
 "path_id": "...", "scope": "...",
 "expected_outcome": "answer|empty|defer",
 "stages": [{"stage": "...", "status": "...", "source": "..."}],
 "final_outcome": {"result": "correct|wrong|abstain|excluded",
                   "positive_expected": true},
 "raw_category": "verbatim source vocabulary label"
}
```

`source` is a provenance string naming the artifact/field the status was
derived from. Every emitted artifact carries `schema` + `schema_version` and
is written sorted-keys with `--check` byte-compare support.

## 8. Outcome mapping (total; totality is a test)

Target: `expected in {answer, empty, defer}` x `result in {correct, wrong,
abstain, excluded}`.

ARKit `{correct -> (answer,correct); wrong -> (answer,wrong); unanswered ->
(answer,abstain); excluded_no_human_answer -> (answer,excluded)}`.

router_qa 9-category vocabulary maps via the per-question record, NOT the
category alone. `miss` splits: incorrect/incomplete cited bindings ->
(answer,wrong); returned empty without deferring -> (answer,wrong); explicit
defer/abstain -> (answer,abstain). `false_answer` splits on the question's
`expected_outcome`. `true_answer -> (answer,correct)`, `true_empty ->
(empty,correct)`, `correct_defer -> (defer,correct)`. `raw_category` is
always preserved.

## 9. Numeric gates (a track's text enters the paper only if its gate passes)

- **G-ARKIT-ARMS**: per-arm ladders, independently derived from
  `eval/results/project_census_v1/arkit_relation_challenge_report.json`:
  delivered_graph 10->8->8->8->0; grounded_graph 10->8->8->8->3->2;
  stored_human_identity 10->7; direct_rgb 10->7.
- **G-ARKIT-LEGACY**: the mixed ladder 10->8->8->8->3->0 reproduced ONLY as a
  legacy-ledger compatibility check vs committed
  `eval/results/paper_statistics.json` (field-by-field).
- **G-REPLICA**: BOTH raw categories 4 true_answer + 27 true_empty + 22 miss
  + 3 false_answer = 56 AND the normalized matrix (answer,correct)=4,
  (answer,wrong)=20, (answer,abstain)=4, (empty,correct)=27, (empty,wrong)=1;
  per-scene n 13/14/16/13; all §6 guards hold. Reads ONLY the packed copies
  in `eval/results/project_census_v1/` (never `runs/`); every internal stage
  the packed scorecards cannot support is `unknown` — delivery, relation
  correctness, and serialization are never inferred from the final answer.
- **G-FAULT**: 24/24 controlled artifact-level injected faults localized
  while the evaluator was masked to the injected class (8 fault classes x 3
  relation types: NEAR, ON_ENTITY_SURFACE, ATTACHED_TO), plus zero failures
  on clean artifacts. Artifact chain: evaluation_key + entity_artifact ->
  relation_artifact -> serialized_graph -> grounded_candidates -> answer.
  Injections mutate exactly one artifact; the evaluator attributes via
  independent per-stage checkers and is never told the injected class.

## 10. File ownership (no agent crosses these lines)

- Agent B: `eval/stagereach/**`, `tools/stagereach_eval.py`,
  `tools/stagereach_numbers.py`, `docs/3dv/sec/generated_numbers.tex`,
  `eval/fixtures/stagereach/**`, `eval/results/stagereach/**`,
  `tests/eval/test_stagereach_*.py`.
- Agent C: `docs/3dv/main.tex`, `docs/3dv/sec/{0..6}_*.tex`, `docs/3dv/refs.bib`
  (via its generator), `tools/paper_figures.py`, `docs/figures/*.svg`,
  `docs/3dv/figures/*.pdf`. May `\input` `generated_numbers.tex`, never edit it.
- Agent D: `docs/3dv/supp.tex`, `docs/3dv/sec/supp_*.tex`,
  `tools/paper_supp_index.py`, `tests/tools/test_paper_supplement.py`.
- Integrator only: `CLAUDE.md`, this file, `tools/stagereach_review_sheet.py`,
  evidence-pack additions, `docs/paper_draft.md`,
  `docs/project_results_registry.csv`, `docs/paper_claim_audit.csv`,
  `tests/tools/test_paper_claim_audit.py` reconciliation, anything reading
  `runs/` (annotator frames, Fig3).
