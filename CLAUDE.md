# Active Objective — Read This First

The historical rules below remain useful for comparability, but their
`graffiti_bathroom` baseline description is no longer the active development
target.

## Overnight amendment 2026-08-27/28, on owner authorization

Owner authorizes an overnight paper/evaluation pivot to StageReach3D
(2026-08-27/28). This does not reverse the product conclusion favoring direct
RGB or reopen graph tuning. The graph pipeline is retained as the system under
evaluation.

Scope of the amendment, for every agent working tonight:

- The 3DV 2027 submission (`docs/3dv/`, paper deadline 2026-08-28 23:59 AoE
  = 2026-08-29 06:59 CDT; supplementary 2026-09-02 11:00 PDT) is
  reframed around a reusable, schema-driven evaluator, **StageReach3D**
  (`eval/stagereach/`), validated by evaluator-masked artifact-level fault
  injection and demonstrated on the frozen ARKit relation challenge (per-arm
  traces) and the frozen Phase-8 Replica scorecards (schema/outcome transfer
  only).
- Frozen design: `docs/stagereach_schema_freeze.md`. No agent may change the
  schema without the integrator.
- `tools/paper_statistics.py` stays byte-for-byte untouched; StageReach3D
  reproduces its committed numbers independently (equivalence gates).
- No graph tuning, no new grounding variants, no new VLM runs, no new scenes.
  Direct multiview RGB remains the product answer path and is disclosed in the
  paper as one named model (claude-opus-5), one frozen pass.
- If a second annotator returns a review, it is stored as evaluation-key
  reproducibility evidence and future calibration data; it must not be used to
  retune the evaluated system, and no returned review arrived before the
  2026-08-28 submission.

Current development target:

- canonical repo: `~/Desktop/surgical-graph-rag-v2` (THE only working copy;
  `~/Desktop/surgical-graph-rag` is historical/reference-only)
- integration branch: `v2-calibration`
- source dataset for the active slice: ARKitScenes
- source perception backend: frozen Mask3D bundle
- required outcome: **direct multiview RGB as the product answer path**, with
  the 3D layer retained as inspectable evidence and explicitly NOT the answer
  engine
- authoritative document: `docs/direct_rgb_product_path.md`
- prior contract, now superseded: `docs/arkit_vertical_slice_72h.md`

## Objective changed 2026-08-19, on owner authorization

The previous objective was "an oracle-free mesh -> delivered entities -> graph
-> inspectable answer vertical slice". It is closed on measured results, not
abandoned. Owner: "The stop is justified. The current graph-centered product
path should end here."

What the three experiments established, over 10 scored items on the
owner-confirmed relation key:

| arm | correct | coverage | deployable |
|---|---:|---:|---|
| direct multiview RGB | 7/10 | 0.90 | yes |
| stored graph + human identity | 7/10 | 0.80 | **no** — identity oracle |
| grounded delivered graph | 2/10 | 0.20 | yes |
| delivered graph | 0/10 | 0.00 | yes |

Useful spatial information exists in the graph and is not deployably
reachable. Serialization added no measured loss (stored-edge replay matched
the geometry ceiling 12/12 under the same convention; semantic correctness
of the relation convention was not independently established); the binding
stage is identity, and the pinned OpenCLIP crop-based grounding bridge
failed all three predeclared gates.

The stop closes THAT bridge, not grounding research in general. Its purpose is
to stop endless variants being fitted to the same seventeen anchors.

Graph results are preserved as the measured negative comparison. Do not delete
them and do not quietly restart the graph answer path; reopening it needs a
persisted per-entity embedding at delivery time or a better instance-delivery
stage, and a new key.

Consolidated 2026-08-09: `codex/arkit-vertical-slice` was fast-forwarded
into `v2-calibration` (no merge, linear history). That branch is retained
only until its artifacts are verified, then retired. Do not commit to it.

Before proposing or editing anything, read the execution contract and inspect
the current branch. Its scope and definition of done take precedence over a
new experiment idea.

## Roles, set by the owner 2026-08-09

- **Claude — primary editor / integrator.** Implements the active
  experiment, owns `v2-calibration`, commits and pushes complete
  checkpoints, maintains this file, `AGENTS.md`, and the execution contract.
- **Codex — independent reviewer.** Checks that a proposed experiment
  addresses the real bottleneck; reviews the commit rather than a pasted
  summary; verifies metrics and the ceiling-versus-delivered distinction;
  names model/data/annotation/evaluation confounders; refuses scope
  expansion that does not move mesh -> answer. Edits ONLY when the owner
  assigns a bounded task, on `codex/<task>`, in its own worktree.

Never let two agents edit the same branch or directory concurrently.

## Handoff block — paste this when switching agents

    Canonical repo: ~/Desktop/surgical-graph-rag-v2
    Integration branch: v2-calibration
    Current commit: <SHA>
    Working tree: clean
    Active objective: <one outcome>
    Read first: CLAUDE.md + docs/arkit_vertical_slice_72h.md
    Your role: editor OR reviewer
    Authorized files/actions: <scope>
    Do not start: <excluded work>
    Validation command: .venv/bin/python3 tools/run_tests.py

The receiving agent replies with: branch, HEAD SHA, `git status`, the
objective in its own words, and the ONE task it will own. Incomplete work
goes on a pushed WIP branch — never handed over as an unexplained dirty
tree.

Active workflow rules:

1. Integration comes before another isolated experiment or protocol.
2. Do not tune P1/SAM, repair pooled selection, add dynamics, or expand the
   benchmark while the vertical slice is incomplete.
3. The deployable lane must not read annotations, human keys, oracle labels,
   oracle boxes, or oracle structural surfaces.
4. `end-to-end` means mesh to final answer. Use `proposal ceiling`,
   `delivered instances`, and `relation evaluation` for narrower results.
5. Development candidates are not frozen. Freeze only an accepted comparison
   baseline or a result being publicly claimed.
6. Human feedback must become relation training/calibration data after the
   slice runs; do not use it only to produce another scorecard.
7. When multiple assistants are involved, use isolated worktrees and
   file-bounded tasks. One integration branch owns the outcome.
8. Every handoff states: commit, dirty/clean state, files changed, commands
   run, measured result, remaining blocker, and any oracle dependency.
9. Do not silently describe the current AABB plan/elevation inspector as a 3D
   mesh inspector.
10. Do not replace the active objective without explicit owner authorization.

# Historical Project Overview
This repository implements a coarse spatial scene-graph pruning / retrieval pipeline over a real captured scene.

Current baseline:
- baseline_id: v1
- primary implementation: tiny_graph_demo.py
- primary scene: graffiti_bathroom
- exported artifacts:
  - scene_graph.json
  - expected_answers.json
  - evaluation_table.json
  - manifest.json

The current system supports:
- relation-aware retrieval
- zone-aware retrieval
- benchmarked query answering over a structured scene graph

# Current Baseline Facts
Unless explicitly changed, assume the current baseline is:

- 1 scene: graffiti_bathroom
- 12 labeled objects with zones, xyz positions, and attributes
- explicit typed relations such as:
  - LEFT_OF
  - RIGHT_OF
  - BELOW
  - ABOVE
  - IN_FRONT_OF
  - BEHIND
  - NEAR
  - ATTACHED_TO

Current benchmark summary for v1:
- n_queries: 10
- top1_accuracy: 1.0
- topk_recall: 1.0
- avg_false_positives_per_query: 0.0

Because the current baseline is already perfect on the existing benchmark, any claimed "improvement" must be interpreted carefully.
Improvement may mean:
- better robustness
- better generalization to new queries/scenes
- lower latency
- lower context cost
- cleaner architecture
- more faithful ranking behavior under harder cases

It does NOT automatically mean higher benchmark accuracy on the current 10-query set.

# Main Priorities
When working in this repo, prioritize in this order:
1. understand the current baseline flow
2. preserve benchmark comparability
3. make the smallest useful experimental change
4. keep experimental logic easy to isolate and revert
5. explain risks, confounders, and interpretation limits clearly

# Working Style
- Prefer minimal diffs over rewrites
- Reuse current data structures and evaluation outputs where possible
- Avoid framework-like abstraction unless clearly justified
- Keep prototype code local and legible
- Be skeptical of "improvements" that come only from changed benchmark semantics
- For non-trivial tasks, inspect relevant files and propose a plan before editing

# Repo-Specific Rules
- Do not claim the system improved unless the baseline and candidate are fairly comparable
- Preserve the current v1 path unless explicitly creating a new candidate path
- If changing scoring, pruning, ranking, or filtering logic, explain exactly what behavior should change
- If changing evaluation logic, explicitly state whether older results remain comparable
- If changing expected answers, benchmark rows, or artifact formats, call that out as a benchmark-definition change, not a model improvement
- Prefer experimental changes that are easy to A/B against v1

# Scene Graph Rules
The scene graph currently includes:
- object ids
- labels
- zones
- xyz positions
- object attributes
- typed directed relations
- optional edge weights on some relations

When modifying scene-graph logic:
- preserve semantic meaning of existing relations unless explicitly redefining them
- do not casually change relation directionality
- do not silently reinterpret zones or object labels
- if changing edge weights, explain why and what ranking behavior should shift
- if adding new objects or relations, explain how this affects current expected answers and benchmark comparability

# Scoring / Retrieval Rules
Current scorer facts from the manifest:
- min_score_default = 0.2
- relation_weights_on_edges = true
- spatial_xy_salience is active for:
  - BELOW
  - ABOVE
- lambda_xy = 0.38
- floor = 0.05

When changing retrieval or ranking behavior:
- identify whether the change affects candidate generation, pruning, ranking, or final scoring
- explain whether the score distribution should become narrower, sharper, or more permissive
- do not mix multiple scoring changes at once unless explicitly requested
- preserve a path for fair baseline comparison whenever possible

# Benchmark / Eval Rules
The current benchmark uses:
- expected_answers.json as the answer key
- evaluation_table.json / csv as the result output
- manifest.json as the baseline summary and scorer metadata

Always identify:
- baseline
- candidate
- scene set
- expected-answer source
- scoring rules
- threshold / k settings
- output format
- any changed benchmark semantics

Watch for confounders such as:
- changed expected answers
- changed scene graph contents
- changed relation weights
- changed scoring thresholds
- changed top-k settings
- changed evaluation logic
- changed query wording
- changed output interpretation

Do not report "improvement" if baseline and candidate are not run under equivalent settings.

# Debugging Rules
When debugging unexpected behavior, first classify the likely failure source:
- scene graph construction
- object metadata / zones / coordinates
- relation definitions or weights
- candidate generation
- pruning / filtering
- ranking / scoring
- evaluation logic
- benchmark artifact generation
- runtime/config mismatch

Prefer one-variable-at-a-time debugging.
Do not rewrite the pipeline before identifying the likely failure point.

# Implementation Rules
When implementing a new idea:
1. restate the hypothesis clearly
2. identify the minimum insertion point
3. preserve the v1 path when possible
4. keep the change localized
5. define how success will be measured before claiming success

If the current benchmark is already saturated, success may need to be measured by:
- harder benchmark cases
- additional scenes
- better calibration
- lower false positives under broader candidate sets
- latency / token / context reduction
- improved maintainability or debuggability

# Output Style I Usually Want
For most tasks, respond in this order:
1. relevant files / code paths
2. current baseline behavior
3. minimal plan
4. implementation or proposed change
5. validation / benchmark steps
6. risks / confounders
7. next experiment

# Preferred Engineering Style
- boring and testable over clever
- explicit over magical
- benchmark-grounded over intuition-grounded
- preserve comparability
- honest about uncertainty
- do not pretend a benchmark-definition change is a model improvement
