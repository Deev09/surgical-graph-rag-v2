# Active Objective — Read This First

The historical rules below remain useful for comparability, but their
`graffiti_bathroom` baseline description is no longer the active development
target.

Current development target:

- branch: `codex/arkit-vertical-slice`
- base commit: `be668c3`
- source dataset for the active slice: ARKitScenes
- source perception backend: frozen Mask3D bundle
- required outcome: an oracle-free mesh -> delivered entities -> graph ->
  inspectable answer vertical slice
- authoritative execution contract:
  `docs/arkit_vertical_slice_72h.md`

Before proposing or editing anything, read the execution contract and inspect
the current branch. Its scope and definition of done take precedence over a
new experiment idea.

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
