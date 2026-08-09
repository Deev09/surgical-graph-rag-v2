# ARKit vertical slice — 72-hour execution contract

Status: ACTIVE DEVELOPMENT

Branch: `codex/arkit-vertical-slice`

Base: `be668c3`

Owner-facing outcome: one unseen-compatible mesh-to-answer path, not another
proposal-only result.

## Checkpoint 0 — completed

The first integration wave is complete on development scene `41069021`:

- The saved Colab dense partition was discovered to use `min_score=0.4`,
  despite the ARKit contract and proposal ceiling using `0.2`. It delivers
  5/18 entities at IoU 0.50 versus the 7/18 proposal ceiling (71.4%
  retention), losing one table and the TV monitor.
- A zero-GPU local re-resolution at the already-declared `min_score=0.2`
  produces 34 delivered instances and retains the complete 7/18 ceiling.
  This is recorded as an operating-point artifact with the original `0.4`
  provenance preserved; it is not a model improvement.
- The oracle-free vertical path now runs on that artifact: 34 anonymous
  `EntityArtifacts`, 139 AABB-surface `NEAR` edges, one deterministic query,
  serialized entities/graph/diagnostics, and a self-contained 2D evidence
  inspector. It explicitly withholds labels, directions, structural
  relations, support, attachment, hierarchy, and conversation.
- Full verification passes: 89/89 test files.

Development outputs are ignored run artifacts under
`runs/arkit_vertical_slice*`; tracked code and tests reproduce them.

## Checkpoint 1 — completed in the second integration wave

- The same one-command path can now attach oracle-free top-3 OpenCLIP label
  hypotheses over a fixed 41-class indoor vocabulary. On `41069021`, 6/34
  predictions clear the declared raw-score display threshold; 28/34 remain
  visibly anonymous. These are uncalibrated hypotheses, not accuracy claims.
- Geometry-only horizontal patch evidence now runs on the actual canonical
  mesh and dense instance assignment. It processed 1,008,964 vertices and
  1,901,874 faces in about 1.3 seconds, producing 93 geometry-qualified
  candidates across 34 entities. A patch is not yet a support relation:
  target-relative contact and containment remain the missing decision stage.
- The local output contains 34 entities, 139 `NEAR` edges, learned-label
  evidence, horizontal-patch evidence, a deterministic answer, and the
  self-contained inspector. No annotation was read.
- The two sealed transfer meshes (`41069025`, `41069042`) are hash/count pinned
  and staged together. Their notebook runs use the declared `min_score=0.2`;
  evaluation is mechanically locked until both distinct bundles pass the pair
  integrity gate.
- Full verification after integration passes: 92/92 test files.

At Checkpoint 1 the remaining path was deliberately narrow: run the two sealed
GPU bundles, implement target-to-patch resting evidence, then run the identical
slice and stage-wise evaluation on both scenes. Checkpoint 2 records that
execution; it did not open another proposal or benchmark-design project.

## Checkpoint 2 — sealed transfer executed

Both sealed GPU bundles passed the all-or-none integrity gate and the identical
oracle-free Lane A path finalized both scenes before evaluation opened:

- `41069025`: 35 delivered instances, 151 `NEAR` edges, and 9/20 annotated
  entities recovered at vertex IoU 0.50. Delivery retains 100% of the scene's
  Mask3D proposal ceiling.
- `41069042`: 23 delivered instances, 100 `NEAR` edges, and 5/6 annotated
  entities recovered at vertex IoU 0.50. Delivery again retains 100% of the
  proposal ceiling.
- Noise remains substantial: the non-exhaustive annotation diagnostic records
  62.9% and 78.3% zero-overlap delivered instances. This is not interpreted as
  conventional false-positive precision because ARKit annotations are not
  exhaustive.

Learned labels do not transfer reliably. On geometry-matched instances,
top-1/top-3 results are 0/7 and 0/7 on development, 1/9 and 4/9 on `41069025`,
and 0/5 and 0/5 on `41069042`. The fixed point-splat OpenCLIP stage is therefore
an inspectable baseline, not a usable semantic layer; no threshold or
vocabulary was tuned after evaluation.

Target-to-patch resting evidence is integrated but not promoted to graph
truth. It finds one provisional candidate in each sealed scene from 1,225 and
529 evaluated target-owner pairs. The geometry-only floor census is `unknown`
in both scenes, so the manifests preserve that uncertainty and emit no
`ON_ENTITY_SURFACE` edge. Human relation truth is still required to calibrate
and evaluate this stage.

The measured transfer conclusion is now specific: geometry-native instance
delivery transfers across the two sealed ARKit rooms; the present learned
semantic labels do not; support evidence runs but remains uncalibrated; and no
end-to-end spatial-QA generalization claim is available without independent
relation/question keys for the two sealed scenes.

Full verification after this integration passes: 95/95 test files.

## Checkpoint 3 — first sealed-scene human feedback

Owner visual review is recorded separately at
`eval/human_feedback/arkitscenes_sealed_visual_review_2026-08-09.json`; it is
not a QA key and changes no metric or Lane A artifact.

The review decomposes `41069025` more sharply than the aggregate label score:
the sofa is delivered as `obj_13` but mislabeled projector, and two delivered
segments (`obj_9`, `obj_23`) are visually sofa cushions mislabeled rug. The
geometry relation stage admits `obj_9` on the sofa; `obj_23` has a compatible
−1 cm gap but 0.454 footprint overlap, just below the provisional 0.50 gate.
This is a calibration example, not authorization to lower the constant from
one reviewed scene. The whole-plane `obj_8` labeled counter is instead an
instance overmerge, so label repair cannot fix it. Human review also confirms
one rug and one trash can, contradicting the learned label cardinalities.

The supplied `41069042` views show a couch-facing kitchen and curtains beside
the window/pathway. Those visible categories are absent from its six ARKit
annotation boxes, and `curtain` is absent from the current label vocabulary.
Thus this scene cannot support conventional label precision from its box set;
human scene inventory and model accuracy must remain separately reported.

## Definition of done

The slice is done when one command can consume an ARKitScenes canonical mesh
and its frozen Mask3D output and produce all of the following without reading
the ARKitScenes annotation file:

1. delivered object instances as `EntityArtifacts`;
2. a `SceneGraphBundle` using only relation families whose required evidence
   is actually available;
3. inspectable entity and edge evidence in a self-contained HTML artifact;
4. at least one deterministic object-relative query, with an explicit
   `answer`, `unknown`, or `abstain` result;
5. a machine-readable run manifest identifying every unavailable capability
   (labels, wall surfaces, floor surfaces, or query types) rather than silently
   substituting oracle data.

Evaluation is a separate lane. The evaluator may read annotations only after
the deployable outputs above have been finalized, and it must report proposal
ceiling, delivered-instance recovery, and answer/relation metrics separately.

## Scope for this slice

Included:

- ARKitScenes development scene `41069021`;
- the frozen Mask3D bundle already present on disk;
- dense-instance delivery measurement;
- Mask3D-to-`EntityArtifacts` conversion;
- entity-only relations that do not require missing architectural surfaces;
- dataset-neutral relation inspection;
- an explicit handoff for the two sealed Mask3D GPU runs.

Excluded until the vertical slice runs:

- P1/Mask3D pooling or selector repair;
- SAM/render parameter tuning;
- wall attachment;
- dynamic tracking;
- multi-room hierarchy;
- unconstrained conversation;
- paper/release work;
- new benchmark semantics.

## Work lanes

### Lane A — deployable path

This lane may read the canonical mesh, Mask3D sidecar, model outputs, and
declared runtime configuration. It must not read annotation JSON, human keys,
oracle labels, oracle object boxes, or oracle structural surfaces.

### Lane B — evaluation

This lane may read annotations only after Lane A artifacts are finalized. It
must never alter Lane A outputs. Oracle-derived labels are evaluation labels,
not deployable predictions.

### Lane C — improvement

After the slice runs, human feedback becomes pair-level training/calibration
data for relation evidence models. It is no longer used only as a scorecard.
Candidate models are evaluated leave-one-scene-out before a generalization
claim.

## 72-hour schedule

### 0–8 hours: vertical integration

- Measure the delivered Mask3D partition on `41069021`.
- Convert dense instance ids to anonymous, geometry-bearing `EntityArtifacts`.
- Build an entity-only graph without invented structural surfaces.
- Allow the inspector to consume serialized non-Replica bundles.
- Emit one deterministic dev-scene artifact and a failure manifest.

Exit condition: the path runs locally without annotation access. If it does
not, the exact failed seam becomes the only active blocker.

### 8–24 hours: semantic and query minimum

- Attach top-k learned label hypotheses through a global declared vocabulary;
  low-confidence entities remain anonymous.
- Add a Router-compatible adapter for a small, schema-constrained query set;
  the deterministic rules compiler remains the fallback.
- Evaluate three object-relative questions and record unsupported question
  types honestly.

Exit condition: labels and language cannot invent a geometric edge.

### 24–48 hours: surface-backed support

- Extract horizontal candidate patches from each potential supporter's actual
  instance points, not its whole AABB.
- Store continuous evidence: vertical gap, overlap, patch orientation, support
  area, relative scale, and closest-point statistics.
- Compare this candidate against frozen AABB support logic using existing
  human keys; do not change those keys.

Exit condition: report per-scene support precision/recall and leave failures
inspectable. A development-only gain is not called transfer.

### 48–72 hours: transfer and package

- Finalize both sealed Mask3D bundles before evaluating either.
- Run the identical Lane A path on `41069025` and `41069042`.
- Evaluate delivered entities, labels, relations, and final answers by stage.
- Produce one command, one manifest, one inspector artifact per scene, and one
  compact scorecard.

Exit condition: state exactly which capabilities transfer and which do not.

## Collaboration rules

1. This document is the only active priority list for the 72-hour slice.
2. One integration branch owns the outcome. Other assistants use isolated
   worktrees/branches and receive file-bounded tasks.
3. No two assistants edit the same file concurrently.
4. Workers report: files changed, commands run, measured result, blocker, and
   assumptions. They do not redefine scope or gates.
5. Integrate at least twice daily. Run focused tests on each task and the full
   suite at integration points.
6. Development candidates are not frozen. Freeze only (a) the accepted
   comparison baseline and (b) a result being publicly claimed.
7. Negative results close a mechanism only when the replacement path or the
   next integration action is stated in the same handoff.
8. `end-to-end` means mesh to final answer. Proposal generation, entity
   evaluation, and relation evaluation must use their precise names.
9. `3D inspector` means the source mesh/point cloud is rendered. AABB plan and
   elevation projections are labeled as 2D evidence views.
10. The project owner is asked only for external actions or semantic judgments
    that code and existing artifacts cannot supply: GPU execution, unavailable
    data, human relation truth, publication, and release.

## Daily owner update template

```text
Outcome completed:
Measured result:
What is now runnable:
What remains blocked:
Oracle/evaluation-only dependencies:
Next 4-hour task:
Owner action needed (or "none"):
```

## Owner workflow across Codex and Claude

Use one assistant as the integration owner for the entire 72-hour window.
The other assistant is a reviewer or receives a file-bounded task on a
different branch; it does not independently choose the next experiment.

Start every new chat with this exact handoff:

```text
Read AGENTS.md (or CLAUDE.md) and docs/arkit_vertical_slice_72h.md first.
Active branch: codex/arkit-vertical-slice.
Current commit: <paste git rev-parse --short HEAD>.
Do not change the active objective, benchmark keys, or frozen baselines.
First report the dirty state and the one remaining seam you will own.
End with: files changed, commands/tests, measured result, blocker, oracle use.
```

Execution cadence:

1. Keep at most three parallel work items: one deployable seam, one evaluator,
   and one external/GPU handoff. Give each non-overlapping files and a binary
   exit condition.
2. Integrate every four hours. Review the diff, run focused tests, then the
   whole suite. A worker's uncommitted tree is never the project state.
3. After each integration, push one commit and update the checkpoint above.
   The next worker starts from that commit, not from prose copied between
   chats.
4. Background jobs must be deterministic and restartable, write to a unique
   `runs/<task>/<scene>` directory, log their command/config/hash, and write a
   success marker only after integrity checks. Do not leave two jobs writing
   the same bundle.
5. The owner handles only the two actions automation cannot safely replace:
   keeping the external GPU runtime alive and supplying semantic ground truth.
6. Freeze only after an integrated artifact is accepted for comparison or
   publication. Never freeze a draft, a proposed constant, or an unconnected
   module.

The integration owner may reject any task whose output does not move the
definition of done within four hours. Such a task goes to `future_work.md`;
it does not remain half-active.

## Stop conditions

Stop and report instead of expanding scope when:

- an implementation reads annotations in Lane A;
- the delivered Mask3D output loses most of the proposal ceiling;
- a new requirement would delay the vertical slice without changing its user
  outcome;
- a task requires retuning against a sealed scene;
- the same blocker remains after two independent implementation attempts.
