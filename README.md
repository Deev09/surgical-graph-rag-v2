# surgical-graph-rag

> A modular, queryable spatial-graph reasoner over real captured 3D scenes —
> typed relations, honest outcomes, and measured failure isolation under
> imperfect 3D instance extraction.

Given a captured indoor scene, the system builds a typed spatial scene graph
(support, wall contact/proximity, attachment, directional relations) and
answers natural-language structural questions ("what is on the table?",
"what is against the wall?") through a compile → execute → verbalize Router
that can answer, return a grounded empty, abstain, or say **unknown** — it
never fabricates certainty the graph cannot support.

What makes it unusual is the evaluation discipline: every input variant is
hash-pinned, every pipeline stage is isolated so failures attribute to exactly
one stage, experiments run under predeclared gates that are allowed to fail,
and negative results are committed as first-class documentation.

---

## v2 — does any of it transfer? (branch `v2-calibration`)

v1 established the methodology on Replica. **v2 asks whether the system
survives a dataset it was never tuned for**, and answers it with three
predeclared experiments on ARKitScenes — real handheld iPad captures instead
of synthetic meshes.

**Four of the five v2 headline results are negative.** That is the point of
the branch, not an apology for it. Each negative eliminates a named suspect
under gates fixed before the code existed, and the frozen Replica headline
(4 / 27 / 22 / 3, six bundle hashes) is **byte-identical across all 18
commits** — verifiable with `git diff main...v2-calibration`.

| # | result | verdict |
|---|---|---|
| 1 | **Oracle-free proposal selection** — ranks C1-P1 proposals with no ground truth, recovering **100% of the oracle-selection ceiling at k=50–100** on all three Replica scenes | **positive** |
| 2 | **Frame + scale audit** (6 Replica scenes) — found `frame="world"` was false on the live path, worth **26.9% of room_1's directional edges**, and that two importers disagreed on whether the dev scene was loadable at all | **correctness finding** |
| 3 | **Calibrated abstention from geometric margins** — AURC 0.74 → 0.23 looks like a win; random controls show it lands *inside* the noise band of a permutation that knows only the outcome type | **refuted** |
| 4 | **F1: fusion evidence denominator** on ARKitScenes | **refuted** |
| 5 | **R1: render splat density** on ARKitScenes | **refuted** |

### What v2 establishes

Running the frozen C1-P1 pipeline end-to-end on ARKitScenes yields an entity
ceiling of **1/18** (Replica room_2: 25/53). Three experiments then eliminated
the chain by measurement:

```
fusion evidence   → F1,     refuted   (made the bank worse: 12→9 @IoU0.10)
lifting dilation  → R1 arm C, null    (+21pts fill, Δceiling = 0)
render density    → R1 arm A, null    (matched Replica on fill AND mask count; Δceiling = 0)
mask coverage     → residual, newly isolated, untested
```

Arm A is the sharp one. It brought the render **above** Replica on both fill
(74.6% vs 72.8%) and mask density (16.4 vs 15.5 masks/view), SAM produced 43%
more masks — and the ceiling did not move by a single entity. What did *not*
move with it is occupied-pixel mask coverage: 28.9% → 30.2%, against Replica's
46.5%. **That quantity is the residual**, and it is a segmenter-behaviour
property, not a rasterisation one.

The honest reading, committed in advance in the protocol's decision table:
**render-and-lift does not transfer off synthetic meshes on the frozen C1-P1
configuration.** C1-P1's result is a Replica result until something outside
that configuration changes it — a limitation of the mechanism, established
across three experiments, not an implementation defect.

Protocols and verdicts:
[`arkitscenes_fusion_evidence_protocol.md`](docs/arkitscenes_fusion_evidence_protocol.md) ·
[`arkitscenes_render_density_protocol.md`](docs/arkitscenes_render_density_protocol.md) ·
[`selective_prediction_negative.md`](docs/selective_prediction_negative.md) ·
[`selector_v0_results.md`](docs/selector_v0_results.md) ·
[`frame_and_scale_audit.md`](docs/frame_and_scale_audit.md) ·
[`frame_decision.md`](docs/frame_decision.md)

### What actually generalized

| ported unchanged | why it worked |
|---|---|
| the `ReconstructionAdapter` seam | had exactly one implementation for a year; now holds two |
| the splat renderer and `lift_mask` | its `id_buffer` holds **vertex indices from this repo's own renderer**, not Replica `object_id`s — so 2D→3D lifting is dataset-agnostic |
| the gravity estimator | agrees with Replica metadata within **0.33°** on 6 scenes, and lands within **0.065–0.261°** of +z on ARKitScenes first try |
| the selector, ablation table, and ranking helpers | imported, not reimplemented — a test asserts the ablation table is the *same object*, so the two datasets cannot drift to different scoring |

**Did not generalize:** the relation thresholds. Object scale varies **4.33×**
across six Replica scenes alone (storey height only 1.12×), and
`sparse_min_delta=0.5` is 0.34 object-diagonals on room_1 but **1.49** on
office_0 — the axis-dominance gate is wider than a typical object there.

### Isolation discipline in v2

* **Annotations have exactly one door.** `adapters/arkitscenes.py` cannot
  parse JSON at all (AST-asserted), a runtime audit hook proves `reconstruct()`
  opens no annotation file, and a repo-wide test fails if any file outside a
  five-entry allow-list even *references* the annotation suffix.
* **The oracle-free selector is structurally oracle-free** — numpy-only
  imports, no I/O, no path argument, checked by an AST scan *and* a
  `sys.addaudithook` around a live scoring call.
* **Transfer scenes stay sealed.** ARKitScenes `41069025` and `41069042` have
  SAM masks on disk but have **never been fused, evaluated, or inspected**
  under any condition. Every number above is the dev scene.
* Two silent artifact-collision bugs were caught by these gates before they
  could corrupt a result — see the R1 verdict.

---

## What it is NOT (read this before citing numbers)

- It is **not** an end-to-end NeRF/3DGS system. The only real reconstruction
  adapter is Replica/oracle data.
- The learned-segmentation path (C1) starts from a raw `mesh.ply` but injects
  **oracle labels and structural surfaces** for controlled evaluation.
  Learned semantics (C2.0) is a measured **isolation result only** —
  zero-shot labels on C1's matched instances, closed to further
  optimization — and fully-raw operation (C3) is not implemented.
- The current headline is that the system **exposes and measures its own
  failures** — not that it solves raw-scene QA. The completed
  experimental arc (tag `paper-results-v1.0`) traces where failure
  MOVES: perception → relation semantics → the dataset's own annotation
  geometry.

**v2 additions to this list:**

- ARKitScenes numbers are **one dev scene, 18 annotated entities**. The two
  transfer scenes are sealed and unrun. Nothing here is a generalization
  claim; it is a set of eliminations on a single capture.
- The **oracle-free selector makes the pipeline runnable without an answer
  key. It does not make it accurate** — entity recall is unchanged (47% dev,
  29% / 26% transfer on Replica). Those are two different claims and the
  results doc keeps them apart.
- Its v1 default (`connectivity` dropped) was chosen using an ablation
  measured **on the transfer scenes**, so v1 numbers there are no longer a
  clean held-out measurement. `COMPONENTS_V0` reproduces the frozen
  configuration.
- **No dataset is redistributed.** Replica and ARKitScenes must be obtained
  from their own sources under their own licences (ARKitScenes is Apple
  non-commercial). Derived canonical meshes are written next to the data,
  outside this repo.

## The input ladder (stage isolation)

Each variant changes exactly one upstream stage, so adjacent comparisons
attribute differences to that stage (`docs/mesh_pipeline_contract.md`):

| variant | boxes | labels | status |
|---|---|---|---|
| **A** | `info_semantic.json` oracle | oracle | frozen baseline |
| **B** | derived from `mesh_semantic.ply` | oracle | frame parity with A frozen |
| **C1** | learned segmenter on raw `mesh.ply` | oracle via exact vertex correspondence | **measured** (Mask3D reference @ MIN_SCORE=0.2; Segment3D pilot failed its predeclared gate — see below) |
| **C2** | learned (= C1 frozen) | learned (CLIP zero-shot on matched instances) | **measured, evaluation-only** — labels are not the bottleneck; C2 optimization stopped (`docs/c2_matched_labels_protocol.md`) |
| **C3** | learned | learned, mesh-derived surfaces | fully raw path not implemented; C3.0-S and C3.0-SR surface-source isolation both CLOSED as negative results (input-contract, then acceptance-constant collapse on real mesh roughness — `docs/c3_0_sr_mesh_surfaces_protocol.md`) |

## Measured status (final at `paper-results-v1.0`, 2026-08-02)

**Human-verified baseline (Phase 8).** Four Replica scenes
(room_0/room_1/room_2/office_0) have human-reviewed answer keys; the scorecard
(`runs/phase8_scorecard/`) reports against *reality*, not against the system's
own drafts: 56 questions → 4 fully-correct answers, 27 correct empties, 22
misses, 3 false answers. The misses are dominated by known representational
limits (whole-object AABBs cannot model sofa/chair seat surfaces; the 2 cm
wall-contact band is stricter than human "against the wall"; cabinet/nightstand
are missing from the support-class allowlist). A key the system fails is a
successful review — the keys record what is physically true.

**C1 (raw-mesh instances, oracle labels).** Mask3D backend, four scenes,
frozen operating point: entity recall@IoU0.5 0.25–0.38, answer recall vs B
0.39–0.51. Failure attribution: Mask3D's selection stage is near-optimal and
its ceiling is proposal coverage (~32% of oracle entities have a viable raw
mask); Segment3D raises the proposal ceiling (30/53 vs 20/53 on room_2) but
wastes 13 viable masks in composition and failed 4/5 predeclared gate criteria
— so the pilot stopped after one scene, per protocol
(`docs/c1_closeout.md`, `docs/c1_m2_protocol.md`).

**C2.0 (learned labels, isolation only).** Zero-shot CLIP on matched-
instance point-splats: support-owner labels were 9/10 on room_2 but only
2/7 on the later office_0 transfer; overall top-1 spans 0.25–0.57. One
shelf-label error erased room_2's two support answers, while office_0's
delivered support was already zero under C1. Semantic-citation fidelity
(uid-correct answers that also carry the canonical label) spans 0.31–0.62.
Labels are not the current binding constraint — C1 proposal coverage is —
but they are not robustly solved; C2 optimization remains stopped
(`docs/c2_matched_labels_protocol.md`).

**The causal chain (findings 10–12, tag `paper-results-v1.0`).**
(1) C1-P1 — the first gate-passing performance experiment: SAM 2.1 over
40 deterministic renders of the raw mesh, lifted through vertex-id
buffers and fused by cross-view co-membership, fixes the proposal
bottleneck (entity viability 20/53 → 33/53 dev; +6/+5 on both transfer
scenes; evaluation-only). (2) C1-P2.0 — with proposals fixed, an
oracle-guided ceiling shows the bottleneck moved to relation semantics
(31/53 entities at precision 1.00 lift recall only 0.245 → 0.265); the
predeclared rule stopped the composer before a parameter existed.
(3) semantics-v2 — a separately labeled benchmark-definition track:
support becomes representable on the dev scene (5/20 → 16/20 @ P 0.94)
but is miscalibrated across scenes; attachment is unrecoverable from
Replica's annotation boxes (11/14 keyed fixtures lie BEHIND the
annotated wall planes); relation-specific gates stopped the track where
aggregate gates would have passed. Full story:
`docs/results_narrative.md`; paper draft: `docs/paper_draft.md`.

**Committed negative results.** Segment3D pilot; three selection-repair
rules; query-scoped raw-proposal expansion
(`docs/query_scoped_expansion_prototype.md`); the uncertainty-preserving
provisional pool (`docs/uncertainty_policy_prototype.md`); the
mesh-plane surface-estimator family (three-act closure incl. a
read-only measurement that replaced a third premature freeze); the
semantics-v2 track. Each is committed with its predeclared protocol and
verdict.

## Quickstart

```bash
git clone https://github.com/Deev09/surgical-graph-rag-v2.git
cd surgical-graph-rag-v2
pip install -r requirements.txt   # numpy + Pillow for the current pipeline

# Canonical test command (84 script-style test files, each in its own process;
# dataset-guarded tests self-skip without the Replica / ARKitScenes data)
python3 tools/run_tests.py

# With the Replica dataset on disk (see docs/reproduction.md):
python3 tools/fetch_replica_scenes.py                 # hash-check pinned inputs
python3 demo/question_battery.py /path/to/replica/room_0 replica_room_0
python3 tools/scene_scorecard.py                      # human-verified headline
python3 tools/mvp_demo.py                             # deterministic A/B/C1/C2 report
python3 tools/mvp_viewer.py                           # interactive 3D evidence viewer
#   -> open runs/mvp_v1/viewer.html (self-contained, offline)
python3 tools/mvp_captioned_demo.py                   # self-running captioned walkthrough
#   -> open runs/mvp_v1/captioned_demo.html (presentation-only derivative)
```

### v2 — ARKitScenes (needs the dataset; see `DATA.md` upstream)

```bash
# 1. geometry only — no annotations are read anywhere in this chain
python3 tools/arkitscenes_render.py --scene 41069021 --tar

# 2. SAM 2.1 runs in Colab (notebooks/c1p1_sam2_colab.ipynb, unedited except
#    SCENE=). Upload the tar; ids.npz is deliberately withheld from the GPU.

# 3. back locally
python3 tools/arkitscenes_fuse.py          --scene 41069021
python3 tools/arkitscenes_eval.py          --scene 41069021 --bank   # G5 + ceiling
python3 tools/arkitscenes_selector_eval.py --scene 41069021          # ranked AR@k

# the two negative experiments, reproducible from the same masks:
python3 tools/arkitscenes_fuse.py --scene 41069021 --evidence-denominator masked
python3 tools/arkitscenes_render.py --scene 41069021 --rgb-splat 3x3 --id-splat 5x5
```

Only `arkitscenes_eval.py` and `arkitscenes_selector_eval.py` read ground
truth, below their `ORACLE BOUNDARY` comments.

Public MVP walkthrough (v1): **https://deev09.github.io/surgical-graph-rag/**

The experimental arc is CLOSED at tag `paper-results-v1.0`. Surface
estimation from the mesh closed as a three-act negative
([`docs/c3_0_mesh_surfaces_protocol.md`](docs/c3_0_mesh_surfaces_protocol.md)
and successors); the multiview proposal experiment PASSED all gates
([`docs/c1_p1_multiview_proposals_protocol.md`](docs/c1_p1_multiview_proposals_protocol.md));
the composer and semantics-v2 stages stopped under their predeclared
rules ([`docs/c1_p2_composer_protocol.md`](docs/c1_p2_composer_protocol.md),
[`docs/semantics_v2_track_protocol.md`](docs/semantics_v2_track_protocol.md)).
Future work (D2 precision hardening; annotation-aware attachment) is
identified but deliberately unopened — see
[`docs/results_narrative.md`](docs/results_narrative.md).

## Repo layout (current system)

```
common/ extractors/ geometry/   # EntityArtifacts contract, frame, surfaces
  geometry/frame.py             #   v2: gravity / floor / scale from geometry alone
adapters/                       # ReconstructionAdapter implementations
  adapters/arkitscenes.py       #   v2: first non-Replica capture path
graph/                          # typed relation extractors + graph builder
reasoner/                       # RulesCompiler -> RulesExecutor -> Verbalizer (Router)
  reasoner/confidence.py        #   v2: answer confidence (REFUTED, default-off)
segmenter/                      # C1: segmentation sidecar contract, mask resolution,
                                #     anonymous candidates, derived eval bundles
  segmenter/selector_free.py    #   v2: oracle-free proposal scorer
demo/                           # Replica importers (A/B), question battery, review sheets
eval/                           # router QA scoring + Phase 8 answer keys (human_verified)
  eval/selective.py             #   v2: risk-coverage, AURC, E-AURC, tie-spread
tools/                          # evaluators, scorecard, sweeps, dataset fetch, run_tests
  tools/arkitscenes_*.py        #   v2: render / fuse / eval / selector-eval
  tools/frame_scale_audit.py    #   v2: 19 constants x 6 scenes transfer-risk table
notebooks/                      # Colab GPU backends (Mask3D, Segment3D, SAM 2.1)
docs/                           # contracts, closeouts, protocols, phase records
tests/                          # 84 script-style test files (tools/run_tests.py)
```

Reproduction (datasets, checkpoints, environments, hardware):
`docs/reproduction.md`.

## Legacy v1 (graffiti_bathroom)

The original prototype — a hand-authored 12-object scene graph with
relation-aware retrieval and an optional LLM answering step — lives on in
`tiny_graph_demo.py`, `scenes/`, `benchmark/`, `baselines/`, `scoring/`,
`relations/`, `parsers/`. Its 10-query benchmark is saturated
(top-1 accuracy 1.0) and is **not comparable** to the Phase 8 track:

```bash
SKIP_LLM=1 python3 tiny_graph_demo.py --benchmark-only
```

## License

MIT — see [LICENSE](LICENSE).
