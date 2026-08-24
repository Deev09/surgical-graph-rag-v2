# Semantics-v2 evaluation track (DRAFT — do not execute)

**Status: DRAFT — awaiting project-owner sign-off. Nothing here is
implemented, run, or scored. This document exists so the semantic
definitions are frozen BEFORE any number is computed, per the owner's
instruction after the C1-P2.0 verdict.**

Written 2026-08-02. Motivation: P2.0 measured that the QA bottleneck has
moved from perception to relation semantics — with P1's proposals,
composition could deliver 31/53 entities at precision 1.00 and citable
answers barely move, because the frozen 2 cm attachment band, the
AABB-top support test, and the support-class allowlist cannot express
what the human keys already record as true.

**Revision 1 (2026-08-02, owner review response — pre-sign-off, nothing
executed):** (1) D1 gained a floor-reaching thin-panel disjunct after
the review caught a contradiction with the key's measured
floor∩attached case (room_2 obj_57); (2) D2 gained an explicit
supported-class policy (no class restriction on supported entities,
smallest-footprint supporter assignment, chains permitted, draped-
textile limitation declared); (3) the S2 proceed rule gained
relation-specific gates (attached ≥ 8/14 and support ≥ 8/20 on room_2,
each at citation precision ≥ 0.85) — aggregate gates alone cannot
protect individual relations; (4) the companion narrative/P2 wording
error ("13 key-cited" → 13 newly viable, 11 key-cited) was corrected in
both documents.

**Revision 2 (2026-08-02, second review response — still unexecuted):**
(1) the newly-viable accounting in the narrative and P2.0 verdict was
made exact — of 13: SEVEN attached-key positives, THREE
furniture-support positives (one materialized), ONE near-wall-only
(non-exhaustive, outside micro-recall), TWO not key-cited; (2) D1 was
made reproducible — distance/qualification bound to the frozen
`geometry/wall_contact.py::wall_contact` predicate (signed,
interior-side, polygon-clipped) at 0.12 m, depth defined as AABB
projection width along the qualifying wall's normal, and
"floor-reaching" removed in favor of a bottom-height partition at
0.30 m; (3) the door risk was reclassified: exhaustive attachment keys
rule unlisted doors as NEGATIVES, so door admissions are false
positives counting against the precision gate, not unspecified cases.

**Revision 3 (2026-08-02, third review response — still unexecuted):**
(1) D1's quantification corrected — `wall_contact` evaluates one
entity–wall PAIR and selects nothing; D1 now states per-pair
evaluation over every entity–wall pair at `contact_threshold_m = 0.12`,
per-pair depth against that pair's wall normal, edge emission per
satisfying pair, and citation iff ≥ 1 edge; (2) the relation-specific
gates (attached ≥ 8/14, support ≥ 8/20, each @ citation precision
≥ 0.85) are listed in the sign-off checkbox itself, not only in S2.

## Track separation (the non-negotiables)

1. **A separately labeled track: `semantics_v2`.** Every result it
   produces lives in new files (`runs/semantics_v2/…`), carries the
   track id in its schema, and is reported ONLY as a
   **benchmark-definition change** — never as an improvement over the
   frozen track. The two tracks are not comparable and no table may mix
   their rows without both labels.
2. **Every existing artifact is preserved unchanged**: all human keys
   (verbatim — they were written to physical reality and already
   contain the truth this track tries to reach), all frozen v1 rows
   (A/B/C1/C2, P1 banks, ceilings), all extractor defaults, gates, and
   the MVP outputs. Hash-guard tests must prove the v1 battery configs
   and frozen bundles are byte-identical after the v2 code lands; v2
   semantics are opt-in configs only.
3. **The keys are the target, not a variable.** The v2 track re-scores
   the SAME `human_verified` keys with new SYSTEM semantics. No key
   edit, no new questions, no reweighting. (This is what makes the
   track honest: the keys already say the blinds are attached and the
   cushions are on the sofa; v1 semantics simply cannot cite them.)

## Semantic definitions (frozen at sign-off, BEFORE any scoring)

All constants below are physical-reasoning choices declared now; they
may be challenged at sign-off and are frozen findings afterward. No
sweep, no post-hoc adjustment.

### D1 — wall-mounted attachment (`ATTACHED_TO` v2) — REVISED ×2 at review

Every geometric term below is bound to the EXISTING frozen predicate —
no new distance machinery is introduced:

- **Per-pair evaluation (`wall_contact` evaluates one entity–wall
  pair; it does not select a wall).** Evaluate EVERY entity–wall pair
  with the frozen `geometry/wall_contact.py::wall_contact` exactly as
  the v1 `ContactsSurfaceExtractor` consumes it (signed, interior-side,
  polygon-clipped), with ONE changed parameter:
  `contact_threshold_m = 0.12` instead of 2 cm (widened to absorb the
  measured ≥5 cm annotation-plane displacement, Stage 0m finding, plus
  box-source error).
- **Depth** is computed per pair: the width of the entity's AABB
  projected onto THAT pair's wall (unit) plane normal.
- **Edge emission:** an `ATTACHED_TO` v2 edge is emitted for EVERY
  entity–wall pair that satisfies D1 in full; an entity is CITED by the
  attached question iff at least one such edge exists.

A pair satisfies D1 iff `wall_contact` passes at 0.12 m, the pair's
depth ≤ **0.35 m**, and at least ONE mounting disjunct holds:

- **(a) elevated mount:** AABB bottom ≥ **0.30 m** above the calibrated
  floor plane (this alone; at that height the v1 floor-support
  disqualifier is vacuous, and the v1 furniture-rest limitation carries
  over unchanged) — vents, plugs, switches, pictures, clocks, window
  blinds;
- **(b) low thin panel:** AABB bottom < **0.30 m** (including
  below-floor boxes) AND depth ≤ **0.12 m** — the term "floor-reaching"
  is REMOVED; the bottom-height partition fully defines the disjunct.
  This resolves the key's measured floor∩attached case (room_2 obj_57,
  a low blinds panel: z [−0.49, 0.35], depth 0.07 m, ruled BOTH
  on-floor and attached by the human key, which the original draft's
  unconditional floor-support exclusion contradicted).

Declared, quantified risk: thin doors can fire under (b), and because
the attachment keys are EXHAUSTIVE, unlisted doors are ruled negatives
— door admissions are FALSE POSITIVES, not unspecified cases. They
count directly against the S2 attached-precision gate (≥ 0.85); no
constant may be adjusted to avoid them.

### D2 — seat / interior support surfaces (`ON_ENTITY_SURFACE` v2)

v1's test (rest on the supporter's AABB TOP) is kept, and a second
disjunct is added — **contained rest**: entity E is on supporter S also
iff ALL of:
- E's XY footprint center lies inside S's XY footprint, and E's
  footprint area ≤ **0.5 ×** S's;
- E's bbox bottom lies within S's vertical extent extended by the
  existing contact band (`S.bottom ≤ E.bottom ≤ S.top + band`);
- E is not floor-supported and S is not E.
This makes cushions-on-sofa, plate-on-lower-tier, and items-on-seats
expressible from AABBs alone (variant A can reach them).
Declared, accepted imprecision: objects INSIDE furniture volumes (e.g.
drawer contents) also fire — recorded as a v2 semantics property, not a
bug, and visible in precision if it costs.

**Supported-class policy (disambiguated at review):**
- Candidate SUPPORTED entities: ANY non-structural entity, with NO
  class restriction — anchor-class entities may themselves be supported
  (the keys rule room_0's plant-stand ON the table).
- SUPPORTERS: D3-allowlist entities only.
- Support chains are permitted (table → plant-stand → plant).
- An entity whose footprint qualifies under multiple supporters is
  assigned to the SMALLEST-footprint qualifying supporter
  (deterministic tie: lower uid). One supporter per entity per
  disjunct.
- Floor-supported entities are EXCLUDED from the contained-rest
  disjunct. Declared limitation this leaves unsolved: draped textiles
  touching the floor (room_0's blanket obj_86, keyed on-sofa) remain
  v2 misses — AABB draping is out of scope for this track.
- E ≠ S always; the footprint ≤ 0.5× condition precludes mutual
  support.

### D3 — furniture-anchor classes (support allowlist v2)

v1 allowlist + **cabinet, nightstand, bed** (the measured gaps: room_0's
cabinet question, room_1's nightstand questions, bed rest cases). The
battery question set is unchanged; the added classes make existing key
questions answerable, they do not add questions.

## Execution stages (each requires the prior one; NOTHING runs now)

- **S1 — implementation + guards.** v2 extractor configs (opt-in flags
  or v2 extractor classes), a `semantics_v2` battery config, synthetic
  tests for each definition, and hash-guard tests proving every v1
  path is byte-identical. No scene scoring yet.
- **S2 — variant A first: the new representation ceiling.** Run A under
  v2 semantics on all four keyed scenes; report per-relation and micro
  P/R next to (clearly labeled, never merged with) the frozen A rows.
  **Predeclared proceed rule to S3 (REVISED at review — aggregate
  gates alone cannot protect individual relations):** learned variants
  are justified iff ALL of the following hold:
  - room_2 `ATTACHED_TO` hits ≥ **8/14** with ATTACHED_TO citation
    precision ≥ **0.85** (A-v1: 1/14);
  - room_2 `ON_ENTITY_SURFACE` hits ≥ **8/20** with ON_ENTITY_SURFACE
    citation precision ≥ **0.85** (A-v1: 5/20);
  - room_2 aggregate micro-R ≥ **0.55** (A-v1: 0.4082) with micro-P ≥
    **0.85**;
  - no keyed scene's A-v2 micro-P falls below **0.80**.
  Otherwise STOP: the definitions do not unlock meaningful reachable
  recall, the v2 track closes as a measured negative, and the v1 track
  remains the project's only benchmark.
- **S3 — learned variants under v2 (only on S2 pass).** B, the frozen
  C1 (ms02), and — evaluation-only — the P2.0 pooled-bank ceiling
  re-scored under v2 semantics. That last row answers the arc's open
  question: do P1's recovered entities become citable once the
  semantics can express them? (Expected but unproven; this measures
  it.) No new GPU, no new perception, no composer — compositions are
  the frozen artifacts only.
- **S4 — reporting.** A `runs/semantics_v2/` scorecard + a labeled
  section in the narrative. Every table carries: "semantics_v2 track —
  benchmark-definition change; not comparable to the frozen track."

## S1 completion record (2026-08-02)

Implemented exactly to the frozen definitions; no scene was scored:

- `graph/relations/attached_to_v2.py` — D1 per-pair extractor (frozen
  `wall_contact` at 0.12 m as the only changed wall parameter; per-pair
  depth along that pair's wall normal; disjuncts (a)/(b) partitioned at
  0.30 m bottom elevation vs the calibrated floor plane; NO
  floor-support disqualifier; edge per satisfying pair).
- `graph/relations/on_entity_surface_v2.py` — D2/D3 extractor (v1
  top-rest delegated verbatim with the v2 allowlist; contained-rest
  disjunct with the frozen policy: smallest-footprint assignment,
  deterministic ties, frozen ON_SURFACE floor disqualifier, (E,S)
  dedupe against top-rest; allowlist = v1 + cabinet/nightstand/bed).
- `reasoner/compiler_rules.py` — opt-in `extra_on_classes` constructor
  parameter; the default constructs the frozen v1 vocabulary exactly.
- `demo/semantics_v2.py` — `runs_v2()` (v1 stack with only the two
  relation swaps) + `make_v2_compiler()`.
- Tests: `tests/relations/test_semantics_v2.py` (6 synthetic
  definition tests: elevated mount incl. an 8 cm-gap case that v1
  provably rejects, low-thin-panel vs low-deep-box, deep-furniture
  rejection + per-pair dual-wall emission, contained-rest +
  smallest-footprint, floor-supported exclusion, D3 anchors + no
  (E,S) duplicates) and `tests/tools/test_semantics_v2_guards.py`
  (4 guards: golden v1 battery bundle hash `graph_dd30b1f3cecfabd5`
  computed BEFORE S1 code landed; default compiler vocabulary frozen —
  cabinet/nightstand/bed still not compiled; v2 anchors opt-in only;
  `runs_v2` swaps exactly the two relation extractors).

Full suite 71/71. **S2 (variant A first) does not run without its own
owner authorization.**

## 2026-08-02 S2 verdict — STOP_TRACK (frozen gates; relation gates decisive)

One S2 run (`tools/semantics_v2_s2.py`; report
`runs/semantics_v2/s2_report.json`; every scene's frozen v1 A row was
recomputed first and hard-matched its committed anchor).

| scene | A-v1 (P / R) | A-v2 (P / R) |
|---|---|---|
| room_0 | 0.85 / 0.347 | 0.79 / 0.612 |
| room_1 | 0.83 / 0.286 | 0.58 / 0.514 |
| room_2 | 0.95 / 0.408 | 0.94 / 0.612 |
| office_0 | 1.00 / 0.375 | **0.36** / 0.500 |

Gate outcomes: aggregate recall (0.612 ≥ 0.55) and aggregate precision
(0.94) PASS; support hits **16/20 @ P 0.94** PASS (v1: 5/20 — D2's
contained-rest works exactly as designed on the dev scene); attached
**0/14, zero citations** FAIL; all-scenes precision floor FAIL (office_0
0.36, room_1 0.58, **and room_0 0.79**). **Two label corrections, 2026-08-24.** (1) The `0.94` in the sentence above is
the **scene aggregate micro-precision** for room_2 (0.9375). The
`ON_ENTITY_SURFACE` citation precision for the same scene is a *different*
number, 0.9412, and both round to 0.94; earlier revisions used the single figure
for both quantities. The same applies to the transfer figures: `0.58` on room_1
and `0.36` on office_0 are **scene aggregate micro-precision**, not the support
relation's own precision, which is 0.5294 on room_1. (2) The prose named only
office_0 and room_1 as failing the 0.80 all-scenes floor; **room_0 at 0.79 also
fails it** and was omitted. Source for all values:
`eval/results/project_census_v1/replica_semantics_v2_s2_report.json`.

**STOP_TRACK per the frozen rule: the v1 track
remains the project's only benchmark; S3 is cancelled unspent.**

**Why attached scored ZERO (read-only census of all 14 keyed attached
objects, recorded in full):** this is not a wiring failure — the v2
extractor emitted no edge because the DATASET's annotation geometry
contradicts any interior-side proximity definition:

- **11 of 14 keyed attached objects lie BEHIND the oracle wall planes**
  (signed best-wall gap −0.16 m to −1.23 m): windows, blinds, and the
  picture are annotated at or beyond the annotated wall plane, so the
  frozen `on_interior_side` clause fails regardless of any band width.
  This is the Stage 0m annotation-displacement finding measured from
  the variant-A side.
- The 3 objects that DO pass wall contact (vents at gaps −0.01 to
  +0.06 m) carry annotation boxes **0.37–0.62 m deep** — all beyond the
  0.35 m depth ceiling; v2 thereby also lost the single hit v1 had
  (obj_42, depth 0.372, by 0.022 m). Constants stay frozen; no
  adjustment.
- One keyed vent (obj_59) sits **1.26 m from every wall plane** — an
  annotation reality no wall-proximity definition reaches.

**What the verdict decomposes:** D2 (contained-rest support) is a
validated definition on the development scene but over-fires on
office_0/room_1 (precision 0.36/0.58 — interior-volume false positives,
the declared drawer-contents property, at real cost); D1 (attachment)
is structurally unreachable from A's annotation geometry — "attached"
in the keys is a semantic judgment about wall fixtures, not an
interior-side distance property of their boxes. Any successor would
need annotation-geometry-aware attachment evidence (e.g., embedded-in-
wall-plane semantics) — a NEW definition requiring its own protocol,
not a constant change.

**Methodological note:** the aggregate gates alone (R 0.612 @ P 0.94 on
room_2) would have PASSED this run. The relation-specific gates the
owner required at review (attached ≥ 8/14) are precisely what caught
the failure. The review requirement is vindicated by the measurement.

## Budget and prohibitions

Zero GPU. No key edits, no v1-artifact modification, no threshold
sweeps (the D1/D2 constants are single declared values), no new
questions, no frl, no composer work, no C2/C3 reopening. Failures stop
at their stage and are committed as findings.

## Sign-off

- [x] Owner approves the track separation, definitions D1–D3 (with
      their frozen constants and D1's per-pair edge-emission
      semantics), the A-first proceed rule — relation-specific gates
      **room_2 ATTACHED_TO ≥ 8/14 @ citation precision ≥ 0.85** and
      **room_2 ON_ENTITY_SURFACE ≥ 8/20 @ citation precision ≥ 0.85**,
      plus aggregate room_2 micro-R ≥ 0.55 @ micro-P ≥ 0.85 and the
      all-scenes micro-P ≥ 0.80 floor — and the stage order
      (2026-08-02, project owner / deevyaswain — "approved, sign off
      semantics-v2 and start S1", after three recorded review rounds).
      Definitions D1–D3 and all gates are FROZEN as of this sign-off.
