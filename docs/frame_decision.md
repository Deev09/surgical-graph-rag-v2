# Frame decision — what frame the graph is built in, and who says so

**Status: acted-on.** This change modifies the pipeline. It is the follow-up
to `docs/frame_and_scale_audit.md` (diagnostic; changed nothing) and acts on
that document's Findings 1 and 4 only. **No threshold was retuned.** Findings
5–7 (the scale spread) are deliberately untouched — see *Not done here*.

Verification, both required to be unmoved and both confirmed:

- `.venv/bin/python3 tools/run_tests.py` — 76/76 before, 78/78 after (two new
  files, no failures, nothing skipped that was not already dataset-guarded).
- `.venv/bin/python3 tools/scene_scorecard.py` — 4 true_answer / 27 true_empty
  / 22 miss / 3 false_answer, and all six `bundle_hash` values byte-identical:

  | scene | bundle_hash (before == after) |
  |---|---|
  | replica_room_0 | `graph_f91cd343511a6cc7` |
  | replica_room_1 | `graph_b2c45040d87a3eda` |
  | replica_room_2 | `graph_a9aaa272b7bd0755` |
  | replica_office_0 | `graph_10ffdb8f9bf477e0` |
  | replica_frl_apartment_0 | `graph_d00ed152eb56b8ad` |
  | replica_apartment_0 | `graph_335f8fd2abe82f3e` |

---

## 1. What frame the graph is actually built in

**`scene_canonical`, on every scene the live pipeline touches.**

`demo/replica_habitat_import.py` — the importer behind
`tools/scene_scorecard.py`, `demo/question_battery.py` and every Phase 6–8
tool — applies, in order:

1. `R_align`, rotating physical up (`-gravity_dir`) onto exactly `+z`. Replica's
   smallest declared tilt is 0.11° and its largest is 8.72°, so this is
   **never** the identity on real data;
2. a yaw de-rotation about `+z`, when the wall-derived dominant yaw exceeds
   `YAW_DEROTATE_GUARD_DEG = 5.0` (applied on room_1 at +26.6° and room_2 at
   −7.2°);
3. a global `z_translation`.

Only step 3 is frame-preserving: every relation in `graph/relations/**` is
computed from coordinate *differences*, so a translation cannot change an edge.
Steps 1 and 2 can, and do. Finding 4 measured how much: recomputing directional
edges in the raw Habitat world frame changes **26.9% of room_1's** edges and
**9.7% of room_2's**. The label `frame="world"` was therefore false on the live
path, and false in a way with a measured magnitude.

It stayed invisible for a simple reason. On room_0 — the scene every threshold
in the repo was calibrated against — the raw and canonical frames agree on
**2593 of 2594 edges**. The mislabel cost nothing on the one scene anyone
looked at.

**`"scene_canonical"` was already the right word and was used by zero edges.**
It now means, precisely: *some rotation was applied so that up is exactly +z;
possibly a yaw de-rotation on top of it; these are not the capture's axes.*
Two importers can both produce a `scene_canonical` frame without producing the
*same* one (the legacy importer levels gravity but does not de-rotate yaw).
The label states what was guaranteed, not that two canonical frames coincide.

### How it is now enforced

The label is not a new constant. It is an **equality**:

```
for every edge e extracted from bundle b:   e.frame == b.frame.kind
```

- `common.types.SceneFrame` gained `kind: FrameKind = "world"`. The default is
  the conservative one: a bundle that does not consciously canonicalize keeps
  claiming only what it did.
- `graph.relations.base.edge_frame(entities)` returns `entities.frame.kind`.
  All nine edge constructors route through it; none names a frame.
- `graph.schema.Edge.frame` is now typed `FrameKind`, the same domain, so the
  two cannot drift apart.
- `tests/graph/test_edge_frame_label.py` runs every extractor over identical
  geometry under all three frame kinds. A hardcoded literal can match at most
  one of the three. It also introspects `graph.relations` and fails if an
  extractor exists that its roster does not cover, so a tenth extractor cannot
  quietly reintroduce the bug.

Verified to catch the original defect: reverting `directional.py` alone to
`frame="world"` fails 4 of the 6 tests in that file, including on real
room_0 and room_2 data.

Not everything is `scene_canonical`, and that is the point of asserting an
equality rather than swapping one constant for another. The v1 / oracle path
(`importers/replica.py` → `scenes/replica_room_0/` →
`adapters/oracle_replica.py`) really is in the capture's raw axes, and its
edges correctly still say `"world"`.

## 2. Reconciling the two importers

### What the frozen results depend on: ACCEPT-AND-ROTATE

`tools/scene_scorecard.py::_build_scene` calls `import_habitat_room`, which
rotates. Making the demo path refuse tilted scenes would drop room_2 from the
scorecard entirely and move the frozen 4/27/22/3. **That option is excluded by
measurement, not by preference.** The legacy importer is the side that changes.

### The change

`importers/replica.py` no longer raises `SystemExit: Refusing to import` on a
tilted capture. It levels it, using the *same* rotation function as the demo
path — `gravity_align_matrix` moved verbatim into `importers/replica.py` (the
lower module; `demo/replica_habitat_import.py` re-exports it under its old
private name). Two importers computing "the same" rotation two different ways
is how this divergence arose; there is now one implementation.

Levelling applies at `GRAVITY_ALIGN_GUARD_DEG = 5.0`:

| scene | tilt off +Z | before | after |
|---|---|---|---|
| frl_apartment_0 | 0.11° | accepted, raw axes | unchanged, `frame_kind: world` |
| office_0 | 0.21° | accepted, raw axes | unchanged, `frame_kind: world` |
| room_1 | 0.23° | accepted, raw axes | unchanged, `frame_kind: world` |
| room_0 | 0.27° | accepted, raw axes | unchanged, `frame_kind: world` |
| apartment_0 | 1.31° | accepted, raw axes | unchanged, `frame_kind: world` |
| **room_2** | **8.72°** | **`SystemExit`** | **levelled, `frame_kind: scene_canonical`** |

**The guard is not a claim that a 1.31° tilt is negligible.** Finding 4 is
exactly the evidence that a small rotation need not be. The guard exists for
one reason: `scenes/replica_room_0/enriched/v2/` is the frozen Phase 1 replay
fixture the v1 benchmark is defined against, and unconditional levelling would
move every coordinate in it — a benchmark-definition change wearing a bug-fix
costume. `test_room_0_enriched_v2_is_byte_identical_to_the_committed_fixture`
compares bytes and fails if that ever stops holding. 5.0 is also the value and
idiom already used by `YAW_DEROTATE_GUARD_DEG`, and it splits a 1.31° / 8.72°
gap with >3.5× clearance on both sides.

The residual difference between the two importers is now *declared* rather than
silent: below the guard the legacy importer emits `frame_kind: "world"`, and
the demo importer emits `"scene_canonical"` for the same scene. Both are true.
`capture_meta.json`'s `axis_convention` records `frame_kind`,
`gravity_tilt_deg`, `gravity_align_guard_deg`, `gravity_align_applied`, and
`gravity_dir_effective`; `adapters/oracle_replica.py` reads them, so a levelled
scene gets a `SceneFrame` whose gravity matches its own coordinates instead of
the raw tilted vector. Both keys are absent from capture_meta files written
before this change, and absence correctly defaults to the raw-axes reading.

`_gravity_is_neg_z` is kept — it names the assumption the whole v1 path was
built on, and `tools/frame_scale_audit.py` still reports it — but it no longer
gates anything.

## 3. Bundle hashes

`graph.builder._build_bundle_hash` covers `entity_bundle_hash`, `mode`, and the
ordered `(extractor_name, extractor_version, config)` tuples. The frame label is
derived from the entity bundle, which is already inside that hash, so no
extractor version was bumped and no config gained a field — which is why all
six hashes are unchanged. The corollary is worth stating plainly: **a bundle
cached under one of these hashes before this change carries `frame="world"` on
edges that now serialize as `frame="scene_canonical"`.** Nothing in the repo
reads `Edge.frame`, and no such cache is committed, so this is a note for
whoever builds one, not a live problem.

## Not done here

- **No threshold retuned.** Every constant in Findings 5–7 is untouched. That
  work still needs the per-constant *scale / sensor-resolution / content-
  distribution* triage the audit calls for — and the audit's own warning that
  a scale-consistent contact band on office_0 would be 6 mm, under the mesh's
  own reconstruction noise.
- **No yaw de-rotation in `importers/replica.py`.** It levels gravity only.
  room_2's walls come out ~11° off axis. Adding yaw there would move nothing
  frozen, but it is a separate change with its own guard question.
- **`tools/frame_scale_audit.py` and `runs/frame_audit/` are untouched.** They
  are a dated snapshot of a measurement taken against the pre-change code;
  rewriting them retroactively would make the report describe a run that never
  happened. Table 1's column header "`+Z up` guard in importers/replica.py" is
  accurate for the run that produced it and stale for the code today: that
  predicate no longer decides whether a scene imports.
- **`Edge.frame` is still read by nothing.** Making it truthful does not make
  it load-bearing. A consumer that wants to reject cross-frame comparisons now
  has a field it can trust.
