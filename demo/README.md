# Phase 7 demo harness — plug-and-chug on real Replica scenes

This directory is **evidence and exploration**, not the Phase 7 pipeline itself.
The actual relations live in `graph/relations/on_entity_surface.py` (furniture
support) and `graph/relations/attached_to.py` (wall-mounted objects), with gates
in `tools/phase6_*` and `tools/phase7_exit_gate.py`. Everything here *consumes*
that pipeline unchanged to answer one question: **does it work, and generalize,
on real reconstructions?**

Nothing here touches the committed pipeline or its frozen artifacts.

## TL;DR — what was shown

| Claim | How it's backed | Where |
|---|---|---|
| Reproduces ground truth | room_0 imported boxes reproduce the committed `enriched_v2` support pairs (5 table rests + plant-stand) | `real_scene_demo.py` |
| Generalizes to an unseen scene | apartment_0 (454 obj, multi-room) → 10 plausible support pairs across table/desk/plant-stand | `real_scene_demo.py` |
| Runs on raw mesh, not precomputed boxes | boxes recomputed from `mesh_semantic.ply` vertices → **9–10/10 support pairs identical** to the JSON boxes | `mesh_vs_json_demo.py` |
| Robust to global reconstruction drift | simulated backends: support invariant to uniform bias, degrades predictably under per-object jitter | `backend_swap_demo.py` |
| Adds wall-mounted attachment | apartment_0 → 3 plausibility positives (`obj_176` vent, `obj_260` sink, `obj_309` lamp/sconce-like light) while room_0 stays honestly empty | `question_battery.py`, `real_scene_demo.py` |
| Full QA, not just support | floor / against-wall / attached-wall / near-wall / per-class support all answerable | `question_battery.py` |

## Scenes

Scenes are **not** in the repo. They are raw Replica captures under
`~/Desktop/datasets/replica/`:

- `room_0/` — full capture incl. `habitat/mesh_semantic.ply` (44 MB). The **only
  scene with ground truth** (validated against committed `enriched_v2`).
- `apartment_0/` — `habitat/info_semantic.json` (340 KB) **and**
  `habitat/mesh_semantic.ply` (210 MB). Unseen scene; **no answer key**.

The Replica archive (facebookresearch/Replica-Dataset, v1.0) is a 34 GB gzip
stream split into 17×2 GB GitHub-release parts. To extract just one file without
downloading all 34 GB, stream the leading parts and let BSD `tar -q` stop after
the first match (apartment_0 is the first scene, so its files are early):

```bash
BASE="https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0/replica_v1_0.tar.gz"
curl -sL "$BASE.partaa" "$BASE.partab" "$BASE.partac" "$BASE.partad" "$BASE.partae" \
  | gunzip -c \
  | tar -xq -f - -C ~/Desktop/datasets/replica \
        'apartment_0/habitat/mesh_semantic.ply'
```

## Scripts (all take `<room_dir> <scene_id>`; default room_0)

| Script | Purpose |
|---|---|
| `replica_habitat_import.py` | Importer: raw `info_semantic.json` → `EntityArtifacts` (object boxes **and** floor/wall/ceiling surfaces). Gravity-canonicalizes the scene (up→+z) and reuses the canonical `importers/replica.py` surface geometry. |
| `replica_mesh_import.py` | Variant importer: object boxes recomputed **from `mesh_semantic.ply` vertices** (per-face `object_id`), labels still from the JSON. Proves the pipeline runs on real geometry. |
| `real_scene_demo.py` | Build the graph + run the Router on a scene. Validates room_0 support against ground truth; reports support + attachment findings for any other scene. |
| `mesh_vs_json_demo.py` | A/B: same pipeline, JSON boxes vs mesh boxes; diffs the answers. Objective consistency check that needs no answer key. |
| `backend_swap_demo.py` | Simulated reconstruction backends (box perturbation) + Monte-Carlo accuracy-vs-box-error curve. The contact band is the reconstruction-quality budget. |
| `question_battery.py` | Wide reference-free question set (every support class + structural relations) per scene. |
| `visualize_scene.py` | Top-down floorplan PNG with support pairs drawn as labeled connectors. |
| `visualize_questions.py` | 2×2 panel PNG — each relation (floor / against-wall / near-wall / support) isolated. |
| `visualize_3d.py` | Oblique isometric PNG — shaded 3D boxes; height shows objects resting *on* furniture tops. |

### Run everything (room_0 then apartment_0)

```bash
AP=~/Desktop/datasets/replica/apartment_0
python3 demo/real_scene_demo.py                          # room_0: VALIDATED
python3 demo/real_scene_demo.py $AP replica_apartment_0
python3 demo/mesh_vs_json_demo.py                        # room_0 box-source A/B
python3 demo/mesh_vs_json_demo.py $AP replica_apartment_0
python3 demo/question_battery.py $AP replica_apartment_0
python3 demo/visualize_questions.py $AP replica_apartment_0
python3 demo/visualize_3d.py $AP replica_apartment_0
```

## Interpretation limits (read before trusting a number)

- **Only room_0 has ground truth.** apartment_0 is judged by plausibility plus the
  mesh-vs-JSON self-consistency A/B — not an accuracy score.
- **A mesh has geometry, not labels.** `replica_mesh_import.py` takes object boxes
  from the `.ply` but class names (`table`, `book`) still come from
  `info_semantic.json`. A real backend pairs its mesh with a segmentation source.
- **Structural answers are contact-band-sensitive.** Floor/wall calls hinge on a
  2 cm/3 cm band; a ~1.4 cm box-source difference flips borderline objects in/out
  (see the apartment_0 A/B). Support is more robust. This *is* the
  reconstruction-quality budget, made visible — not a bug.
- **`near the wall` is loose.** The 30 cm threshold tags ~271/346 objects in
  apartment_0. Expected for a dense multi-room space.
- **`empty` ≠ "no such object."** Under the oracle completeness profile, "on the
  desk?" returns `empty` whether the desk is bare *or* absent. In Phase 7,
  `attached to the wall?` is answerable when `ATTACHED_TO` edges exist; room_0
  remains an honest empty, while apartment_0 is demo/plausibility evidence only.
- **Box source matters, frame is fixed.** Both importers gravity-canonicalize and
  share one `z_translation`; relations are invariant to that global shift, so
  absolute floor height is cosmetic.
