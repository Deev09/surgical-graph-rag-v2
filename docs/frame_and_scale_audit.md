# Frame and scale audit — six Replica scenes

**Status: diagnostic. Nothing in the pipeline was changed by this work.** No
threshold was retuned, no extractor was edited, no default was moved. This
document reports measurements; acting on them is a separate change.

- Estimator: `geometry/frame.py` (new, additive, imported by nothing in the
  extraction path).
- Driver: `tools/frame_scale_audit.py`.
- Tests: `tests/geometry/test_frame.py`.
- Raw output: `runs/frame_audit/frame_scale_audit.json`,
  `runs/frame_audit/tables.md`, `runs/frame_audit/<scene>_frame_estimate.json`.
- Reproduce: `.venv/bin/python3 tools/frame_scale_audit.py` (about 20 s for
  all six scenes).

## Why

Two assumptions are baked into `graph/relations/**` and neither had ever been
measured:

1. **The world frame is gravity-aligned.** `graph/schema.py` gives `Edge` a
   `frame: Literal["world", "viewpoint", "scene_canonical"]`, and all nine
   edge constructors in `graph/relations/**` write `frame="world"`. Zero
   edges anywhere in the repo carry `"scene_canonical"` or `"viewpoint"`.
   The directional convention itself is documented as un-generalised — see
   `relations/compute.py:6`, *"Conventions (read off the v1 scene; not
   generalized): x small -> LEFT ... z small -> BELOW"*.
2. **Every scene is room_0-sized.** `LEGACY_MIN_DELTA = 0.3`,
   `sparse_max_distance = 2.5`, the 0.02 m contact bands, the 0.30 m
   elevation gate and the rest are absolute metres, calibrated on one scene.
   `DirectionalConfig`'s own docstring already says 2.5 m is "a
   Replica-calibrated provisional default ... NOT evidence that 2.5 m is a
   generally good threshold."

## What was measured, and how

`geometry/frame.py` estimates, from triangle geometry alone — no semantic
labels, no `gravity_dir`, no Replica-specific prior:

- the **gravity axis**, as the peak of the area-weighted normal histogram
  over a 512-direction Fibonacci hemisphere, refined by the top eigenvector
  of the area-weighted normal scatter matrix inside the winning cone;
- the **sign** of that axis, by majority vote of three physical cues
  (up/down-facing area asymmetry; near-floor vs near-ceiling clutter;
  height of the interior horizontal surfaces);
- the **floor and ceiling planes**, as area-weighted peaks of the horizontal
  face-offset histogram (1 cm bins), each least-squares refit over the
  triangle vertices in a 5 cm band;
- the **yaw**, as the 90°-symmetric circular mean of vertical-face azimuths;
- **scale**: robust (p1–p99) extents in the canonical frame, room diagonal,
  floor footprint diagonal, storey height.

The only absolute-length prior in the module is
`min_storey_height_m = 1.8`, used solely to pick the *first* ceiling above
the floor so a two-level capture does not report a 5 m "room height". The
value is inherited from the existing repo precedent,
`MeshSurfaceConfig.floor_ceiling_separation_min_m`.

**Scenes measured: all six.** `apartment_0`, `frl_apartment_0`, `office_0`,
`room_0`, `room_1`, `room_2` — each read from
`habitat/mesh_semantic.ply` (0.6 M–4.6 M vertices). Nothing was skipped.
Note that `geometry.mesh_surfaces.load_raw_triangle_mesh` cannot read these
files: the Habitat instance meshes carry a per-face `uint16 object_id` after
the vertex-index list, so `frame.py` has its own binary face-record reader
(it reuses the shared PLY header grammar).

---

## Finding 1 — the geometry agrees with Replica's declared gravity, and BOTH say the world frame is not gravity-aligned

| scene | est. up vs world +Z (deg) | declared gravity vs -Z (deg) | est. up vs declared up (deg) | `+Z up` guard in importers/replica.py | axis margin (win/runner-up) | sign votes A/B/C |
|---|---|---|---|---|---|---|
| room_0 | 0.301 | 0.271 | 0.059 | yes | 1.647 | -/+/+ |
| room_1 | 0.265 | 0.230 | 0.036 | yes | 1.213 | -/+/+ |
| room_2 | 8.783 | 8.725 | 0.082 | NO | 1.617 | -/+/+ |
| office_0 | 0.180 | 0.206 | 0.330 | yes | 1.152 | -/+/+ |
| frl_apartment_0 | 0.163 | 0.115 | 0.118 | yes | 1.021 | +/+/+ |
| apartment_0 | 1.184 | 1.309 | 0.152 | yes | 1.741 | +/+/+ |

**Agreement.** The mesh-derived up axis matches Replica's declared
`gravity_dir` to within **0.33° on every scene**. Replica's metadata is
trustworthy, and a geometry-only estimator recovers it without being told.

**Disagreement with the pipeline's assumption.** `room_2`'s vertical is
**8.78° away from world +Z**, and `apartment_0`'s is 1.18°. So "z is up"
is false for room_2 by nearly nine degrees. Two consequences already in
the codebase:

- `importers/replica.py::_gravity_is_neg_z` requires `|g_x| < 0.05`.
  room_2 has `g_x = -0.1496`, so **the canonical importer refuses room_2
  outright** (`SystemExit: Refusing to import`). The scorecard path works
  only because `demo/replica_habitat_import.py` takes a different route and
  rotates the scene into alignment.
- Two importers therefore disagree about whether room_2 is importable at
  all. That is a frame problem wearing a validation-error costume.

**Honest caveat on the axis estimator.** The winning axis beats the
runner-up (a wall direction) by a margin of only **1.02× to 1.74×** in
parallel mesh area. On `frl_apartment_0` the margin is 2%. The "floor plus
ceiling outweigh any one wall" premise is true here but it is *thin*, and a
long-corridor or high-ceilinged scene could invert it. The margins are
reported per scene in `runs/frame_audit/<scene>_frame_estimate.json` so this
is never silently assumed.

**A textbook cue that measurably fails.** The standard way to sign a gravity
axis — "there is more up-facing horizontal area than down-facing" — is
**wrong on 4 of 6 scenes** (`-` entries in the sign-votes column). Furniture
occludes the floor while the ceiling scans clean: office_0's observed floor
is 5.6 m² against a 15.1 m² ceiling. The clutter cue and the interior-surface
cue are right 6/6. This is recorded in the module rather than deleted,
because a cue that fails on real scans is information.

## Finding 2 — the mesh floor and the calibrated floor plane agree to ~1–3 cm, and the F2 snap is vindicated

`graph/relations/attached_to_v2.py` measures every object's mount height
against the floor plane it receives from `demo/replica_habitat_import.py`.
That plane is compared here against the mesh-derived one, in the importer's
own frame.

| scene | importer floors | best abs delta (m) | worst abs delta (m) | same delta BEFORE the F2 floor snap (m) | F2 snap applied (m) | floor-normal angle (deg) |
|---|---|---|---|---|---|---|
| room_0 | 1 | 0.0198 | 0.0198 | 0.0198 | (none) | 0.055 |
| room_1 | 1 | 0.0112 | 0.0112 | 0.0985 | {'floor_1': -0.1097} | 0.189 |
| room_2 | 1 | 0.0103 | 0.0103 | 0.0103 | (none) | 0.041 |
| office_0 | 1 | 0.0104 | 0.0104 | 0.0104 | (none) | 0.173 |
| frl_apartment_0 | 1 | 0.0006 | 0.0006 | 0.2819 | {'floor_8': -0.2814} | 0.059 |
| apartment_0 | 4 | 0.0267 | 2.8239 | 0.0267 | (none) | 0.111 |

- **Agreement**: 0.6 mm to 2.7 cm, with floor normals within 0.19°. The
  calibrated floor plane is geometrically correct on every scene.
- **The Phase 8 F2 floor-calibration heuristic is independently confirmed.**
  frl_apartment_0's *labelled* floor sits **28.2 cm above the real floor**;
  the heuristic snapped it down 28.1 cm and landed within 0.6 mm of the
  mesh. room_1: 9.9 cm error, snapped 11.0 cm, landed within 1.1 cm. That
  heuristic was justified by "objects penetrate the floor"; it is now also
  justified by the mesh.
- **The one real disagreement is scope, not error.** apartment_0 has four
  labelled floors across two storeys (mesh up-facing peaks at −1.45 m and
  +1.31 m; down-facing at +1.10 m and +3.69 m). The mesh estimator returns the lowest
  floor only, so the worst per-floor delta is 2.82 m — it is comparing the
  ground-floor plane against a first-floor label. A single-floor-plane
  abstraction does not survive a multi-storey capture, and apartment_0 is
  flagged `multi_level_suspected: true`.

## Finding 3 — the yaw estimate, and a 5° guard sitting inside the disagreement

`demo/replica_habitat_import.py` de-rotates the room yaw only when a
wall-label estimate exceeds `YAW_DEROTATE_GUARD_DEG = 5.0`.

| scene | mesh dominant yaw (deg) | importer label yaw, pre-guard (deg) | disagreement (deg) | yaw actually applied (deg) | guard would flip on the other estimate |
|---|---|---|---|---|---|
| room_0 | -0.449 | -0.369 | 0.080 | 0.000 | no |
| room_1 | 25.992 | 26.555 | 0.564 | 26.555 | no |
| room_2 | -11.386 | -7.201 | 4.184 | -7.201 | no |
| office_0 | 5.900 | 3.107 | 2.793 | 0.000 | YES |
| frl_apartment_0 | 0.463 | -0.481 | 0.944 | 0.000 | no |
| apartment_0 | 7.569 | 1.659 | 5.909 | 0.000 | YES |

Two reasonable estimators of the same quantity — labelled wall normals vs.
all vertical mesh faces — differ by up to 5.9°, which is larger than the
guard that decides whether to correct at all. **On office_0 and
apartment_0 the de-rotation decision flips depending on which estimator you
use.** office_0 currently gets no correction; the mesh says it is rotated
5.9°. (For apartment_0, a single room yaw is arguably ill-posed — it is a
multi-room apartment — which is itself part of the finding.)

## Finding 4 — `frame="world"` is a load-bearing label, and it is the wrong one

Directional edges recomputed in the raw Habitat world frame vs. the
gravity-aligned, yaw-de-rotated frame the importer actually builds, using a
read-only port of the compat dominant-axis rule:

| scene | pairs with an edge (aligned frame) | type changes | share | dominant axis changes | share |
|---|---|---|---|---|---|
| room_0 | 2594 | 1 | 0.000 | 1 | 0.000 |
| room_1 | 975 | 262 | 0.269 | 262 | 0.269 |
| room_2 | 1347 | 130 | 0.097 | 128 | 0.095 |
| office_0 | 1046 | 1 | 0.001 | 1 | 0.001 |
| frl_apartment_0 | 18415 | 3 | 0.000 | 2 | 0.000 |
| apartment_0 | 59580 | 435 | 0.007 | 435 | 0.007 |

**26.9% of room_1's directional edges and 9.7% of room_2's change type
depending on which frame the extractor runs in.** The pipeline already picks
the right frame — the importer gravity-aligns and yaw-de-rotates — but every
resulting edge is stamped `frame="world"`, which is not the frame it was
computed in. The correct existing enum value, `"scene_canonical"`, is used
by nothing. On room_0 (the calibration scene) the two frames agree on 2593
of 2594 edges, which is exactly why the mislabel was invisible: it costs
nothing on the one scene the system was built against.

---

## Finding 5 — scene scale, measured

| scene | room diag (m) | floor diag (m) | storey height (m) | median object diag (m) | entity-set diag (m) | n entities | median pair dist (m) | observed floor area (m2) |
|---|---|---|---|---|---|---|---|---|
| room_0 | 9.402 | 8.991 | 2.728 | 1.089 | 9.750 | 73 | 3.513 | 27.76 |
| room_1 | 7.472 | 6.959 | 2.700 | 1.455 | 8.460 | 45 | 2.963 | 12.15 |
| room_2 | 7.495 | 6.980 | 2.708 | 1.091 | 7.820 | 53 | 2.895 | 20.71 |
| office_0 | 6.734 | 6.085 | 2.855 | 0.336 | 7.437 | 47 | 2.632 | 5.62 |
| frl_apartment_0 | 14.720 | 14.415 | 2.732 | 0.539 | 16.840 | 194 | 4.734 | 48.76 |
| apartment_0 | 16.799 | 15.983 | 2.552 | 0.822 | 18.462 | 346 | 5.912 | 75.52 |

| denominator | min | max | max/min | mean | CV |
|---|---|---|---|---|---|
| room_diag | 6.734 | 16.799 | 2.49 | 10.437 | 0.373 |
| floor_diag | 6.085 | 15.983 | 2.63 | 9.902 | 0.391 |
| storey_height | 2.552 | 2.855 | 1.12 | 2.713 | 0.032 |
| obj_diag | 0.336 | 1.455 | 4.33 | 0.889 | 0.420 |

The headline: **object scale varies more than room scale.** Median object
diagonal spans 4.33× (office_0's 0.34 m of desk clutter vs room_1's 1.46 m
of beds and wardrobes) while room diagonal spans 2.49× and storey height
spans only 1.12×. `observed floor area` is the *scanned* floor and is
heavily occlusion-dependent (office_0: 5.6 m² observed in a ~18 m² room);
it is reported for completeness but is not used as a scale denominator.

## Finding 6 — MAIN DELIVERABLE: every hardcoded metre constant as a fraction of each scene's own scale

Each constant is read directly out of its live module by
`tools/frame_scale_audit.py::_constants` (never re-typed), and divided by the
scene scale it is physically comparing against.

**Read the max/min and CV columns honestly.** They are a property of the
*denominator*, so every row sharing a denominator shares them. What differs
per constant is *where* its fraction sits: whether it is a
few-percent-of-room quantity that survives a 2.5× scale change, or a
comparable-to-one-object quantity that a 4.3× object-size change moves
across the decision boundary.

| constant | value (m) | denominator | room_0 | room_1 | room_2 | office_0 | frl_apartment_0 | apartment_0 | min | max | max/min | CV |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `directional.LEGACY_MIN_DELTA` | 0.3 | obj_diag | 0.2755 | 0.2062 | 0.2749 | 0.8937 | 0.5570 | 0.3649 | 0.2062 | 0.8937 | 4.33 | 0.550 |
| `directional.sparse_min_delta` | 0.5 | obj_diag | 0.4591 | 0.3437 | 0.4581 | 1.4896 | 0.9283 | 0.6081 | 0.3437 | 1.4896 | 4.33 | 0.550 |
| `directional.sparse_max_distance` | 2.5 | room_diag | 0.2659 | 0.3346 | 0.3336 | 0.3712 | 0.1698 | 0.1488 | 0.1488 | 0.3712 | 2.49 | 0.313 |
| `proximity.LEGACY_NEAR_THRESHOLD` | 1 | room_diag | 0.1064 | 0.1338 | 0.1334 | 0.1485 | 0.0679 | 0.0595 | 0.0595 | 0.1485 | 2.49 | 0.313 |
| `proximity.sparse_near_threshold` | 1 | room_diag | 0.1064 | 0.1338 | 0.1334 | 0.1485 | 0.0679 | 0.0595 | 0.0595 | 0.1485 | 2.49 | 0.313 |
| `contacts_surface.DEFAULT_CONTACT_THRESHOLD_M` | 0.02 | obj_diag | 0.0184 | 0.0137 | 0.0183 | 0.0596 | 0.0371 | 0.0243 | 0.0137 | 0.0596 | 4.33 | 0.550 |
| `contacts_surface.DEFAULT_PENETRATION_TOLERANCE_M` | 0.02 | obj_diag | 0.0184 | 0.0137 | 0.0183 | 0.0596 | 0.0371 | 0.0243 | 0.0137 | 0.0596 | 4.33 | 0.550 |
| `contacts_surface.DEFAULT_NEAR_SURFACE_THRESHOLD_M` | 0.3 | obj_diag | 0.2755 | 0.2062 | 0.2749 | 0.8937 | 0.5570 | 0.3649 | 0.2062 | 0.8937 | 4.33 | 0.550 |
| `contacts_surface.DEFAULT_ROOM_SCALE_FLAT_MAX_HEIGHT_M` | 0.2 | storey_height | 0.0733 | 0.0741 | 0.0739 | 0.0701 | 0.0732 | 0.0784 | 0.0701 | 0.0784 | 1.12 | 0.033 |
| `on_surface.DEFAULT_CONTACT_THRESHOLD_M` | 0.02 | obj_diag | 0.0184 | 0.0137 | 0.0183 | 0.0596 | 0.0371 | 0.0243 | 0.0137 | 0.0596 | 4.33 | 0.550 |
| `on_surface.DEFAULT_PENETRATION_TOLERANCE_M` | 0.03 | obj_diag | 0.0275 | 0.0206 | 0.0275 | 0.0894 | 0.0557 | 0.0365 | 0.0206 | 0.0894 | 4.33 | 0.550 |
| `on_surface.DEFAULT_NEAR_SURFACE_THRESHOLD_M` | 0.05 | obj_diag | 0.0459 | 0.0344 | 0.0458 | 0.1490 | 0.0928 | 0.0608 | 0.0344 | 0.1490 | 4.33 | 0.550 |
| `surface.DEFAULT_FLOOR_THRESHOLD_M` | 0.05 | obj_diag | 0.0459 | 0.0344 | 0.0458 | 0.1490 | 0.0928 | 0.0608 | 0.0344 | 0.1490 | 4.33 | 0.550 |
| `surface.DEFAULT_WALL_THRESHOLD_M` | 0.3 | obj_diag | 0.2755 | 0.2062 | 0.2749 | 0.8937 | 0.5570 | 0.3649 | 0.2062 | 0.8937 | 4.33 | 0.550 |
| `surface.DEFAULT_CEILING_THRESHOLD_M` | 0.1 | obj_diag | 0.0918 | 0.0687 | 0.0916 | 0.2979 | 0.1857 | 0.1216 | 0.0687 | 0.2979 | 4.33 | 0.550 |
| `attached_to_v2.contact_threshold_m` | 0.12 | obj_diag | 0.1102 | 0.0825 | 0.1099 | 0.3575 | 0.2228 | 0.1459 | 0.0825 | 0.3575 | 4.33 | 0.550 |
| `attached_to_v2.depth_max_m` | 0.35 | obj_diag | 0.3214 | 0.2406 | 0.3207 | 1.0427 | 0.6498 | 0.4257 | 0.2406 | 1.0427 | 4.33 | 0.550 |
| `attached_to_v2.elevated_bottom_min_m` | 0.3 | storey_height | 0.1100 | 0.1111 | 0.1108 | 0.1051 | 0.1098 | 0.1175 | 0.1051 | 0.1175 | 1.12 | 0.033 |
| `attached_to_v2.thin_panel_depth_max_m` | 0.12 | obj_diag | 0.1102 | 0.0825 | 0.1099 | 0.3575 | 0.2228 | 0.1459 | 0.0825 | 0.3575 | 4.33 | 0.550 |

### Risk tiers this table produces

**Tier 1 — highest transfer risk (object-scale constants that cross a
semantic boundary).**
`sparse_min_delta = 0.5` is 0.34 median-object-diagonals on room_1 and
**1.49 on office_0** — on office_0 the axis-dominance gate is wider than a
typical object, so "is A left of B" is being asked at a granularity coarser
than the objects themselves. `attached_to_v2.depth_max_m = 0.35` crosses
1.0× object diagonal on office_0 too (1.04): every office_0 object is
"shallow enough to be wall-mounted" by that test. `LEGACY_MIN_DELTA = 0.3`
and `surface.DEFAULT_WALL_THRESHOLD_M = 0.3` reach 0.89 object diagonals on
office_0 versus 0.21 on room_1 — the same number means "a small nudge" in
one scene and "most of an object" in another.

**Tier 2 — moderate risk (room-scale constants).**
`sparse_max_distance = 2.5` ranges 0.149–0.371 of the room diagonal, a
2.49× spread. `NEAR = 1.0` likewise. These do not cross a semantic
boundary, but they change how much of the scene each relation covers (see
Finding 7).

**Tier 3 — low risk (storey-height constants).**
`DEFAULT_ROOM_SCALE_FLAT_MAX_HEIGHT_M = 0.20` and
`elevated_bottom_min_m = 0.30` vary by only 1.12× (CV 0.033) because storey
height is nearly constant across human buildings. These two are the *least*
in need of rescaling — a genuinely useful negative result, since 0.30 m
looked like exactly the kind of arbitrary metre constant that would fail to
transfer.

**Tier 0 — no scale risk at all.**

| constant | value | why it is scale-free |
|---|---|---|
| `contacts_surface.DEFAULT_MAX_WALL_TILT_DEG` | 30.0 | angular, dimensionless under uniform scaling |
| `on_surface.DEFAULT_MAX_TILT_DEG` | 30.0 | angular, dimensionless under uniform scaling |
| `contacts_surface.DEFAULT_FOOTPRINT_TOLERANCE_M` | 0.0 | zero length; scale-invariant |
| `on_surface.DEFAULT_FOOTPRINT_TOLERANCE_M` | 0.0 | zero length; scale-invariant |
| `contacts_surface.DEFAULT_ROOM_SCALE_FLAT_MIN_AREA_FRAC` | 0.6 | already a ratio of two same-scene quantities |
| `on_entity_surface_v2.footprint_area_max_frac` | 0.5 | already a ratio of two same-scene quantities |

### What the constants would be if the room_0 fraction were held fixed

Same fraction, each scene's own scale. This is the metre-level size of the
disagreement, not a proposal.

| constant | frozen value (m) | denominator | room_0 | room_1 | room_2 | office_0 | frl_apartment_0 | apartment_0 |
|---|---|---|---|---|---|---|---|---|
| `directional.LEGACY_MIN_DELTA` | 0.3 | obj_diag | 0.300 | 0.401 | 0.301 | 0.092 | 0.148 | 0.226 |
| `directional.sparse_min_delta` | 0.5 | obj_diag | 0.500 | 0.668 | 0.501 | 0.154 | 0.247 | 0.377 |
| `directional.sparse_max_distance` | 2.5 | room_diag | 2.500 | 1.987 | 1.993 | 1.791 | 3.914 | 4.467 |
| `proximity.LEGACY_NEAR_THRESHOLD` | 1 | room_diag | 1.000 | 0.795 | 0.797 | 0.716 | 1.566 | 1.787 |
| `proximity.sparse_near_threshold` | 1 | room_diag | 1.000 | 0.795 | 0.797 | 0.716 | 1.566 | 1.787 |
| `contacts_surface.DEFAULT_CONTACT_THRESHOLD_M` | 0.02 | obj_diag | 0.020 | 0.027 | 0.020 | 0.006 | 0.010 | 0.015 |
| `contacts_surface.DEFAULT_PENETRATION_TOLERANCE_M` | 0.02 | obj_diag | 0.020 | 0.027 | 0.020 | 0.006 | 0.010 | 0.015 |
| `contacts_surface.DEFAULT_NEAR_SURFACE_THRESHOLD_M` | 0.3 | obj_diag | 0.300 | 0.401 | 0.301 | 0.092 | 0.148 | 0.226 |
| `contacts_surface.DEFAULT_ROOM_SCALE_FLAT_MAX_HEIGHT_M` | 0.2 | storey_height | 0.200 | 0.198 | 0.199 | 0.209 | 0.200 | 0.187 |
| `on_surface.DEFAULT_CONTACT_THRESHOLD_M` | 0.02 | obj_diag | 0.020 | 0.027 | 0.020 | 0.006 | 0.010 | 0.015 |
| `on_surface.DEFAULT_PENETRATION_TOLERANCE_M` | 0.03 | obj_diag | 0.030 | 0.040 | 0.030 | 0.009 | 0.015 | 0.023 |
| `on_surface.DEFAULT_NEAR_SURFACE_THRESHOLD_M` | 0.05 | obj_diag | 0.050 | 0.067 | 0.050 | 0.015 | 0.025 | 0.038 |
| `surface.DEFAULT_FLOOR_THRESHOLD_M` | 0.05 | obj_diag | 0.050 | 0.067 | 0.050 | 0.015 | 0.025 | 0.038 |
| `surface.DEFAULT_WALL_THRESHOLD_M` | 0.3 | obj_diag | 0.300 | 0.401 | 0.301 | 0.092 | 0.148 | 0.226 |
| `surface.DEFAULT_CEILING_THRESHOLD_M` | 0.1 | obj_diag | 0.100 | 0.134 | 0.100 | 0.031 | 0.049 | 0.075 |
| `attached_to_v2.contact_threshold_m` | 0.12 | obj_diag | 0.120 | 0.160 | 0.120 | 0.037 | 0.059 | 0.091 |
| `attached_to_v2.depth_max_m` | 0.35 | obj_diag | 0.350 | 0.467 | 0.351 | 0.108 | 0.173 | 0.264 |
| `attached_to_v2.elevated_bottom_min_m` | 0.3 | storey_height | 0.300 | 0.297 | 0.298 | 0.314 | 0.300 | 0.281 |
| `attached_to_v2.thin_panel_depth_max_m` | 0.12 | obj_diag | 0.120 | 0.160 | 0.120 | 0.037 | 0.059 | 0.091 |

An important caution on this table: the contact-band row says a
scale-consistent `contact_threshold` on office_0 would be **6 mm**. That is
at or below the mesh's own reconstruction noise (the refit floor and ceiling
planes have RMS 3.7–16 mm across the six scenes; office_0's floor is
9.1 mm). Naïvely rescaling the small bands by object size would push them
under the sensor floor. Contact bands are a *sensor-resolution* quantity, not
a scale quantity; this table shows that treating them as scale quantities
would be a mistake, which is itself a useful result for whoever acts on this.

## Finding 7 — the same constant, wildly different selectivity per scene

Fractions of each scene's own entity pairs / entities, computed on the
imported entity set the extractors actually consume.

| measure | room_0 | room_1 | room_2 | office_0 | frl_apartment_0 | apartment_0 |
|---|---|---|---|---|---|---|
| pairs within sparse_max_distance = 2.5 m | 0.280 | 0.356 | 0.363 | 0.424 | 0.188 | 0.080 |
| pairs within NEAR threshold = 1.0 m | 0.043 | 0.070 | 0.126 | 0.102 | 0.079 | 0.012 |
| pairs REJECTED by LEGACY_MIN_DELTA = 0.3 m | 0.013 | 0.015 | 0.022 | 0.032 | 0.016 | 0.002 |
| pairs REJECTED by sparse_min_delta = 0.5 m | 0.025 | 0.026 | 0.057 | 0.047 | 0.036 | 0.004 |
| entities below elevated_bottom_min_m = 0.30 m | 0.288 | 0.244 | 0.340 | 0.319 | 0.320 | 0.731 |

- `sparse_max_distance = 2.5 m` admits **42.4% of office_0's pairs but only
  8.0% of apartment_0's** — a 5.3× swing in what the same constant does.
  This is the density guardrail the docstring was calibrated against, and it
  is not doing the same job on a different scene.
- `NEAR = 1.0 m` swings 10.5× (12.6% on room_2, 1.2% on apartment_0).
- `elevated_bottom_min_m = 0.30 m` is the interesting counterexample: its
  *fraction of storey height* is nearly constant (Tier 3 above), yet the
  share of entities below it jumps to **73.1% on apartment_0** versus
  24–34% elsewhere. That is not a scale effect — apartment_0's entity set is
  dominated by small floor-level objects. **A constant can be
  scale-invariant and still transfer badly**, because content distribution,
  not just room size, changes between datasets. Rescaling alone will not fix
  this one.

---

## Limitations

- Six scenes, one dataset, one capture rig. Everything above characterises
  variation *within* Replica. Cross-dataset variation is expected to be
  larger, and the object-scale spread in particular is a property of how
  Replica annotates instances (office_0's small desk clutter vs room_1's
  furniture-only labelling).
- The axis estimator's premise (parallel horizontal area beats any single
  wall) holds here with margins of 1.02×–1.74×. It is not proven for
  corridors, atria, or heavily furnished single-wall scenes.
- `median object diagonal` comes from the imported Habitat OBB→AABB boxes,
  not from the mesh. It is the right denominator because it is what the
  extractors see, but it inherits any labelling bias in the dataset.
- Yaw for a multi-room scene (apartment_0, frl_apartment_0) is not
  well-defined as a single number; the value is reported anyway and should
  be read as "the dominant wall direction", not "the room's rotation".
- `apartment_0` is two storeys. Single-floor-plane and single-room-diagonal
  abstractions do not describe it, and its numbers should be read with that
  in mind.

## What this does and does not authorise

It establishes: the frame is measurable and currently mislabelled; the
calibrated floor plane is correct; the scale spread across six same-dataset
scenes is 2.5× (room), 4.3× (object), 1.1× (storey height); and which
constants sit where in that spread.

It does not authorise retuning any threshold. A later change acting on this
should decide, per constant, whether it is a *scale* quantity (rescale),
a *sensor-resolution* quantity (leave alone — see the 6 mm caution above),
or a *content-distribution* quantity (rescaling will not help). And it
should relabel edges computed in the gravity-aligned, yaw-de-rotated frame
as `frame="scene_canonical"`, which is what they are.

---

## Follow-up (added after this audit; this document is otherwise unedited)

`docs/frame_decision.md` acts on **Findings 1 and 4 only**: edges now carry the
frame they were computed in, and `importers/replica.py` levels tilted captures
instead of refusing them, so it and `demo/replica_habitat_import.py` no longer
disagree about whether room_2 is importable. No threshold was retuned —
Findings 5–7 remain unacted-on, and the scorecard and all six bundle hashes are
unchanged.

Two things above are now stale, and are left as written because they record
what was measured at the time:

- Table 1's "`+Z up` guard in importers/replica.py" column. That predicate
  (`_gravity_is_neg_z`) still exists and still evaluates the same way, but it
  no longer decides whether a scene imports.
- "the canonical importer refuses room_2 outright" under Finding 1. It no
  longer does.
