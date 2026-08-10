# UID confirmation sheet — arkitscenes_41069025

Regions: **35** delivered. Bundle `ent_openclip_global_indoor_labels_35de101d48b05a87`.

Candidate descriptors are **geometry only**. The system's predicted labels are deliberately not shown, so a model guess cannot become ground truth by suggestion.

## Section 1 — human object facts (confirmed; not up for revision here)

| object | confirmed count | source |
|---|---|---|
| rug | 1 | owner confirmation 2026-08-10 |
| trash can | 1 | owner confirmation 2026-08-10 |
| main kitchen counter | 1 | owner confirmation 2026-08-10 |
| visible sofa cushion | 2 | owner confirmation 2026-08-10 |

These are scene facts. They stay true whatever the segmentation does.

## Section 2 — predicted-UID correspondence (what needs your answer)

Separate from Section 1 on purpose: this mapping is revisable and dies with the current segmentation.

For each row choose ONE:

- `uid = obj_N` — this delivered region is the object
- `none/missing` — the object is in the room but was not delivered
- `overmerged into obj_N` — it is inside a bigger region covering other things too
- `ambiguous` — cannot tell from the views available

**rug** (expect 1)

- [ ] instance: uid = `______`  |  [ ] none/missing  |  [ ] overmerged into `______`  |  [ ] ambiguous

**trash can** (expect 1)

- [ ] instance: uid = `______`  |  [ ] none/missing  |  [ ] overmerged into `______`  |  [ ] ambiguous

**main kitchen counter** (expect 1)

- [ ] instance: uid = `______`  |  [ ] none/missing  |  [ ] overmerged into `______`  |  [ ] ambiguous

**visible sofa cushion** (expect 2)

- [ ] instance #1: uid = `______`  |  [ ] none/missing  |  [ ] overmerged into `______`  |  [ ] ambiguous
- [ ] instance #2: uid = `______`  |  [ ] none/missing  |  [ ] overmerged into `______`  |  [ ] ambiguous

## Section 3 — delivered regions, geometry only (reference)

Sorted largest first. `under` is the height of the region's underside above the lowest point in the scene — 0.00 means it sits on the floor.

| uid | verts | w×d×h (m) | footprint m² | under (m) |
|---|---|---|---|---|
| `obj_8` | 132173 | 7.38×5.95×0.2 | 43.88 | 2.29 |
| `obj_24` | 33889 | 2.8×1.5×2.05 | 4.19 | 0.1 |
| `obj_27` | 33769 | 1.49×2.0×0.9 | 2.99 | 0.11 |
| `obj_13` | 28460 | 1.86×1.76×0.84 | 3.28 | 0.09 |
| `obj_7` | 26450 | 1.12×1.72×2.23 | 1.92 | 0.12 |
| `obj_36` | 25597 | 0.93×1.07×1.77 | 0.99 | 0.64 |
| `obj_10` | 15501 | 0.65×0.98×2.25 | 0.63 | 0.11 |
| `obj_14` | 14019 | 1.26×1.13×0.87 | 1.42 | 0.09 |
| `obj_19` | 13137 | 0.88×0.57×2.26 | 0.5 | 0.11 |
| `obj_16` | 10553 | 2.67×1.79×0.62 | 4.77 | 0.53 |
| `obj_32` | 9289 | 0.48×0.81×2.29 | 0.39 | 0.0 |
| `obj_12` | 7812 | 0.81×0.55×0.66 | 0.44 | 0.12 |
| `obj_18` | 7159 | 2.06×1.06×0.86 | 2.17 | 1.5 |
| `obj_5` | 6736 | 0.92×0.89×0.2 | 0.82 | 0.67 |
| `obj_22` | 6277 | 1.35×4.24×0.24 | 5.73 | 2.26 |
| `obj_34` | 6129 | 1.14×1.0×0.44 | 1.14 | 0.11 |
| `obj_2` | 5894 | 0.71×0.54×0.71 | 0.39 | 0.09 |
| `obj_4` | 5247 | 0.73×0.6×0.58 | 0.44 | 1.21 |
| `obj_0` | 4490 | 0.53×0.43×0.52 | 0.23 | 0.38 |
| `obj_17` | 4217 | 0.6×0.81×1.88 | 0.48 | 0.18 |
| `obj_25` | 4190 | 0.39×0.96×1.93 | 0.37 | 0.12 |
| `obj_15` | 3931 | 0.7×0.68×0.47 | 0.47 | 0.09 |
| `obj_6` | 3545 | 0.62×0.4×0.75 | 0.25 | 1.08 |
| `obj_21` | 2975 | 0.42×0.24×1.07 | 0.1 | 0.22 |
| `obj_1` | 2797 | 0.35×0.35×0.32 | 0.12 | 0.1 |
| `obj_28` | 2430 | 0.42×0.64×0.35 | 0.27 | 0.22 |
| `obj_20` | 2069 | 0.82×1.34×1.34 | 1.1 | 0.85 |
| `obj_3` | 2037 | 0.29×0.34×0.36 | 0.1 | 1.47 |
| `obj_23` | 1972 | 0.49×0.42×0.35 | 0.21 | 0.55 |
| `obj_11` | 1595 | 0.75×0.87×0.83 | 0.65 | 1.46 |
| `obj_26` | 1478 | 0.16×0.22×1.34 | 0.04 | 0.13 |
| `obj_9` | 1471 | 0.48×0.46×0.25 | 0.22 | 0.56 |
| `obj_33` | 1053 | 0.82×0.97×0.34 | 0.8 | 2.11 |
| `obj_43` | 220 | 0.31×0.32×0.8 | 0.1 | 1.51 |
| `obj_29` | 39 | 0.09×0.1×0.03 | 0.01 | 0.14 |

---

Return the filled Section 2 (or the JSON skeleton beside this file). The key is finalized and scored once, after that.
