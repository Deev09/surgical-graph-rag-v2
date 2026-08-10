# Real capture-RGB crops as the labeler input — measured results

Development comparison, not a protocol. Constants are recorded so a run can
be reproduced or superseded; nothing here is frozen.

Reproduce:

    python3 tools/arkitscenes_label_image_ab.py                     # dev
    python3 tools/arkitscenes_label_image_ab.py \
        --scene-dir <ARKitScenes>/Validation/<vid> \
        --segmentation-dir runs/arkitscenes_mask3d_transfer/bundle_arkitscenes_<vid> \
        --out runs/arkit_label_image_ab_<vid> --arms splat rgb_tight

(the label stage needs `open_clip` + `torch`; the project venv is
deliberately numpy + Pillow only.)

## What varied

**Only the images.** Same OpenCLIP ViT-B-32/openai weights, same 41-class
`GLOBAL_INDOOR_VOCABULARY_V1`, same `top_k=3`, same `min_top1_score=0.28`,
same evaluator, same delivered Mask3D partitions, same greedy IoU≥0.50
matching.

| arm | images |
|---|---|
| `splat` | three isolated point-splat renders of the instance alone (previous behaviour) |
| `rgb_tight` | real capture RGB, `context_pad=0.15`, target not marked |
| `rgb_context` | real capture RGB, `context_pad=0.60`, outside-target dimmed to 0.7 |

## Results

| scene | arm | top-1 | top-3 | admitted | adm. precision | 3-view coverage |
|---|---|---|---|---|---|---|
| 41069021 (dev) | splat | 0/7 | 0/7 | 1 | 0.00 | — |
| 41069021 | **rgb_tight** | **5/7** | **7/7** | 7 | **0.71** | 34/34 (med 0.99) |
| 41069021 | rgb_context | 3/7 | 5/7 | 7 | 0.43 | 34/34 |
| 41069025 (sealed) | splat | 1/9 | 4/9 | 5 | 0.20 | — |
| 41069025 | **rgb_tight** | **5/9** | **8/9** | 9 | **0.56** | 35/35 (med 0.97) |
| 41069042 (sealed) | splat | 0/5 | 0/5 | 3 | 0.00 | — |
| 41069042 | **rgb_tight** | **2/5** | **3/5** | 4 | **0.50** | 23/23 (med 0.99) |
| **pooled** | splat | **1/21** | **4/21** | | | |
| **pooled** | **rgb_tight** | **12/21** | **18/21** | | | |

No tuning between scenes: identical tool and constants, only `--scene-dir`
and `--segmentation-dir` differ. Every `splat` row reproduces its committed
baseline exactly, which is the evidence that the harness did not move.

## The control that makes this interpretable

`rgb_tight` **beats** `rgb_context` (5/7 vs 3/7). The tight arm exists to
catch the confound that would make a naive RGB win worthless: a wide crop of
a doorway reads as a kitchen, so a gain could have been CLIP recognising
rooms rather than objects. More context measurably hurts, so the gain comes
from real texture on the object.

Verified by inspection as well as score: the winning dev crops are
unambiguous photographs of a chair, a white round table, a dark sofa, a
wall-mounted TV and a glass coffee table. Both dev misses are near-synonyms
recovered in top-3 — cabinet read as `shelf`, a narrow console table read as
`counter`.

## Qualitative audit against the owner's visual review

Used as a **fixed audit**, never as tuning input.
Source: `eval/human_feedback/arkitscenes_sealed_visual_review_2026-08-09.json`.

| uid | truth | splat | rgb_tight | |
|---|---|---|---|---|
| obj_0 | chair | stool | **chair** | fixed |
| obj_5 | table | counter | **table** | fixed |
| obj_18 | cabinet | stool | **cabinet** | fixed |
| obj_6 | bed | counter | **bed** | fixed |
| obj_14 | tv-monitor | counter | **tv-monitor** | fixed |
| obj_13 | sofa | projector | cushion | still wrong |
| obj_36 | cabinet | trash-can | microwave | still wrong |
| obj_34 | table | table | stool | **regressed** |

`obj_13` is the sofa the owner corrected personally. RGB does not fix it,
but the error moves from an implausible class to an adjacent one, plausibly
reading the cushions the same review identified as `obj_9`/`obj_23`.

**RGB is not uniformly better per instance — it is better in aggregate.**
`obj_34` was correct under splats and is wrong under RGB. Reporting the
pooled gain without that row would overstate the result.

## Limits

* 21 matched instances across three scenes. Screening evidence.
* Oracle is annotation-box-derived vertex sets; classes absent from the
  41-class vocabulary (notably `curtain`) cannot be scored correctly by any
  arm.
* `min_top1_score=0.28` was chosen against splat scores. It is retained for a
  controlled comparison and is **not** calibrated for RGB.
* Matched-instance accuracy only. **End-to-end QA has not been run** — the
  RGB image source is not yet wired into `tools/arkit_vertical_slice.py`,
  which still labels from splats.
* This fixes naming, not detection. It cannot split the overmerged plane or
  add an instance Mask3D never proposed.

## Two bugs that produced earlier invalid numbers

Recorded because both were silent and both produced confident wrong results.

1. **Wrong camera convention.** ARKitScenes poses are plain pinhole/OpenCV;
   an ARKit-native `[1,-1,-1]` axis flip was applied. Fixed in
   `codex/camera-convention`, verified against synchronized sensor depth
   (1.4–5.2 cm median error direct, 23.8–68.3 cm flipped).
2. **Stride applied before pose matching.** `load_frames` strided the raw
   60 Hz frame list, then filtered to the ~10 Hz posed subset, intersecting
   two unrelated subsamples and leaving 122 usable frames instead of 1878.

Under the wrong projection, coverage still read 34/34 with median visible
fraction 0.99 — identical to the correct run. **Coverage statistics cannot
validate the geometry that produced them.** Only an independent instrument
(sensor depth) could.
