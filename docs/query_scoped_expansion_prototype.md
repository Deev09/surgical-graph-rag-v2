# Query-scoped raw-proposal expansion prototype

Date: 2026-07-26.

## Outcome

The prototype works mechanically, but it does **not** improve the tested support
answers.

- Mask3D: no eligible raw proposal satisfies the fixed tabletop geometry on any
  of the four saved scenes.
- Segment3D room_2: seven raw proposals satisfy the shelf-local predicate and
  local recomposition changes which predicted masks are cited, but the
  oracle-space answer stays unchanged: one correct vase plus the same
  indoor-plant false citation, with two reference vases still missed.
- Global inclusion adds no recovered support answer and, for the Segment3D
  shelf case, grows the graph from 645 to 1,360 edges.

This is a useful negative result. It rules out “lower the threshold around the
support anchor” as a sufficient fix for these saved proposals under the frozen
rest-contact semantics.

## Interface and isolation

The pipeline is:

```text
support query
  -> RulesCompiler: EntityClassRef anchor
  -> accepted hard-graph support surfaces
  -> candidate pool from independent raw masks
       - score-provisional masks
       - accepted masks swallowed during composition
  -> geometry-only rest-contact selection
  -> query-local recomposition
       - selected raw masks may reclaim non-anchor vertices
       - accepted anchor masks are protected
  -> existing EntityArtifacts -> graph builder -> Router
```

Candidate selection does not read oracle labels, expected answers, or relation
edges. Replica labels and surfaces are injected only after selection through the
existing C1 evaluation path. The hard graph uses an `unknown` completeness
profile, so a missed learned answer is reported as unknown rather than a
confident empty.

At the time of this experiment (2026-07-26) no Phase 8 key had been promoted
to `human_verified`. (Three keys — room_0/room_1/room_2 — landed 2026-07-31;
this document's evaluation predates them and was NOT rescored.) Evaluation
therefore uses two diagnostics:

1. Habitat JSON boxes (reference A);
2. semantic-mesh boxes (reference B).

A reference answer is called stable only when A and B agree. These diagnostics
do not authorize an accuracy claim.

## Backend score contract

Thresholds remain in native backend score space:

- Mask3D: `identity_probability`;
- Segment3D: `sigmoid_logit` for evidence accumulation, while the frozen
  resolver thresholds still apply to the native logits.

Empty scored Segment3D masks remain evidence records but are excluded from the
activation pool because they contain no geometry.

## Real results

### Mask3D table query

Fixed query: `what is on the table?`; accepted score 0.2, provisional floor
0.05, minimum 20 vertices.

| scene | hard anchors | raw activation candidates | selected | hard answer | scoped answer |
|---|---:|---:|---:|---|---|
| office_0 | 2 | 8 | 0 | unknown | unknown |
| room_2 | 1 | 10 | 0 | unknown | unknown |
| room_1 | 0 | 17 | 0 | unknown | unknown |
| frl_apartment_0 | 7 | 157 | 0 | obj_112, obj_113, anonymous obj_93 | unchanged |

The room_2 A/B reference is stable at `obj_14` and `obj_18`; neither is
recovered. The frl A/B reference is stable at `obj_112` and `obj_113`; the hard
graph already returns both plus an unmatched anonymous citation, and expansion
does not change it. Office A and B disagree, so no stable key is inferred.

### Segment3D room_2 support queries

Native accepted score 0.2, provisional floor 0.05, sigmoid evidence transform,
minimum 20 vertices. The raw pool contains 85 non-empty candidates absent from
the hard graph: 38 score-provisional and 47 accepted-but-composition-lost.

| query | anchors | selected raw masks | hard oracle-space answer | scoped oracle-space answer | stable A/B reference |
|---|---:|---:|---|---|---|
| table | 1 | 0 | unknown | unknown | obj_14, obj_18 |
| shelf | 1 | 7 | obj_17, obj_47 | unchanged | obj_17, obj_27, obj_46 |
| chair | 6 | 0 | unknown | unknown | obj_56 |

For the shelf query, masks `40, 56, 80, 91, 124, 125, 162` pass local
rest-contact. Only `40` and `56` survive recomposition; they displace hard masks
`0` and `13`. The graph hash and raw cited prediction ids change, proving that
the activation path executed. After prediction-to-oracle translation, however,
the cited set remains `obj_17, obj_47`; diagnostic recall remains 1/3 and
precision 1/2.

## Interpretation

The bottleneck is not simply eager thresholding:

- Mask3D lacks task-relevant raw support proposals in the provisional and
  composition-lost pool.
- Segment3D contains local alternatives, but they are alternate masks for the
  same recovered objects rather than the two missing shelf objects.
- Inclusive materialization increases graph cost without changing these
  answers.

PUF-style uncertainty remains useful for honest state and abstention, while
JITOMA-style local expansion needs a perception operation capable of creating
new evidence—not merely reselecting the current raw masks.

The next defensible experiment is local re-perception or fragment construction
around a support region, guarded by reprojection/mask consistency. Another
threshold sweep over the same masks is not justified by this result.

## Reproduction

Mask3D example:

```bash
./.venv/bin/python tools/query_scoped_expansion_demo.py \
  ~/Desktop/datasets/replica/room_2 \
  notebooks/bundle_room_2 \
  replica_room_2 \
  --query "what is on the table?" \
  --score-transform identity_probability
```

Segment3D shelf example:

```bash
./.venv/bin/python tools/query_scoped_expansion_demo.py \
  ~/Desktop/datasets/replica/room_2 \
  notebooks/s3d_bundle_room_2 \
  replica_room_2_segment3d \
  --query "what is on the shelf?" \
  --score-transform sigmoid_logit
```

Reports are written under `runs/phase8_c1/*_query_expansion.json`.
