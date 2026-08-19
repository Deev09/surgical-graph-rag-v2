# Direct multiview RGB as the product answer path

Decision taken 2026-08-19, on measured results across three experiments. This
document is the entry point for the product path; the graph-centered path is
closed and preserved as the negative comparison.

## What is established

All four figures below are read from
`runs/arkit_relation_challenge/report.json`, scored once on the owner-confirmed
key over 10 scored items (2 of the 12 excluded as owner-ambiguous).

| arm | correct | coverage | deployable |
|---|---:|---:|---|
| **direct multiview RGB** | **7 / 10** | **0.90** | yes |
| stored graph + human identity | 7 / 10 | 0.80 | **no** — identity oracle |
| grounded delivered graph | 2 / 10 | 0.20 | yes |
| delivered graph (exact label match) | 0 / 10 | 0.00 | yes |

Read together these say one thing: **useful spatial information exists in the
graph and is not deployably reachable.** The identity oracle matches direct RGB
on correctness, so the content is there; every deployable route to it scores at
or near zero.

Supporting results, each measured rather than argued:

- Relation extraction is **cleared** on this slice — the stored-edge replay
  matched the recomputed geometry ceiling 12/12, so nothing is lost between the
  delivered boxes and the serialized graph.
- The pinned OpenCLIP crop-based grounding bridge **failed every predeclared
  gate**: precision 0.583 against ≥0.80, coverage 0.467 against ≥0.60, and 0
  graph-unique wins against ≥2.
- The striped rug and the white radiator are **instance-delivery failures**,
  not grounding or relation failures. No grounding mechanism can reach an
  object that was never delivered.

### Wording guard

The stop closes **the pinned OpenCLIP crop-based grounding bridge**, not all
possible grounding research. The stop rule's purpose is to prevent endless
variants being tried against the same seventeen anchors, each selected on
numbers it has already seen. That is a discipline about this key, not a claim
about the field.

## The product path

1. **Keep the existing structured response contract** — `answer` / `unknown`,
   `confidence`, and cited frame ids. It already exists, is schema-validated,
   and its abstention semantics are what make an honest demo possible.
2. **Show the cited RGB frames beside every answer.** The citation is the
   evidence; an answer shown without it is a claim shown without it.
3. **Retain the 3D viewer as an inspectable evidence and state layer,
   explicitly not the primary answer engine.** It must be labelled as such
   wherever it appears, for the same reason a ceiling number is never quoted as
   deployable performance.
4. **Preserve the graph results as the measured negative comparison.** They are
   the reason the product is shaped this way, and deleting them would leave the
   design looking like a preference instead of a finding.
5. **Carry the grounding failure and the ceiling-versus-deployable distinction
   into the paper narrative.** The interesting claim is not "RGB wins"; it is
   that a representation can hold correct information that no deployable query
   path can reach, and that the gap is identity, not geometry or relations.

## Two implementation options

| | precomputed sidecars | live vision API |
|---|---|---|
| dependency | none at demo time | credentials, network |
| determinism | byte-reproducible | varies per call |
| cost | zero at runtime | per-query, needs caps |
| failure modes | none at runtime | timeouts, rate limits, outages |
| answers | fixed question set only | arbitrary questions |
| blinding | preserved by construction | must be re-established per run |

**Recommended sequencing, owner's call and recorded here:**

1. Ship the **precomputed demo first**. No API key, no runtime dependency, and
   it reuses the blinded responses already collected and hash-pinned.
2. Then run direct RGB **once** on the untouched `47331972` scene, for actual
   transfer evidence. One scene, one run, blinded the same way.
3. Do **not** spend another cycle tuning graph grounding before that.

Step 2 is the first genuine transfer test in this line of work: every result so
far lives on `41069025` and `41069042`, and `47331972` has never been
downloaded or looked at.

## Not started

Neither implementation option has been started, and `47331972` has not been
downloaded. The option fork is a product decision with a credentials
implication, so it is recorded here rather than chosen unilaterally.

## What would reopen the graph path

Not a better prompt or a swept threshold. Concretely: a persisted per-entity
embedding written at delivery time — `embedding_ref` is `None` on every entity
today, which is why the bridge had to re-encode crops at query time — or an
instance-delivery stage that actually delivers the objects a human names. Both
are upstream of everything measured here, and neither is licensed by these
results; they are hypotheses to be paid for with a new key, not this one.
