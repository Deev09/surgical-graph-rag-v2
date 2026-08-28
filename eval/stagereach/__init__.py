"""StageReach3D — a schema-driven, stage-resolved reachability evaluator.

Implements docs/stagereach_schema_freeze.md (the frozen design, 2026-08-28).
Pure library: schema.py declares vocabularies, paths and trace dataclasses;
evaluator.py applies gating and validates invariants; metrics.py computes
survival ladders and outcome matrices. No module here performs I/O — file
reading lives in eval/stagereach/adapters/ and tools/stagereach_eval.py,
following the house rule from adapters/base.py: adapters do not score
themselves, and scorers do not read files.
"""
