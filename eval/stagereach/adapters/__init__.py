"""StageReach3D source adapters.

Each adapter reads exactly one frozen, committed evidence source and derives
per-question StageReach traces from fields that source states. Adapters do
not score themselves (house rule, adapters/base.py): all counting lives in
eval/stagereach/metrics.py, and adapters never infer a stage the source does
not state — an unstated stage is observed as `unknown`.
"""
