"""Serde for CompletenessProfile and ExecutionContext.

These are configuration artifacts attached to eval runs, not 'bundles' in
the data-producing-stage sense. They round-trip so a calibration run's
output can be persisted and reloaded by later eval runs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, get_args

from graph.schema import EdgeType
from reasoner.base import CompletenessProfile, ExecutionContext

_VALID_SOURCES = ("oracle", "measured", "unknown")
_VALID_EDGE_TYPES = frozenset(get_args(EdgeType))


def _check_edge_type(t: str) -> str:
    if t not in _VALID_EDGE_TYPES:
        raise ValueError(f"unknown EdgeType {t!r}")
    return t


def completeness_to_dict(c: CompletenessProfile) -> dict[str, Any]:
    return {
        "source": c.source,
        "entity_recall_by_class": dict(c.entity_recall_by_class),
        "edge_recall_by_type": dict(c.edge_recall_by_type),
        "calibration_dataset": c.calibration_dataset,
    }


def completeness_from_dict(d: dict[str, Any]) -> CompletenessProfile:
    source = d["source"]
    if source not in _VALID_SOURCES:
        raise ValueError(f"unknown CompletenessProfile source {source!r}")
    return CompletenessProfile(
        source=source,
        entity_recall_by_class={
            str(k): float(v) for k, v in d.get("entity_recall_by_class", {}).items()
        },
        edge_recall_by_type={
            _check_edge_type(str(k)): float(v)
            for k, v in d.get("edge_recall_by_type", {}).items()
        },
        calibration_dataset=d.get("calibration_dataset"),
    )


def dump_completeness_profile(c: CompletenessProfile, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(completeness_to_dict(c), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return out_path


def load_completeness_profile(in_path: Path) -> CompletenessProfile:
    return completeness_from_dict(json.loads(in_path.read_text(encoding="utf-8")))


def dump_execution_context(ctx: ExecutionContext, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "completeness": completeness_to_dict(ctx.completeness),
        "empty_recall_threshold": ctx.empty_recall_threshold,
        "answer_tau": ctx.answer_tau,
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def load_execution_context(in_path: Path) -> ExecutionContext:
    """Note: ExecutionContext.rejections is deliberately NOT round-tripped.
    It is per-build evidence (BuildDiagnostics.rejection_samples), not
    calibration configuration, and it is re-attached by whoever built the
    graph. A loaded context therefore always has rejections=()."""
    payload = json.loads(in_path.read_text(encoding="utf-8"))
    return ExecutionContext(
        completeness=completeness_from_dict(payload["completeness"]),
        empty_recall_threshold=float(payload.get("empty_recall_threshold", 0.95)),
        answer_tau=float(payload.get("answer_tau", 0.0)),
    )
