"""Anthropic Claude vision client for spatial-QA evaluation.

Sends a question + scene images + scene canonical-id list to Claude and forces
structured output via the tool-use API. Returns a RunnerOutput-shaped dict:

    {
      "answer_entity_ids": [str, ...],
      "abstained": bool,
      "answer_text": str,
      "evidence": {
        "entity_ids": [str, ...],
        "source_frame_idx": int | null,
        "crop_bbox": [x, y, w, h] | null,
        "confidence": float in [0, 1]
      },
      "raw_response": str       # debug aid; not part of scoring
    }

Bumps PROMPT_TEMPLATE_VERSION when the schema or system prompt changes —
DiskCache uses it as part of the key, so old entries are invalidated.

MockVLMClient returns deterministic stub responses without calling the API.
Use it (eval_vlm.py --mock) to validate the runner pipeline end-to-end without
spending money or needing an API key.

ABSTENTION (P-SEL)
------------------
Comparing a calibrated system against a VLM that is forced to always answer
is a strawman. The VLM baseline must be able to abstain, and must emit a raw
per-question confidence so it can be scored on the same risk-coverage curve
(eval/selective.py). Two independent mechanisms are provided:

  1. VERBALIZED CONFIDENCE (VerbalizedConfidenceVLMClient, template v0.2-conf)
     One call. The tool schema requires a top-level `confidence` in [0,1] and
     the system prompt states that abstaining is a valid, unpenalised answer.
     Signal recorded as `confidence_verbalized`.

  2. SELF-CONSISTENCY (SelfConsistencyVLMClient, template <base>+sc<N>)
     N sampled calls through any base client. Confidence = the agreement
     rate of the modal answer set. Signal recorded as
     `confidence_self_consistency`, with all N normalized samples kept.
     NOTE: sampling diversity needs temperature. `claude-haiku-4-5` accepts
     `temperature`; Opus 5 / Sonnet 5 / Opus 4.7+ REJECT it with a 400, so on
     those models the only source of variation is natural nondeterminism and
     the agreement rate will be near-degenerate. `temperature` is only sent
     when explicitly set.

Both signals are recorded raw. Thresholding happens downstream in
eval/selective.py, never here.

PROMPT_TEMPLATE_VERSION stays "v0.1" so the pre-existing baseline artifact
under runs/vlm/ remains byte-reproducible; the confidence path declares its
own version, and DiskCache keys on it, so the two never collide.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


PROMPT_TEMPLATE_VERSION = "v0.1"
PROMPT_TEMPLATE_VERSION_CONF = "v0.2-conf"
DEFAULT_MODEL_ID = "claude-haiku-4-5-20251001"


SYSTEM_PROMPT = """You are a careful visual-spatial QA system. You answer questions about a physical room shown in the provided photos.

Rules:
- Answer using only the canonical entity IDs from the list the user provides. If the answer is not one of those IDs, abstain.
- It is OK to abstain (set "abstained": true and return an empty answer_entity_ids list) when the photos do not let you decide.
- For multi-instance questions, return all matching IDs.
- For "where is X?" questions, return the entity IDs corresponding to X.
- Cite the photo whose 0-based index most clearly shows your answer in evidence.source_frame_idx.
- Always submit your answer via the submit_answer tool. Do not respond in plain text.
"""


def _entity_block(scene_objects: list[dict[str, Any]]) -> str:
    lines = []
    for o in scene_objects:
        cid = str(o["label"])
        m = re.match(r"^(.*)_\d+$", cid)
        display = m.group(1).replace("_", " ") if m else cid.replace("_", " ")
        lines.append(f"  - {cid}  (a {display})")
    return "\n".join(lines)


USER_PROMPT_TEMPLATE = """Question: {question}

Available canonical entity IDs in this scene (you must use ONLY these in your answer):
{entities_block}

Expected answer_type: {answer_type}
Question category: {category}

The photos below show the same room from different viewpoints. They are 0-indexed in the order shown.

Submit your answer via the submit_answer tool."""


ANSWER_TOOL_SCHEMA: dict[str, Any] = {
    "name": "submit_answer",
    "description": "Submit the final spatial-QA answer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer_entity_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ordered list of canonical entity IDs that answer the question. Empty list if abstaining.",
            },
            "abstained": {
                "type": "boolean",
                "description": "True if the question cannot be answered from the photos.",
            },
            "answer_text": {
                "type": "string",
                "description": "Short natural-language explanation of the answer.",
            },
            "evidence": {
                "type": "object",
                "properties": {
                    "entity_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Canonical IDs cited as evidence (typically same as answer_entity_ids).",
                    },
                    "source_frame_idx": {
                        "type": ["integer", "null"],
                        "description": "0-based index of the photo most clearly showing the answer, or null.",
                    },
                    "crop_bbox": {
                        "type": ["array", "null"],
                        "items": {"type": "number"},
                        "description": "[x, y, w, h] normalized to [0,1] for the relevant region in source frame, or null.",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Confidence in the answer.",
                    },
                },
                "required": ["entity_ids", "confidence"],
            },
        },
        "required": ["answer_entity_ids", "abstained", "answer_text", "evidence"],
    },
}


# --------------------------------------------------------------------------
# Mechanism 1: verbalized confidence (prompt template v0.2-conf)
# --------------------------------------------------------------------------

SYSTEM_PROMPT_CONFIDENCE = """You are a careful visual-spatial QA system. You answer questions about a physical room shown in the provided photos.

Rules:
- Answer using only the canonical entity IDs from the list the user provides. If the answer is not one of those IDs, abstain.
- For multi-instance questions, return all matching IDs.
- For "where is X?" questions, return the entity IDs corresponding to X.
- Cite the photo whose 0-based index most clearly shows your answer in evidence.source_frame_idx.
- Always submit your answer via the submit_answer tool. Do not respond in plain text.

Abstention and confidence:
- Abstaining is a FIRST-CLASS answer, not a failure. If the photos do not let you decide -- occlusion, ambiguity, the object is not visible, or you cannot tell two candidate objects apart -- set "abstained": true, return an empty answer_entity_ids list, and say why in abstain_reason.
- You are NOT rewarded for guessing. A wrong answer is worse than an abstention.
- Report "confidence": your honest probability in [0, 1] that the answer you submitted is exactly correct. Calibrate it: of all the answers you would give a confidence of 0.8, about 80% should turn out correct. Do not default to round or high numbers. Low confidence on a hard question is the correct behaviour.
- Report confidence even when abstaining: there it means your probability that abstaining is the right call.
"""


ANSWER_TOOL_SCHEMA_CONFIDENCE: dict[str, Any] = {
    "name": "submit_answer",
    "description": "Submit the final spatial-QA answer with a calibrated confidence.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer_entity_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ordered list of canonical entity IDs that answer the question. Empty list if abstaining.",
            },
            "abstained": {
                "type": "boolean",
                "description": "True if the question cannot be answered from the photos.",
            },
            "abstain_reason": {
                "type": ["string", "null"],
                "description": "Why you abstained (occlusion / not visible / ambiguous / cannot disambiguate). Null if you answered.",
            },
            "answer_text": {
                "type": "string",
                "description": "Short natural-language explanation of the answer.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Calibrated probability in [0,1] that the submitted answer is exactly correct (or, if abstaining, that abstaining is correct).",
            },
            "evidence": {
                "type": "object",
                "properties": {
                    "entity_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Canonical IDs cited as evidence (typically same as answer_entity_ids).",
                    },
                    "source_frame_idx": {
                        "type": ["integer", "null"],
                        "description": "0-based index of the photo most clearly showing the answer, or null.",
                    },
                    "crop_bbox": {
                        "type": ["array", "null"],
                        "items": {"type": "number"},
                        "description": "[x, y, w, h] normalized to [0,1] for the relevant region in source frame, or null.",
                    },
                },
                "required": ["entity_ids"],
            },
        },
        "required": [
            "answer_entity_ids", "abstained", "answer_text", "confidence", "evidence",
        ],
    },
}


def normalize_answer_key(output: dict[str, Any]) -> tuple[str, ...]:
    """Canonical key for one sampled answer, used to measure self-consistency
    agreement. An abstention is its own bucket -- two abstentions agree with
    each other, and an abstention never agrees with an answer.

    Order-insensitive: the question sets are unordered, so ["a","b"] and
    ["b","a"] must not count as disagreement.
    """
    if not output:
        return ("__error__",)
    if bool(output.get("abstained", False)):
        return ("__abstain__",)
    ids = output.get("answer_entity_ids") or []
    return tuple(sorted(str(x) for x in ids))


@dataclass
class VLMResponse:
    output: dict[str, Any]            # parsed structured answer
    raw_response: str                 # debug; the assistant's full message text or block dump
    latency_ms: float
    tokens: dict[str, int] | None = None
    error: str | None = None
    # --- P-SEL: raw abstention/confidence signals, never thresholded here ---
    confidence_verbalized: float | None = None
    confidence_self_consistency: float | None = None
    samples: list[dict[str, Any]] = field(default_factory=list)


def _clamp_unit(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return max(0.0, min(1.0, v))


def extract_verbalized_confidence(output: dict[str, Any]) -> float | None:
    """Pull the model's self-reported confidence out of a submit_answer
    payload. Prefers the top-level v0.2 field, falls back to the nested
    evidence.confidence of the v0.1 schema. None if absent/unparseable --
    a missing confidence is recorded as missing, not silently defaulted."""
    if not output:
        return None
    if "confidence" in output:
        v = _clamp_unit(output.get("confidence"))
        if v is not None:
            return v
    ev = output.get("evidence") or {}
    if isinstance(ev, dict):
        return _clamp_unit(ev.get("confidence"))
    return None


class BaseVLMClient(ABC):
    model_id: str
    prompt_template_version: str = PROMPT_TEMPLATE_VERSION
    system_prompt: str = SYSTEM_PROMPT
    tool_schema: dict[str, Any] = ANSWER_TOOL_SCHEMA

    @abstractmethod
    def call(
        self,
        *,
        question_text: str,
        category: str,
        answer_type: str,
        scene_objects: list[dict[str, Any]],
        image_paths: list[Path],
    ) -> VLMResponse: ...

    def build_prompt_payload(
        self,
        *,
        question_text: str,
        category: str,
        answer_type: str,
        scene_objects: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "system": self.system_prompt,
            "user": USER_PROMPT_TEMPLATE.format(
                question=question_text,
                entities_block=_entity_block(scene_objects),
                answer_type=answer_type,
                category=category,
            ),
            "tool_schema_name": self.tool_schema["name"],
            "tool_schema_version": self.prompt_template_version,
        }


class AnthropicVLMClient(BaseVLMClient):
    """Anthropic Claude with vision + tool-use forced output.

    Requires:
      - `pip install anthropic`
      - ANTHROPIC_API_KEY env var
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        max_tokens: int = 1024,
        *,
        temperature: float | None = None,
    ):
        self.model_id = model_id
        self.max_tokens = max_tokens
        # Only forwarded when explicitly set: Opus 4.7+ / Opus 5 / Sonnet 5
        # reject `temperature` with a 400. haiku-4-5 accepts it.
        self.temperature = temperature
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "anthropic SDK not installed. Run: pip install anthropic"
            ) from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Export it before running eval_vlm.py."
            )
        from anthropic import Anthropic
        self._client = Anthropic()

    def _image_block(self, path: Path) -> dict[str, Any]:
        suffix = path.suffix.lower().lstrip(".")
        media_type = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "gif": "image/gif",
            "webp": "image/webp",
        }.get(suffix, "image/png")
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": b64},
        }

    def call(
        self,
        *,
        question_text: str,
        category: str,
        answer_type: str,
        scene_objects: list[dict[str, Any]],
        image_paths: list[Path],
    ) -> VLMResponse:
        payload = self.build_prompt_payload(
            question_text=question_text,
            category=category,
            answer_type=answer_type,
            scene_objects=scene_objects,
        )
        content_blocks: list[dict[str, Any]] = [
            self._image_block(p) for p in image_paths
        ]
        content_blocks.append({"type": "text", "text": payload["user"]})

        kwargs: dict[str, Any] = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        t0 = time.perf_counter()
        try:
            msg = self._client.messages.create(
                model=self.model_id,
                max_tokens=self.max_tokens,
                system=payload["system"],
                tools=[self.tool_schema],
                tool_choice={"type": "tool", "name": "submit_answer"},
                messages=[{"role": "user", "content": content_blocks}],
                **kwargs,
            )
        except Exception as e:
            return VLMResponse(
                output={},
                raw_response="",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                error=f"vlm_format: {type(e).__name__}: {e}",
            )
        latency = (time.perf_counter() - t0) * 1000.0

        tool_use = None
        text_blocks: list[str] = []
        for block in msg.content:
            btype = getattr(block, "type", None)
            if btype == "tool_use" and getattr(block, "name", "") == "submit_answer":
                tool_use = getattr(block, "input", {})
            elif btype == "text":
                text_blocks.append(getattr(block, "text", ""))

        if tool_use is None:
            return VLMResponse(
                output={},
                raw_response="\n".join(text_blocks),
                latency_ms=latency,
                error="vlm_format: no submit_answer tool_use block",
            )

        usage = getattr(msg, "usage", None)
        tokens = None
        if usage is not None:
            tokens = {
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
            }
        out = dict(tool_use)
        return VLMResponse(
            output=out,
            raw_response=json.dumps(tool_use),
            latency_ms=latency,
            tokens=tokens,
            confidence_verbalized=extract_verbalized_confidence(out),
        )


class VerbalizedConfidenceVLMClient(AnthropicVLMClient):
    """Mechanism 1: one call, model reports a calibrated 0-1 confidence.

    Differs from AnthropicVLMClient only in the system prompt and tool
    schema (template v0.2-conf), which
      - state that abstaining is a valid, unpenalised answer,
      - require a top-level `confidence` in [0, 1],
      - add `abstain_reason`.

    Everything else -- caching, scoring, artifact shape -- is unchanged, so
    v0.1 and v0.2-conf runs are A/B-comparable on the same questions. They
    are NOT the same prompt, so accuracy differences between them are a
    prompt change, not a model improvement.
    """

    prompt_template_version = PROMPT_TEMPLATE_VERSION_CONF
    system_prompt = SYSTEM_PROMPT_CONFIDENCE
    tool_schema = ANSWER_TOOL_SCHEMA_CONFIDENCE


class SelfConsistencyVLMClient(BaseVLMClient):
    """Mechanism 2: N sampled responses; confidence = modal agreement rate.

    Wraps any BaseVLMClient. For each question it calls the base client
    n_samples times and computes

        confidence_self_consistency = count(modal answer key) / n_samples

    where the answer key is the order-insensitive set of returned entity
    IDs, with abstention as its own bucket (see normalize_answer_key). The
    returned output is the modal sample; every sample is kept in
    VLMResponse.samples so the raw distribution is auditable.

    Cost: n_samples x the base client's per-question cost. Budget for it.

    Diversity caveat: with temperature unset (or on a model that rejects
    temperature) the N samples may be near-identical and the agreement rate
    collapses to 1.0 for every question -- a degenerate confidence, which
    eval/selective.py will flag rather than silently score.
    """

    def __init__(self, base: BaseVLMClient, n_samples: int = 5):
        if n_samples < 2:
            raise ValueError("self-consistency needs n_samples >= 2")
        self.base = base
        self.n_samples = n_samples
        self.model_id = base.model_id
        self.prompt_template_version = f"{base.prompt_template_version}+sc{n_samples}"
        self.system_prompt = base.system_prompt
        self.tool_schema = base.tool_schema

    def build_prompt_payload(self, **kwargs: Any) -> dict[str, Any]:
        payload = self.base.build_prompt_payload(**kwargs)
        # n_samples is part of the cache identity: a 3-sample and a 5-sample
        # run are different measurements of the same question.
        payload["n_samples"] = self.n_samples
        payload["tool_schema_version"] = self.prompt_template_version
        return payload

    def call(
        self,
        *,
        question_text: str,
        category: str,
        answer_type: str,
        scene_objects: list[dict[str, Any]],
        image_paths: list[Path],
    ) -> VLMResponse:
        t0 = time.perf_counter()
        responses: list[VLMResponse] = []
        for _ in range(self.n_samples):
            responses.append(self.base.call(
                question_text=question_text,
                category=category,
                answer_type=answer_type,
                scene_objects=scene_objects,
                image_paths=image_paths,
            ))
        latency = (time.perf_counter() - t0) * 1000.0

        keys = [normalize_answer_key(r.output) for r in responses]
        counts = Counter(keys)
        # Deterministic tie-break on equal counts: lexicographic key order.
        modal_key, modal_n = sorted(
            counts.items(), key=lambda kv: (-kv[1], kv[0])
        )[0]
        agreement = modal_n / len(keys)
        modal_idx = keys.index(modal_key)
        modal = responses[modal_idx]

        samples = [
            {
                "answer_key": list(k),
                "abstained": bool((r.output or {}).get("abstained", False)),
                "confidence_verbalized": r.confidence_verbalized,
                "error": r.error,
                "latency_ms": r.latency_ms,
            }
            for k, r in zip(keys, responses)
        ]
        tokens = _sum_tokens([r.tokens for r in responses])

        return VLMResponse(
            output=modal.output,
            raw_response=modal.raw_response,
            latency_ms=latency,
            tokens=tokens,
            error=modal.error,
            confidence_verbalized=modal.confidence_verbalized,
            confidence_self_consistency=agreement,
            samples=samples,
        )


def _sum_tokens(all_tokens: Sequence[dict[str, int] | None]) -> dict[str, int] | None:
    present = [t for t in all_tokens if t]
    if not present:
        return None
    return {
        "input_tokens": sum(int(t.get("input_tokens", 0)) for t in present),
        "output_tokens": sum(int(t.get("output_tokens", 0)) for t in present),
    }


class MockVLMClient(BaseVLMClient):
    """Deterministic stub. Always abstains by default. Used by
    eval_vlm.py --mock to validate the runner pipeline end-to-end without API
    spend or photos.

    `scripted` (optional) replaces the always-abstain behaviour with a cycle
    of canned outputs, so the self-consistency and confidence code paths can
    be exercised offline. Leaving it None preserves the exact v0.1 behaviour
    and therefore the existing cached artifacts under runs/vlm/.
    """

    def __init__(
        self,
        model_id: str = "mock-vlm-v0",
        scripted: Sequence[dict[str, Any]] | None = None,
    ):
        self.model_id = model_id
        self._scripted = list(scripted) if scripted else None
        self._i = 0

    def call(
        self,
        *,
        question_text: str,
        category: str,
        answer_type: str,
        scene_objects: list[dict[str, Any]],
        image_paths: list[Path],
    ) -> VLMResponse:
        if self._scripted is not None:
            out = dict(self._scripted[self._i % len(self._scripted)])
            self._i += 1
        else:
            out = {
                "answer_entity_ids": [],
                "abstained": True,
                "answer_text": "(mock) abstained",
                "evidence": {
                    "entity_ids": [],
                    "source_frame_idx": None,
                    "crop_bbox": None,
                    "confidence": 0.0,
                },
            }
        return VLMResponse(
            output=out,
            raw_response=json.dumps(out),
            latency_ms=0.1,
            tokens={"input_tokens": 0, "output_tokens": 0},
            confidence_verbalized=extract_verbalized_confidence(out),
        )
