"""Evaluator request policy from the catalog; no server or agent tuning restrictions."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .config import OUTPUT_ALLOWANCE, TABLES, WORKLOAD_POOL
from .config import model as get_model

CANDIDATE = "candidate"
RECORDING = "vllm"


def pool_of(workload: str) -> str:
    try:
        return WORKLOAD_POOL[workload]
    except KeyError:
        raise ValueError(f"unknown workload {workload!r}") from None


def _copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _copy(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_copy(child) for child in value]
    return copy.deepcopy(value)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(child) for child in value)
    return value


def merge(left: Mapping, right: Mapping) -> dict:
    """Recursively merge JSON objects without sharing nested values with either input."""
    out = _copy(left)
    for key, value in right.items():
        out[key] = (
            merge(out[key], value)
            if isinstance(value, Mapping) and isinstance(out.get(key), Mapping)
            else _copy(value)
        )
    return out


def get_path(obj: Any, path: str) -> Any:
    for key in path.split("."):
        if not isinstance(obj, Mapping) or key not in obj:
            return None
        obj = obj[key]
    return obj


def set_path(body: dict, path: str, value: Any) -> dict:
    *parents, last = path.split(".")
    node = body
    for key in parents:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ValueError(f"{path!r} crosses a non-object field")
    node[last] = copy.deepcopy(value)
    return body


@dataclass(frozen=True)
class RequestPolicy:
    model: str
    baseline: str
    api: str
    endpoint: str
    stream: Mapping[str, Any]
    max_tokens: str
    output_count: str
    force: Mapping[str, Any]
    thinking: Mapping[str, Any]
    sampling: Mapping[str, Any]
    sampling_agentic: Mapping[str, Any]
    status: str
    note: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("stream", "force", "thinking", "sampling", "sampling_agentic", "extra"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))

    def sampling_for(self, workload: str) -> dict:
        pool_of(workload)
        return _copy(self.sampling if workload.startswith("S") else self.sampling_agentic)

    def fixed(self, workload: str, output_limit: int = OUTPUT_ALLOWANCE) -> dict:
        if isinstance(output_limit, bool) or not isinstance(output_limit, int) or output_limit <= 0:
            raise ValueError("output_limit must be a positive integer")
        body = merge(merge(self.thinking, self.sampling_for(workload)), self.extra)
        return set_path(body, self.max_tokens, output_limit)

    def fields(
        self, workload: str, output_limit: int = OUTPUT_ALLOWANCE, forced: bool = False
    ) -> dict:
        """Return fresh streaming fields; agentic workloads always require their exact recorded count."""
        body = merge(self.stream, self.fixed(workload, output_limit))
        forced = forced or workload.startswith("A")
        if forced and not self.force:
            raise ValueError(
                f"{self.baseline} does not support the exact output-count contract; {self.note}"
            )
        return merge(body, self.force) if forced else body

    def request(
        self,
        messages: list[dict],
        *,
        workload: str,
        output_limit: int = OUTPUT_ALLOWANCE,
        served_model: str | None = None,
        tools: list[dict] | None = None,
        forced: bool = False,
    ) -> dict:
        """Build evaluator requests from recorded messages/tools and authored generation policy."""
        if served_model is None and self.api != "ollama-chat":
            served_model = "served"
        if not isinstance(served_model, str) or not served_model:
            raise ValueError("served_model must be a nonempty name (the native tag for Ollama)")
        body = self.fields(workload, output_limit, forced)
        body.update(model=served_model, messages=copy.deepcopy(messages))
        if tools is not None:
            body["tools"] = copy.deepcopy(tools)
        return body

    def count(self, event: dict) -> int | None:
        count = get_path(event, self.output_count)
        return (
            count if isinstance(count, int) and not isinstance(count, bool) and count >= 0 else None
        )


def request_fields(model: str, baseline: str = CANDIDATE) -> RequestPolicy:
    """Resolve model-specific policy and API translation from one normalized JSON catalog."""
    catalog = json.loads((TABLES / "request_fields.json").read_text())
    name = get_model(model).name
    engine = CANDIDATE if baseline == RECORDING else baseline
    try:
        spec = catalog["models"][name]
        backend = catalog["baselines"][engine]
    except KeyError:
        raise ValueError(f"no request policy for {name} on {baseline!r}") from None
    if engine not in spec["baselines"]:
        raise ValueError(f"no request policy for {name} on {baseline!r}")
    api = backend["api"]
    translated = copy.deepcopy(catalog["apis"][api])
    thinking = copy.deepcopy(spec["thinking"])
    sampling = copy.deepcopy(spec["sampling"])
    sampling_agentic = copy.deepcopy(spec.get("sampling_agentic", sampling))
    if api == "ollama-chat":
        # Native Ollama takes think/options. It has no equivalent for every template option.
        thinking = {"think": True}
        sampling = {"options": {**sampling, "repeat_penalty": 1.0}}
        sampling_agentic = {"options": {**sampling_agentic, "repeat_penalty": 1.0}}
    return RequestPolicy(
        name,
        baseline,
        api,
        **translated,
        thinking=thinking,
        sampling=sampling,
        sampling_agentic=sampling_agentic,
        status=backend["status"],
        note=backend["note"],
        extra={"return_token_ids": True} if baseline == RECORDING else {},
    )
