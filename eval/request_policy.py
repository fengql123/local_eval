"""Build baseline and candidate OpenAI requests from the authored request policy."""

import copy
import json
from dataclasses import dataclass, field

from .config import OUTPUT_ALLOWANCE, TABLES, WORKLOAD_POOL
from .config import model as get_model


def pool_of(workload: str) -> str:
    return WORKLOAD_POOL[workload]


@dataclass
class RequestPolicy:
    model: str
    baseline: str
    api: str
    endpoint: str
    stream: dict
    max_tokens: str
    output_count: str
    force: dict
    thinking: dict
    sampling: dict
    sampling_agentic: dict
    overrides: dict = field(default_factory=dict)

    def sampling_for(self, workload: str) -> dict:
        return copy.deepcopy(self.sampling if workload.startswith("S") else self.sampling_agentic)

    def fixed(self, workload: str, output_limit: int = OUTPUT_ALLOWANCE) -> dict:
        return {
            **copy.deepcopy(self.thinking),
            **self.sampling_for(workload),
            self.max_tokens: output_limit,
        }

    def fields(
        self, workload: str, output_limit: int = OUTPUT_ALLOWANCE, forced: bool = False
    ) -> dict:
        """Agentic replay requests reproduce the recorded output count with ignore_eos."""
        return {
            **copy.deepcopy(self.stream),
            **self.fixed(workload, output_limit),
            **copy.deepcopy(self.overrides),
            **copy.deepcopy(self.force if forced or workload.startswith("A") else {}),
        }

    def request(
        self,
        messages: list[dict],
        *,
        workload: str,
        output_limit: int = OUTPUT_ALLOWANCE,
        served_model: str = "served",
        tools: list[dict] | None = None,
        forced: bool = False,
    ) -> dict:
        body = self.fields(workload, output_limit, forced)
        body.update(model=served_model, messages=copy.deepcopy(messages))
        if tools is not None:
            body["tools"] = copy.deepcopy(tools)
        return body

    def count(self, event: dict) -> int | None:
        parent, key = self.output_count.split(".")
        return (event.get(parent) or {}).get(key)


def request_fields(model: str, baseline: str = "candidate") -> RequestPolicy:
    catalog = json.loads((TABLES / "request_fields.json").read_text())
    name = get_model(model).name
    spec = catalog["models"][name]
    backend = catalog["baselines"]["candidate" if baseline == "vllm" else baseline]
    return RequestPolicy(
        name,
        baseline,
        backend["api"],
        **catalog["apis"][backend["api"]],
        thinking=spec["thinking"],
        sampling=spec["sampling"],
        sampling_agentic=spec.get("sampling_agentic", spec["sampling"]),
        overrides=backend.get("fields", {}),
    )
