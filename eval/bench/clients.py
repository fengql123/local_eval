"""Send recorded OpenAI chat requests and collect streaming timing evidence.

Messages and tools are JSON strings in the prepared dataset. Parsing happens at
this HTTP boundary; replay never rewrites the recorded conversation. Server usage
counts include generated tokens that are not visible in text or tool deltas.
"""

import asyncio
import json
from time import perf_counter

import httpx

from ..request_policy import RequestPolicy


async def _events(response: httpx.Response):
    """Frame SSE data independently of HTTP packet boundaries."""
    data = []
    async for line in response.aiter_lines():
        if not line:
            if data:
                yield "\n".join(data), perf_counter()
                data.clear()
        elif line.startswith("data:"):
            value = line[5:]
            data.append(value[1:] if value.startswith(" ") else value)


class ChatClient:
    """One pooled client per server, shared by all concurrent replay requests."""

    def __init__(
        self,
        endpoint: str,
        policy: RequestPolicy,
        *,
        served_model: str = "served",
        timeout: float = 3600,
    ):
        self.url = endpoint.rstrip("/") + policy.endpoint
        self.policy = policy
        self.served_model = served_model
        self.timeout = timeout
        self._client = httpx.AsyncClient(
            timeout=timeout,
            trust_env=False,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
        )

    async def __aenter__(self):
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *exc):
        await self._client.__aexit__(*exc)

    async def request(self, request: dict, workload: str) -> dict:
        """Return stream evidence; HTTP/transport failures propagate to replay.

        Protocol failures produce an invalid request with explicit errors. The
        caller owns scheduling and answer grading belongs to the judge.
        """
        body = self.policy.request(
            json.loads(request["messages"]),
            workload=workload,
            output_limit=request["out"],
            served_model=self.served_model,
            tools=json.loads(request["tools"]) if request["tools"] is not None else None,
        )
        result = dict(
            ok=False,
            request_id=request["id"],
            t_send=perf_counter(),
            t_first=None,
            t_last=None,
            t_end=None,
            output_tokens=None,
            cached_tokens=None,
            text="",
            reasoning="",
            tool_calls=[],
            finish_reason=None,
            errors=[],
        )
        tools = {}
        done = False
        async with (
            asyncio.timeout(self.timeout),
            self._client.stream("POST", self.url, json=body) as response,
        ):
            response.raise_for_status()
            if "text/event-stream" not in response.headers.get("content-type", ""):
                result["errors"].append("response is not an SSE stream")
            async for data, received in _events(response):
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if event.get("error") is not None:
                    result["errors"].append(f"server error: {json.dumps(event['error'])}")
                    continue
                count = self.policy.count(event)
                if count is not None:
                    if type(count) is int and count >= 0:
                        result["output_tokens"] = count
                    else:
                        result["errors"].append("invalid completion token count")
                usage = event.get("usage")
                if usage:
                    result["cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get(
                        "cached_tokens"
                    )
                choices = event.get("choices", [])
                if len(choices) > 1:
                    result["errors"].append("expected one completion choice")
                for choice in choices:
                    delta = choice.get("delta", {})
                    content = delta.get("content") or ""
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    fragments = delta.get("tool_calls") or []
                    result["text"] += content
                    result["reasoning"] += reasoning
                    for fragment in fragments:
                        index = fragment["index"]
                        tool = tools.setdefault(
                            index,
                            dict(id="", type="function", function=dict(name="", arguments="")),
                        )
                        tool["id"] += fragment.get("id") or ""
                        tool["type"] = fragment.get("type", tool["type"])
                        function = fragment.get("function", {})
                        tool["function"]["name"] += function.get("name") or ""
                        tool["function"]["arguments"] += function.get("arguments") or ""
                    if content or reasoning or fragments:
                        if result["t_first"] is None:
                            result["t_first"] = received
                        result["t_last"] = received
                    if choice.get("finish_reason") is not None:
                        result["finish_reason"] = choice["finish_reason"]
                        # Forced generation may continue after the last visible delta.
                        result["t_last"] = received
        result["t_end"] = perf_counter()
        result["tool_calls"] = [tools[index] for index in sorted(tools)]
        if not done:
            result["errors"].append("missing [DONE] stream terminator")
        if result["finish_reason"] is None:
            result["errors"].append("missing completion finish marker")
        if result["t_first"] is None:
            result["errors"].append("no generated token observed; TTFT is unavailable")
        count = result["output_tokens"]
        if count is None:
            result["errors"].append("missing completion token usage")
        elif count == 0:
            result["errors"].append("no completion tokens generated")
        elif workload.startswith("A") and count != request["out"]:
            result["errors"].append(
                f"expected {request['out']} completion tokens, received {count}"
            )
        elif count > request["out"]:
            result["errors"].append(f"completion tokens exceed allowance {request['out']}")
        result["ok"] = not result["errors"]
        return result
