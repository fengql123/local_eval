"""Share OpenAI chat transport and stream assembly across replay and recording.

Messages and tools are JSON strings in the prepared dataset. Parsing happens at
this HTTP boundary; replay never rewrites the recorded conversation. Server usage
counts include generated tokens that are not visible in text or tool deltas.
"""

import asyncio
import json
import os
from time import perf_counter

import httpx

from ..request_policy import RequestPolicy


async def _events(response: httpx.Response, *, timing: bool):
    """Frame SSE data independently of HTTP packet boundaries."""
    data = []
    async for line in response.aiter_lines():
        if not line:
            if data:
                yield "\n".join(data), perf_counter() if timing else None
                data.clear()
        elif line.startswith("data:"):
            value = line[5:]
            data.append(value[1:] if value.startswith(" ") else value)


class ChatClient:
    """Share transport with one bounded queue/serve budget for reads and completion."""

    def __init__(
        self,
        endpoint: str,
        policy: RequestPolicy,
        *,
        served_model: str = "served",
        timeout: float = 3600,
        total_timeout: float | None = None,
        mode: str = "timing",
        tokenizer=None,
    ):
        self.url = endpoint.rstrip("/") + policy.endpoint
        self.policy = policy
        self.served_model = served_model
        self.total_timeout = timeout if total_timeout is None else total_timeout
        self.timing = mode == "timing"
        self.tokenizer = tokenizer
        self.observer = None
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, read=self.total_timeout),
            trust_env=False,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
        )

    async def __aenter__(self):
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *exc):
        await self._client.__aexit__(*exc)

    async def complete(self, body: dict, *, result: dict | None = None, on_event=None) -> dict:
        """Send an authored body, retaining partial replies on interruption.

        Recording supplies a result dictionary and an async event callback to
        capture a reply while forwarding it live. Replay uses the same assembly
        without a callback. HTTP and transport failures propagate to the caller.
        """
        if result is None:
            result = {}
        result.update(
            ok=False,
            text="",
            reasoning="",
            tool_calls=[],
            finish_reason=None,
            usage=None,
            errors=[],
        )
        if self.timing:
            result.update(
                t_send=perf_counter(),
                t_first=None,
                t_last=None,
                t_end=None,
                output_tokens=None,
                cached_tokens=None,
                output_events=0,
                stream_gap_max=0.0,
            )
        tools = {}
        done = False
        if self.observer:
            self.observer.start(result)
        try:
            async with (
                asyncio.timeout(self.total_timeout),
                self._client.stream("POST", self.url, json=body) as response,
            ):
                response.raise_for_status()
                if "text/event-stream" not in response.headers.get("content-type", ""):
                    result["errors"].append("response is not an SSE stream")
                async for data, received in _events(response, timing=self.timing):
                    if data == "[DONE]":
                        done = True
                        break
                    event = json.loads(data)
                    if event.get("error") is not None:
                        result["errors"].append(f"server error: {json.dumps(event['error'])}")
                    if event.get("usage"):
                        result["usage"] = event["usage"]
                    if self.timing:
                        count = self.policy.count(event)
                        if count is not None:
                            if type(count) is int and count >= 0:
                                result["output_tokens"] = count
                            else:
                                result["errors"].append("invalid completion token count")
                        if event.get("usage"):
                            result["cached_tokens"] = (
                                event["usage"].get("prompt_tokens_details") or {}
                            ).get("cached_tokens")
                    choices = event.get("choices", [])
                    if len(choices) > 1:
                        result["errors"].append("expected one completion choice")
                    for choice in choices:
                        delta = choice.get("delta", {})
                        token_ids = delta.get("token_ids")
                        if token_ids is not None:
                            if not isinstance(token_ids, list) or any(
                                type(token) is not int or token < 0 for token in token_ids
                            ):
                                result["errors"].append("invalid accepted token IDs")
                            else:
                                result.setdefault("token_ids", []).extend(token_ids)
                        if "raw_text" in delta:
                            if isinstance(delta["raw_text"], str):
                                result["raw_text"] = result.get("raw_text", "") + delta["raw_text"]
                            else:
                                result["errors"].append("invalid raw output text")
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
                        generated = (
                            token_ids
                            or content
                            or reasoning
                            or any(
                                fragment.get("function", {}).get("name")
                                or fragment.get("function", {}).get("arguments")
                                for fragment in fragments
                            )
                        )
                        if self.timing and generated:
                            if result["t_first"] is None:
                                result["t_first"] = received
                            else:
                                result["stream_gap_max"] = max(
                                    result["stream_gap_max"], received - result["t_last"]
                                )
                            result["t_last"] = received
                            result["output_events"] += 1
                        if choice.get("finish_reason") is not None:
                            result["finish_reason"] = choice["finish_reason"]
                            if self.timing:
                                # Forced generation can outlast the last visible delta.
                                result["t_last"] = received
                    if self.observer:
                        self.observer.update(result)
                    if on_event is not None:
                        await on_event(event)
        except (httpx.HTTPError, ValueError):
            if self.observer:
                self.observer.failed = True
            raise
        finally:
            result["tool_calls"] = [tools[index] for index in sorted(tools)]
            if self.timing:
                result["t_end"] = perf_counter()
            if self.observer:
                self.observer.finish(result)
        if not done:
            result["errors"].append("missing [DONE] stream terminator")
        if result["finish_reason"] is None:
            result["errors"].append("missing completion finish marker")
        if self.tokenizer is not None:
            ids = result.get("token_ids", [])
            count = self.policy.count({"usage": result["usage"]})
            if not ids:
                result["errors"].append("missing accepted token IDs")
            elif any(
                token >= self.tokenizer.get_vocab_size(with_added_tokens=True) for token in ids
            ):
                result["errors"].append("accepted token ID is outside the fixed tokenizer")
            elif self.tokenizer.decode(ids, skip_special_tokens=False) != result.get("raw_text"):
                result["errors"].append("raw output does not match fixed-tokenizer decoding")
            if type(count) is not int or count != len(ids):
                result["errors"].append("completion usage does not match accepted token IDs")
            if self.timing:
                result["output_tokens"] = len(ids)
        if self.timing:
            if result["t_first"] is None:
                result["errors"].append("no generated token observed; TTFT is unavailable")
            count = result["output_tokens"]
            if count is None:
                result["errors"].append("missing completion token usage")
            elif count == 0:
                result["errors"].append("no completion tokens generated")
        result["ok"] = not result["errors"]
        return result

    def partial_errors(self, result: dict, allowance: int) -> list[str]:
        """Check what a cut stream delivered: its count and fixed-tokenizer prefix."""
        errors = []
        ids = result.get("token_ids") or []
        if len(ids) > allowance:
            errors.append(f"completion tokens exceed allowance {allowance}")
        if self.tokenizer is None or result.get("t_first") is None:
            return errors
        if not ids:
            errors.append("missing accepted token IDs")
        elif any(token >= self.tokenizer.get_vocab_size(with_added_tokens=True) for token in ids):
            errors.append("accepted token ID is outside the fixed tokenizer")
        else:
            decoded = self.tokenizer.decode(ids, skip_special_tokens=False)
            raw = result.get("raw_text", "")
            # A cut may split one multi-byte character between the two texts.
            if max(len(decoded), len(raw)) - len(os.path.commonprefix([decoded, raw])) > 4:
                errors.append("raw output does not match fixed-tokenizer decoding")
        return errors

    async def request(self, request: dict, workload: str, *, result: dict | None = None) -> dict:
        """Apply replay policy and validate timing's recorded output allowance.

        A caller-supplied ``result`` keeps partial evidence if the request is cut.
        """
        body = self.policy.request(
            json.loads(request["messages"]),
            workload=workload,
            output_limit=request["out"],
            served_model=self.served_model,
            tools=json.loads(request["tools"]) if request["tools"] is not None else None,
        )
        result = await self.complete(body, result=result)
        result["request_id"] = request["id"]
        if self.timing and result["output_tokens"]:
            count = result["output_tokens"]
            if workload.startswith("A") and count != request["out"]:
                result["errors"].append(
                    f"expected {request['out']} completion tokens, received {count}"
                )
            elif count > request["out"]:
                result["errors"].append(f"completion tokens exceed allowance {request['out']}")
        result["ok"] = not result["errors"]
        return result
