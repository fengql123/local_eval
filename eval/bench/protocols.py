"""Translate native harness messages and streams to/from canonical Chat Completions.

The backend conversation stays in OpenAI chat form. These adapters preserve native
conversation/tool structure; backend generation settings come from request policy.
"""

import copy
import json
import time
from uuid import uuid4


def text(content):
    if isinstance(content, str):
        return content
    parts = []
    for part in content or []:
        if part.get("type", "text") not in ("text", "input_text", "output_text"):
            raise ValueError(f"unsupported text content block: {part['type']}")
        parts.append(part["text"])
    return "".join(parts)


def _arguments(value):
    return value if isinstance(value, str) else json.dumps(value or {}, ensure_ascii=False)


def _definitions(tools, namespace=None):
    for tool in tools or []:
        if tool.get("type") == "namespace":
            yield from _definitions(tool["tools"], tool["name"])
        else:
            yield tool, namespace


def _request_tools(body):
    """Include tool definitions that Responses added through conversation history."""
    tools = list(body.get("tools") or [])
    items = body.get("input", [])
    if isinstance(items, list):
        for item in items:
            if item.get("type") == "additional_tools":
                tools.extend(item["tools"])
    return tools


def _tools(tools):
    """Flatten Responses namespaces and normalize function/custom tool definitions."""
    result = []
    for tool, _ in _definitions(tools):
        if "function" in tool:
            result.append(tool)
            continue
        if tool.get("type") not in (None, "function", "custom"):
            raise ValueError(f"recording supports callable tools, not {tool['type']}")
        parameters = tool.get("parameters", tool.get("input_schema"))
        if tool.get("type") == "custom" and parameters is None:
            parameters = {
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
            }
        result.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": parameters or {"type": "object", "properties": {}},
                },
            }
        )
    names = [tool["function"]["name"] for tool in result]
    if len(names) != len(set(names)):
        raise ValueError("canonical chat requires unique tool names across namespaces")
    return result


def chat(body, api):
    """Translate the three harness APIs to the canonical chat request shape."""
    messages = []
    if api == "chat":
        messages = copy.deepcopy(body["messages"])
        for message in messages:
            text(message.get("content"))
    elif api == "messages":
        if body.get("system"):
            messages.append({"role": "system", "content": text(body["system"])})
        for message in body["messages"]:
            content = message["content"]
            if isinstance(content, str):
                messages.append(dict(message))
                continue
            parts, reasoning, calls = [], [], []
            for block in content:
                kind = block["type"]
                if kind == "tool_result":
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": text(block.get("content")),
                        }
                    )
                elif kind == "tool_use":
                    calls.append(
                        {
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": _arguments(block["input"]),
                            },
                        }
                    )
                elif kind == "thinking":
                    reasoning.append(block["thinking"])
                elif kind == "text":
                    parts.append(block["text"])
                else:
                    raise ValueError(f"unsupported Anthropic content block: {kind}")
            if parts or calls or reasoning:
                converted = {"role": message["role"], "content": "".join(parts)}
                if calls:
                    converted["tool_calls"] = calls
                if reasoning:
                    converted["reasoning_content"] = "".join(reasoning)
                messages.append(converted)
    else:
        if body.get("previous_response_id"):
            raise ValueError("recording requires full Responses input, not previous_response_id")
        if body.get("instructions"):
            messages.append({"role": "system", "content": body["instructions"]})
        items = body.get("input", [])
        if isinstance(items, str):
            items = [{"role": "user", "content": items}]
        reasoning = []
        for item in items:
            kind = item.get("type", "message")
            if kind == "message":
                message = {"role": item["role"], "content": text(item["content"])}
                if reasoning:
                    if message["role"] == "assistant":
                        message["reasoning_content"] = "".join(reasoning)
                    else:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": "",
                                "reasoning_content": "".join(reasoning),
                            }
                        )
                    reasoning.clear()
                messages.append(message)
            elif kind in ("function_call", "custom_tool_call"):
                call = {
                    "id": item["call_id"],
                    "type": "function",
                    "function": {
                        "name": item["name"],
                        "arguments": item.get(
                            "arguments", _arguments({"input": item.get("input")})
                        ),
                    },
                }
                if not messages or messages[-1]["role"] != "assistant":
                    messages.append({"role": "assistant", "content": ""})
                messages[-1].setdefault("tool_calls", []).append(call)
                if reasoning:
                    messages[-1]["reasoning_content"] = "".join(reasoning)
                    reasoning.clear()
            elif kind in ("function_call_output", "custom_tool_call_output"):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item["call_id"],
                        "content": text(item["output"]),
                    }
                )
            elif kind == "additional_tools":
                # These definitions enter the callable tool set, not message text.
                continue
            elif kind == "reasoning":
                if item.get("encrypted_content"):
                    raise ValueError("canonical chat cannot replay encrypted Responses reasoning")
                reasoning.extend(part["text"] for part in item.get("summary", []))
            else:
                raise ValueError(f"unsupported Responses input item: {kind}")
        if reasoning:
            messages.append(
                {"role": "assistant", "content": "", "reasoning_content": "".join(reasoning)}
            )
    # Qwen templates expect one leading system message, including developer instructions.
    system = [text(m.get("content")) for m in messages if m["role"] in ("system", "developer")]
    messages = [m for m in messages if m["role"] not in ("system", "developer")]
    if system:
        messages.insert(0, {"role": "system", "content": "\n\n".join(system)})
    tools = _tools(_request_tools(body) if api == "responses" else body.get("tools"))
    result = {"messages": messages}
    if not tools:
        return result
    result["tools"] = tools
    parallel = body.get("parallel_tool_calls") if api != "messages" else None
    choice = body.get("tool_choice")
    if isinstance(choice, dict):
        kind = choice["type"]
        if api == "messages" and "disable_parallel_tool_use" in choice:
            parallel = not choice["disable_parallel_tool_use"]
        if kind in ("auto", "any", "none"):
            choice = {"auto": "auto", "any": "required", "none": "none"}[kind]
        elif kind in ("function", "custom", "tool"):
            choice = {
                "type": "function",
                "function": choice["function"]
                if "function" in choice
                else {"name": choice["name"]},
            }
        else:
            raise ValueError(f"unsupported tool choice: {kind}")
    if choice is not None:
        result["tool_choice"] = choice
    if parallel is not None:
        result["parallel_tool_calls"] = parallel
    return result


def sse(kind, payload):
    prefix = f"event: {kind}\n" if kind else ""
    return (prefix + "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode()


class Emitter:
    """Translate chat chunks and completed answers to the harness's native API."""

    def __init__(self, api, request):
        self.api, self.request = api, request
        self.tools = _request_tools(request) if api == "responses" else request.get("tools", [])
        self.id = ("resp_" if api == "responses" else "msg_") + uuid4().hex
        self.sequence, self.index = -1, -1
        self.phase, self.text, self.items = None, "", []

    def event(self, kind, **fields):
        if self.api == "responses":
            self.sequence += 1
            fields["sequence_number"] = self.sequence
        return sse(kind, {"type": kind, **fields})

    def count(self, usage):
        prompt, completion = usage.get("prompt_tokens", 0), usage["completion_tokens"]
        count = {"input_tokens": prompt, "output_tokens": completion}
        if self.api == "responses":
            count.update(
                total_tokens=prompt + completion,
                input_tokens_details={
                    "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                    or 0
                },
                output_tokens_details={
                    "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
                        "reasoning_tokens"
                    )
                    or 0
                },
            )
        return count

    def message(self, content, usage, stop=None):
        return {
            "id": self.id,
            "type": "message",
            "role": "assistant",
            "model": "served",
            "content": content,
            "usage": usage,
            "stop_reason": stop,
            "stop_sequence": None,
        }

    def response(self, status, usage=None):
        return {
            "id": self.id,
            "object": "response",
            "created_at": int(time.time()),
            "status": status,
            "model": "served",
            "output": self.items,
            "usage": usage,
            "error": None,
            "incomplete_details": {"reason": "max_output_tokens"}
            if status == "incomplete"
            else None,
            "parallel_tool_calls": self.request.get("parallel_tool_calls", True),
            "tools": self.tools,
        }

    def start(self):
        if self.api == "responses":
            return b"".join(
                self.event(kind, response=self.response("in_progress"))
                for kind in ("response.created", "response.in_progress")
            )
        return self.event(
            "message_start", message=self.message([], {"input_tokens": 0, "output_tokens": 0})
        )

    def part_event(self, kind, **fields):
        index = "summary_index" if self.phase == "reasoning" else "content_index"
        return self.event(
            kind, item_id=self.item["id"], output_index=self.index, **{index: 0}, **fields
        )

    def close(self):
        if self.phase is None:
            return b""
        if self.api == "messages":
            out = self.event("content_block_stop", index=self.index)
        else:
            reasoning = self.phase == "reasoning"
            self.part["text"] = self.text
            self.item["summary" if reasoning else "content"].append(self.part)
            if not reasoning:
                self.item["status"] = "completed"
            stem = "reasoning_summary_text" if reasoning else "output_text"
            out = self.part_event(
                "response." + stem + ".done",
                text=self.text,
                **({} if reasoning else {"logprobs": []}),
            )
            stem = "reasoning_summary_part" if reasoning else "content_part"
            out += self.part_event("response." + stem + ".done", part=self.part)
            self.items.append(self.item)
            out += self.event("response.output_item.done", output_index=self.index, item=self.item)
        self.phase = None
        return out

    def delta(self, phase, text):
        out, reasoning = b"", phase == "reasoning"
        if phase != self.phase:
            out = self.close()
            self.index += 1
            self.phase, self.text = phase, ""
            if self.api == "messages":
                block = (
                    {"type": "thinking", "thinking": ""}
                    if reasoning
                    else {"type": "text", "text": ""}
                )
                out += self.event("content_block_start", index=self.index, content_block=block)
            else:
                self.item = {
                    "id": ("rs_" if reasoning else "msg_") + uuid4().hex,
                    "type": "reasoning" if reasoning else "message",
                    "summary" if reasoning else "content": [],
                }
                self.part = {"type": "summary_text" if reasoning else "output_text", "text": ""}
                if not reasoning:
                    self.item.update(status="in_progress", role="assistant")
                    self.part["annotations"] = []
                out += self.event(
                    "response.output_item.added", output_index=self.index, item=self.item
                )
                stem = "reasoning_summary_part" if reasoning else "content_part"
                out += self.part_event("response." + stem + ".added", part=self.part)
        self.text += text
        if self.api == "messages":
            delta = (
                {"type": "thinking_delta", "thinking": text}
                if reasoning
                else {"type": "text_delta", "text": text}
            )
            return out + self.event("content_block_delta", index=self.index, delta=delta)
        stem = "reasoning_summary_text" if reasoning else "output_text"
        return out + self.part_event(
            "response." + stem + ".delta", delta=text, **({} if reasoning else {"logprobs": []})
        )

    def chunk(self, event):
        out = b""
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            for phase, text in (
                ("reasoning", delta.get("reasoning_content") or delta.get("reasoning")),
                ("content", delta.get("content")),
            ):
                if text:
                    out += self.delta(phase, text)
        return out

    def stop(self, answer, finish):
        return (
            "max_tokens"
            if finish == "length"
            else "tool_use"
            if answer["tool_calls"]
            else "end_turn"
        )

    def finish(self, answer, usage, finish):
        out = self.close()
        for call in answer["tool_calls"]:
            self.index += 1
            function = call["function"]
            if self.api == "messages":
                block = {
                    "type": "tool_use",
                    "id": call["id"],
                    "name": function["name"],
                    "input": {},
                }
                out += self.event("content_block_start", index=self.index, content_block=block)
                out += self.event(
                    "content_block_delta",
                    index=self.index,
                    delta={"type": "input_json_delta", "partial_json": function["arguments"]},
                )
                out += self.event("content_block_stop", index=self.index)
                continue
            definition, namespace = next(
                (
                    (tool, namespace)
                    for tool, namespace in _definitions(self.tools)
                    if (tool.get("function") or tool)["name"] == function["name"]
                ),
                ({}, None),
            )
            value = function["arguments"]
            custom = False
            if definition.get("type") == "custom":
                try:
                    payload = json.loads(value)
                except json.JSONDecodeError:
                    payload = None
                custom = isinstance(payload, dict) and isinstance(payload.get("input"), str)
                if custom:
                    value = payload["input"]
                # Otherwise preserve the call for the native tool handler to reject.
            item = {
                "id": "fc_" + uuid4().hex,
                "type": "custom_tool_call" if custom else "function_call",
                "call_id": call["id"],
                "name": function["name"],
                "status": "in_progress",
            }
            if namespace:
                item["namespace"] = namespace
            field = "input" if custom else "arguments"
            item[field] = ""
            out += self.event("response.output_item.added", output_index=self.index, item=item)
            stem = "custom_tool_call_input" if custom else "function_call_arguments"
            fields = {"item_id": item["id"], "output_index": self.index}
            out += self.event("response." + stem + ".delta", **fields, delta=value)
            out += self.event("response." + stem + ".done", **fields, **{field: value})
            item.update(status="completed", **{field: value})
            self.items.append(item)
            out += self.event("response.output_item.done", output_index=self.index, item=item)
        if self.index < 0:
            out += self.delta("content", "") + self.close()
        if self.api == "responses":
            status = "incomplete" if finish == "length" else "completed"
            return out + self.event(
                "response." + status, response=self.response(status, self.count(usage))
            )
        return (
            out
            + self.event(
                "message_delta",
                delta={"stop_reason": self.stop(answer, finish), "stop_sequence": None},
                usage=self.count(usage),
            )
            + self.event("message_stop")
        )

    def reply(self, answer, usage, finish):
        if self.api == "responses":
            for phase, field in (("reasoning", "reasoning_content"), ("content", "content")):
                if answer.get(field):
                    self.delta(phase, answer[field])
            self.finish(answer, usage, finish)
            return self.response(
                "incomplete" if finish == "length" else "completed", self.count(usage)
            )
        content = []
        if answer.get("reasoning_content"):
            content.append({"type": "thinking", "thinking": answer["reasoning_content"]})
        content.append({"type": "text", "text": answer["content"]})
        content.extend(
            {
                "type": "tool_use",
                "id": call["id"],
                "name": call["function"]["name"],
                "input": json.loads(call["function"]["arguments"]),
            }
            for call in answer["tool_calls"]
        )
        return self.message(content, self.count(usage), self.stop(answer, finish))

    def failure(self, message):
        if self.api == "responses":
            response = self.response("failed")
            response["error"] = {"code": "server_error", "message": message}
            return self.event("response.failed", response=response)
        return self.event("error", error={"type": "api_error", "message": message})
