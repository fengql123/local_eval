import pytest

from eval.tables import request_fields


def test_openai_request_applies_catalog_policy_without_mutating_recorded_data():
    policy = request_fields("C2")
    messages = [{"role": "user", "content": "Explain caching."}]
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]
    body = policy.request(messages, workload="A-N", output_limit=19, tools=tools)
    assert (policy.api, policy.endpoint) == ("openai-chat", "/v1/chat/completions")
    assert body["model"] == "served"
    assert body["max_tokens"] == 19
    assert body["ignore_eos"] is True
    assert body["presence_penalty"] == 1.5
    assert body["chat_template_kwargs"] == {"enable_thinking": True, "preserve_thinking": True}
    assert policy.count({"usage": {"completion_tokens": 19}}) == 19
    body["messages"][0]["content"] = "changed"
    body["tools"][0]["function"]["name"] = "changed"
    body["chat_template_kwargs"]["enable_thinking"] = False
    assert messages[0]["content"] == "Explain caching."
    assert tools[0]["function"]["name"] == "read_file"
    assert policy.fields("S-1")["chat_template_kwargs"]["enable_thinking"] is True
    assert "ignore_eos" not in policy.fields("S-1")


def test_native_ollama_request_uses_native_endpoint_fields_and_count():
    policy = request_fields("C2", "ollama")
    body = policy.request(
        [{"role": "user", "content": "Explain caching."}],
        workload="S-1",
        output_limit=12,
        served_model="qwen3.6:35b-a3b-mtp-q4_K_M",
    )
    assert (policy.api, policy.endpoint) == ("ollama-chat", "/api/chat")
    assert body["options"]["num_predict"] == 12
    assert body["options"]["presence_penalty"] == 1.5
    assert body["think"] is True
    assert "chat_template_kwargs" not in body
    assert policy.count({"eval_count": 12}) == 12


def test_unsupported_native_agentic_output_policy_is_rejected():
    with pytest.raises(ValueError, match="exact output-count"):
        request_fields("C2", "ollama").fields("A-1", 12)


def test_boolean_output_limit_is_rejected():
    with pytest.raises(ValueError, match="output_limit must be a positive integer"):
        request_fields("C2").fields("S-1", True)
