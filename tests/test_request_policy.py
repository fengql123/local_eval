from eval.request_policy import pool_of, request_fields


def test_candidate_request_uses_model_policy_and_openai_endpoint():
    policy = request_fields("C2")
    body = policy.request([{"role": "user", "content": "Explain caching."}], workload="S-1")
    assert (policy.api, policy.endpoint) == ("openai-chat", "/v1/chat/completions")
    assert body["model"] == "served"
    assert body["stream_options"] == {"include_usage": True}
    assert body["presence_penalty"] == 1.5
    assert body["chat_template_kwargs"] == {"enable_thinking": True, "preserve_thinking": True}
    assert "ignore_eos" not in body
    assert pool_of("A-MA") == "A-MA"


def test_ollama_request_keeps_openai_fields_and_applies_backend_overrides():
    policy = request_fields("C2", "ollama")
    body = policy.request([], workload="A-N", output_limit=12, served_model="baseline")
    assert policy.endpoint == "/v1/chat/completions"
    assert body["model"] == "baseline"
    assert body["max_tokens"] == 12
    assert body["ignore_eos"] is True
    assert body["think"] is True
    assert body["repeat_penalty"] == 1.0
    assert body["chat_template_kwargs"]["preserve_thinking"] is True
    assert body["presence_penalty"] == 1.5
    assert "options" not in body


def test_stream_usage_can_be_absent_or_report_zero_and_agentic_sampling_differs():
    policy = request_fields("N3", "vllm")
    assert policy.count({"choices": [{"delta": {"content": "Hello"}}], "usage": None}) is None
    assert policy.count({"usage": {"completion_tokens": 0}}) == 0
    assert policy.count({"usage": {"completion_tokens": 17}}) == 17
    assert policy.sampling_for("S-1")["top_p"] == 1.0
    assert policy.sampling_for("A-MA")["top_p"] == 0.95
    assert policy.fields("S-1", forced=True)["ignore_eos"] is True


def test_request_changes_do_not_modify_recorded_data_or_future_requests():
    policy = request_fields("C2", "ollama")
    messages = [{"role": "user", "content": "Explain caching."}]
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]
    body = policy.request(messages, workload="A-N", tools=tools)
    body["messages"][0]["content"] = "changed"
    body["tools"][0]["function"]["name"] = "changed"
    body["chat_template_kwargs"]["enable_thinking"] = False
    body["stream_options"]["include_usage"] = False
    assert messages[0]["content"] == "Explain caching."
    assert tools[0]["function"]["name"] == "read_file"
    assert policy.fields("S-1")["chat_template_kwargs"]["enable_thinking"] is True
    assert policy.fields("S-1")["stream_options"]["include_usage"] is True
