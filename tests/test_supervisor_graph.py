from __future__ import annotations

from collections import deque
from types import SimpleNamespace
import pytest
from graphs.supervisor_graph import build_supervisor_graph
from shared.settings import AppSettings

class FakeResponses:
    def __init__(self, outputs):
        self.outputs = deque(outputs)
        self.calls = []
    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.outputs.popleft())

class FakeClient:
    def __init__(self, *outputs):
        self.responses = FakeResponses(outputs)

@pytest.fixture
def settings():
    return AppSettings(
        foundry_project_endpoint="https://example.services.ai.azure.com/api/projects/example",
        foundry_model_deployment="test-deployment",
        app_name="test-app",
        app_environment="test",
        app_version="0.1.0",
        model_max_output_tokens=200,
        system_prompt="You are a test assistant.",
        router_max_output_tokens=10,
        router_prompt="Return only general or time.",
    )

def test_general_question_is_routed_and_answered_by_foundry(settings):
    client = FakeClient("general", "Resposta do Foundry.")
    graph = build_supervisor_graph(settings=settings, client_factory=lambda: client)
    result = graph.invoke({"user_input":"Explique o Foundry.","intent":"general","answer":""})
    assert result["intent"] == "general"
    assert result["answer"] == "Resposta do Foundry."
    assert len(client.responses.calls) == 2

def test_time_question_uses_local_tool_after_llm_routing(settings):
    client = FakeClient("time")
    graph = build_supervisor_graph(settings=settings, client_factory=lambda: client)
    result = graph.invoke({"user_input":"Você pode informar o horário universal?","intent":"general","answer":""})
    assert result["intent"] == "time"
    assert "UTC" in result["answer"]
    assert len(client.responses.calls) == 1

def test_router_normalizes_case_and_punctuation(settings):
    client = FakeClient(" TIME. ")
    graph = build_supervisor_graph(settings=settings, client_factory=lambda: client)
    result = graph.invoke({"user_input":"What time is it in UTC?","intent":"general","answer":""})
    assert result["intent"] == "time"

def test_empty_input_is_rejected(settings):
    client = FakeClient("general")
    graph = build_supervisor_graph(settings=settings, client_factory=lambda: client)
    with pytest.raises(ValueError, match="user_input cannot be empty"):
        graph.invoke({"user_input":"   ","intent":"general","answer":""})

def test_unsupported_router_intent_is_rejected(settings):
    client = FakeClient("inventory")
    graph = build_supervisor_graph(settings=settings, client_factory=lambda: client)
    with pytest.raises(RuntimeError, match="unsupported intent"):
        graph.invoke({"user_input":"Consulte o estoque.","intent":"general","answer":""})

def test_empty_model_response_is_rejected(settings):
    client = FakeClient("general", "")
    graph = build_supervisor_graph(settings=settings, client_factory=lambda: client)
    with pytest.raises(RuntimeError, match="empty response"):
        graph.invoke({"user_input":"Hello","intent":"general","answer":""})
