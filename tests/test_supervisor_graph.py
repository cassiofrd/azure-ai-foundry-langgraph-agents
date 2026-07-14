from __future__ import annotations

from types import SimpleNamespace

import pytest

from graphs.supervisor_graph import build_supervisor_graph
from shared.settings import AppSettings


class FakeResponses:
    def __init__(self, text: str):
        self.text = text
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.text)


class FakeClient:
    def __init__(self, text: str):
        self.responses = FakeResponses(text)


@pytest.fixture
def settings() -> AppSettings:
    return AppSettings(
        foundry_project_endpoint=(
            "https://example.services.ai.azure.com/api/projects/example"
        ),
        foundry_model_deployment="test-deployment",
        app_name="test-app",
        app_environment="test",
        app_version="0.1.0",
        model_max_output_tokens=200,
        system_prompt="You are a test assistant.",
    )


def test_general_question_calls_foundry_and_returns_answer(settings):
    client = FakeClient("Resposta do Foundry.")
    graph = build_supervisor_graph(
        settings=settings,
        client_factory=lambda: client,
    )

    result = graph.invoke(
        {
            "user_input": "Explique o Foundry.",
            "intent": "general",
            "answer": "",
        }
    )

    assert result["intent"] == "general"
    assert result["answer"] == "Resposta do Foundry."
    assert client.responses.calls[0]["model"] == "test-deployment"
    assert client.responses.calls[0]["instructions"] == (
        "You are a test assistant."
    )


def test_time_question_uses_local_tool_without_calling_foundry(settings):
    client = FakeClient("unused")
    graph = build_supervisor_graph(
        settings=settings,
        client_factory=lambda: client,
    )

    result = graph.invoke(
        {
            "user_input": "Que horas são em UTC?",
            "intent": "general",
            "answer": "",
        }
    )

    assert result["intent"] == "time"
    assert "UTC" in result["answer"]
    assert client.responses.calls == []


def test_graph_rejects_empty_input(settings):
    graph = build_supervisor_graph(
        settings=settings,
        client_factory=lambda: FakeClient("unused"),
    )

    with pytest.raises(ValueError, match="user_input cannot be empty"):
        graph.invoke(
            {
                "user_input": "   ",
                "intent": "general",
                "answer": "",
            }
        )


def test_graph_rejects_empty_model_response(settings):
    graph = build_supervisor_graph(
        settings=settings,
        client_factory=lambda: FakeClient(""),
    )

    with pytest.raises(RuntimeError, match="empty response"):
        graph.invoke(
            {
                "user_input": "Hello",
                "intent": "general",
                "answer": "",
            }
        )
