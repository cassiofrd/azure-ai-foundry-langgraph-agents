from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

from graphs.supervisor_graph import build_supervisor_graph
from shared.settings import AppSettings


class FakeResponses:
    def __init__(self, responses: list[SimpleNamespace]):
        self.responses = deque(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        if not self.responses:
            raise AssertionError("No fake response was configured.")

        return self.responses.popleft()


class FakeClient:
    def __init__(self, *responses: SimpleNamespace):
        self.responses = FakeResponses(list(responses))


def direct_response(
    text: str,
    *,
    response_id: str = "resp-direct",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=response_id,
        output=[],
        output_text=text,
    )


def tool_call_response(
    *,
    tool_name: str = "get_current_utc_time",
    arguments: str = "{}",
    call_id: str = "call-1",
    response_id: str = "resp-tool",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=response_id,
        output=[
            SimpleNamespace(
                type="function_call",
                name=tool_name,
                arguments=arguments,
                call_id=call_id,
            )
        ],
        output_text="",
    )


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
        router_max_output_tokens=16,
        router_prompt="Return only general or time.",
    )


def invoke_graph(graph, question: str):
    return graph.invoke(
        {
            "user_input": question,
            "intent": "general",
            "answer": "",
        }
    )


def test_general_question_returns_direct_foundry_answer(settings):
    client = FakeClient(
        direct_response("Resposta direta do Foundry."),
    )
    graph = build_supervisor_graph(
        settings=settings,
        client_factory=lambda: client,
    )

    result = invoke_graph(graph, "Explique o Foundry.")

    assert result["intent"] == "general"
    assert result["answer"] == "Resposta direta do Foundry."
    assert len(client.responses.calls) == 1

    first_call = client.responses.calls[0]
    assert first_call["model"] == "test-deployment"
    assert first_call["instructions"] == settings.system_prompt
    assert first_call["input"] == "Explique o Foundry."
    assert "tools" in first_call


def test_time_question_executes_tool_and_requests_final_answer(settings):
    client = FakeClient(
        tool_call_response(),
        direct_response(
            "Agora são 13:35 UTC.",
            response_id="resp-final",
        ),
    )
    graph = build_supervisor_graph(
        settings=settings,
        client_factory=lambda: client,
    )

    result = invoke_graph(graph, "Que horas são em UTC?")

    assert result["intent"] == "time"
    assert result["answer"] == "Agora são 13:35 UTC."
    assert len(client.responses.calls) == 2

    second_call = client.responses.calls[1]
    assert second_call["previous_response_id"] == "resp-tool"
    assert second_call["input"][0]["type"] == "function_call_output"
    assert second_call["input"][0]["call_id"] == "call-1"
    assert "UTC" in second_call["input"][0]["output"]


def test_empty_input_is_rejected_before_calling_foundry(settings):
    client = FakeClient(
        direct_response("unused"),
    )
    graph = build_supervisor_graph(
        settings=settings,
        client_factory=lambda: client,
    )

    with pytest.raises(ValueError, match="user_input cannot be empty"):
        invoke_graph(graph, "   ")

    assert client.responses.calls == []


def test_unknown_tool_is_rejected(settings):
    client = FakeClient(
        tool_call_response(tool_name="unknown_tool"),
    )
    graph = build_supervisor_graph(
        settings=settings,
        client_factory=lambda: client,
    )

    with pytest.raises(RuntimeError, match="unknown tool"):
        invoke_graph(graph, "Use uma ferramenta desconhecida.")


def test_invalid_tool_arguments_are_rejected(settings):
    client = FakeClient(
        tool_call_response(arguments="{invalid-json"),
    )
    graph = build_supervisor_graph(
        settings=settings,
        client_factory=lambda: client,
    )

    with pytest.raises(RuntimeError, match="invalid arguments"):
        invoke_graph(graph, "Que horas são?")


def test_empty_direct_response_is_rejected(settings):
    client = FakeClient(
        direct_response(""),
    )
    graph = build_supervisor_graph(
        settings=settings,
        client_factory=lambda: client,
    )

    with pytest.raises(RuntimeError, match="empty response"):
        invoke_graph(graph, "Explique o Foundry.")


def test_empty_final_response_after_tool_is_rejected(settings):
    client = FakeClient(
        tool_call_response(),
        direct_response("", response_id="resp-final"),
    )
    graph = build_supervisor_graph(
        settings=settings,
        client_factory=lambda: client,
    )

    with pytest.raises(
        RuntimeError,
        match="empty response after tool execution",
    ):
        invoke_graph(graph, "Que horas são em UTC?")
