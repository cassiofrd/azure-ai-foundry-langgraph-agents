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


def direct_response(text: str, *, response_id: str = "resp-direct"):
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
):
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


def invoke_graph(
    graph,
    question: str,
    conversation_response_id: str | None = None,
):
    return graph.invoke(
        {
            "user_input": question,
            "intent": "general",
            "answer": "",
            "conversation_response_id": conversation_response_id,
        }
    )


def test_general_question_returns_direct_foundry_answer(settings):
    client = FakeClient(
        direct_response(
            "Resposta direta do Foundry.",
            response_id="resp-1",
        ),
    )
    graph = build_supervisor_graph(
        settings=settings,
        client_factory=lambda: client,
    )

    result = invoke_graph(graph, "Explique o Foundry.")

    assert result["intent"] == "general"
    assert result["answer"] == "Resposta direta do Foundry."
    assert result["conversation_response_id"] == "resp-1"
    assert len(client.responses.calls) == 1


def test_second_turn_uses_previous_response_id(settings):
    client = FakeClient(
        direct_response(
            "Seu nome é Cássio.",
            response_id="resp-2",
        ),
    )
    graph = build_supervisor_graph(
        settings=settings,
        client_factory=lambda: client,
    )

    result = invoke_graph(
        graph,
        "Qual é o meu nome?",
        conversation_response_id="resp-1",
    )

    assert result["conversation_response_id"] == "resp-2"
    assert (
        client.responses.calls[0]["previous_response_id"]
        == "resp-1"
    )


def test_time_question_stores_final_response_as_memory(settings):
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
    assert result["conversation_response_id"] == "resp-final"
    assert len(client.responses.calls) == 2


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
