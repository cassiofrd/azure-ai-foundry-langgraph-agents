from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

from graphs.supervisor_graph import build_supervisor_graph
from shared.settings import AppSettings


class FakeResponses:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("No fake response was configured.")
        return self.responses.popleft()


class FakeClient:
    def __init__(self, *responses):
        self.responses = FakeResponses(responses)


def direct_response(text: str, response_id: str = "resp-direct"):
    return SimpleNamespace(
        id=response_id,
        output=[],
        output_text=text,
    )


def tool_call_response(name: str, response_id: str = "resp-tool"):
    return SimpleNamespace(
        id=response_id,
        output=[
            SimpleNamespace(
                type="function_call",
                name=name,
                arguments='{"query":"M10"}' if "search" in name else "{}",
                call_id="call-1",
            )
        ],
        output_text="",
    )


@pytest.fixture
def settings() -> AppSettings:
    return AppSettings(
        foundry_project_endpoint="https://example.services.ai.azure.com/api/projects/example",
        foundry_model_deployment="test-deployment",
        foundry_embedding_deployment="embedding-test-deployment",
        azure_openai_endpoint="https://example.openai.azure.com",
        azure_openai_api_key="test-api-key",
        app_name="test-app",
        app_environment="test",
        app_version="0.1.0",
        model_max_output_tokens=200,
        system_prompt="general prompt",
        router_max_output_tokens=16,
        router_prompt="router prompt",
        inventory_prompt="inventory prompt",
        supplier_prompt="supplier prompt",
        time_prompt="time prompt",
    )


def invoke_graph(graph, question, conversation_response_id=None):
    return graph.invoke(
        {
            "user_input": question,
            "intent": "general",
            "agent": "general",
            "answer": "",
            "conversation_response_id": conversation_response_id,
        }
    )


def test_general_question_routes_and_returns_direct_answer(settings):
    client = FakeClient(
        direct_response("general", "router-1"),
        direct_response("Resposta direta.", "resp-1"),
    )
    graph = build_supervisor_graph(
        settings=settings,
        client_factory=lambda: client,
        telemetry_sink=None,
    )
    result = invoke_graph(graph, "Explique o Foundry.")

    assert result["intent"] == "general"
    assert result["agent"] == "general"
    assert result["answer"] == "Resposta direta."
    assert result["conversation_response_id"] == "resp-1"
    assert client.responses.calls[0]["instructions"] == "router prompt"
    assert "tools" not in client.responses.calls[1]


def test_inventory_route_exposes_only_inventory_tool(settings, monkeypatch):
    monkeypatch.setitem(
        __import__("shared.tools", fromlist=["TOOL_REGISTRY"]).TOOL_REGISTRY,
        "search_inventory_documents",
        lambda query: "inventory result",
    )
    client = FakeClient(
        direct_response("inventory", "router-1"),
        tool_call_response("search_inventory_documents"),
        direct_response("Política encontrada.", "resp-final"),
    )
    graph = build_supervisor_graph(
        settings=settings,
        client_factory=lambda: client,
        telemetry_sink=None,
    )
    result = invoke_graph(graph, "Qual é a política do M10?")

    assert result["intent"] == "inventory"
    assert result["agent"] == "inventory"
    assert result["answer"] == "Política encontrada."
    assert [tool["name"] for tool in client.responses.calls[1]["tools"]] == [
        "search_inventory_documents"
    ]
    assert client.responses.calls[1]["instructions"] == "inventory prompt"
    assert client.responses.calls[1]["tool_choice"] == "required"
    assert client.responses.calls[2]["instructions"] == "inventory prompt"


def test_supplier_route_exposes_only_supplier_tool(settings, monkeypatch):
    monkeypatch.setitem(
        __import__("shared.tools", fromlist=["TOOL_REGISTRY"]).TOOL_REGISTRY,
        "search_supplier_documents",
        lambda query: "supplier result",
    )
    client = FakeClient(
        direct_response("supplier", "router-1"),
        tool_call_response("search_supplier_documents"),
        direct_response("Fornecedor encontrado.", "resp-final"),
    )
    graph = build_supervisor_graph(
        settings=settings,
        client_factory=lambda: client,
        telemetry_sink=None,
    )
    result = invoke_graph(graph, "Quem fornece o M10?")

    assert result["intent"] == "supplier"
    assert result["agent"] == "supplier"
    assert [tool["name"] for tool in client.responses.calls[1]["tools"]] == [
        "search_supplier_documents"
    ]
    assert client.responses.calls[1]["tool_choice"] == "required"


def test_time_route_uses_time_tool(settings):
    client = FakeClient(
        direct_response("time", "router-1"),
        tool_call_response("get_current_utc_time"),
        direct_response("Agora são 13:35 UTC.", "resp-final"),
    )
    graph = build_supervisor_graph(
        settings=settings,
        client_factory=lambda: client,
        telemetry_sink=None,
    )
    result = invoke_graph(graph, "Que horas são em UTC?")

    assert result["intent"] == "time"
    assert result["agent"] == "time"
    assert result["answer"] == "Agora são 13:35 UTC."


def test_invalid_router_output_falls_back_to_general(settings):
    client = FakeClient(
        direct_response("unknown-route", "router-1"),
        direct_response("Resposta geral.", "resp-1"),
    )
    graph = build_supervisor_graph(
        settings=settings,
        client_factory=lambda: client,
        telemetry_sink=None,
    )
    result = invoke_graph(graph, "Olá")
    assert result["intent"] == "general"


def test_second_turn_uses_memory_only_in_specialist_call(settings):
    client = FakeClient(
        direct_response("general", "router-2"),
        direct_response("Seu nome é Cássio.", "resp-2"),
    )
    graph = build_supervisor_graph(
        settings=settings,
        client_factory=lambda: client,
        telemetry_sink=None,
    )
    result = invoke_graph(graph, "Qual é meu nome?", "resp-1")

    assert "previous_response_id" not in client.responses.calls[0]
    assert client.responses.calls[1]["previous_response_id"] == "resp-1"
    assert result["conversation_response_id"] == "resp-2"


def test_empty_input_is_rejected(settings):
    client = FakeClient()
    graph = build_supervisor_graph(
        settings=settings,
        client_factory=lambda: client,
        telemetry_sink=None,
    )
    with pytest.raises(ValueError, match="user_input cannot be empty"):
        invoke_graph(graph, "   ")


def test_inventory_supplier_route_runs_both_specialists_and_synthesizes(
    settings,
    monkeypatch,
):
    registry = __import__(
        "shared.tools",
        fromlist=["TOOL_REGISTRY"],
    ).TOOL_REGISTRY
    monkeypatch.setitem(
        registry,
        "search_inventory_documents",
        lambda query: "inventory result",
    )
    monkeypatch.setitem(
        registry,
        "search_supplier_documents",
        lambda query: "supplier result",
    )
    client = FakeClient(
        direct_response("inventory_supplier", "router-1"),
        tool_call_response("search_inventory_documents", "inv-tool"),
        direct_response("Inventory answer with Evidence.", "inv-final"),
        tool_call_response("search_supplier_documents", "sup-tool"),
        direct_response("Supplier answer with Evidence.", "sup-final"),
        direct_response("Combined grounded answer.", "synthesis-final"),
    )
    graph = build_supervisor_graph(
        settings=settings,
        client_factory=lambda: client,
        telemetry_sink=None,
    )

    result = invoke_graph(
        graph,
        "O fornecedor do M10 atende à política de estoque?",
    )

    assert result["intent"] == "inventory_supplier"
    assert result["agent"] == "supervisor"
    assert result["answer"] == "Combined grounded answer."
    assert result["conversation_response_id"] == "synthesis-final"
    assert result["specialist_outputs"] == {
        "inventory": "Inventory answer with Evidence.",
        "supplier": "Supplier answer with Evidence.",
    }
    assert "exact entity_id M10" in result["specialist_queries"]["inventory"]
    assert "exact entity_id M10" in result["specialist_queries"]["supplier"]
    assert "target stock level" in client.responses.calls[1]["input"]
    assert "contractual lead time" in client.responses.calls[3]["input"]
    assert client.responses.calls[1]["input"] != result["user_input"]
    assert client.responses.calls[3]["input"] != result["user_input"]
    assert [tool["name"] for tool in client.responses.calls[1]["tools"]] == [
        "search_inventory_documents"
    ]
    assert [tool["name"] for tool in client.responses.calls[3]["tools"]] == [
        "search_supplier_documents"
    ]
    synthesis_call = client.responses.calls[5]
    assert synthesis_call["instructions"] == settings.multi_agent_prompt
    assert "Inventory answer with Evidence." in synthesis_call["input"]
    assert "Supplier answer with Evidence." in synthesis_call["input"]
    assert "tools" not in synthesis_call


def test_composite_route_does_not_branch_from_previous_response(settings):
    client = FakeClient(
        direct_response("inventory_supplier", "router-1"),
        direct_response("Inventory answer.", "inv-final"),
        direct_response("Supplier answer.", "sup-final"),
        direct_response("Combined answer.", "synthesis-final"),
    )
    graph = build_supervisor_graph(
        settings=settings,
        client_factory=lambda: client,
        telemetry_sink=None,
    )

    result = invoke_graph(
        graph,
        "Compare estoque e fornecedor do M10.",
        "previous-conversation-response",
    )

    assert result["answer"] == "Combined answer."
    assert all(
        "previous_response_id" not in call
        for call in (
            client.responses.calls[0],
            client.responses.calls[1],
            client.responses.calls[2],
            client.responses.calls[3],
        )
    )


def test_logistics_route_exposes_only_logistics_tool(settings, monkeypatch):
    registry = __import__(
        "shared.tools", fromlist=["TOOL_REGISTRY"]
    ).TOOL_REGISTRY
    monkeypatch.setitem(
        registry,
        "search_logistics_documents",
        lambda query: "logistics result",
    )
    client = FakeClient(
        direct_response("logistics", "router-1"),
        tool_call_response("search_logistics_documents"),
        direct_response("Política logística encontrada.", "resp-final"),
    )
    graph = build_supervisor_graph(
        settings=settings,
        client_factory=lambda: client,
        telemetry_sink=None,
    )

    result = invoke_graph(graph, "Qual modal devo usar para uma urgência?")

    assert result["intent"] == "logistics"
    assert result["agent"] == "logistics"
    assert result["answer"] == "Política logística encontrada."
    assert [tool["name"] for tool in client.responses.calls[1]["tools"]] == [
        "search_logistics_documents"
    ]
    assert client.responses.calls[1]["instructions"] == settings.logistics_prompt


def test_three_specialist_route_runs_all_agents_and_synthesizes(
    settings,
    monkeypatch,
):
    registry = __import__(
        "shared.tools", fromlist=["TOOL_REGISTRY"]
    ).TOOL_REGISTRY
    for name, result in (
        ("search_inventory_documents", "inventory result"),
        ("search_supplier_documents", "supplier result"),
        ("search_logistics_documents", "logistics result"),
    ):
        monkeypatch.setitem(registry, name, lambda query, value=result: value)

    client = FakeClient(
        direct_response("inventory_supplier_logistics", "router-1"),
        tool_call_response("search_inventory_documents", "inv-tool"),
        direct_response("Inventory evidence.", "inv-final"),
        tool_call_response("search_supplier_documents", "sup-tool"),
        direct_response("Supplier evidence.", "sup-final"),
        tool_call_response("search_logistics_documents", "log-tool"),
        direct_response("Logistics evidence.", "log-final"),
        direct_response("Plano consolidado em português.", "synthesis-final"),
    )
    graph = build_supervisor_graph(
        settings=settings,
        client_factory=lambda: client,
        telemetry_sink=None,
    )

    result = invoke_graph(
        graph,
        "Crie um plano para o M10 considerando estoque, fornecedor e logística.",
    )

    assert result["intent"] == "inventory_supplier_logistics"
    assert result["agent"] == "supervisor"
    assert list(result["specialist_outputs"]) == [
        "inventory",
        "supplier",
        "logistics",
    ]
    assert set(result["specialist_queries"]) == {
        "inventory",
        "supplier",
        "logistics",
    }
    assert "M10" not in result["specialist_queries"]["logistics"]
    assert result["answer"] == "Plano consolidado em português."
    synthesis_call = client.responses.calls[7]
    assert "Original user request" in synthesis_call["input"]
    assert "Logistics Agent output" in synthesis_call["input"]
    assert synthesis_call["instructions"] == settings.multi_agent_prompt
