from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from apps.api.main import ApiRuntime, create_app
from shared.execution_context import ExecutionContext


class FakeGraph:
    def __init__(self):
        self.calls: list[dict] = []

    def invoke(self, state: dict):
        self.calls.append(state)
        execution = ExecutionContext()
        execution.finish()
        return {
            **state,
            "intent": "inventory_supplier",
            "agent": "supervisor",
            "answer": (
                "Resposta fundamentada.\n\nEvidence:\n"
                "- Title: Bolt M10 inventory policy, "
                "Source: internal_inventory_manual, Entity_id: M10\n"
                "- Title: Bolt M10 approved supplier, "
                "Source: supplier_master_data, Entity_id: M10"
            ),
            "specialist_outputs": {
                "inventory": "inventory output",
                "supplier": "supplier output",
            },
            "specialist_queries": {
                "inventory": "inventory query",
                "supplier": "supplier query",
            },
            "execution": execution,
        }


class FakeStore:
    backend_name = "memory"
    is_persistent = False
    startup_warning = None

    def __init__(self):
        self.cleared: list[str] = []

    def clear(self, session_id: str) -> None:
        self.cleared.append(session_id)


def build_client():
    graph = FakeGraph()
    store = FakeStore()
    settings = SimpleNamespace(
        app_name="test-app",
        app_version="1.2.3",
        app_environment="test",
        default_session_id="default-session",
    )
    app = create_app(
        runtime=ApiRuntime(
            settings=settings,
            graph=graph,
            conversation_store=store,
        )
    )
    return TestClient(app), graph, store


def test_health_reports_runtime_configuration():
    client, _, _ = build_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app_name": "test-app",
        "app_version": "1.2.3",
        "environment": "test",
        "conversation_store": "memory",
        "conversation_persistence": False,
        "startup_warning": None,
    }


def test_copilot_returns_structured_response():
    client, graph, _ = build_client()
    response = client.post(
        "/copilot",
        json={"session_id": "demo", "message": "Pergunta"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "demo"
    assert payload["route"] == "inventory_supplier"
    assert payload["specialist"] == "supervisor"
    assert payload["participants"] == ["inventory", "supplier"]
    assert len(payload["evidence"]) == 2
    assert payload["execution"]["status"] == "success"
    assert graph.calls[0]["session_id"] == "demo"


def test_copilot_uses_default_session():
    client, graph, _ = build_client()
    response = client.post(
        "/copilot",
        json={"message": "Pergunta"},
    )
    assert response.status_code == 200
    assert response.json()["session_id"] == "default-session"
    assert graph.calls[0]["session_id"] == "default-session"


def test_copilot_rejects_blank_message():
    client, _, _ = build_client()
    response = client.post(
        "/copilot",
        json={"message": "   "},
    )
    assert response.status_code == 422


def test_delete_session_clears_store():
    client, _, store = build_client()
    response = client.delete("/sessions/demo")
    assert response.status_code == 200
    assert response.json() == {"session_id": "demo", "cleared": True}
    assert store.cleared == ["demo"]
