from __future__ import annotations

from collections import deque
from types import SimpleNamespace

from shared.foundry_service import FoundryService
from shared.settings import AppSettings


class FakeResponses:
    def __init__(self, *responses):
        self._responses = deque(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.popleft()


class FakeClient:
    def __init__(self, *responses):
        self.responses = FakeResponses(*responses)


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


def test_ask_normalizes_direct_response():
    client = FakeClient(
        SimpleNamespace(
            id="resp-1",
            output=[],
            output_text="Resposta direta.",
        )
    )
    service = FoundryService(
        settings=settings(),
        client_factory=lambda: client,
    )

    response = service.ask(
        user_input="Olá",
        tools=[{"type": "function", "name": "example"}],
    )

    assert response.response_id == "resp-1"
    assert response.output_text == "Resposta direta."
    assert response.tool_calls == ()
    assert client.responses.calls[0]["tools"][0]["name"] == "example"


def test_ask_normalizes_function_call():
    client = FakeClient(
        SimpleNamespace(
            id="resp-tool",
            output=[
                SimpleNamespace(
                    type="function_call",
                    name="get_current_utc_time",
                    arguments="{}",
                    call_id="call-1",
                )
            ],
            output_text="",
        )
    )
    service = FoundryService(
        settings=settings(),
        client_factory=lambda: client,
    )

    response = service.ask(
        user_input="Que horas são?",
        tools=[{"type": "function", "name": "get_current_utc_time"}],
    )

    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "get_current_utc_time"
    assert response.tool_calls[0].call_id == "call-1"


def test_continue_after_tools_uses_previous_response_id():
    client = FakeClient(
        SimpleNamespace(
            id="resp-final",
            output=[],
            output_text="Resposta final.",
        )
    )
    service = FoundryService(
        settings=settings(),
        client_factory=lambda: client,
    )

    response = service.continue_after_tools(
        previous_response_id="resp-tool",
        tool_outputs=[
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "resultado",
            }
        ],
    )

    assert response.output_text == "Resposta final."
    assert client.responses.calls[0]["previous_response_id"] == "resp-tool"
