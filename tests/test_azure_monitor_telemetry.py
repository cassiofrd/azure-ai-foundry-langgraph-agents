from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

import shared.telemetry as telemetry
from shared.settings import AppSettings


@pytest.fixture
def settings() -> AppSettings:
    return AppSettings(
        foundry_project_endpoint="https://example.test/project",
        foundry_model_deployment="chat",
        foundry_embedding_deployment="embedding",
        azure_openai_endpoint="https://example.test/openai",
        azure_openai_api_key="secret",
        app_name="test-agent",
        app_environment="test",
        app_version="1.0.0",
        model_max_output_tokens=800,
        system_prompt="test",
        router_max_output_tokens=16,
        router_prompt="test",
    )


def test_build_telemetry_sink_returns_none_when_disabled(
    settings: AppSettings,
) -> None:
    disabled = replace(
        settings,
        telemetry_console_enabled=False,
        azure_monitor_enabled=False,
    )

    assert telemetry.build_telemetry_sink(disabled) is None


def test_composite_sink_emits_console_and_azure(
    monkeypatch: pytest.MonkeyPatch,
    settings: AppSettings,
) -> None:
    enabled = replace(
        settings,
        telemetry_console_enabled=True,
        azure_monitor_enabled=True,
        applicationinsights_connection_string=(
            "InstrumentationKey="
            "00000000-0000-0000-0000-000000000000"
        ),
    )
    calls: list[str] = []

    fake_logger = object()

    monkeypatch.setattr(
        telemetry,
        "initialize_azure_monitor",
        lambda *_args, **_kwargs: fake_logger,
    )
    monkeypatch.setattr(
        telemetry,
        "emit_execution_summary",
        lambda *_args, **_kwargs: calls.append("console"),
    )
    monkeypatch.setattr(
        telemetry,
        "emit_execution_to_azure_monitor",
        lambda *_args, **_kwargs: calls.append("azure"),
    )

    sink = telemetry.build_telemetry_sink(enabled)

    assert sink is not None

    sink(object())

    assert calls == ["console", "azure"]


def test_azure_attributes_are_flat_and_queryable(
    settings: AppSettings,
) -> None:
    context = SimpleNamespace(
        to_dict=lambda: {
            "execution_id": "exec-123",
            "status": "success",
            "duration_ms": 1500.5,
            "metrics": {
                "llm_call_count": 2,
                "successful_llm_calls": 2,
                "total_llm_duration_ms": 1400.0,
                "total_input_tokens": 100,
                "total_output_tokens": 50,
                "total_tokens": 150,
                "tool_call_count": 1,
                "successful_tool_calls": 1,
                "total_tool_duration_ms": 0.1,
                "error_count": 0,
            },
            "tools": [
                {
                    "tool_name": "get_current_utc_time",
                    "error_type": None,
                }
            ],
            "errors": [],
        }
    )

    attributes = telemetry._build_azure_monitor_attributes(
        context,
        settings,
    )

    assert attributes["agent.execution_id"] == "exec-123"
    assert attributes["agent.total_tokens"] == 150
    assert attributes["agent.tool_names"] == "get_current_utc_time"
    assert (
        attributes["microsoft.custom_event.name"]
        == "agent.execution.completed"
    )
