from __future__ import annotations

import json

from shared.execution_context import (
    ExecutionContext,
    LLMCallInfo,
    ToolExecutionInfo,
)
from shared.telemetry import (
    build_execution_summary,
    emit_execution_json,
    format_execution_summary,
)


def populated_execution() -> ExecutionContext:
    execution = ExecutionContext()
    execution.llm_calls.extend(
        [
            LLMCallInfo(
                operation="initial_response",
                model="test-model",
                duration_ms=100.0,
                response_id="resp-1",
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                returned_tool_count=1,
            ),
            LLMCallInfo(
                operation="continue_after_tools",
                model="test-model",
                duration_ms=50.0,
                response_id="resp-2",
                input_tokens=8,
                output_tokens=7,
                total_tokens=15,
            ),
        ]
    )
    execution.tools.append(
        ToolExecutionInfo(
            tool_name="example_tool",
            arguments={"value": 1},
            duration_ms=25.0,
            call_id="call-1",
        )
    )
    execution.finish()
    return execution


def test_build_execution_summary_aggregates_metrics():
    summary = build_execution_summary(populated_execution())

    assert summary["status"] == "success"
    assert summary["metrics"]["llm_call_count"] == 2
    assert summary["metrics"]["tool_call_count"] == 1
    assert summary["metrics"]["total_input_tokens"] == 18
    assert summary["metrics"]["total_output_tokens"] == 12
    assert summary["metrics"]["total_tokens"] == 30
    assert summary["metrics"]["total_llm_duration_ms"] == 150.0
    assert summary["metrics"]["total_tool_duration_ms"] == 25.0


def test_format_execution_summary_contains_key_information():
    output = format_execution_summary(populated_execution())

    assert "Execution Summary" in output
    assert "Status: success" in output
    assert "LLM calls: 2 (2 successful)" in output
    assert "Total tokens: 30" in output
    assert "example_tool: 25.00 ms [success]" in output
    assert "Errors: 0" in output


def test_emit_execution_json_is_valid_json():
    captured: list[str] = []

    emit_execution_json(
        populated_execution(),
        writer=captured.append,
    )

    payload = json.loads(captured[0])
    assert payload["status"] == "success"
    assert payload["metrics"]["total_tokens"] == 30


def test_failed_execution_has_failed_status():
    execution = ExecutionContext()
    execution.record_error(
        component="test",
        operation="run",
        error=RuntimeError("boom"),
    )
    execution.finish()

    assert execution.status == "failed"
    assert build_execution_summary(execution)["metrics"]["error_count"] == 1
