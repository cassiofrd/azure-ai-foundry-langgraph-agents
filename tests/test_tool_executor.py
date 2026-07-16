from __future__ import annotations

import pytest

from shared.foundry_service import ToolCall
from shared.tool_executor import ToolExecutor


def test_executes_registered_tool_and_builds_output():
    def example_tool(value: int) -> int:
        return value * 2

    executor = ToolExecutor(
        tool_registry={
            "example_tool": example_tool,
        }
    )

    result = executor.execute(
        (
            ToolCall(
                name="example_tool",
                arguments='{"value": 21}',
                call_id="call-1",
            ),
        )
    )

    assert result.executed_tool_names == ["example_tool"]
    assert result.tool_outputs == [
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "42",
        }
    ]


def test_executes_multiple_tools_in_order():
    def first_tool() -> str:
        return "first-result"

    def second_tool(name: str) -> str:
        return f"Hello, {name}"

    executor = ToolExecutor(
        tool_registry={
            "first_tool": first_tool,
            "second_tool": second_tool,
        }
    )

    result = executor.execute(
        (
            ToolCall(
                name="first_tool",
                arguments="{}",
                call_id="call-1",
            ),
            ToolCall(
                name="second_tool",
                arguments='{"name": "Cássio"}',
                call_id="call-2",
            ),
        )
    )

    assert result.executed_tool_names == [
        "first_tool",
        "second_tool",
    ]

    assert result.tool_outputs == [
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "first-result",
        },
        {
            "type": "function_call_output",
            "call_id": "call-2",
            "output": "Hello, Cássio",
        },
    ]


def test_unknown_tool_is_rejected():
    executor = ToolExecutor(
        tool_registry={}
    )

    with pytest.raises(RuntimeError, match="unknown tool"):
        executor.execute(
            (
                ToolCall(
                    name="missing_tool",
                    arguments="{}",
                    call_id="call-1",
                ),
            )
        )


def test_invalid_json_arguments_are_rejected():
    executor = ToolExecutor(
        tool_registry={
            "example_tool": lambda: "unused",
        }
    )

    with pytest.raises(
        RuntimeError,
        match="invalid arguments",
    ):
        executor.execute(
            (
                ToolCall(
                    name="example_tool",
                    arguments="{invalid-json",
                    call_id="call-1",
                ),
            )
        )


def test_non_object_arguments_are_rejected():
    executor = ToolExecutor(
        tool_registry={
            "example_tool": lambda: "unused",
        }
    )

    with pytest.raises(
        RuntimeError,
        match="must be a JSON object",
    ):
        executor.execute(
            (
                ToolCall(
                    name="example_tool",
                    arguments='["not", "an", "object"]',
                    call_id="call-1",
                ),
            )
        )


def test_empty_tool_call_collection_returns_empty_result():
    executor = ToolExecutor(
        tool_registry={}
    )

    result = executor.execute(())

    assert result.executed_tool_names == []
    assert result.tool_outputs == []