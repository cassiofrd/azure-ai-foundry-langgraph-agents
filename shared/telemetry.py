from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from shared.execution_context import ExecutionContext


OutputWriter = Callable[[str], Any]


def build_execution_summary(
    execution_context: ExecutionContext,
) -> dict[str, Any]:
    """Build the transport-neutral payload used by local and Azure telemetry."""
    return execution_context.to_dict()


def format_execution_summary(
    execution_context: ExecutionContext,
) -> str:
    """Format a concise human-readable summary for local development."""
    summary = build_execution_summary(execution_context)
    metrics = summary["metrics"]

    duration = summary["duration_ms"]
    duration_text = (
        f"{duration:.2f} ms"
        if duration is not None
        else "not finished"
    )

    lines = [
        "",
        "Execution Summary",
        "-----------------",
        f"Execution ID: {summary['execution_id']}",
        f"Status: {summary['status']}",
        f"Duration: {duration_text}",
        "",
        (
            "LLM calls: "
            f"{metrics['llm_call_count']} "
            f"({metrics['successful_llm_calls']} successful)"
        ),
        f"LLM duration: {metrics['total_llm_duration_ms']:.2f} ms",
        f"Input tokens: {metrics['total_input_tokens']}",
        f"Output tokens: {metrics['total_output_tokens']}",
        f"Total tokens: {metrics['total_tokens']}",
        "",
        (
            "Tool calls: "
            f"{metrics['tool_call_count']} "
            f"({metrics['successful_tool_calls']} successful)"
        ),
    ]

    for tool in summary["tools"]:
        outcome = "success" if tool["error_type"] is None else "failed"
        lines.append(
            f"- {tool['tool_name']}: "
            f"{tool['duration_ms']:.2f} ms [{outcome}]"
        )

    lines.extend(
        [
            f"Tool duration: {metrics['total_tool_duration_ms']:.2f} ms",
            "",
            f"Errors: {metrics['error_count']}",
        ]
    )

    for error in summary["errors"]:
        lines.append(
            f"- {error['component']}.{error['operation']}: "
            f"{error['error_type']}: {error['error_message']}"
        )

    return "\n".join(lines)


def emit_execution_summary(
    execution_context: ExecutionContext,
    *,
    writer: OutputWriter = print,
) -> None:
    """Emit the local terminal representation."""
    writer(format_execution_summary(execution_context))


def emit_execution_json(
    execution_context: ExecutionContext,
    *,
    writer: OutputWriter = print,
    indent: int | None = None,
) -> None:
    """Emit structured JSON, useful before Application Insights integration."""
    writer(
        json.dumps(
            build_execution_summary(execution_context),
            ensure_ascii=False,
            indent=indent,
        )
    )
