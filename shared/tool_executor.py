from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable

from shared.execution_context import ExecutionContext, ToolExecutionInfo
from shared.foundry_service import ToolCall
from shared.tools import TOOL_REGISTRY


@dataclass(frozen=True)
class ToolExecutionResult:
    tool_outputs: list[dict[str, str]]
    executed_tool_names: list[str]
    execution_context: ExecutionContext


class ToolExecutor:
    def __init__(
        self,
        tool_registry: dict[str, Callable] | None = None,
    ) -> None:
        self._tool_registry = (
            TOOL_REGISTRY if tool_registry is None else tool_registry
        )

    def execute(
        self,
        tool_calls: tuple[ToolCall, ...],
        execution_context: ExecutionContext | None = None,
    ) -> ToolExecutionResult:
        context = execution_context or ExecutionContext()
        tool_outputs: list[dict[str, str]] = []
        executed_tool_names: list[str] = []

        for tool_call in tool_calls:
            tool = self._tool_registry.get(tool_call.name)

            if tool is None:
                error = RuntimeError(
                    "Foundry requested an unknown tool: "
                    f"{tool_call.name}."
                )
                context.record_error(
                    component="tool_executor",
                    operation=f"resolve:{tool_call.name}",
                    error=error,
                )
                raise error

            try:
                arguments = json.loads(tool_call.arguments)
            except json.JSONDecodeError as exc:
                error = RuntimeError(
                    "Foundry returned invalid arguments for "
                    f"{tool_call.name}."
                )
                context.record_error(
                    component="tool_executor",
                    operation=f"parse_arguments:{tool_call.name}",
                    error=error,
                )
                raise error from exc

            if not isinstance(arguments, dict):
                error = RuntimeError(
                    f"Tool arguments for {tool_call.name} "
                    "must be a JSON object."
                )
                context.record_error(
                    component="tool_executor",
                    operation=f"validate_arguments:{tool_call.name}",
                    error=error,
                )
                raise error

            started = time.perf_counter()

            try:
                tool_result = tool(**arguments)
            except Exception as exc:
                duration_ms = (time.perf_counter() - started) * 1000
                context.tools.append(
                    ToolExecutionInfo(
                        tool_name=tool_call.name,
                        arguments=arguments,
                        duration_ms=duration_ms,
                        call_id=tool_call.call_id,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
                context.record_error(
                    component="tool_executor",
                    operation=f"execute:{tool_call.name}",
                    error=exc,
                )
                raise

            duration_ms = (time.perf_counter() - started) * 1000
            executed_tool_names.append(tool_call.name)
            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": str(tool_result),
                }
            )
            context.tools.append(
                ToolExecutionInfo(
                    tool_name=tool_call.name,
                    arguments=arguments,
                    duration_ms=duration_ms,
                    call_id=tool_call.call_id,
                )
            )

        return ToolExecutionResult(
            tool_outputs=tool_outputs,
            executed_tool_names=executed_tool_names,
            execution_context=context,
        )
