from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from shared.foundry_service import ToolCall
from shared.tools import TOOL_REGISTRY


@dataclass(frozen=True)
class ToolExecutionResult:
    tool_outputs: list[dict[str, str]]
    executed_tool_names: list[str]


class ToolExecutor:
    def __init__(
        self,
        tool_registry: dict[str, Callable] | None = None,
    ) -> None:
        self._tool_registry = tool_registry or TOOL_REGISTRY

    def execute(
        self,
        tool_calls: tuple[ToolCall, ...],
    ) -> ToolExecutionResult:

        tool_outputs: list[dict[str, str]] = []
        executed_tool_names: list[str] = []

        for tool_call in tool_calls:

            tool = self._tool_registry.get(tool_call.name)

            if tool is None:
                raise RuntimeError(
                    f"Foundry requested an unknown tool: "
                    f"{tool_call.name}."
                )

            try:
                arguments = json.loads(tool_call.arguments)

            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "Foundry returned invalid arguments for "
                    f"{tool_call.name}."
                ) from exc

            if not isinstance(arguments, dict):
                raise RuntimeError(
                    f"Tool arguments for "
                    f"{tool_call.name} "
                    "must be a JSON object."
                )

            tool_result = tool(**arguments)

            executed_tool_names.append(tool_call.name)

            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": str(tool_result),
                }
            )

        return ToolExecutionResult(
            tool_outputs=tool_outputs,
            executed_tool_names=executed_tool_names,
        )