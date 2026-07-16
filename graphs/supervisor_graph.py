from __future__ import annotations

import json
from typing import Callable

from langgraph.graph import END, START, StateGraph

from shared.foundry_client import ResponsesClient
from shared.foundry_service import FoundryService
from shared.settings import AppSettings
from shared.state import SupervisorState
from shared.tools import TOOL_REGISTRY, TOOLS


def build_supervisor_graph(
    *,
    settings: AppSettings,
    client_factory: Callable[[], ResponsesClient],
):
    foundry_service = FoundryService(
        settings=settings,
        client_factory=client_factory,
    )

    def call_foundry_with_tools(
        state: SupervisorState,
    ) -> SupervisorState:
        user_input = state["user_input"].strip()

        if not user_input:
            raise ValueError("user_input cannot be empty.")

        response = foundry_service.ask(
            user_input=user_input,
            tools=TOOLS,
        )

        if not response.tool_calls:
            if not response.output_text:
                raise RuntimeError("Foundry returned an empty response.")

            return {
                "user_input": user_input,
                "intent": "general",
                "answer": response.output_text,
            }

        tool_outputs: list[dict[str, str]] = []
        executed_tool_names: list[str] = []

        for tool_call in response.tool_calls:
            tool = TOOL_REGISTRY.get(tool_call.name)

            if tool is None:
                raise RuntimeError(
                    f"Foundry requested an unknown tool: {tool_call.name}."
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
                    f"Tool arguments for {tool_call.name} "
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

        final_response = foundry_service.continue_after_tools(
            previous_response_id=response.response_id,
            tool_outputs=tool_outputs,
        )

        if not final_response.output_text:
            raise RuntimeError(
                "Foundry returned an empty response after tool execution."
            )

        intent = (
            "time"
            if "get_current_utc_time" in executed_tool_names
            else "general"
        )

        return {
            "user_input": user_input,
            "intent": intent,
            "answer": final_response.output_text,
        }

    graph = StateGraph(SupervisorState)
    graph.add_node(
        "call_foundry_with_tools",
        call_foundry_with_tools,
    )
    graph.add_edge(START, "call_foundry_with_tools")
    graph.add_edge("call_foundry_with_tools", END)

    return graph.compile()
