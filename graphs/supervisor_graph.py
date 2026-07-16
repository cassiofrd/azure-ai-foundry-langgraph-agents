from __future__ import annotations

import json
from typing import Callable

from langgraph.graph import END, START, StateGraph

from shared.foundry_client import ResponsesClient
from shared.settings import AppSettings
from shared.state import SupervisorState
from shared.tools import TOOL_REGISTRY, TOOLS


def build_supervisor_graph(
    *,
    settings: AppSettings,
    client_factory: Callable[[], ResponsesClient],
):
    def call_foundry_with_tools(
        state: SupervisorState,
    ) -> SupervisorState:
        user_input = state["user_input"].strip()

        if not user_input:
            raise ValueError("user_input cannot be empty.")

        client = client_factory()

        response = client.responses.create(
            model=settings.foundry_model_deployment,
            instructions=settings.system_prompt,
            input=user_input,
            tools=TOOLS,
            max_output_tokens=settings.model_max_output_tokens,
        )

        function_calls = [
            item
            for item in response.output
            if getattr(item, "type", None) == "function_call"
        ]

        if not function_calls:
            answer = (response.output_text or "").strip()

            if not answer:
                raise RuntimeError("Foundry returned an empty response.")

            return {
                "user_input": user_input,
                "intent": "general",
                "answer": answer,
            }

        tool_outputs: list[dict[str, str]] = []
        executed_tool_names: list[str] = []

        for function_call in function_calls:
            tool_name = function_call.name
            tool = TOOL_REGISTRY.get(tool_name)

            if tool is None:
                raise RuntimeError(
                    f"Foundry requested an unknown tool: {tool_name}."
                )

            try:
                arguments = json.loads(function_call.arguments or "{}")
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Foundry returned invalid arguments for {tool_name}."
                ) from exc

            if not isinstance(arguments, dict):
                raise RuntimeError(
                    f"Tool arguments for {tool_name} must be a JSON object."
                )

            tool_result = tool(**arguments)
            executed_tool_names.append(tool_name)

            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": function_call.call_id,
                    "output": str(tool_result),
                }
            )

        final_response = client.responses.create(
            model=settings.foundry_model_deployment,
            previous_response_id=response.id,
            input=tool_outputs,
            max_output_tokens=settings.model_max_output_tokens,
        )

        answer = (final_response.output_text or "").strip()

        if not answer:
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
            "answer": answer,
        }

    graph = StateGraph(SupervisorState)
    graph.add_node(
        "call_foundry_with_tools",
        call_foundry_with_tools,
    )
    graph.add_edge(START, "call_foundry_with_tools")
    graph.add_edge("call_foundry_with_tools", END)

    return graph.compile()
