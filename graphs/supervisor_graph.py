from __future__ import annotations

from typing import Callable

from langgraph.graph import END, START, StateGraph

from shared.foundry_client import ResponsesClient
from shared.foundry_service import FoundryService
from shared.settings import AppSettings
from shared.state import SupervisorState
from shared.tool_executor import ToolExecutor
from shared.tools import TOOLS


def build_supervisor_graph(
    *,
    settings: AppSettings,
    client_factory: Callable[[], ResponsesClient],
):
    foundry_service = FoundryService(
        settings=settings,
        client_factory=client_factory,
    )

    tool_executor = ToolExecutor()

    def call_foundry_with_tools(
        state: SupervisorState,
    ) -> SupervisorState:

        user_input = state["user_input"].strip()

        if not user_input:
            raise ValueError("user_input cannot be empty.")

        response = foundry_service.ask(
            user_input=user_input,
            tools=TOOLS,
            previous_response_id=state.get(
                "conversation_response_id"
            ),
        )

        if not response.tool_calls:

            if not response.output_text:
                raise RuntimeError(
                    "Foundry returned an empty response."
                )

            return {
                "user_input": user_input,
                "intent": "general",
                "answer": response.output_text,
                "conversation_response_id": response.response_id,
            }

        execution = tool_executor.execute(
            response.tool_calls
        )

        final_response = (
            foundry_service.continue_after_tools(
                previous_response_id=response.response_id,
                tool_outputs=execution.tool_outputs,
            )
        )

        if not final_response.output_text:
            raise RuntimeError(
                "Foundry returned an empty response "
                "after tool execution."
            )

        intent = (
            "time"
            if "get_current_utc_time"
            in execution.executed_tool_names
            else "general"
        )

        return {
            "user_input": user_input,
            "intent": intent,
            "answer": final_response.output_text,
            "conversation_response_id": (
                final_response.response_id
            ),
        }

    graph = StateGraph(SupervisorState)

    graph.add_node(
        "call_foundry_with_tools",
        call_foundry_with_tools,
    )

    graph.add_edge(
        START,
        "call_foundry_with_tools",
    )

    graph.add_edge(
        "call_foundry_with_tools",
        END,
    )

    return graph.compile()