from __future__ import annotations

from typing import Callable

from langgraph.graph import END, START, StateGraph

from shared.execution_context import ExecutionContext
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
        execution_context = ExecutionContext()
        user_input = state["user_input"].strip()

        if not user_input:
            error = ValueError("user_input cannot be empty.")
            execution_context.record_error(
                component="supervisor_graph",
                operation="validate_input",
                error=error,
            )
            execution_context.finish()
            raise error

        try:
            response = foundry_service.ask(
                user_input=user_input,
                tools=TOOLS,
                previous_response_id=state.get(
                    "conversation_response_id"
                ),
                execution_context=execution_context,
            )

            if not response.tool_calls:
                if not response.output_text:
                    raise RuntimeError(
                        "Foundry returned an empty response."
                    )

                execution_context.finish()
                return {
                    "user_input": user_input,
                    "intent": "general",
                    "answer": response.output_text,
                    "conversation_response_id": response.response_id,
                    "execution": execution_context,
                }

            execution = tool_executor.execute(
                response.tool_calls,
                execution_context=execution_context,
            )

            final_response = foundry_service.continue_after_tools(
                previous_response_id=response.response_id,
                tool_outputs=execution.tool_outputs,
                execution_context=execution_context,
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

            execution_context.finish()
            return {
                "user_input": user_input,
                "intent": intent,
                "answer": final_response.output_text,
                "conversation_response_id": final_response.response_id,
                "execution": execution_context,
            }
        except Exception as exc:
            already_recorded = any(
                error.error_type == type(exc).__name__
                and error.error_message == str(exc)
                for error in execution_context.errors
            )
            if not already_recorded:
                execution_context.record_error(
                    component="supervisor_graph",
                    operation="call_foundry_with_tools",
                    error=exc,
                )
            execution_context.finish()
            raise

    graph = StateGraph(SupervisorState)
    graph.add_node(
        "call_foundry_with_tools",
        call_foundry_with_tools,
    )
    graph.add_edge(START, "call_foundry_with_tools")
    graph.add_edge("call_foundry_with_tools", END)

    return graph.compile()
