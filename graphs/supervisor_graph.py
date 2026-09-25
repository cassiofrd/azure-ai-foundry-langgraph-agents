from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph

from shared.conversation_store import (
    ConversationStore,
    build_memory_context,
    resolve_entity_reference,
)
from shared.execution_context import ExecutionContext
from shared.foundry_client import ResponsesClient
from shared.foundry_service import FoundryService
from shared.query_planner import extract_entity_id, plan_specialist_queries
from shared.settings import AppSettings
from shared.state import AgentName, Intent, SupervisorState
from shared.telemetry import TelemetrySink, build_telemetry_sink
from shared.tool_executor import ToolExecutor
from shared.tools import (
    DEMAND_TOOLS,
    INVENTORY_TOOLS,
    LOGISTICS_TOOLS,
    SUPPLIER_TOOLS,
    TIME_TOOLS,
)


_DEFAULT_TELEMETRY_SINK = object()
_VALID_ROUTES = {
    "general",
    "time",
    "inventory",
    "supplier",
    "logistics",
    "inventory_supplier",
    "inventory_logistics",
    "supplier_logistics",
    "inventory_supplier_logistics",
}
_DOMAIN_ORDER = ("inventory", "supplier", "logistics")


def _route_specialists(route: Intent) -> tuple[str, ...]:
    return tuple(
        domain
        for domain in _DOMAIN_ORDER
        if domain in route.split("_")
    )


def build_supervisor_graph(
    *,
    settings: AppSettings,
    client_factory: Callable[[], ResponsesClient],
    telemetry_sink: TelemetrySink | None | Any = _DEFAULT_TELEMETRY_SINK,
    conversation_store: ConversationStore | None = None,
):
    if telemetry_sink is _DEFAULT_TELEMETRY_SINK:
        telemetry_sink = build_telemetry_sink(settings)

    foundry_service = FoundryService(
        settings=settings,
        client_factory=client_factory,
    )
    tool_executor = ToolExecutor()

    specialist_config: dict[str, tuple[list[dict[str, Any]], str]] = {
        "inventory": ([*INVENTORY_TOOLS, *DEMAND_TOOLS], settings.inventory_prompt),
        "supplier": (SUPPLIER_TOOLS, settings.supplier_prompt),
        "logistics": (LOGISTICS_TOOLS, settings.logistics_prompt),
    }

    def finalize_execution(execution_context: ExecutionContext) -> None:
        execution_context.finish()
        if telemetry_sink is None:
            return
        try:
            telemetry_sink(execution_context)
        except Exception as exc:
            execution_context.record_error(
                component="telemetry",
                operation="emit_execution",
                error=exc,
            )

    def load_memory(state: SupervisorState) -> SupervisorState:
        user_input = state["user_input"].strip()
        if not user_input:
            raise ValueError("user_input cannot be empty.")

        session_id = state.get("session_id", settings.default_session_id)
        execution_context = state.get("execution") or ExecutionContext()

        if conversation_store is None:
            return {
                **state,
                "user_input": user_input,
                "session_id": session_id,
                "resolved_user_input": user_input,
                "memory_context": "",
                "memory_last_entity": None,
                "execution": execution_context,
            }

        try:
            session = conversation_store.load(session_id)
            resolved_user_input = resolve_entity_reference(
                user_input,
                session.last_entity,
            )
            return {
                **state,
                "user_input": user_input,
                "session_id": session.session_id,
                "resolved_user_input": resolved_user_input,
                "memory_context": build_memory_context(session),
                "memory_last_entity": session.last_entity,
                "conversation_response_id": (
                    state.get("conversation_response_id")
                    or session.previous_response_id
                ),
                "execution": execution_context,
            }
        except Exception as exc:
            execution_context.record_error(
                component="conversation_store",
                operation="load",
                error=exc,
            )
            return {
                **state,
                "user_input": user_input,
                "session_id": session_id,
                "resolved_user_input": user_input,
                "memory_context": "",
                "memory_last_entity": None,
                "execution": execution_context,
            }

    def route_request(state: SupervisorState) -> SupervisorState:
        execution_context = state.get("execution") or ExecutionContext()
        router_input = state.get(
            "resolved_user_input",
            state["user_input"],
        )
        response = foundry_service.ask(
            user_input=router_input,
            instructions=settings.router_prompt,
            max_output_tokens=settings.router_max_output_tokens,
            execution_context=execution_context,
        )
        route = response.output_text.strip().lower()
        if route not in _VALID_ROUTES:
            route = "general"

        specialists = _route_specialists(route)  # type: ignore[arg-type]
        agent: AgentName = (
            "supervisor" if len(specialists) > 1 else route  # type: ignore[assignment]
        )

        return {
            **state,
            "intent": route,
            "agent": agent,
            "execution": execution_context,
        }

    def execute_specialist(
        *,
        user_input: str,
        tools: list[dict[str, Any]] | None,
        instructions: str,
        execution_context: ExecutionContext,
        previous_response_id: str | None = None,
    ) -> tuple[str, str]:
        response = foundry_service.ask(
            user_input=user_input,
            tools=tools,
            previous_response_id=previous_response_id,
            execution_context=execution_context,
            instructions=instructions,
            tool_choice="required" if tools else None,
        )

        if not response.tool_calls:
            if not response.output_text:
                raise RuntimeError("Foundry returned an empty response.")
            return response.output_text, response.response_id

        execution = tool_executor.execute(
            response.tool_calls,
            execution_context=execution_context,
        )
        final_response = foundry_service.continue_after_tools(
            previous_response_id=response.response_id,
            tool_outputs=execution.tool_outputs,
            execution_context=execution_context,
            instructions=instructions,
        )
        if not final_response.output_text:
            raise RuntimeError(
                "Foundry returned an empty response after tool execution."
            )
        return final_response.output_text, final_response.response_id

    def persist_turn(
        *,
        state: SupervisorState,
        answer: str,
        response_id: str,
        execution_context: ExecutionContext,
    ) -> None:
        if conversation_store is None:
            return

        resolved_input = state.get(
            "resolved_user_input",
            state["user_input"],
        )
        last_entity = (
            extract_entity_id(resolved_input)
            or state.get("memory_last_entity")
        )
        try:
            conversation_store.append_turn(
                session_id=state.get(
                    "session_id",
                    settings.default_session_id,
                ),
                user_input=state["user_input"],
                assistant_output=answer,
                last_entity=last_entity,
                previous_response_id=response_id,
            )
        except Exception as exc:
            execution_context.record_error(
                component="conversation_store",
                operation="append_turn",
                error=exc,
            )

    def run_agent(state: SupervisorState) -> SupervisorState:
        execution_context = state.get("execution") or ExecutionContext()
        user_input = state.get(
            "resolved_user_input",
            state["user_input"],
        )
        route: Intent = state["intent"]

        try:
            specialists = _route_specialists(route)
            if len(specialists) > 1:
                query_plan = plan_specialist_queries(user_input, specialists)
                specialist_queries = query_plan.as_dict()
                specialist_outputs: dict[str, str] = {}

                for specialist in specialists:
                    tools, instructions = specialist_config[specialist]
                    answer, _ = execute_specialist(
                        user_input=specialist_queries[specialist],
                        tools=tools,
                        instructions=instructions,
                        execution_context=execution_context,
                    )
                    specialist_outputs[specialist] = answer

                synthesis_sections = [
                    f"Original user request:\n{state['user_input']}",
                    (
                        "Resolved user request:\n"
                        f"{user_input}"
                    ),
                    (
                        "Extracted product entity_id:\n"
                        f"{query_plan.entity_id or 'not identified'}"
                    ),
                ]
                for specialist in specialists:
                    label = specialist.capitalize()
                    synthesis_sections.append(
                        f"{label} specialist request:\n"
                        f"{specialist_queries[specialist]}\n\n"
                        f"{label} Agent output:\n"
                        f"{specialist_outputs[specialist]}"
                    )

                synthesis = foundry_service.ask(
                    user_input="\n\n".join(synthesis_sections),
                    instructions=settings.multi_agent_prompt,
                    execution_context=execution_context,
                )
                if not synthesis.output_text:
                    raise RuntimeError(
                        "Foundry returned an empty multi-agent synthesis."
                    )
                persist_turn(
                    state=state,
                    answer=synthesis.output_text,
                    response_id=synthesis.response_id,
                    execution_context=execution_context,
                )
                finalize_execution(execution_context)
                return {
                    **state,
                    "agent": "supervisor",
                    "answer": synthesis.output_text,
                    "specialist_outputs": specialist_outputs,
                    "specialist_queries": specialist_queries,
                    "conversation_response_id": synthesis.response_id,
                    "execution": execution_context,
                }

            if route in specialist_config:
                tools, instructions = specialist_config[route]
                agent_input = user_input
            elif route == "time":
                tools = TIME_TOOLS
                instructions = settings.time_prompt
                agent_input = user_input
            else:
                tools = None
                instructions = settings.system_prompt
                memory_context = state.get("memory_context", "")
                agent_input = (
                    f"{memory_context}\n\nCurrent user request:\n{user_input}"
                    if memory_context
                    else user_input
                )

            answer, response_id = execute_specialist(
                user_input=agent_input,
                tools=tools,
                instructions=instructions,
                execution_context=execution_context,
                previous_response_id=state.get("conversation_response_id"),
            )
            persist_turn(
                state=state,
                answer=answer,
                response_id=response_id,
                execution_context=execution_context,
            )
            finalize_execution(execution_context)
            return {
                **state,
                "answer": answer,
                "conversation_response_id": response_id,
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
                    operation=f"run_{route}_agent",
                    error=exc,
                )
            if not execution_context.is_finished:
                finalize_execution(execution_context)
            raise

    graph = StateGraph(SupervisorState)
    graph.add_node("load_memory", load_memory)
    graph.add_node("route_request", route_request)
    graph.add_node("run_agent", run_agent)
    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "route_request")
    graph.add_edge("route_request", "run_agent")
    graph.add_edge("run_agent", END)
    return graph.compile()
