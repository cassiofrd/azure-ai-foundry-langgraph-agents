from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from langgraph.graph import END, START, StateGraph

from shared.foundry_client import ResponsesClient
from shared.settings import AppSettings
from shared.state import Intent, SupervisorState


def build_supervisor_graph(
    *,
    settings: AppSettings,
    client_factory: Callable[[], ResponsesClient],
):
    def classify_request(state: SupervisorState) -> SupervisorState:
        user_input = state["user_input"].strip()
        if not user_input:
            raise ValueError("user_input cannot be empty.")

        normalized = user_input.casefold()
        time_markers = (
            "que horas",
            "qual é a hora",
            "qual a hora",
            "horário",
            "hora atual",
            "current time",
            "what time",
            "utc time",
        )
        intent: Intent = (
            "time"
            if any(marker in normalized for marker in time_markers)
            else "general"
        )

        return {
            "user_input": user_input,
            "intent": intent,
            "answer": "",
        }

    def route_request(state: SupervisorState) -> Intent:
        return state["intent"]

    def call_foundry_model(state: SupervisorState) -> SupervisorState:
        client = client_factory()
        response = client.responses.create(
            model=settings.foundry_model_deployment,
            instructions=settings.system_prompt,
            input=state["user_input"],
            max_output_tokens=settings.model_max_output_tokens,
        )

        answer = (response.output_text or "").strip()
        if not answer:
            raise RuntimeError("Foundry returned an empty response.")

        return {
            "user_input": state["user_input"],
            "intent": state["intent"],
            "answer": answer,
        }

    def get_current_utc_time(state: SupervisorState) -> SupervisorState:
        current_time = datetime.now(timezone.utc)
        return {
            "user_input": state["user_input"],
            "intent": state["intent"],
            "answer": (
                "The current UTC time is "
                f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC."
            ),
        }

    graph = StateGraph(SupervisorState)
    graph.add_node("classify_request", classify_request)
    graph.add_node("call_foundry_model", call_foundry_model)
    graph.add_node("get_current_utc_time", get_current_utc_time)

    graph.add_edge(START, "classify_request")
    graph.add_conditional_edges(
        "classify_request",
        route_request,
        {
            "general": "call_foundry_model",
            "time": "get_current_utc_time",
        },
    )
    graph.add_edge("call_foundry_model", END)
    graph.add_edge("get_current_utc_time", END)

    return graph.compile()
