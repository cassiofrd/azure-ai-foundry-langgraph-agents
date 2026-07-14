from __future__ import annotations

from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from shared.foundry_client import ResponsesClient
from shared.settings import AppSettings


class SupervisorState(TypedDict):
    user_input: str
    answer: str


def build_supervisor_graph(
    *,
    settings: AppSettings,
    client_factory: Callable[[], ResponsesClient],
):
    def call_foundry_model(state: SupervisorState) -> SupervisorState:
        user_input = state["user_input"].strip()
        if not user_input:
            raise ValueError("user_input cannot be empty.")

        client = client_factory()
        response = client.responses.create(
            model=settings.foundry_model_deployment,
            instructions=settings.system_prompt,
            input=user_input,
            max_output_tokens=settings.model_max_output_tokens,
        )

        answer = (response.output_text or "").strip()
        if not answer:
            raise RuntimeError("Foundry returned an empty response.")

        return {
            "user_input": user_input,
            "answer": answer,
        }

    builder = StateGraph(SupervisorState)
    builder.add_node("call_foundry_model", call_foundry_model)
    builder.add_edge(START, "call_foundry_model")
    builder.add_edge("call_foundry_model", END)
    return builder.compile()
