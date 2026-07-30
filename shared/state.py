from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

from shared.execution_context import ExecutionContext


Intent = Literal[
    "general",
    "time",
    "inventory",
    "supplier",
    "logistics",
    "inventory_supplier",
    "inventory_logistics",
    "supplier_logistics",
    "inventory_supplier_logistics",
]
AgentName = Literal[
    "general",
    "time",
    "inventory",
    "supplier",
    "logistics",
    "supervisor",
]


class SupervisorState(TypedDict):
    user_input: str
    intent: Intent
    agent: AgentName
    answer: str
    conversation_response_id: str | None
    specialist_outputs: NotRequired[dict[str, str]]
    specialist_queries: NotRequired[dict[str, str]]
    execution: NotRequired[ExecutionContext]
