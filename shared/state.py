from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

from shared.execution_context import ExecutionContext


Intent = Literal["general", "time"]


class SupervisorState(TypedDict):
    user_input: str
    intent: Intent
    answer: str
    conversation_response_id: str | None
    execution: NotRequired[ExecutionContext]