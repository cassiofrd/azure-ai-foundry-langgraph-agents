from __future__ import annotations

from typing import Literal, TypedDict


Intent = Literal["general", "time"]


class SupervisorState(TypedDict):
    user_input: str
    intent: Intent
    answer: str
    conversation_response_id: str | None
