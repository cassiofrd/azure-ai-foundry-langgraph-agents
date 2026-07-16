from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable


ToolFunction = Callable[..., Any]


def get_current_utc_time() -> str:
    current_time = datetime.now(timezone.utc)

    return (
        "The current UTC time is "
        f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC."
    )


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "get_current_utc_time",
        "description": "Returns the current UTC time.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    }
]


TOOL_REGISTRY: dict[str, ToolFunction] = {
    "get_current_utc_time": get_current_utc_time,
}