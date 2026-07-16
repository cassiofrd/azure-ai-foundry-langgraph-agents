from __future__ import annotations

from datetime import datetime, timezone


def get_current_utc_time() -> str:
    current_time = datetime.now(timezone.utc)

    return (
        f"The current UTC time is "
        f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC."
    )


TOOLS = [
    {
        "type": "function",
        "name": "get_current_utc_time",
        "description": "Returns the current UTC time.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }
]

TOOL_REGISTRY = {
    "get_current_utc_time": get_current_utc_time,
}