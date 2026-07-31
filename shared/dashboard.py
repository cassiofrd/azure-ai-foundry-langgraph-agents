from __future__ import annotations

from typing import Any


def execution_summary(response: dict[str, Any]) -> dict[str, Any]:
    execution = response.get("execution", {}) or {}
    metrics = execution.get("metrics", {}) or {}
    return {
        "route": response.get("route", "general"),
        "specialist": response.get("specialist", "general"),
        "participants": list(response.get("participants", []) or []),
        "duration_ms": execution.get("duration_ms"),
        "llm_calls": int(metrics.get("llm_call_count", 0) or 0),
        "tool_calls": int(metrics.get("tool_call_count", 0) or 0),
        "total_tokens": int(metrics.get("total_tokens", 0) or 0),
        "errors": int(metrics.get("error_count", 0) or 0),
        "tools": [
            item.get("tool_name", "")
            for item in execution.get("tools", []) or []
            if isinstance(item, dict)
        ],
        "evidence_count": len(response.get("evidence", []) or []),
    }
