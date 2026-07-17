from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable

from shared.search_service import SearchService
from shared.settings import load_settings


ToolFunction = Callable[..., Any]


_settings = load_settings()

_search_service: SearchService | None = None


def get_current_utc_time() -> str:
    current_time = datetime.now(timezone.utc)

    return (
        "The current UTC time is "
        f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC."
    )


def _get_search_service() -> SearchService:
    global _search_service

    if _search_service is None:
        _search_service = SearchService.from_settings(
            _settings
        )

    return _search_service


def search_documents(
    query: str,
) -> str:

    documents = (
        _get_search_service()
        .search_documents(query)
    )

    if not documents:
        return (
            "No relevant documents were found."
        )

    payload = []

    for document in documents:
        payload.append(
            {
                "title": document.title,
                "content": document.content,
                "agent": document.agent,
                "entity_type": document.entity_type,
                "entity_id": document.entity_id,
                "source": document.source,
                "score": document.score,
            }
        )

    return json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
    )


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "get_current_utc_time",
        "description": (
            "Returns the current UTC time."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "search_documents",
        "description": (
            "Searches the enterprise knowledge base "
            "for inventory, supplier, procurement "
            "and logistics information."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural language search query."
                    ),
                }
            },
            "required": [
                "query",
            ],
            "additionalProperties": False,
        },
    },
]


TOOL_REGISTRY: dict[str, ToolFunction] = {
    "get_current_utc_time": get_current_utc_time,
    "search_documents": search_documents,
}