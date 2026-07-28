from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Callable

from shared.search_service import SearchService
from shared.settings import load_settings


ToolFunction = Callable[..., Any]


_settings = load_settings()

_search_service: SearchService | None = None

_ENTITY_CODE_PATTERN = re.compile(
    r"\b[A-Za-z]+(?:-[A-Za-z]+)*-?\d+\b",
    re.IGNORECASE,
)


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


def _extract_entity_codes(
    query: str,
) -> set[str]:
    return {
        match.group(0).upper()
        for match in _ENTITY_CODE_PATTERN.finditer(query)
    }


def _normalize_entity_id(
    value: str,
) -> str:
    return value.strip().upper()


def search_documents(
    query: str,
) -> str:
    documents = (
        _get_search_service()
        .search_documents(query)
    )

    requested_codes = _extract_entity_codes(query)

    if requested_codes:
        exact_documents = [
            document
            for document in documents
            if _normalize_entity_id(document.entity_id)
            in requested_codes
        ]

        if not exact_documents:
            return json.dumps(
                {
                    "query": query,
                    "count": 0,
                    "documents": [],
                    "requested_entity_codes": sorted(
                        requested_codes
                    ),
                    "message": (
                        "No document was found with an exact "
                        "entity identifier matching the code "
                        "requested by the user. Do not use "
                        "similar entities as substitutes and "
                        "do not infer a policy from them."
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )

        documents = exact_documents

    if not documents:
        return json.dumps(
            {
                "query": query,
                "count": 0,
                "documents": [],
                "message": (
                    "No relevant documents were found."
                ),
            },
            indent=2,
            ensure_ascii=False,
        )

    payload: dict[str, Any] = {
        "query": query,
        "count": len(documents),
        "documents": [],
    }

    if requested_codes:
        payload["requested_entity_codes"] = sorted(
            requested_codes
        )

    for document in documents:
        payload["documents"].append(
            {
                "title": document.title,
                "content": document.content,
                "agent": document.agent,
                "doc_type": document.doc_type,
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
            "Searches the enterprise knowledge base for "
            "documents related to inventory, logistics, "
            "suppliers, procurement, products, policies, "
            "manuals and internal business information. "
            "Use this tool whenever the user's question "
            "depends on enterprise knowledge instead of "
            "the model's general knowledge. When the user "
            "provides an entity code, only an exact entity "
            "identifier match may be used as evidence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A natural language search query "
                        "describing the information to "
                        "retrieve from the enterprise "
                        "knowledge base. Whenever possible, "
                        "reuse the user's request without "
                        "rewriting it."
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
