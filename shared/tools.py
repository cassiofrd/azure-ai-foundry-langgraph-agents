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
        _search_service = SearchService.from_settings(_settings)
    return _search_service


def _extract_entity_codes(query: str) -> set[str]:
    return {
        match.group(0).upper()
        for match in _ENTITY_CODE_PATTERN.finditer(query)
    }


def _normalize_entity_id(value: str) -> str:
    return value.strip().upper()


def _search_documents(query: str, *, agent: str | None = None) -> str:
    search_service = _get_search_service()
    documents = (
        search_service.search_documents(query, agent=agent)
        if agent is not None
        else search_service.search_documents(query)
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
                    "agent": agent,
                    "count": 0,
                    "documents": [],
                    "requested_entity_codes": sorted(requested_codes),
                    "message": (
                        "No document was found with an exact entity "
                        "identifier matching the code requested by the "
                        "user. Do not use similar entities as substitutes "
                        "and do not infer information from them."
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
                "agent": agent,
                "count": 0,
                "documents": [],
                "message": "No relevant documents were found.",
            },
            indent=2,
            ensure_ascii=False,
        )

    payload: dict[str, Any] = {
        "query": query,
        "agent": agent,
        "count": len(documents),
        "documents": [],
    }
    if requested_codes:
        payload["requested_entity_codes"] = sorted(requested_codes)

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

    return json.dumps(payload, indent=2, ensure_ascii=False)


def search_documents(query: str) -> str:
    """Backward-compatible enterprise search."""
    return _search_documents(query)


def search_inventory_documents(query: str) -> str:
    return _search_documents(query, agent="inventory")


def search_supplier_documents(query: str) -> str:
    return _search_documents(query, agent="supplier")


def search_logistics_documents(query: str) -> str:
    return _search_documents(query, agent="logistics")


def _search_tool(name: str, domain: str, description: str) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        f"Natural-language query for the {domain} "
                        "knowledge base. Preserve entity codes exactly."
                    ),
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    }


TIME_TOOLS = [
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

INVENTORY_TOOLS = [
    _search_tool(
        "search_inventory_documents",
        "inventory",
        (
            "Searches only inventory policies, stock levels, reorder "
            "points, replenishment quantities and shortage procedures."
        ),
    )
]

LOGISTICS_TOOLS = [
    _search_tool(
        "search_logistics_documents",
        "logistics",
        (
            "Searches only transportation policies, transit times, freight "
            "modes, dispatch constraints and logistics approvals."
        ),
    )
]

SUPPLIER_TOOLS = [
    _search_tool(
        "search_supplier_documents",
        "supplier",
        (
            "Searches only supplier, procurement, approved-source and "
            "supplier lead-time information."
        ),
    )
]

GENERIC_SEARCH_TOOLS = [
    _search_tool(
        "search_documents",
        "enterprise",
        (
            "Searches the enterprise knowledge base for inventory, "
            "logistics, suppliers, procurement and internal business "
            "information. Prefer a domain-specific tool when available."
        ),
    )
]

TOOLS = (
    TIME_TOOLS
    + GENERIC_SEARCH_TOOLS
    + INVENTORY_TOOLS
    + SUPPLIER_TOOLS
    + LOGISTICS_TOOLS
)

TOOL_REGISTRY: dict[str, ToolFunction] = {
    "get_current_utc_time": get_current_utc_time,
    "search_documents": search_documents,
    "search_inventory_documents": search_inventory_documents,
    "search_supplier_documents": search_supplier_documents,
    "search_logistics_documents": search_logistics_documents,
}
