from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable

from shared.search_service_v2 import (
    SearchServiceV2,
    build_metadata_filter,
    infer_intent,
    resolve_entity,
    resolve_location,
)


ToolFunction = Callable[..., Any]
_search_service: SearchServiceV2 | None = None


def get_current_utc_time() -> str:
    current_time = datetime.now(timezone.utc)
    return (
        "The current UTC time is "
        f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC."
    )


def _get_search_service() -> SearchServiceV2:
    global _search_service
    if _search_service is None:
        _search_service = SearchServiceV2()
    return _search_service


def _serialize_result(result: Any) -> dict[str, Any]:
    return {
        "id": result.get("id"),
        "chunk_id": result.get("chunk_id"),
        "source_path": result.get("source_path"),
        "domain": result.get("domain"),
        "document_type": result.get("document_type"),
        "content": result.get("content"),
        "entity_ids": list(result.get("entity_ids") or []),
        "aliases": list(result.get("aliases") or []),
        "location_ids": list(result.get("location_ids") or []),
        "supplier_ids": list(result.get("supplier_ids") or []),
        "is_global": result.get("is_global"),
        "score": result.get("@search.score"),
    }


def _search_documents(
    query: str,
    *,
    domain: str | None = None,
    top: int = 5,
) -> str:
    search_service = _get_search_service()

    search_result = search_service.smart_hybrid_search(
        query,
        top=top,
        candidate_pool=max(10, top),
        domain=domain,
    )

    results = search_result["results"]

    payload: dict[str, Any] = {
        "query": query,
        "domain": search_result["domain"],
        "entity_id": search_result["entity_id"],
        "location_id": search_result["location_id"],
        "intent": search_result["intent"],
        "preferred_document_types": search_result[
            "preferred_document_types"
        ],
        "filter": search_result["filter"],
        "count": len(results),
        "documents": [_serialize_result(result) for result in results],
    }

    if not results:
        payload["message"] = "No relevant documents were found."

    return json.dumps(payload, indent=2, ensure_ascii=False)


def search_documents(query: str) -> str:
    """Search the enterprise knowledge base using Retrieval V2."""
    return _search_documents(query)


def search_inventory_documents(query: str) -> str:
    return _search_documents(query, domain="inventory")


def search_demand_documents(query: str) -> str:
    return _search_documents(query, domain="demand")


def search_procedure_documents(query: str) -> str:
    """Search replenishment and shortage procedures using Retrieval V2."""
    return _search_documents(query, domain="procedures")



def _parse_int(value: str | None) -> int | None:
    """Parse an integer field from structured document content."""
    if value is None:
        return None

    normalized = str(value).strip().replace(",", "")
    if not normalized:
        return None

    try:
        return int(normalized)
    except ValueError:
        try:
            return int(float(normalized))
        except ValueError:
            return None


def _parse_key_value_content(content: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in content.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def get_current_inventory(query: str) -> str:
    """Return the exact current inventory snapshot for one item and location."""
    search_service = _get_search_service()
    entity_id = resolve_entity(query)
    location_id = resolve_location(query)

    if not entity_id or not location_id:
        return json.dumps({
            "query": query,
            "entity_id": entity_id,
            "location_id": location_id,
            "count": 0,
            "record": None,
            "message": "Both an item and a location are required for an exact current-inventory lookup.",
        })

    result = search_service.structured_search(
        domain="inventory",
        document_type="inventory_snapshot",
        entity_id=entity_id,
        location_id=location_id,
        top=100,
    )

    records = []
    for document in result["results"]:
        fields = _parse_key_value_content(str(document.get("content", "")))
        records.append({
            "source_path": document.get("source_path"),
            "chunk_id": document.get("chunk_id"),
            "item": fields.get("Item"),
            "location": fields.get("Location"),
            "on_hand": _parse_int(fields.get("On Hand")),
            "reserved": _parse_int(fields.get("Reserved")),
            "available": _parse_int(fields.get("Available")),
            "snapshot_date": fields.get("Snapshot Date"),
        })

    if len(records) != 1:
        return json.dumps({
            "query": query,
            "entity_id": entity_id,
            "location_id": location_id,
            "count": len(records),
            "record": None,
            "records": records,
            "message": "Expected exactly one current inventory record for the resolved item and location.",
        })

    return json.dumps({
        "query": query,
        "entity_id": entity_id,
        "location_id": location_id,
        "count": 1,
        "record": records[0],
    })



def get_demand_forecast(query: str) -> str:
    """Return the complete structured demand forecast for one item and location."""
    entity_id = resolve_entity(query)
    location_id = resolve_location(query)

    if not entity_id or not location_id:
        return json.dumps(
            {
                "query": query,
                "error": (
                    "Both an item identifier and a location identifier are required "
                    "for structured demand-forecast lookup."
                ),
                "entity_id": entity_id,
                "location_id": location_id,
            },
            indent=2,
            ensure_ascii=False,
        )

    search_result = _get_search_service().structured_search(
        domain="demand",
        document_type="demand_forecast",
        entity_id=entity_id,
        location_id=location_id,
        top=100,
    )

    forecast_rows: list[dict[str, Any]] = []
    total_forecast_qty = 0

    for result in search_result["results"]:
        row = _parse_key_value_content(result.get("content") or "")
        period = (
            row.get("Week")
            or row.get("Forecast Week")
            or row.get("Period")
            or row.get("Date")
        )
        forecast_qty = _parse_int(
            row.get("Forecast Demand")
            or row.get("Forecast Qty")
            or row.get("Forecast Quantity")
            or row.get("Forecast")
            or row.get("Demand Forecast")
        )

        if forecast_qty is None:
            continue

        total_forecast_qty += forecast_qty
        forecast_rows.append(
            {
                "item": row.get("Item"),
                "location": row.get("Location") or row.get("Plant"),
                "period": period,
                "forecast_qty": forecast_qty,
                "source_path": result.get("source_path"),
                "chunk_id": result.get("chunk_id"),
            }
        )

    forecast_rows.sort(
        key=lambda forecast: (
            forecast.get("period") or "",
            forecast.get("chunk_id") or "",
        )
    )

    return json.dumps(
        {
            "query": query,
            "mode": "structured",
            "entity_id": entity_id,
            "location_id": location_id,
            "filter": search_result["filter"],
            "search_result_count": len(search_result["results"]),
            "parsed_forecast_count": len(forecast_rows),
            "forecast_row_count": len(forecast_rows),
            "total_forecast_qty": total_forecast_qty,
            "forecast": forecast_rows,
        },
        indent=2,
        ensure_ascii=False,
    )


def get_open_production_orders(query: str) -> str:
    """Return all open production orders for one item/location and aggregate quantity."""
    entity_id = resolve_entity(query)
    location_id = resolve_location(query)

    if not entity_id or not location_id:
        return json.dumps(
            {
                "query": query,
                "error": (
                    "Both an item identifier and a location identifier are required "
                    "for structured production-order lookup."
                ),
                "entity_id": entity_id,
                "location_id": location_id,
            },
            indent=2,
            ensure_ascii=False,
        )

    search_result = _get_search_service().structured_search(
        domain="demand",
        document_type="production_order",
        entity_id=entity_id,
        location_id=location_id,
    )

    open_orders: list[dict[str, Any]] = []
    total_required_qty = 0

    for result in search_result["results"]:
        row = _parse_key_value_content(result.get("content") or "")
        if row.get("Status", "").strip().lower() != "open":
            continue

        try:
            required_qty = int(row.get("Required Qty", "0"))
        except ValueError:
            required_qty = 0

        total_required_qty += required_qty
        open_orders.append(
            {
                "order": row.get("Order"),
                "plant": row.get("Plant"),
                "item": row.get("Item"),
                "required_qty": required_qty,
                "required_date": row.get("Required Date"),
                "status": row.get("Status"),
                "source_path": result.get("source_path"),
                "chunk_id": result.get("chunk_id"),
            }
        )

    open_orders.sort(
        key=lambda order: (
            order.get("required_date") or "",
            order.get("order") or "",
        )
    )

    return json.dumps(
        {
            "query": query,
            "mode": "structured",
            "entity_id": entity_id,
            "location_id": location_id,
            "filter": search_result["filter"],
            "open_order_count": len(open_orders),
            "total_required_qty": total_required_qty,
            "orders": open_orders,
        },
        indent=2,
        ensure_ascii=False,
    )


def calculate_inventory_projection(query: str) -> str:
    """Calculate deterministic inventory scenarios without assuming demand overlap."""
    inventory_payload = json.loads(get_current_inventory(query))
    orders_payload = json.loads(get_open_production_orders(query))
    forecast_payload = json.loads(get_demand_forecast(query))

    inventory_record = inventory_payload.get("record")
    if not inventory_record:
        return json.dumps(
            {
                "query": query,
                "error": "A unique current inventory record is required for projection.",
                "inventory": inventory_payload,
            },
            indent=2,
            ensure_ascii=False,
        )

    available = inventory_record.get("available")
    if available is None:
        return json.dumps(
            {
                "query": query,
                "error": "Current available inventory is missing or not numeric.",
                "inventory": inventory_payload,
            },
            indent=2,
            ensure_ascii=False,
        )

    if orders_payload.get("error"):
        return json.dumps(
            {
                "query": query,
                "error": "Open production orders could not be retrieved exactly.",
                "orders": orders_payload,
            },
            indent=2,
            ensure_ascii=False,
        )

    if forecast_payload.get("error"):
        return json.dumps(
            {
                "query": query,
                "error": "Demand forecast could not be retrieved exactly.",
                "forecast": forecast_payload,
            },
            indent=2,
            ensure_ascii=False,
        )

    total_open_orders = int(orders_payload.get("total_required_qty") or 0)
    total_forecast = int(forecast_payload.get("total_forecast_qty") or 0)

    committed_orders_scenario = {
        "starting_available": available,
        "demand_qty": total_open_orders,
        "projected_available": available - total_open_orders,
        "orders": orders_payload.get("orders", []),
    }

    forecast_timeline: list[dict[str, Any]] = []
    running_forecast_inventory = available
    for row in forecast_payload.get("forecast", []):
        qty = row.get("forecast_qty")
        if qty is None:
            continue
        running_forecast_inventory -= int(qty)
        forecast_timeline.append(
            {
                "period": row.get("period"),
                "forecast_qty": int(qty),
                "projected_available_after_period": running_forecast_inventory,
            }
        )

    forecast_scenario = {
        "starting_available": available,
        "demand_qty": total_forecast,
        "projected_available": available - total_forecast,
        "timeline": forecast_timeline,
    }

    return json.dumps(
        {
            "query": query,
            "mode": "deterministic_projection",
            "entity_id": inventory_payload.get("entity_id"),
            "location_id": inventory_payload.get("location_id"),
            "snapshot_date": inventory_record.get("snapshot_date"),
            "starting_available": available,
            "scenarios": {
                "committed_open_orders": committed_orders_scenario,
                "forecast": forecast_scenario,
            },
            "important_note": (
                "Open production orders and demand forecast are intentionally "
                "calculated as separate scenarios. This tool does not add them "
                "together because the available source data does not establish "
                "whether open orders are already included in the forecast. "
                "Combining them without that knowledge could double-count demand."
            ),
        },
        indent=2,
        ensure_ascii=False,
    )


def search_supplier_documents(query: str) -> str:
    search_service = _get_search_service()

    # Supplier eligibility in this dataset is product-specific, not plant-specific.
    # Keep the full query for relevance, but do not turn a location mentioned in
    # the query into a mandatory supplier metadata filter.
    entity_id = resolve_entity(query)

    intent, preferred_document_types = infer_intent(query)
    metadata_filter = build_metadata_filter(
        domain="suppliers",
        entity_id=entity_id,
        location_id=None,
        include_global=True,
    )

    # Call hybrid_search directly. smart_hybrid_search intentionally auto-resolves
    # a missing location from the query, which is not appropriate for supplier
    # eligibility in this dataset.
    candidates = search_service.hybrid_search(
        query,
        top=10,
        filter=metadata_filter,
    )
    results = search_service._rerank_by_intent(
        candidates,
        preferred_types=preferred_document_types,
    )[:5]

    payload: dict[str, Any] = {
        "query": query,
        "domain": "suppliers",
        "entity_id": entity_id,
        "location_id": None,
        "intent": intent,
        "preferred_document_types": list(preferred_document_types),
        "filter": metadata_filter,
        "count": len(results),
        "documents": [_serialize_result(result) for result in results],
    }

    if not results:
        payload["message"] = "No relevant supplier documents were found."

    return json.dumps(payload, indent=2, ensure_ascii=False)


def search_logistics_documents(query: str) -> str:
    return _search_documents(query, domain="logistics")


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
                        "knowledge base. Preserve entity codes and "
                        "location identifiers exactly when present."
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
            "Searches the inventory knowledge domain using hybrid retrieval, "
            "metadata filters and intent-aware reranking. Use for inventory "
            "snapshots, stock levels, reorder points, safety stock and "
            "inventory policies."
        ),
    ),
    _search_tool(
        "search_procedure_documents",
        "procedures",
        (
            "Searches replenishment and shortage procedures using hybrid "
            "retrieval, metadata filters and intent-aware reranking. Use for "
            "standard replenishment workflow, shortage escalation, emergency "
            "actions, approval requirements and decision rules."
        ),
    ),
    _search_tool(
        "get_current_inventory",
        "structured inventory",
        (
            "Returns the exact current inventory snapshot for one item and "
            "location using deterministic metadata filters. Use when exact "
            "on-hand, reserved, available, and snapshot-date values are required."
        ),
    ),
    _search_tool(
        "calculate_inventory_projection",
        "deterministic planning",
        (
            "Calculates deterministic inventory projections from the exact current "
            "inventory, open production orders, and demand forecast. It returns "
            "committed-order and forecast scenarios separately to avoid unsupported "
            "double counting. Use for replenishment-risk calculations before "
            "interpreting inventory policies and procedures."
        ),
    ),
]

DEMAND_TOOLS = [
    _search_tool(
        "search_demand_documents",
        "demand",
        (
            "Searches demand knowledge using hybrid retrieval. Use for demand "
            "forecasts, historical consumption and other relevance-oriented "
            "demand questions. Do not use it when all open production orders "
            "must be returned exactly."
        ),
    ),
    _search_tool(
        "get_demand_forecast",
        "structured demand",
        (
            "Returns the complete demand forecast for an exact item and location "
            "using deterministic metadata filters, preserving forecast periods "
            "and quantities and calculating the total forecast quantity. Use this "
            "instead of RAG when forecast rows must be complete and exact."
        ),
    ),
    _search_tool(
        "get_open_production_orders",
        "structured demand",
        (
            "Returns all open production orders for an exact item and location "
            "using deterministic metadata filters, and calculates the total "
            "required quantity. Use this instead of RAG when open production "
            "orders must be complete and aggregated."
        ),
    ),
]

LOGISTICS_TOOLS = [
    _search_tool(
        "search_logistics_documents",
        "logistics",
        (
            "Searches the logistics knowledge domain using hybrid retrieval, "
            "metadata filters and intent-aware reranking. Use for "
            "transportation policies, shipping options, transit times, "
            "freight rates, carrier restrictions and logistics approvals."
        ),
    )
]

SUPPLIER_TOOLS = [
    _search_tool(
        "search_supplier_documents",
        "suppliers",
        (
            "Searches the supplier knowledge domain using hybrid retrieval, "
            "metadata filters and intent-aware reranking. Use for approved "
            "suppliers, supplier master data, contracts, lead times, MOQ "
            "and supplier conditions."
        ),
    )
]

GENERIC_SEARCH_TOOLS = [
    _search_tool(
        "search_documents",
        "enterprise",
        (
            "Searches the enterprise knowledge base using Retrieval V2. "
            "Prefer a domain-specific tool when the specialist domain is "
            "already known."
        ),
    )
]

TOOLS = (
    TIME_TOOLS
    + GENERIC_SEARCH_TOOLS
    + INVENTORY_TOOLS
    + DEMAND_TOOLS
    + SUPPLIER_TOOLS
    + LOGISTICS_TOOLS
)

TOOL_REGISTRY: dict[str, ToolFunction] = {
    "get_current_utc_time": get_current_utc_time,
    "search_documents": search_documents,
    "search_inventory_documents": search_inventory_documents,
    "search_demand_documents": search_demand_documents,
    "search_procedure_documents": search_procedure_documents,
    "get_current_inventory": get_current_inventory,
    "calculate_inventory_projection": calculate_inventory_projection,
    "get_demand_forecast": get_demand_forecast,
    "get_open_production_orders": get_open_production_orders,
    "search_supplier_documents": search_supplier_documents,
    "search_logistics_documents": search_logistics_documents,
}
