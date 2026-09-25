from __future__ import annotations

import re
from dataclasses import dataclass


_ENTITY_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?=[A-Za-z0-9-]*[A-Za-z])(?=[A-Za-z0-9-]*\d)[A-Za-z][A-Za-z0-9-]{1,31}(?![A-Za-z0-9])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SpecialistQueries:
    inventory: str | None = None
    supplier: str | None = None
    logistics: str | None = None
    entity_id: str | None = None

    def as_dict(self) -> dict[str, str]:
        return {
            name: query
            for name, query in (
                ("inventory", self.inventory),
                ("supplier", self.supplier),
                ("logistics", self.logistics),
            )
            if query is not None
        }


def extract_entity_id(user_input: str) -> str | None:
    """Extract a compact product/material identifier such as M10 or A100."""
    matches = _ENTITY_PATTERN.findall(user_input)
    if not matches:
        return None
    return matches[0].upper()


def plan_specialist_queries(
    user_input: str,
    specialists: tuple[str, ...],
) -> SpecialistQueries:
    """Create focused requests only for the specialists selected by routing."""
    normalized = user_input.strip()
    if not normalized:
        raise ValueError("user_input cannot be empty.")

    allowed = {"inventory", "supplier", "logistics"}
    invalid = set(specialists) - allowed
    if invalid:
        raise ValueError(
            "Unsupported specialist(s): " + ", ".join(sorted(invalid))
        )

    entity_id = extract_entity_id(normalized)
    target = (
        f"the item identified as {entity_id}"
        if entity_id
        else "the item mentioned in the original request"
    )

    original_request_context = f"Original request: {normalized} "

    inventory = None
    if "inventory" in specialists:
        inventory = (
            original_request_context
            + f"Retrieve the current inventory snapshot and inventory policy for {target}, "
            "preserving any location identifier from the original request. "
            "Focus on current on-hand, reserved, and available inventory; snapshot date; "
            "target stock level; reorder point; safety stock; preferred replenishment "
            "quantity; shortage rules; and planning instructions. Also retrieve demand "
            "evidence relevant to replenishment planning, including open production "
            "orders, demand forecasts, and recent historical consumption for the item "
            "and location when available. Also retrieve the applicable standard "
            "replenishment and emergency-shortage procedures, including escalation rules, "
            "approval requirements, and decision criteria. For replenishment planning, "
            "shortage-risk analysis, or projected inventory, use calculate_inventory_projection "
            "to obtain deterministic inventory projections instead of asking the LLM to perform "
            "inventory arithmetic. The projection tool already retrieves the exact current "
            "inventory, complete open production orders, and exact demand forecast; do not call "
            "get_current_inventory, get_open_production_orders, or get_demand_forecast again "
            "unless the projection output is missing required data or the original request "
            "explicitly asks for an underlying dataset. Treat the tool's committed-open-orders "
            "and forecast results as separate scenarios unless enterprise evidence explicitly "
            "establishes that they can be combined without double counting. Use the inventory search tool "
            "for inventory policy and other relevance-oriented inventory knowledge. Use "
            "search_procedure_documents once with a sufficiently broad query for the applicable "
            "standard and emergency procedural guidance; avoid duplicate procedure searches for "
            "the same planning request. Use the demand search tool for historical consumption "
            "and other relevance-oriented "
            "demand evidence. "
            "Preserve the product identifier and location from the original request in all "
            "tool queries "
            "so Retrieval V2 can resolve aliases and apply metadata filters. Do not "
            "require a literal exact entity-id match."
        )

    supplier = None
    if "supplier" in specialists:
        supplier = (
            original_request_context
            + f"Retrieve the approved supplier information for {target}. "
            "Focus on supplier name, approval status, contractual lead time, minimum "
            "and maximum order quantities, accelerated-processing options, capacity "
            "constraints, and other facts relevant to whether replenishment demand can "
            "be served. Use only the supplier search tool. Preserve the product "
            "identifier from the original request so Retrieval V2 can resolve aliases "
            "and apply metadata filters. Do not require a literal exact entity-id match."
        )

    logistics = None
    if "logistics" in specialists:
        logistics = (
            original_request_context
            + f"Retrieve the transportation policies, freight information, and carrier "
            f"restrictions relevant to {target} and the destination or location in the "
            "original request. Focus on available transportation modes, transit times "
            "after supplier dispatch, urgency criteria, approvals, cost trade-offs, "
            "carrier restrictions, and production-shortage constraints. Use only the "
            "logistics search tool. Preserve the product identifier and location from "
            "the original request when useful; Retrieval V2 can combine entity-specific "
            "documents with global logistics documents."
        )

    return SpecialistQueries(
        inventory=inventory,
        supplier=supplier,
        logistics=logistics,
        entity_id=entity_id,
    )

def plan_inventory_supplier_queries(user_input: str) -> SpecialistQueries:
    """Backward-compatible planner for the original two-specialist route."""
    return plan_specialist_queries(
        user_input,
        ("inventory", "supplier"),
    )
