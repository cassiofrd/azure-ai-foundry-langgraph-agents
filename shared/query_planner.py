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
        f"the item with exact entity_id {entity_id}"
        if entity_id
        else "the item mentioned in the original request"
    )

    inventory = None
    if "inventory" in specialists:
        inventory = (
            f"Retrieve the inventory policy for {target}. "
            "Focus on target stock level, reorder point, preferred replenishment "
            "quantity, shortage rules, and any planning instructions. "
            "Use only the inventory search tool and require an exact entity match."
        )

    supplier = None
    if "supplier" in specialists:
        supplier = (
            f"Retrieve the approved supplier information for {target}. "
            "Focus on supplier name, contractual lead time, order capacity, and "
            "other facts relevant to whether replenishment demand can be served. "
            "Use only the supplier search tool and require an exact entity match."
        )

    logistics = None
    if "logistics" in specialists:
        logistics = (
            "Retrieve the transportation policies relevant to the original "
            "replenishment request. Focus on available transportation modes, "
            "transit times after supplier dispatch, urgency criteria, approvals, "
            "cost trade-offs, and production-shortage constraints. Use only the "
            "logistics search tool. Do not search by the product entity_id because "
            "logistics documents are organized by transportation mode."
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
