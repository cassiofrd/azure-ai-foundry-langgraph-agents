from __future__ import annotations

import re

from ingestion.models import EnrichedDocument, ParsedDocument


ENTITY_PATTERN = re.compile(r"\b(?:M10|A100|V300|B200)\b", re.IGNORECASE)
LOCATION_PATTERN = re.compile(r"\b(?:PLANT-BH|PLANT-SP|DC-SP|DC-RJ)\b", re.IGNORECASE)
SUPPLIER_PATTERN = re.compile(r"\bSUP-(?:CONTOSO|NORTHWIND|FABRIKAM)\b", re.IGNORECASE)

ALIASES_BY_ENTITY = {
    "M10": ("BOLT-M10", "Bolt M10", "M10 Bolt"),
    "A100": ("BEARING-A100", "Bearing A100"),
    "V300": ("VALVE-V300", "Valve V300"),
    "B200": ("BOLT-B200", "Bolt B200", "B200 Bolt"),
}


def _unique_upper(pattern: re.Pattern[str], text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(match.upper() for match in pattern.findall(text)))


def _infer_document_type(document: ParsedDocument) -> str:
    name = document.metadata.get("source_file", "").lower()

    if "inventory_policy" in name:
        return "inventory_policy"
    if "current_inventory" in name:
        return "inventory_snapshot"
    if "supplier_master" in name:
        return "supplier_master"
    if "contract" in name:
        return "supplier_contract"
    if "freight_rates" in name:
        return "freight_rate"
    if "transportation_policy" in name:
        return "transportation_policy"
    if "carrier_restrictions" in name:
        return "carrier_restriction"
    if "demand_forecast" in name:
        return "demand_forecast"
    if "open_production_orders" in name:
        return "production_order"
    if "historical_consumption" in name:
        return "historical_consumption"
    if "replenishment_procedure" in name:
        return "replenishment_procedure"
    if "emergency_shortage_procedure" in name:
        return "emergency_shortage_procedure"

    return document.metadata.get("record_type", "unknown")


def _infer_global(document: ParsedDocument) -> bool:
    content = document.content.lower()

    if re.search(r"\bis_global\s*\n?\s*true\b", content):
        return True

    if re.search(r"\bscope\s*\n?\s*global\b", content):
        return True

    return False


def enrich_document(document: ParsedDocument) -> EnrichedDocument:
    searchable_text = " ".join(
        [
            document.source_path,
            document.content,
        ]
    )

    entity_ids = _unique_upper(ENTITY_PATTERN, searchable_text)
    location_ids = _unique_upper(LOCATION_PATTERN, searchable_text)
    supplier_ids = _unique_upper(SUPPLIER_PATTERN, searchable_text)

    aliases: list[str] = []
    for entity_id in entity_ids:
        aliases.extend(ALIASES_BY_ENTITY.get(entity_id, ()))

    return EnrichedDocument(
        document_id=document.document_id,
        source_path=document.source_path,
        file_type=document.file_type,
        content=document.content,
        domain=document.metadata.get("domain", "").lower(),
        document_type=_infer_document_type(document),
        entity_ids=entity_ids,
        aliases=tuple(dict.fromkeys(aliases)),
        location_ids=location_ids,
        supplier_ids=supplier_ids,
        is_global=_infer_global(document),
        metadata=dict(document.metadata),
    )


def enrich_documents(
    documents: list[ParsedDocument],
) -> list[EnrichedDocument]:
    return [enrich_document(document) for document in documents]
