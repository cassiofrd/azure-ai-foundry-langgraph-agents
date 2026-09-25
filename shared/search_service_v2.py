from __future__ import annotations

import re

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

from ingestion.embedding_service import IngestionEmbeddingService
from shared.settings import load_settings


INDEX_NAME = "supply-chain-docs-v2"
VECTOR_FIELD = "content_vector"

ENTITY_ALIASES = {
    "M10": ("M10", "BOLT-M10", "BOLT M10", "M10 BOLT"),
    "A100": ("A100", "BEARING-A100", "BEARING A100"),
    "V300": ("V300", "VALVE-V300", "VALVE V300"),
    "B200": ("B200", "BOLT-B200", "BOLT B200", "B200 BOLT"),
}

LOCATION_IDS = ("PLANT-BH", "PLANT-SP", "DC-SP", "DC-RJ")

DOMAIN_RULES = {
    "inventory": (
        "inventory", "on hand", "available", "stock",
        "reorder point", "safety stock",
    ),
    "suppliers": (
        "supplier", "vendor", "contract", "lead time", "moq",
    ),
    "logistics": (
        "shipping", "transport", "freight", "carrier",
        "ground", "air freight",
    ),
    "demand": (
        "demand", "forecast", "consumption", "production order",
    ),
    "procedures": (
        "procedure", "process", "workflow", "approval",
    ),
}

# Intent is narrower than domain. It is used only as a soft reranking hint:
# neighboring relevant documents remain available to the RAG.
INTENT_RULES = (
    (
        "inventory_snapshot",
        ("current inventory", "current stock", "on hand", "available inventory"),
        ("inventory_snapshot",),
    ),
    (
        "inventory_policy",
        ("reorder point", "safety stock", "target inventory", "inventory policy"),
        ("inventory_policy",),
    ),
    (
        "transportation_policy",
        (
            "shipping options", "shipping option", "transportation options",
            "transport options", "shipping methods", "transportation modes",
        ),
        ("transportation_policy", "carrier_restriction", "freight_rate"),
    ),
    (
        "replenishment_procedure",
        (
            "replenishment procedure", "standard replenishment",
            "replenishment process",
        ),
        ("replenishment_procedure",),
    ),
    (
        "emergency_shortage_procedure",
        ("emergency shortage", "critical shortage", "emergency procedure"),
        ("emergency_shortage_procedure",),
    ),
    (
        "supplier_master",
        ("approved supplier", "approved vendor", "which supplier", "supplier for"),
        ("supplier_master", "supplier_contract"),
    ),
    (
        "demand_forecast",
        ("demand forecast", "forecast demand", "forecast for"),
        ("demand_forecast",),
    ),
    (
        "production_order",
        (
            "open production order", "open production orders",
            "production order",
        ),
        ("production_order",),
    ),
)


def _contains_token(text: str, token: str) -> bool:
    pattern = rf"(?<![A-Z0-9]){re.escape(token)}(?![A-Z0-9])"
    return re.search(pattern, text.upper()) is not None


def resolve_entity(query: str) -> str | None:
    upper = query.upper()
    for canonical, aliases in ENTITY_ALIASES.items():
        if any(_contains_token(upper, alias) for alias in aliases):
            return canonical
    return None


def resolve_location(query: str) -> str | None:
    upper = query.upper()
    for location in LOCATION_IDS:
        if _contains_token(upper, location):
            return location
    return None


def infer_domain(query: str) -> str | None:
    q = query.lower()
    matches = [
        domain
        for domain, terms in DOMAIN_RULES.items()
        if any(term in q for term in terms)
    ]
    return matches[0] if len(matches) == 1 else None


def infer_intent(query: str) -> tuple[str | None, tuple[str, ...]]:
    q = query.lower()
    for intent, terms, preferred_types in INTENT_RULES:
        if any(term in q for term in terms):
            return intent, preferred_types
    return None, ()


def build_metadata_filter(
    *,
    domain: str | None = None,
    entity_id: str | None = None,
    location_id: str | None = None,
    include_global: bool = True,
) -> str | None:
    clauses = []

    if domain:
        clauses.append(f"domain eq '{domain}'")

    if entity_id:
        entity_clause = f"entity_ids/any(e: e eq '{entity_id}')"
        if include_global:
            entity_clause = f"({entity_clause} or is_global eq true)"
        clauses.append(entity_clause)

    if location_id:
        location_clause = f"location_ids/any(l: l eq '{location_id}')"
        if include_global:
            location_clause = f"({location_clause} or is_global eq true)"
        clauses.append(location_clause)

    return " and ".join(clauses) if clauses else None


class SearchServiceV2:
    def __init__(self):
        settings = load_settings()
        self._client = SearchClient(
            endpoint=settings.azure_search_endpoint,
            index_name=INDEX_NAME,
            credential=AzureKeyCredential(settings.azure_search_admin_key),
        )
        self._embedding_service = IngestionEmbeddingService()

    def hybrid_search(
        self,
        query: str,
        *,
        top: int = 5,
        filter: str | None = None,
    ):
        vector = self._embedding_service.embed_texts([query])[0]
        vector_query = VectorizedQuery(
            vector=vector,
            k_nearest_neighbors=max(top, 10),
            fields=VECTOR_FIELD,
        )

        return list(
            self._client.search(
                search_text=query,
                vector_queries=[vector_query],
                filter=filter,
                top=top,
                select=self._select_fields(),
            )
        )

    def structured_search(
        self,
        *,
        domain: str,
        document_type: str,
        entity_id: str | None = None,
        location_id: str | None = None,
        top: int = 1000,
    ):
        """Return every indexed record matching deterministic metadata filters."""
        clauses = [
            f"domain eq '{domain}'",
            f"document_type eq '{document_type}'",
        ]

        if entity_id:
            clauses.append(
                f"entity_ids/any(e: e eq '{entity_id}')"
            )

        if location_id:
            clauses.append(
                f"location_ids/any(l: l eq '{location_id}')"
            )

        metadata_filter = " and ".join(clauses)

        results = list(
            self._client.search(
                search_text="*",
                filter=metadata_filter,
                top=top,
                select=self._select_fields(),
            )
        )

        return {
            "domain": domain,
            "document_type": document_type,
            "entity_id": entity_id,
            "location_id": location_id,
            "filter": metadata_filter,
            "results": results,
        }

    def smart_hybrid_search(
        self,
        query: str,
        *,
        top: int = 5,
        candidate_pool: int = 10,
        domain: str | None = None,
        entity_id: str | None = None,
        location_id: str | None = None,
        include_global: bool = True,
    ):
        resolved_entity = entity_id or resolve_entity(query)
        resolved_location = location_id or resolve_location(query)
        resolved_domain = domain or infer_domain(query)
        intent, preferred_types = infer_intent(query)

        metadata_filter = build_metadata_filter(
            domain=resolved_domain,
            entity_id=resolved_entity,
            location_id=resolved_location,
            include_global=include_global,
        )

        candidates = self.hybrid_search(
            query,
            top=max(top, candidate_pool),
            filter=metadata_filter,
        )

        reranked = self._rerank_by_intent(
            candidates,
            preferred_types=preferred_types,
        )

        return {
            "query": query,
            "entity_id": resolved_entity,
            "location_id": resolved_location,
            "domain": resolved_domain,
            "intent": intent,
            "preferred_document_types": list(preferred_types),
            "filter": metadata_filter,
            "results": reranked[:top],
        }

    @staticmethod
    def _rerank_by_intent(results, *, preferred_types: tuple[str, ...]):
        if not preferred_types:
            return results

        priority = {
            document_type: index
            for index, document_type in enumerate(preferred_types)
        }

        decorated = []
        for original_rank, result in enumerate(results):
            document_type = result.get("document_type")
            preferred = document_type in priority

            decorated.append(
                (
                    0 if preferred else 1,
                    priority.get(document_type, len(priority)),
                    original_rank,
                    result,
                )
            )

        decorated.sort(key=lambda item: item[:3])
        return [item[3] for item in decorated]

    @staticmethod
    def _select_fields():
        return [
            "id",
            "chunk_id",
            "source_path",
            "domain",
            "document_type",
            "content",
            "entity_ids",
            "aliases",
            "location_ids",
            "supplier_ids",
            "is_global",
        ]
