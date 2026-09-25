from __future__ import annotations

import math

from ingestion.blob_reader import (
    download_source_document,
    list_source_documents,
)
from ingestion.chunking.chunker import chunk_documents
from ingestion.embedding_service import IngestionEmbeddingService
from ingestion.metadata.enricher import enrich_documents
from ingestion.parsers.csv_parser import parse_csv_bytes
from ingestion.parsers.pdf_parser import parse_pdf_bytes
from ingestion.parsers.xlsx_parser import parse_xlsx_bytes


def _parse(path: str):
    data = download_source_document(path)
    lower_path = path.lower()

    if lower_path.endswith(".csv"):
        return parse_csv_bytes(data, path)
    if lower_path.endswith(".xlsx"):
        return parse_xlsx_bytes(data, path)
    if lower_path.endswith(".pdf"):
        return parse_pdf_bytes(data, path)

    return []


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def main():
    source_paths = sorted(list_source_documents())
    parsed = []

    for path in source_paths:
        parsed.extend(_parse(path))

    enriched = enrich_documents(parsed)
    chunks = chunk_documents(enriched)

    selected = [
        chunk
        for chunk in chunks
        if (
            chunk.source_path == "inventory/inventory_policy_M10.pdf"
            or chunk.source_path == "procedures/replenishment_procedure.pdf"
            or chunk.document_id.endswith(
                "current_inventory.xlsx#sheet=Inventory Snapshot#row=2"
            )
        )
    ]

    if not selected:
        print("No validation chunks found.")
        return

    service = IngestionEmbeddingService()
    vectors = service.embed_texts(
        [chunk.content for chunk in selected]
    )

    print(f"Validation chunks: {len(selected)}")
    print(f"Embeddings returned: {len(vectors)}\n")

    for chunk, vector in zip(selected, vectors):
        print(chunk.chunk_id)
        print(f"  dimensions: {len(vector)}")
        print(f"  vector_norm: {_norm(vector):.6f}")
        print(f"  first_8_values: {[round(value, 6) for value in vector[:8]]}")
        print()


if __name__ == "__main__":
    main()
