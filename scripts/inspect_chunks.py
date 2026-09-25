from collections import Counter

from ingestion.blob_reader import (
    download_source_document,
    list_source_documents,
)
from ingestion.chunking.chunker import chunk_documents
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


def main():
    source_paths = sorted(list_source_documents())
    parsed = []

    for path in source_paths:
        parsed.extend(_parse(path))

    enriched = enrich_documents(parsed)
    chunks = chunk_documents(enriched)

    by_type = Counter(chunk.file_type for chunk in chunks)

    print(f"Source files: {len(source_paths)}")
    print(f"Parsed documents: {len(parsed)}")
    print(f"Enriched documents: {len(enriched)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Chunks by file type: {dict(sorted(by_type.items()))}\n")

    selected = [
        chunk
        for chunk in chunks
        if (
            chunk.source_path == "inventory/inventory_policy_M10.pdf"
            or chunk.source_path == "procedures/replenishment_procedure.pdf"
            or chunk.document_id.endswith("current_inventory.xlsx#sheet=Inventory Snapshot#row=2")
        )
    ]

    for chunk in selected:
        print(chunk.chunk_id)
        print(f"  document_type: {chunk.document_type}")
        print(f"  entity_ids: {list(chunk.entity_ids)}")
        print(f"  aliases: {list(chunk.aliases)}")
        print(f"  location_ids: {list(chunk.location_ids)}")
        print(f"  is_global: {chunk.is_global}")
        print(f"  content_chars: {len(chunk.content)}")
        print("  content:")
        for line in chunk.content.splitlines()[:20]:
            print(f"    {line}")
        if len(chunk.content.splitlines()) > 20:
            print("    ...")
        print()


if __name__ == "__main__":
    main()
