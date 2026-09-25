from ingestion.blob_reader import (
    download_source_document,
    list_source_documents,
)
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

    print(f"Source files: {len(source_paths)}")
    print(f"Parsed documents: {len(parsed)}")
    print(f"Enriched documents: {len(enriched)}\n")

    interesting = [
        document
        for document in enriched
        if (
            "M10" in document.entity_ids
            or document.is_global
        )
    ]

    for document in interesting[:15]:
        print(document.document_id)
        print(f"  domain: {document.domain}")
        print(f"  document_type: {document.document_type}")
        print(f"  entity_ids: {list(document.entity_ids)}")
        print(f"  aliases: {list(document.aliases)}")
        print(f"  location_ids: {list(document.location_ids)}")
        print(f"  supplier_ids: {list(document.supplier_ids)}")
        print(f"  is_global: {document.is_global}")
        print()


if __name__ == "__main__":
    main()
