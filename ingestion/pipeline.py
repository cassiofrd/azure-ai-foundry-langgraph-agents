from __future__ import annotations

import base64
import hashlib

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

from ingestion.blob_reader import download_source_document, list_source_documents
from ingestion.chunking.chunker import chunk_documents
from ingestion.embedding_service import IngestionEmbeddingService
from ingestion.metadata.enricher import enrich_documents
from ingestion.parsers.csv_parser import parse_csv_bytes
from ingestion.parsers.pdf_parser import parse_pdf_bytes
from ingestion.parsers.xlsx_parser import parse_xlsx_bytes
from shared.settings import load_settings

INDEX_NAME = "supply-chain-docs-v2"
EMBEDDING_BATCH_SIZE = 16
UPLOAD_BATCH_SIZE = 100


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


def _search_key(chunk_id: str) -> str:
    digest = hashlib.sha256(chunk_id.encode("utf-8")).digest()
    encoded = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return f"d-{encoded}"


def _optional_int(metadata: dict[str, str], key: str):
    value = metadata.get(key)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _to_search_document(chunk, vector: list[float]) -> dict:
    metadata = chunk.metadata
    return {
        "id": _search_key(chunk.chunk_id),
        "document_id": chunk.document_id,
        "chunk_id": chunk.chunk_id,
        "source_path": chunk.source_path,
        "source_file": metadata.get("source_file", ""),
        "file_type": chunk.file_type,
        "domain": chunk.domain,
        "document_type": chunk.document_type,
        "content": chunk.content,
        "entity_ids": list(chunk.entity_ids),
        "aliases": list(chunk.aliases),
        "location_ids": list(chunk.location_ids),
        "supplier_ids": list(chunk.supplier_ids),
        "is_global": chunk.is_global,
        "page_number": _optional_int(metadata, "page_number"),
        "row_number": _optional_int(metadata, "row_number"),
        "sheet_name": metadata.get("sheet_name"),
        "content_vector": vector,
    }


def build_chunks():
    source_paths = sorted(list_source_documents())
    parsed = []
    for path in source_paths:
        parsed.extend(_parse(path))
    enriched = enrich_documents(parsed)
    chunks = chunk_documents(enriched)
    return source_paths, parsed, enriched, chunks


def generate_search_documents(chunks):
    service = IngestionEmbeddingService()
    documents = []

    for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[start:start + EMBEDDING_BATCH_SIZE]
        vectors = service.embed_texts([chunk.content for chunk in batch])

        if len(vectors) != len(batch):
            raise RuntimeError("Embedding count does not match chunk count.")

        for chunk, vector in zip(batch, vectors):
            documents.append(_to_search_document(chunk, vector))

        print(f"Embedded {min(start + len(batch), len(chunks))}/{len(chunks)} chunks")

    return documents


def upload_search_documents(documents):
    settings = load_settings()
    client = SearchClient(
        endpoint=settings.azure_search_endpoint,
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(settings.azure_search_admin_key),
    )

    succeeded = 0
    for start in range(0, len(documents), UPLOAD_BATCH_SIZE):
        batch = documents[start:start + UPLOAD_BATCH_SIZE]
        results = client.upload_documents(documents=batch)
        failures = [result for result in results if not result.succeeded]

        if failures:
            details = "; ".join(
                f"{result.key}: {result.error_message}" for result in failures
            )
            raise RuntimeError(f"Azure AI Search upload failed: {details}")

        succeeded += len(results)
        print(f"Uploaded {succeeded}/{len(documents)} documents")


def run_ingestion():
    source_paths, parsed, enriched, chunks = build_chunks()
    print(f"Source files: {len(source_paths)}")
    print(f"Parsed documents: {len(parsed)}")
    print(f"Enriched documents: {len(enriched)}")
    print(f"Chunks: {len(chunks)}")

    documents = generate_search_documents(chunks)
    print(f"Search documents prepared: {len(documents)}")
    upload_search_documents(documents)

    print("\nIngestion completed successfully.")
    print(f"Index: {INDEX_NAME}")
    print(f"Indexed documents: {len(documents)}")


if __name__ == "__main__":
    run_ingestion()
