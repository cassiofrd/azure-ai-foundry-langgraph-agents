from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

from shared.embedding_service import EmbeddingService
from shared.settings import AppSettings, load_settings


DEFAULT_DOCUMENTS_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "documents"
)

REQUIRED_FIELDS = (
    "id",
    "title",
    "content",
    "agent",
    "doc_type",
    "entity_type",
    "entity_id",
    "source",
)


@dataclass(frozen=True)
class SourceDocument:
    id: str
    title: str
    content: str
    agent: str
    doc_type: str
    entity_type: str
    entity_id: str
    source: str


class EmbeddingServiceProtocol(Protocol):
    def create_embedding(
        self,
        text: str,
    ) -> Any:
        """Create an embedding for the supplied text."""


class SearchClientProtocol(Protocol):
    def upload_documents(
        self,
        *,
        documents: list[dict[str, Any]],
    ) -> Sequence[Any]:
        """Upload documents to Azure AI Search."""


def validate_search_settings(
    settings: AppSettings,
) -> None:
    if not settings.azure_search_endpoint:
        raise ValueError(
            "AZURE_SEARCH_ENDPOINT is required."
        )

    if not settings.azure_search_index_name:
        raise ValueError(
            "AZURE_SEARCH_INDEX_NAME is required."
        )

    if not settings.azure_search_admin_key:
        raise ValueError(
            "AZURE_SEARCH_ADMIN_KEY is required."
        )

    if not settings.azure_search_vector_field:
        raise ValueError(
            "AZURE_SEARCH_VECTOR_FIELD is required."
        )

    if settings.azure_search_vector_dimensions < 1:
        raise ValueError(
            "AZURE_SEARCH_VECTOR_DIMENSIONS must be "
            "greater than zero."
        )


def _required_string(
    raw_document: dict[str, Any],
    field_name: str,
    file_path: Path,
) -> str:
    value = raw_document.get(field_name)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Field '{field_name}' must be a non-empty "
            f"string in {file_path}."
        )

    return value.strip()


def parse_source_document(
    raw_document: dict[str, Any],
    file_path: Path,
) -> SourceDocument:
    missing_fields = [
        field_name
        for field_name in REQUIRED_FIELDS
        if field_name not in raw_document
    ]

    if missing_fields:
        formatted_fields = ", ".join(missing_fields)

        raise ValueError(
            f"Missing required fields in {file_path}: "
            f"{formatted_fields}."
        )

    return SourceDocument(
        id=_required_string(
            raw_document,
            "id",
            file_path,
        ),
        title=_required_string(
            raw_document,
            "title",
            file_path,
        ),
        content=_required_string(
            raw_document,
            "content",
            file_path,
        ),
        agent=_required_string(
            raw_document,
            "agent",
            file_path,
        ),
        doc_type=_required_string(
            raw_document,
            "doc_type",
            file_path,
        ),
        entity_type=_required_string(
            raw_document,
            "entity_type",
            file_path,
        ),
        entity_id=_required_string(
            raw_document,
            "entity_id",
            file_path,
        ),
        source=_required_string(
            raw_document,
            "source",
            file_path,
        ),
    )


def load_source_documents(
    documents_directory: Path,
) -> list[SourceDocument]:
    if not documents_directory.exists():
        raise FileNotFoundError(
            "Documents directory was not found: "
            f"{documents_directory}"
        )

    if not documents_directory.is_dir():
        raise ValueError(
            "Documents path must be a directory: "
            f"{documents_directory}"
        )

    json_files = sorted(
        documents_directory.glob("*.json")
    )

    if not json_files:
        raise ValueError(
            "No JSON document files were found in: "
            f"{documents_directory}"
        )

    documents: list[SourceDocument] = []
    document_ids: set[str] = set()

    for file_path in json_files:
        try:
            raw_data = json.loads(
                file_path.read_text(encoding="utf-8")
            )

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON file: {file_path}"
            ) from exc

        if not isinstance(raw_data, list):
            raise ValueError(
                "Each document file must contain a JSON "
                f"array: {file_path}"
            )

        for raw_document in raw_data:
            if not isinstance(raw_document, dict):
                raise ValueError(
                    "Each JSON array item must be an "
                    f"object: {file_path}"
                )

            document = parse_source_document(
                raw_document=raw_document,
                file_path=file_path,
            )

            if document.id in document_ids:
                raise ValueError(
                    "Duplicate document id found: "
                    f"{document.id}"
                )

            document_ids.add(document.id)
            documents.append(document)

    return documents


def build_embedding_input(
    document: SourceDocument,
) -> str:
    return (
        f"Title: {document.title}\n"
        f"Agent: {document.agent}\n"
        f"Document type: {document.doc_type}\n"
        f"Entity type: {document.entity_type}\n"
        f"Entity id: {document.entity_id}\n"
        f"Content: {document.content}"
    )


def build_index_documents(
    *,
    source_documents: Sequence[SourceDocument],
    settings: AppSettings,
    embedding_service: EmbeddingServiceProtocol,
) -> list[dict[str, Any]]:
    index_documents: list[dict[str, Any]] = []

    for position, document in enumerate(
        source_documents,
        start=1,
    ):
        print(
            "Generating embedding "
            f"{position}/{len(source_documents)}: "
            f"{document.id}"
        )

        embedding_result = (
            embedding_service.create_embedding(
                build_embedding_input(document)
            )
        )

        embedding = list(embedding_result.embedding)

        if (
            len(embedding)
            != settings.azure_search_vector_dimensions
        ):
            raise ValueError(
                "Embedding dimension mismatch for "
                f"document '{document.id}'. Expected "
                f"{settings.azure_search_vector_dimensions}, "
                f"received {len(embedding)}."
            )

        index_documents.append(
            {
                "id": document.id,
                "title": document.title,
                "content": document.content,
                "agent": document.agent,
                "doc_type": document.doc_type,
                "entity_type": document.entity_type,
                "entity_id": document.entity_id,
                "source": document.source,
                settings.azure_search_vector_field: (
                    embedding
                ),
            }
        )

    return index_documents


def create_search_client(
    settings: AppSettings,
) -> SearchClient:
    return SearchClient(
        endpoint=settings.azure_search_endpoint,
        index_name=settings.azure_search_index_name,
        credential=AzureKeyCredential(
            settings.azure_search_admin_key
        ),
    )


def upload_index_documents(
    *,
    search_client: SearchClientProtocol,
    documents: list[dict[str, Any]],
) -> None:
    if not documents:
        raise ValueError(
            "At least one document is required for upload."
        )

    results = search_client.upload_documents(
        documents=documents
    )

    failures: list[str] = []

    for result in results:
        if result.succeeded:
            continue

        key = getattr(result, "key", "unknown")
        error_message = getattr(
            result,
            "error_message",
            "Unknown indexing error.",
        )

        failures.append(
            f"{key}: {error_message}"
        )

    if failures:
        details = "\n".join(failures)

        raise RuntimeError(
            "One or more documents could not be "
            f"uploaded:\n{details}"
        )


def main() -> None:
    settings = load_settings()

    validate_search_settings(settings)

    source_documents = load_source_documents(
        DEFAULT_DOCUMENTS_DIRECTORY
    )

    print(
        f"Loaded {len(source_documents)} source documents."
    )

    embedding_service = EmbeddingService(
        settings=settings,
        client=None,
    )

    index_documents = build_index_documents(
        source_documents=source_documents,
        settings=settings,
        embedding_service=embedding_service,
    )

    search_client = create_search_client(settings)

    upload_index_documents(
        search_client=search_client,
        documents=index_documents,
    )

    print(
        "Documents uploaded successfully to Azure AI "
        f"Search: {len(index_documents)}"
    )
    print(
        "Index name: "
        f"{settings.azure_search_index_name}"
    )
    print(
        "Vector field: "
        f"{settings.azure_search_vector_field}"
    )


if __name__ == "__main__":
    main()