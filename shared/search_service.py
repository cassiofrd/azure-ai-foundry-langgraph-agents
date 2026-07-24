from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

from shared.embedding_service import EmbeddingService
from shared.settings import AppSettings


class SearchClientProtocol(Protocol):
    def search(
        self,
        *,
        search_text: str,
        top: int,
        select: list[str],
    ) -> Iterable[dict[str, Any]]:
        ...


SearchClientFactory = Callable[
    [str, str, str],
    SearchClientProtocol,
]


@dataclass(frozen=True)
class SearchDocument:
    document_id: str
    title: str
    content: str
    agent: str
    doc_type: str
    entity_type: str
    entity_id: str
    source: str
    score: float | None


class SearchService:
    _SELECT_FIELDS = [
        "id",
        "title",
        "content",
        "agent",
        "doc_type",
        "entity_type",
        "entity_id",
        "source",
    ]

    _MAX_CONTENT_LENGTH = 1000

    def __init__(
        self,
        *,
        endpoint: str,
        index_name: str,
        admin_key: str,
        top_k: int = 3,
        embedding_service: EmbeddingService | None = None,
        client_factory: SearchClientFactory | None = None,
    ) -> None:
        self._endpoint = endpoint.strip()
        self._index_name = index_name.strip()
        self._admin_key = admin_key.strip()
        self._top_k = top_k
        self._embedding_service = embedding_service
        self._client_factory = (
            client_factory or self._create_search_client
        )

        self._validate_configuration()

    @classmethod
    def from_settings(
        cls,
        settings: AppSettings,
        *,
        embedding_service: EmbeddingService | None = None,
        client_factory: SearchClientFactory | None = None,
    ) -> SearchService:
        resolved_embedding_service = (
            embedding_service
            or EmbeddingService(
                settings=settings,
                client=None,
            )
        )

        return cls(
            endpoint=settings.azure_search_endpoint,
            index_name=settings.azure_search_index_name,
            admin_key=settings.azure_search_admin_key,
            top_k=settings.azure_search_top_k,
            embedding_service=resolved_embedding_service,
            client_factory=client_factory,
        )

    def search_documents(
        self,
        query: str,
    ) -> list[SearchDocument]:

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError(
                "Search query cannot be empty."
            )

        client = self._client_factory(
            self._endpoint,
            self._index_name,
            self._admin_key,
        )

        search_kwargs = {
            "search_text": normalized_query,
            "top": self._top_k,
            "select": self._SELECT_FIELDS,
        }

        results = client.search(**search_kwargs)

        documents = [
            self._normalize_document(document)
            for document in results
        ]

        documents = self._remove_duplicates(
            documents
        )

        documents = self._remove_empty_documents(
            documents
        )

        documents.sort(
            key=lambda document: (
                document.score is None,
                -(document.score or 0.0),
            )
        )

        return documents

    def _remove_duplicates(
        self,
        documents: list[SearchDocument],
    ) -> list[SearchDocument]:

        unique: dict[str, SearchDocument] = {}

        for document in documents:
            unique.setdefault(
                document.document_id,
                document,
            )

        return list(unique.values())

    def _remove_empty_documents(
        self,
        documents: list[SearchDocument],
    ) -> list[SearchDocument]:

        return [
            document
            for document in documents
            if document.content.strip()
        ]

    def _validate_configuration(self) -> None:

        if not self._endpoint:
            raise ValueError(
                "AZURE_SEARCH_ENDPOINT is required to use "
                "Azure AI Search."
            )

        if not self._index_name:
            raise ValueError(
                "AZURE_SEARCH_INDEX_NAME is required to use "
                "Azure AI Search."
            )

        if not self._admin_key:
            raise ValueError(
                "AZURE_SEARCH_ADMIN_KEY is required to use "
                "Azure AI Search."
            )

        if self._top_k < 1:
            raise ValueError(
                "Azure Search top_k must be greater than zero."
            )

    @staticmethod
    def _create_search_client(
        endpoint: str,
        index_name: str,
        admin_key: str,
    ) -> SearchClient:

        return SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(admin_key),
        )

    @classmethod
    def _normalize_document(
        cls,
        document: dict[str, Any],
    ) -> SearchDocument:

        raw_score = document.get("@search.score")

        score = (
            float(raw_score)
            if raw_score is not None
            else None
        )

        content = str(
            document.get("content", "")
        )

        if len(content) > cls._MAX_CONTENT_LENGTH:
            content = (
                content[
                    : cls._MAX_CONTENT_LENGTH
                ].rstrip()
                + "..."
            )

        return SearchDocument(
            document_id=str(document.get("id", "")),
            title=str(document.get("title", "")),
            content=content,
            agent=str(document.get("agent", "")),
            doc_type=str(
                document.get("doc_type", "")
            ),
            entity_type=str(
                document.get("entity_type", "")
            ),
            entity_id=str(
                document.get("entity_id", "")
            ),
            source=str(
                document.get("source", "")
            ),
            score=score,
        )