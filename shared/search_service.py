from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

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

    def __init__(
        self,
        *,
        endpoint: str,
        index_name: str,
        admin_key: str,
        top_k: int = 3,
        client_factory: SearchClientFactory | None = None,
    ) -> None:
        self._endpoint = endpoint.strip()
        self._index_name = index_name.strip()
        self._admin_key = admin_key.strip()
        self._top_k = top_k
        self._client_factory = (
            client_factory or self._create_search_client
        )

        self._validate_configuration()

    @classmethod
    def from_settings(
        cls,
        settings: AppSettings,
        *,
        client_factory: SearchClientFactory | None = None,
    ) -> SearchService:
        return cls(
            endpoint=settings.azure_search_endpoint,
            index_name=settings.azure_search_index_name,
            admin_key=settings.azure_search_admin_key,
            top_k=settings.azure_search_top_k,
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

        results = client.search(
            search_text=normalized_query,
            top=self._top_k,
            select=self._SELECT_FIELDS,
        )

        return [
            self._normalize_document(document)
            for document in results
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

    @staticmethod
    def _normalize_document(
        document: dict[str, Any],
    ) -> SearchDocument:
        raw_score = document.get("@search.score")

        score = (
            float(raw_score)
            if raw_score is not None
            else None
        )

        return SearchDocument(
            document_id=str(document.get("id", "")),
            title=str(document.get("title", "")),
            content=str(document.get("content", "")),
            agent=str(document.get("agent", "")),
            doc_type=str(document.get("doc_type", "")),
            entity_type=str(
                document.get("entity_type", "")
            ),
            entity_id=str(document.get("entity_id", "")),
            source=str(document.get("source", "")),
            score=score,
        )