from __future__ import annotations

from typing import Any

import pytest

from shared.embedding_service import EmbeddingResult
from shared.search_service import SearchService


class FakeEmbeddingService:
    def create_embedding(
        self,
        text: str,
    ) -> EmbeddingResult:
        assert text == "PARAFUSO-M10"

        return EmbeddingResult(
            embedding=(0.1, 0.2, 0.3),
        )


class FakeSearchClient:
    def search(
        self,
        *,
        search_text: str,
        top: int,
        select: list[str],
        vector_queries: list[Any],
    ):
        assert search_text == "PARAFUSO-M10"
        assert top == 3

        assert select == [
            "id",
            "title",
            "content",
            "agent",
            "doc_type",
            "entity_type",
            "entity_id",
            "source",
        ]

        assert len(vector_queries) == 1

        vector_query = vector_queries[0]

        assert vector_query.vector == [0.1, 0.2, 0.3]
        assert vector_query.k_nearest_neighbors == 3
        assert vector_query.fields == "content_vector"

        return [
            {
                "id": "duplicate",
                "title": "Parafuso M10",
                "content": "Fornecedor ABC",
                "agent": "inventory",
                "doc_type": "structured_reference",
                "entity_type": "product",
                "entity_id": "PARAFUSO-M10",
                "source": "ERP",
                "@search.score": 2.45,
            },
            {
                "id": "duplicate",
                "title": "Parafuso M10 duplicado",
                "content": "Este resultado deve ser removido.",
                "agent": "inventory",
                "doc_type": "structured_reference",
                "entity_type": "product",
                "entity_id": "PARAFUSO-M10",
                "source": "ERP",
                "@search.score": 9.99,
            },
            {
                "id": "empty",
                "title": "Documento vazio",
                "content": "   ",
                "agent": "inventory",
                "doc_type": "structured_reference",
                "entity_type": "product",
                "entity_id": "EMPTY",
                "source": "ERP",
                "@search.score": 8.0,
            },
            {
                "id": "lower-score",
                "title": "Política de estoque",
                "content": "Documento complementar.",
                "agent": "inventory",
                "doc_type": "policy",
                "entity_type": "inventory_policy",
                "entity_id": "POLICY-001",
                "source": "SharePoint",
                "@search.score": 1.25,
            },
            {
                "id": "no-score",
                "title": "Manual interno",
                "content": "Documento sem score.",
                "agent": "logistics",
                "doc_type": "manual",
                "entity_type": "procedure",
                "entity_id": "MANUAL-001",
                "source": "Blob Storage",
            },
        ]


class FakeLongContentSearchClient:
    def search(
        self,
        *,
        search_text: str,
        top: int,
        select: list[str],
        vector_queries: list[Any],
    ):
        assert search_text == "PARAFUSO-M10"
        assert top == 3
        assert select
        assert len(vector_queries) == 1

        return [
            {
                "id": "long-document",
                "title": "Documento longo",
                "content": "A" * 1500,
                "agent": "procurement",
                "doc_type": "manual",
                "entity_type": "procedure",
                "entity_id": "LONG-001",
                "source": "Blob Storage",
                "@search.score": 3.0,
            }
        ]


def fake_factory(
    endpoint: str,
    index_name: str,
    admin_key: str,
):
    assert endpoint == "endpoint"
    assert index_name == "index"
    assert admin_key == "key"

    return FakeSearchClient()


def fake_long_content_factory(
    endpoint: str,
    index_name: str,
    admin_key: str,
):
    assert endpoint == "endpoint"
    assert index_name == "index"
    assert admin_key == "key"

    return FakeLongContentSearchClient()


def build_service(
    *,
    client_factory=fake_factory,
    embedding_service: Any | None = None,
    vector_field: str = "content_vector",
) -> SearchService:
    return SearchService(
        endpoint="endpoint",
        index_name="index",
        admin_key="key",
        top_k=3,
        vector_field=vector_field,
        embedding_service=(
            embedding_service
            if embedding_service is not None
            else FakeEmbeddingService()
        ),
        client_factory=client_factory,
    )


def test_search_documents_sends_hybrid_query():
    service = build_service()

    documents = service.search_documents(
        "PARAFUSO-M10"
    )

    assert documents


def test_search_documents_enriches_results():
    service = build_service()

    documents = service.search_documents(
        "PARAFUSO-M10"
    )

    assert len(documents) == 3

    first_document = documents[0]

    assert first_document.document_id == "duplicate"
    assert first_document.title == "Parafuso M10"
    assert first_document.content == "Fornecedor ABC"
    assert first_document.agent == "inventory"
    assert (
        first_document.doc_type
        == "structured_reference"
    )
    assert first_document.entity_type == "product"
    assert (
        first_document.entity_id
        == "PARAFUSO-M10"
    )
    assert first_document.source == "ERP"
    assert first_document.score == 2.45


def test_search_documents_removes_duplicates():
    service = build_service()

    documents = service.search_documents(
        "PARAFUSO-M10"
    )

    document_ids = [
        document.document_id
        for document in documents
    ]

    assert document_ids.count("duplicate") == 1


def test_search_documents_removes_empty_content():
    service = build_service()

    documents = service.search_documents(
        "PARAFUSO-M10"
    )

    document_ids = {
        document.document_id
        for document in documents
    }

    assert "empty" not in document_ids


def test_search_documents_orders_by_score():
    service = build_service()

    documents = service.search_documents(
        "PARAFUSO-M10"
    )

    assert [
        document.document_id
        for document in documents
    ] == [
        "duplicate",
        "lower-score",
        "no-score",
    ]


def test_search_documents_truncates_long_content():
    service = build_service(
        client_factory=fake_long_content_factory,
    )

    documents = service.search_documents(
        "PARAFUSO-M10"
    )

    content = documents[0].content

    assert len(content) == 1003
    assert content == ("A" * 1000) + "..."


def test_empty_query_raises():
    service = build_service()

    with pytest.raises(
        ValueError,
        match="Search query cannot be empty",
    ):
        service.search_documents("   ")


def test_missing_embedding_service_raises():
    service = SearchService(
        endpoint="endpoint",
        index_name="index",
        admin_key="key",
        top_k=3,
        vector_field="content_vector",
        embedding_service=None,
        client_factory=fake_factory,
    )

    with pytest.raises(
        ValueError,
        match="EmbeddingService is required for hybrid search",
    ):
        service.search_documents("PARAFUSO-M10")


@pytest.mark.parametrize(
    (
        "endpoint",
        "index_name",
        "admin_key",
        "top_k",
        "vector_field",
        "expected_message",
    ),
    [
        (
            "",
            "index",
            "key",
            3,
            "content_vector",
            "AZURE_SEARCH_ENDPOINT",
        ),
        (
            "endpoint",
            "",
            "key",
            3,
            "content_vector",
            "AZURE_SEARCH_INDEX_NAME",
        ),
        (
            "endpoint",
            "index",
            "",
            3,
            "content_vector",
            "AZURE_SEARCH_ADMIN_KEY",
        ),
        (
            "endpoint",
            "index",
            "key",
            0,
            "content_vector",
            "top_k",
        ),
        (
            "endpoint",
            "index",
            "key",
            3,
            "",
            "AZURE_SEARCH_VECTOR_FIELD",
        ),
    ],
)
def test_invalid_configuration_raises(
    endpoint: str,
    index_name: str,
    admin_key: str,
    top_k: int,
    vector_field: str,
    expected_message: str,
):
    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        SearchService(
            endpoint=endpoint,
            index_name=index_name,
            admin_key=admin_key,
            top_k=top_k,
            vector_field=vector_field,
            embedding_service=FakeEmbeddingService(),
            client_factory=fake_factory,
        )


def test_search_documents_applies_agent_filter():
    captured = {}

    class FilterClient:
        def search(self, **kwargs):
            captured.update(kwargs)
            return []

    service = build_service(
        client_factory=lambda endpoint, index_name, admin_key: FilterClient()
    )
    service.search_documents("PARAFUSO-M10", agent="supplier")
    assert captured["filter"] == "agent eq 'supplier'"
