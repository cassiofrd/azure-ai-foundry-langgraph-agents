from __future__ import annotations

from shared.search_service import SearchService


class FakeSearchClient:
    def search(
        self,
        *,
        search_text: str,
        top: int,
        select: list[str],
    ):
        assert search_text == "PARAFUSO-M10"
        assert top == 3

        return [
            {
                "id": "1",
                "title": "Parafuso M10",
                "content": "Fornecedor ABC",
                "agent": "inventory",
                "doc_type": "structured_reference",
                "entity_type": "product",
                "entity_id": "PARAFUSO-M10",
                "source": "ERP",
                "@search.score": 2.45,
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


def test_search_documents_returns_documents():

    service = SearchService(
        endpoint="endpoint",
        index_name="index",
        admin_key="key",
        top_k=3,
        client_factory=fake_factory,
    )

    documents = service.search_documents(
        "PARAFUSO-M10"
    )

    assert len(documents) == 1

    document = documents[0]

    assert document.title == "Parafuso M10"
    assert document.agent == "inventory"
    assert document.entity_id == "PARAFUSO-M10"
    assert document.score == 2.45


def test_empty_query_raises():

    service = SearchService(
        endpoint="endpoint",
        index_name="index",
        admin_key="key",
        client_factory=fake_factory,
    )

    import pytest

    with pytest.raises(ValueError):
        service.search_documents("")