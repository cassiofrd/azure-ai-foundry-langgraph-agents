import json

from shared.tools import TOOL_REGISTRY
from shared.tools import TOOLS
from shared.tools import search_documents


def test_search_tool_registered():

    assert "search_documents" in TOOL_REGISTRY


def test_search_tool_declared():

    names = {
        tool["name"]
        for tool in TOOLS
    }

    assert "search_documents" in names


def test_search_tool_has_query_parameter():

    tool = next(
        tool
        for tool in TOOLS
        if tool["name"] == "search_documents"
    )

    properties = tool["parameters"]["properties"]

    assert "query" in properties

    assert (
        properties["query"]["type"]
        == "string"
    )

    assert (
        tool["parameters"]["required"]
        == ["query"]
    )


def test_search_tool_description_mentions_enterprise_knowledge():

    tool = next(
        tool
        for tool in TOOLS
        if tool["name"] == "search_documents"
    )

    description = tool["description"].lower()

    assert "enterprise" in description
    assert "knowledge" in description
    assert "suppliers" in description


def test_search_documents_returns_structured_payload(
    monkeypatch,
):

    class FakeDocument:
        title = "Parafuso M10"
        content = "Fornecedor ABC"
        agent = "inventory"
        doc_type = "structured_reference"
        entity_type = "product"
        entity_id = "PARAFUSO-M10"
        source = "ERP"
        score = 2.45

    class FakeSearchService:
        def search_documents(
            self,
            query: str,
        ):
            return [FakeDocument()]

    monkeypatch.setattr(
        "shared.tools._search_service",
        FakeSearchService(),
    )

    payload = json.loads(
        search_documents(
            "Quem fornece o PARAFUSO-M10?"
        )
    )

    assert payload["count"] == 1

    assert (
        payload["query"]
        == "Quem fornece o PARAFUSO-M10?"
    )

    assert len(payload["documents"]) == 1

    document = payload["documents"][0]

    assert (
        document["doc_type"]
        == "structured_reference"
    )

    assert document["score"] == 2.45


def test_search_documents_returns_empty_payload(
    monkeypatch,
):

    class FakeSearchService:
        def search_documents(
            self,
            query: str,
        ):
            return []

    monkeypatch.setattr(
        "shared.tools._search_service",
        FakeSearchService(),
    )

    payload = json.loads(
        search_documents("teste")
    )

    assert payload["count"] == 0
    assert payload["documents"] == []

    assert "message" in payload