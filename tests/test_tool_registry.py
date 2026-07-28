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

    assert payload["requested_entity_codes"] == [
        "PARAFUSO-M10"
    ]

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


def test_search_documents_keeps_exact_entity_match(
    monkeypatch,
):
    class ExactDocument:
        title = "Bolt M10 inventory policy"
        content = "Target inventory is 900 units."
        agent = "inventory"
        doc_type = "inventory_policy"
        entity_type = "material"
        entity_id = "M10"
        source = "manual"
        score = 0.033

    class SimilarDocument:
        title = "Bolt B200 inventory policy"
        content = "Target inventory is 700 units."
        agent = "inventory"
        doc_type = "inventory_policy"
        entity_type = "material"
        entity_id = "B200"
        source = "manual"
        score = 0.032

    class FakeSearchService:
        def search_documents(
            self,
            query: str,
        ):
            return [
                ExactDocument(),
                SimilarDocument(),
            ]

    monkeypatch.setattr(
        "shared.tools._search_service",
        FakeSearchService(),
    )

    payload = json.loads(
        search_documents(
            "Qual é a política do parafuso M10?"
        )
    )

    assert payload["count"] == 1
    assert payload["requested_entity_codes"] == ["M10"]
    assert payload["documents"][0]["entity_id"] == "M10"


def test_search_documents_matches_entity_code_case_insensitively(
    monkeypatch,
):
    class FakeDocument:
        title = "Bolt M10 inventory policy"
        content = "Target inventory is 900 units."
        agent = "inventory"
        doc_type = "inventory_policy"
        entity_type = "material"
        entity_id = "m10"
        source = "manual"
        score = 0.033

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
            "Qual é a política do parafuso M10?"
        )
    )

    assert payload["count"] == 1
    assert payload["documents"][0]["entity_id"] == "m10"


def test_search_documents_rejects_similar_entity(
    monkeypatch,
):
    class FakeDocument:
        title = "Bearing A100 inventory policy"
        content = "Target inventory is 500 units."
        agent = "inventory"
        doc_type = "inventory_policy"
        entity_type = "material"
        entity_id = "A100"
        source = "manual"
        score = 0.033

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
            "Qual é a política do rolamento Z500?"
        )
    )

    assert payload["count"] == 0
    assert payload["documents"] == []
    assert payload["requested_entity_codes"] == ["Z500"]
    assert "exact" in payload["message"].lower()
    assert "substitutes" in payload["message"].lower()
