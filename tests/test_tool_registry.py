import json

from shared.tools import TOOL_REGISTRY, TOOLS, search_documents


def _result(**overrides):
    item = {
        "id": "doc-1",
        "chunk_id": "chunk-1",
        "source_path": "inventory/inventory_policy_M10.pdf",
        "domain": "inventory",
        "document_type": "inventory_policy",
        "content": "Target inventory is 900 units.",
        "entity_ids": ["M10"],
        "aliases": ["BOLT-M10"],
        "location_ids": ["PLANT-BH"],
        "supplier_ids": [],
        "is_global": False,
        "@search.score": 2.45,
    }
    item.update(overrides)
    return item


def _smart_payload(query, *, domain=None, results=None):
    return {
        "query": query,
        "entity_id": "M10" if "M10" in query.upper() else None,
        "location_id": "PLANT-BH" if "PLANT-BH" in query.upper() else None,
        "domain": domain,
        "intent": "inventory_policy" if "policy" in query.lower() or "política" in query.lower() else None,
        "preferred_document_types": ["inventory_policy"],
        "filter": None,
        "results": list(results or []),
    }


def test_search_tool_registered():
    assert "search_documents" in TOOL_REGISTRY


def test_search_tool_declared():
    assert "search_documents" in {tool["name"] for tool in TOOLS}


def test_search_tool_has_query_parameter():
    tool = next(tool for tool in TOOLS if tool["name"] == "search_documents")
    properties = tool["parameters"]["properties"]
    assert properties["query"]["type"] == "string"
    assert tool["parameters"]["required"] == ["query"]


def test_search_tool_description_mentions_retrieval_v2():
    tool = next(tool for tool in TOOLS if tool["name"] == "search_documents")
    description = tool["description"].lower()
    assert "enterprise" in description
    assert "knowledge" in description
    assert "retrieval v2" in description


def test_search_documents_returns_structured_payload(monkeypatch):
    class FakeSearchService:
        def smart_hybrid_search(self, query, **kwargs):
            return _smart_payload(query, domain=kwargs.get("domain"), results=[_result()])

    monkeypatch.setattr("shared.tools._search_service", FakeSearchService())
    payload = json.loads(search_documents("What is the inventory policy for M10?"))
    assert payload["count"] == 1
    assert payload["query"] == "What is the inventory policy for M10?"
    assert payload["entity_id"] == "M10"
    assert payload["documents"][0]["document_type"] == "inventory_policy"
    assert payload["documents"][0]["score"] == 2.45


def test_search_documents_returns_empty_payload(monkeypatch):
    class FakeSearchService:
        def smart_hybrid_search(self, query, **kwargs):
            return _smart_payload(query, domain=kwargs.get("domain"), results=[])

    monkeypatch.setattr("shared.tools._search_service", FakeSearchService())
    payload = json.loads(search_documents("teste"))
    assert payload["count"] == 0
    assert payload["documents"] == []
    assert "message" in payload


def test_specialist_search_tools_are_registered_and_declared():
    names = {tool["name"] for tool in TOOLS}
    for name in (
        "search_inventory_documents",
        "search_supplier_documents",
        "search_logistics_documents",
        "search_procedure_documents",
    ):
        assert name in TOOL_REGISTRY
        assert name in names


def test_inventory_search_filters_domain(monkeypatch):
    calls = []
    class FakeSearchService:
        def smart_hybrid_search(self, query, **kwargs):
            calls.append((query, kwargs))
            return _smart_payload(query, domain=kwargs.get("domain"), results=[])
    monkeypatch.setattr("shared.tools._search_service", FakeSearchService())
    from shared.tools import search_inventory_documents
    search_inventory_documents("M10")
    assert calls[0][0] == "M10"
    assert calls[0][1]["domain"] == "inventory"


def test_logistics_search_filters_domain(monkeypatch):
    calls = []
    class FakeSearchService:
        def smart_hybrid_search(self, query, **kwargs):
            calls.append((query, kwargs))
            return _smart_payload(query, domain=kwargs.get("domain"), results=[])
    monkeypatch.setattr("shared.tools._search_service", FakeSearchService())
    from shared.tools import search_logistics_documents
    search_logistics_documents("transporte urgente")
    assert calls[0][1]["domain"] == "logistics"


def test_procedure_search_filters_domain(monkeypatch):
    calls = []
    class FakeSearchService:
        def smart_hybrid_search(self, query, **kwargs):
            calls.append((query, kwargs))
            return _smart_payload(query, domain=kwargs.get("domain"), results=[])
    monkeypatch.setattr("shared.tools._search_service", FakeSearchService())
    from shared.tools import search_procedure_documents
    search_procedure_documents("standard replenishment procedure for M10")
    assert calls[0][1]["domain"] == "procedures"


def test_supplier_search_uses_supplier_domain_without_location_filter(monkeypatch):
    calls = []
    class FakeSearchService:
        def hybrid_search(self, query, **kwargs):
            calls.append((query, kwargs))
            return []
        def _rerank_by_intent(self, results, *, preferred_types):
            return results
    monkeypatch.setattr("shared.tools._search_service", FakeSearchService())
    from shared.tools import search_supplier_documents
    payload = json.loads(search_supplier_documents("approved supplier for M10 at PLANT-BH"))
    assert calls[0][0] == "approved supplier for M10 at PLANT-BH"
    assert "domain eq 'suppliers'" in calls[0][1]["filter"]
    assert "M10" in calls[0][1]["filter"]
    assert "PLANT-BH" not in calls[0][1]["filter"]
    assert payload["domain"] == "suppliers"
    assert payload["location_id"] is None


def test_logistics_tool_is_registered_and_declared():
    names = {tool["name"] for tool in TOOLS}
    assert "search_logistics_documents" in TOOL_REGISTRY
    assert "search_logistics_documents" in names
