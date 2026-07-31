from shared.dashboard import execution_summary


def test_execution_summary_extracts_metrics():
    response = {
        "route": "inventory_supplier",
        "specialist": "supervisor",
        "participants": ["inventory", "supplier"],
        "evidence": [{"entity_id": "M10"}],
        "execution": {
            "duration_ms": 5000,
            "metrics": {
                "llm_call_count": 6,
                "tool_call_count": 2,
                "total_tokens": 1800,
                "error_count": 0,
            },
            "tools": [
                {"tool_name": "search_inventory_documents"},
                {"tool_name": "search_supplier_documents"},
            ],
        },
    }
    summary = execution_summary(response)
    assert summary["route"] == "inventory_supplier"
    assert summary["participants"] == ["inventory", "supplier"]
    assert summary["llm_calls"] == 6
    assert summary["tool_calls"] == 2
    assert summary["total_tokens"] == 1800
    assert summary["evidence_count"] == 1
