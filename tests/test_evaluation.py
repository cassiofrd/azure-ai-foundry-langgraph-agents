from __future__ import annotations

from evaluation.models import EvaluationCase
from evaluation.runner import evaluate_case, run_evaluation


def response_payload():
    return {
        "route": "inventory_supplier",
        "participants": ["inventory", "supplier"],
        "answer": "Contoso atende o M10 em 7 dias.",
        "evidence": [
            {"title": "Supplier", "source": "master", "entity_id": "M10"}
        ],
        "execution": {
            "duration_ms": 2500,
            "metrics": {"total_tokens": 400},
            "tools": [
                {"tool_name": "search_inventory_documents"},
                {"tool_name": "search_supplier_documents"},
            ],
        },
    }


def test_evaluate_case_passes_all_deterministic_checks():
    case = EvaluationCase(
        case_id="case-1",
        message="question",
        expected_route="inventory_supplier",
        expected_participants=("inventory", "supplier"),
        expected_tools=("search_inventory_documents", "search_supplier_documents"),
        expected_entity_ids=("M10",),
        required_answer_terms=("Contoso", "7 dias"),
        forbidden_answer_terms=("Fabrikam",),
        max_duration_ms=3000,
    )
    result = evaluate_case(case, response_payload())
    assert result.passed is True
    assert all(check.passed for check in result.checks)


def test_evaluate_case_reports_route_failure():
    case = EvaluationCase(
        case_id="case-1",
        message="question",
        expected_route="logistics",
    )
    result = evaluate_case(case, response_payload())
    assert result.passed is False
    route_check = next(check for check in result.checks if check.name == "route")
    assert route_check.passed is False


def test_run_evaluation_aggregates_results():
    class FakeClient:
        def ask(self, *, session_id: str, message: str):
            assert session_id.startswith("test-")
            return response_payload()

    cases = [
        EvaluationCase(
            case_id="one",
            message="question",
            expected_route="inventory_supplier",
            expected_participants=("inventory", "supplier"),
        )
    ]
    report = run_evaluation(client=FakeClient(), cases=cases, session_prefix="test")
    assert report.total_cases == 1
    assert report.passed_cases == 1
    assert report.pass_rate == 1.0
    assert report.average_total_tokens == 400
