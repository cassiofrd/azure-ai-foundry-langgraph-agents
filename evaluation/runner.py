from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from evaluation.models import (
    CaseResult,
    CheckResult,
    EvaluationCase,
    EvaluationReport,
)


class CopilotClientProtocol(Protocol):
    def ask(self, *, session_id: str, message: str) -> dict[str, Any]:
        ...


def load_cases(path: str | Path) -> list[EvaluationCase]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Evaluation dataset must contain a JSON list.")
    return [EvaluationCase.from_dict(item) for item in payload]


def evaluate_case(
    case: EvaluationCase,
    response: dict[str, Any],
) -> CaseResult:
    route = str(response.get("route", ""))
    participants = tuple(str(item) for item in response.get("participants", []))
    answer = str(response.get("answer", ""))
    answer_folded = answer.casefold()
    evidence = response.get("evidence", []) or []
    entity_ids = {
        str(item.get("entity_id", "")).casefold()
        for item in evidence
        if isinstance(item, dict)
    }
    execution = response.get("execution", {}) or {}
    metrics = execution.get("metrics", {}) or {}
    tools = execution.get("tools", []) or []
    tool_names = {
        str(item.get("tool_name", ""))
        for item in tools
        if isinstance(item, dict)
    }
    duration_ms = _optional_float(execution.get("duration_ms"))
    total_tokens = int(metrics.get("total_tokens", 0) or 0)

    checks = [
        CheckResult(
            name="route",
            passed=route == case.expected_route,
            expected=case.expected_route,
            actual=route,
        ),
        CheckResult(
            name="participants",
            passed=set(participants) == set(case.expected_participants),
            expected=sorted(case.expected_participants),
            actual=sorted(participants),
        ),
        CheckResult(
            name="tools",
            passed=set(case.expected_tools).issubset(tool_names),
            expected=sorted(case.expected_tools),
            actual=sorted(tool_names),
        ),
        CheckResult(
            name="evidence_entity_ids",
            passed={item.casefold() for item in case.expected_entity_ids}.issubset(entity_ids),
            expected=sorted(case.expected_entity_ids),
            actual=sorted(entity_ids),
        ),
        CheckResult(
            name="required_answer_terms",
            passed=all(term.casefold() in answer_folded for term in case.required_answer_terms),
            expected=list(case.required_answer_terms),
            actual=answer,
        ),
        CheckResult(
            name="forbidden_answer_terms",
            passed=all(term.casefold() not in answer_folded for term in case.forbidden_answer_terms),
            expected=list(case.forbidden_answer_terms),
            actual=answer,
        ),
    ]

    if case.max_duration_ms is not None:
        checks.append(
            CheckResult(
                name="max_duration_ms",
                passed=duration_ms is not None and duration_ms <= case.max_duration_ms,
                expected=case.max_duration_ms,
                actual=duration_ms,
            )
        )

    return CaseResult(
        case_id=case.case_id,
        passed=all(check.passed for check in checks),
        checks=tuple(checks),
        route=route,
        participants=participants,
        duration_ms=duration_ms,
        total_tokens=total_tokens,
    )


def run_evaluation(
    *,
    client: CopilotClientProtocol,
    cases: list[EvaluationCase],
    session_prefix: str = "evaluation",
) -> EvaluationReport:
    report = EvaluationReport()
    for index, case in enumerate(cases, start=1):
        session_id = f"{session_prefix}-{index}-{case.case_id}"
        try:
            response = client.ask(session_id=session_id, message=case.message)
            report.results.append(evaluate_case(case, response))
        except Exception as exc:
            report.results.append(
                CaseResult(
                    case_id=case.case_id,
                    passed=False,
                    checks=(),
                    route="",
                    participants=(),
                    duration_ms=None,
                    total_tokens=0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return report


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
