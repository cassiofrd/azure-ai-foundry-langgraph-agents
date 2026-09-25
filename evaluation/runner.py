from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol

from evaluation.models import CaseResult, CheckResult, EvaluationCase, EvaluationReport


class CopilotClientProtocol(Protocol):
    def ask(self, *, session_id: str, message: str) -> dict[str, Any]:
        ...


def load_cases(path: str | Path) -> list[EvaluationCase]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Evaluation dataset must contain a JSON list.")
    return [EvaluationCase.from_dict(item) for item in payload]


def evaluate_case(case: EvaluationCase, response: dict[str, Any]) -> CaseResult:
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
    tool_names_list = [
        str(item.get("tool_name", ""))
        for item in tools
        if isinstance(item, dict)
    ]
    tool_names = set(tool_names_list)
    duration_ms = _optional_float(execution.get("duration_ms"))
    total_tokens = int(metrics.get("total_tokens", 0) or 0)
    tool_call_count = len(tool_names_list)

    checks = [
        CheckResult("route", route == case.expected_route, case.expected_route, route),
        CheckResult(
            "participants",
            set(participants) == set(case.expected_participants),
            sorted(case.expected_participants),
            sorted(participants),
        ),
        CheckResult(
            "required_tools",
            set(case.expected_tools).issubset(tool_names),
            sorted(case.expected_tools),
            sorted(tool_names),
        ),
        CheckResult(
            "forbidden_tools",
            set(case.forbidden_tools).isdisjoint(tool_names),
            sorted(case.forbidden_tools),
            sorted(tool_names),
        ),
        CheckResult(
            "evidence_entity_ids",
            {item.casefold() for item in case.expected_entity_ids}.issubset(entity_ids),
            sorted(case.expected_entity_ids),
            sorted(entity_ids),
        ),
        CheckResult(
            "required_answer_terms",
            all(term.casefold() in answer_folded for term in case.required_answer_terms),
            list(case.required_answer_terms),
            answer,
        ),
        CheckResult(
            "forbidden_answer_terms",
            all(term.casefold() not in answer_folded for term in case.forbidden_answer_terms),
            list(case.forbidden_answer_terms),
            answer,
        ),
        CheckResult(
            "required_answer_regex",
            all(re.search(pattern, answer) is not None for pattern in case.required_answer_regex),
            list(case.required_answer_regex),
            answer,
        ),
        CheckResult(
            "forbidden_answer_regex",
            all(re.search(pattern, answer) is None for pattern in case.forbidden_answer_regex),
            list(case.forbidden_answer_regex),
            answer,
        ),
    ]

    if case.max_tool_calls is not None:
        checks.append(
            CheckResult(
                "max_tool_calls",
                tool_call_count <= case.max_tool_calls,
                case.max_tool_calls,
                tool_call_count,
            )
        )

    if case.max_duration_ms is not None:
        checks.append(
            CheckResult(
                "max_duration_ms",
                duration_ms is not None and duration_ms <= case.max_duration_ms,
                case.max_duration_ms,
                duration_ms,
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
        tool_calls=tool_call_count,
    )


def run_evaluation(
    *,
    client: CopilotClientProtocol,
    cases: list[EvaluationCase],
    session_prefix: str = "evaluation-v2",
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
                    tool_calls=0,
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
