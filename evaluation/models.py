from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    message: str
    expected_route: str
    expected_participants: tuple[str, ...] = ()
    expected_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    expected_entity_ids: tuple[str, ...] = ()
    required_answer_terms: tuple[str, ...] = ()
    forbidden_answer_terms: tuple[str, ...] = ()
    required_answer_regex: tuple[str, ...] = ()
    forbidden_answer_regex: tuple[str, ...] = ()
    max_tool_calls: int | None = None
    max_duration_ms: float | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvaluationCase":
        return cls(
            case_id=str(payload["case_id"]),
            message=str(payload["message"]),
            expected_route=str(payload["expected_route"]),
            expected_participants=tuple(payload.get("expected_participants", [])),
            expected_tools=tuple(payload.get("expected_tools", [])),
            forbidden_tools=tuple(payload.get("forbidden_tools", [])),
            expected_entity_ids=tuple(payload.get("expected_entity_ids", [])),
            required_answer_terms=tuple(payload.get("required_answer_terms", [])),
            forbidden_answer_terms=tuple(payload.get("forbidden_answer_terms", [])),
            required_answer_regex=tuple(payload.get("required_answer_regex", [])),
            forbidden_answer_regex=tuple(payload.get("forbidden_answer_regex", [])),
            max_tool_calls=(
                int(payload["max_tool_calls"])
                if payload.get("max_tool_calls") is not None
                else None
            ),
            max_duration_ms=(
                float(payload["max_duration_ms"])
                if payload.get("max_duration_ms") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    expected: Any
    actual: Any


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    passed: bool
    checks: tuple[CheckResult, ...]
    route: str
    participants: tuple[str, ...]
    duration_ms: float | None
    total_tokens: int
    tool_calls: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "route": self.route,
            "participants": list(self.participants),
            "duration_ms": self.duration_ms,
            "total_tokens": self.total_tokens,
            "tool_calls": self.tool_calls,
            "error": self.error,
            "checks": [
                {
                    "name": check.name,
                    "passed": check.passed,
                    "expected": check.expected,
                    "actual": check.actual,
                }
                for check in self.checks
            ],
        }


@dataclass
class EvaluationReport:
    results: list[CaseResult] = field(default_factory=list)

    @property
    def total_cases(self) -> int:
        return len(self.results)

    @property
    def passed_cases(self) -> int:
        return sum(result.passed for result in self.results)

    @property
    def pass_rate(self) -> float:
        return self.passed_cases / self.total_cases if self.results else 0.0

    @property
    def average_duration_ms(self) -> float:
        values = [r.duration_ms for r in self.results if r.duration_ms is not None]
        return sum(values) / len(values) if values else 0.0

    @property
    def average_total_tokens(self) -> float:
        return (
            sum(result.total_tokens for result in self.results) / len(self.results)
            if self.results else 0.0
        )

    @property
    def average_tool_calls(self) -> float:
        return (
            sum(result.tool_calls for result in self.results) / len(self.results)
            if self.results else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "total_cases": self.total_cases,
                "passed_cases": self.passed_cases,
                "failed_cases": self.total_cases - self.passed_cases,
                "pass_rate": self.pass_rate,
                "average_duration_ms": self.average_duration_ms,
                "average_total_tokens": self.average_total_tokens,
                "average_tool_calls": self.average_tool_calls,
            },
            "results": [result.to_dict() for result in self.results],
        }
