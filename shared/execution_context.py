from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class LLMCallInfo:
    operation: str
    model: str
    duration_ms: float
    response_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    requested_tool_count: int = 0
    returned_tool_count: int = 0
    error_type: str | None = None
    error_message: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error_type is None


@dataclass(frozen=True)
class ToolExecutionInfo:
    tool_name: str
    arguments: dict[str, Any]
    duration_ms: float
    call_id: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error_type is None


@dataclass(frozen=True)
class EvidenceInfo:
    title: str
    source: str
    entity_id: str


@dataclass(frozen=True)
class ExecutionErrorInfo:
    component: str
    operation: str
    error_type: str
    error_message: str


@dataclass
class ExecutionContext:
    execution_id: str = field(default_factory=lambda: str(uuid4()))
    started_at: datetime = field(default_factory=_utc_now)
    finished_at: datetime | None = None
    duration_ms: float | None = None
    llm_calls: list[LLMCallInfo] = field(default_factory=list)
    tools: list[ToolExecutionInfo] = field(default_factory=list)
    evidence: list[EvidenceInfo] = field(default_factory=list)
    errors: list[ExecutionErrorInfo] = field(default_factory=list)

    @property
    def is_finished(self) -> bool:
        return self.finished_at is not None

    @property
    def status(self) -> str:
        if not self.is_finished:
            return "running"

        functional_errors = [
            error
            for error in self.errors
            if error.component != "telemetry"
        ]
        return "failed" if functional_errors else "success"

    @property
    def total_input_tokens(self) -> int:
        return sum(call.input_tokens or 0 for call in self.llm_calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(call.output_tokens or 0 for call in self.llm_calls)

    @property
    def total_tokens(self) -> int:
        return sum(
            call.total_tokens
            if call.total_tokens is not None
            else (call.input_tokens or 0) + (call.output_tokens or 0)
            for call in self.llm_calls
        )

    @property
    def total_llm_duration_ms(self) -> float:
        return sum(call.duration_ms for call in self.llm_calls)

    @property
    def total_tool_duration_ms(self) -> float:
        return sum(tool.duration_ms for tool in self.tools)

    @property
    def successful_llm_calls(self) -> int:
        return sum(call.succeeded for call in self.llm_calls)

    @property
    def successful_tool_calls(self) -> int:
        return sum(tool.succeeded for tool in self.tools)

    def record_error(
        self,
        *,
        component: str,
        operation: str,
        error: BaseException,
    ) -> None:
        self.errors.append(
            ExecutionErrorInfo(
                component=component,
                operation=operation,
                error_type=type(error).__name__,
                error_message=str(error),
            )
        )

    def finish(self) -> None:
        if self.is_finished:
            return

        self.finished_at = _utc_now()
        self.duration_ms = (
            self.finished_at - self.started_at
        ).total_seconds() * 1000

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the execution."""
        return {
            "execution_id": self.execution_id,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "finished_at": (
                self.finished_at.isoformat()
                if self.finished_at is not None
                else None
            ),
            "duration_ms": self.duration_ms,
            "metrics": {
                "llm_call_count": len(self.llm_calls),
                "successful_llm_calls": self.successful_llm_calls,
                "tool_call_count": len(self.tools),
                "successful_tool_calls": self.successful_tool_calls,
                "error_count": len(self.errors),
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens": self.total_tokens,
                "total_llm_duration_ms": self.total_llm_duration_ms,
                "total_tool_duration_ms": self.total_tool_duration_ms,
            },
            "llm_calls": [asdict(call) for call in self.llm_calls],
            "tools": [asdict(tool) for tool in self.tools],
            "evidence": [asdict(item) for item in self.evidence],
            "errors": [asdict(error) for error in self.errors],
        }
