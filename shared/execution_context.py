from __future__ import annotations

from dataclasses import dataclass, field
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
    errors: list[ExecutionErrorInfo] = field(default_factory=list)

    @property
    def is_finished(self) -> bool:
        return self.finished_at is not None

    @property
    def total_input_tokens(self) -> int:
        return sum(call.input_tokens or 0 for call in self.llm_calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(call.output_tokens or 0 for call in self.llm_calls)

    @property
    def total_tokens(self) -> int:
        explicit_totals = [
            call.total_tokens
            for call in self.llm_calls
            if call.total_tokens is not None
        ]
        if len(explicit_totals) == len(self.llm_calls):
            return sum(explicit_totals)
        return self.total_input_tokens + self.total_output_tokens

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
