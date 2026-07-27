from __future__ import annotations

from typing import TypedDict


class ToolExecutionContext(TypedDict, total=False):
    tool_name: str
    latency_ms: float


class RetrievalContext(TypedDict, total=False):
    query: str
    document_count: int
    retrieved_documents: list[str]


class ExecutionContext(TypedDict, total=False):
    model: str
    total_latency_ms: float
    tool_executions: list[ToolExecutionContext]
    retrieval: RetrievalContext | None