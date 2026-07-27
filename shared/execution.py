from __future__ import annotations

"""Backward-compatible execution types.

ExecutionContext now has a single canonical implementation in
``shared.execution_context``. Importing it from this module remains supported
while older retrieval typing is retained until the retrieval telemetry sprint.
"""

from typing import TypedDict

from shared.execution_context import (
    ExecutionContext,
    ExecutionErrorInfo,
    LLMCallInfo,
    ToolExecutionInfo,
)


class RetrievalContext(TypedDict, total=False):
    query: str
    document_count: int
    retrieved_documents: list[str]


__all__ = [
    "ExecutionContext",
    "ExecutionErrorInfo",
    "LLMCallInfo",
    "RetrievalContext",
    "ToolExecutionInfo",
]
