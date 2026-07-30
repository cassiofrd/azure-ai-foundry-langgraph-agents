from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from shared.execution_context import ExecutionContext, LLMCallInfo
from shared.foundry_client import ResponsesClient
from shared.settings import AppSettings


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: str
    call_id: str


@dataclass(frozen=True)
class FoundryResponse:
    response_id: str
    output_text: str
    tool_calls: tuple[ToolCall, ...]


class FoundryService:
    def __init__(
        self,
        *,
        settings: AppSettings,
        client_factory: Callable[[], ResponsesClient],
    ) -> None:
        self._settings = settings
        self._client_factory = client_factory

    def ask(
        self,
        *,
        user_input: str,
        tools: list[dict[str, Any]] | None = None,
        previous_response_id: str | None = None,
        execution_context: ExecutionContext | None = None,
        instructions: str | None = None,
        max_output_tokens: int | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> FoundryResponse:
        request: dict[str, Any] = {
            "model": self._settings.foundry_model_deployment,
            "instructions": instructions or self._settings.system_prompt,
            "input": user_input,
            "max_output_tokens": (
                max_output_tokens
                or self._settings.model_max_output_tokens
            ),
        }

        if tools:
            request["tools"] = tools

        if tool_choice is not None:
            if not tools:
                raise ValueError(
                    "tool_choice can only be used when tools are provided."
                )
            request["tool_choice"] = tool_choice

        if previous_response_id:
            request["previous_response_id"] = previous_response_id

        return self._create_response(
            operation="initial_response",
            request=request,
            requested_tool_count=len(tools or []),
            execution_context=execution_context,
        )

    def continue_after_tools(
        self,
        *,
        previous_response_id: str,
        tool_outputs: Iterable[dict[str, str]],
        execution_context: ExecutionContext | None = None,
        instructions: str | None = None,
    ) -> FoundryResponse:
        request: dict[str, Any] = {
            "model": self._settings.foundry_model_deployment,
            "instructions": instructions or self._settings.system_prompt,
            "previous_response_id": previous_response_id,
            "input": list(tool_outputs),
            "max_output_tokens": self._settings.model_max_output_tokens,
        }

        return self._create_response(
            operation="continue_after_tools",
            request=request,
            requested_tool_count=0,
            execution_context=execution_context,
        )

    def _create_response(
        self,
        *,
        operation: str,
        request: dict[str, Any],
        requested_tool_count: int,
        execution_context: ExecutionContext | None,
    ) -> FoundryResponse:
        client = self._client_factory()
        started = time.perf_counter()

        try:
            raw_response = client.responses.create(**request)
            normalized = self._normalize_response(raw_response)
        except Exception as exc:
            duration_ms = (time.perf_counter() - started) * 1000
            if execution_context is not None:
                execution_context.llm_calls.append(
                    LLMCallInfo(
                        operation=operation,
                        model=self._settings.foundry_model_deployment,
                        duration_ms=duration_ms,
                        requested_tool_count=requested_tool_count,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
                execution_context.record_error(
                    component="foundry_service",
                    operation=operation,
                    error=exc,
                )
            raise

        duration_ms = (time.perf_counter() - started) * 1000
        input_tokens, output_tokens, total_tokens = self._read_usage(
            raw_response
        )

        if execution_context is not None:
            execution_context.llm_calls.append(
                LLMCallInfo(
                    operation=operation,
                    model=self._settings.foundry_model_deployment,
                    duration_ms=duration_ms,
                    response_id=normalized.response_id,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    requested_tool_count=requested_tool_count,
                    returned_tool_count=len(normalized.tool_calls),
                )
            )

        return normalized

    @staticmethod
    def _read_usage(
        response: Any,
    ) -> tuple[int | None, int | None, int | None]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None, None, None

        def read(name: str) -> int | None:
            value = (
                usage.get(name)
                if isinstance(usage, dict)
                else getattr(usage, name, None)
            )
            return value if isinstance(value, int) else None

        return (
            read("input_tokens"),
            read("output_tokens"),
            read("total_tokens"),
        )

    @staticmethod
    def _normalize_response(response: Any) -> FoundryResponse:
        tool_calls = tuple(
            ToolCall(
                name=item.name,
                arguments=item.arguments or "{}",
                call_id=item.call_id,
            )
            for item in getattr(response, "output", [])
            if getattr(item, "type", None) == "function_call"
        )

        return FoundryResponse(
            response_id=str(response.id),
            output_text=(getattr(response, "output_text", "") or "").strip(),
            tool_calls=tool_calls,
        )
