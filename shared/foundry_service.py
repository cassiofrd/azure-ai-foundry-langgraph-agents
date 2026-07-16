from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

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
    ) -> FoundryResponse:
        client = self._client_factory()

        request: dict[str, Any] = {
            "model": self._settings.foundry_model_deployment,
            "instructions": self._settings.system_prompt,
            "input": user_input,
            "max_output_tokens": self._settings.model_max_output_tokens,
        }

        if tools:
            request["tools"] = tools

        if previous_response_id:
            request["previous_response_id"] = previous_response_id

        response = client.responses.create(**request)
        return self._normalize_response(response)

    def continue_after_tools(
        self,
        *,
        previous_response_id: str,
        tool_outputs: Iterable[dict[str, str]],
    ) -> FoundryResponse:
        client = self._client_factory()

        response = client.responses.create(
            model=self._settings.foundry_model_deployment,
            previous_response_id=previous_response_id,
            input=list(tool_outputs),
            max_output_tokens=self._settings.model_max_output_tokens,
        )

        return self._normalize_response(response)

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
