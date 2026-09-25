from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from shared.response_parser import extract_evidence


class GraphProtocol(Protocol):
    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        ...


class ConversationStoreProtocol(Protocol):
    backend_name: str
    is_persistent: bool
    startup_warning: str | None

    def clear(self, session_id: str) -> None:
        ...


@dataclass(frozen=True)
class ApiRuntime:
    settings: Any
    graph: GraphProtocol
    conversation_store: ConversationStoreProtocol


class CopilotRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    session_id: str | None = Field(default=None, max_length=128)


class EvidenceResponse(BaseModel):
    title: str
    source: str
    entity_id: str


class CopilotResponse(BaseModel):
    session_id: str
    route: str
    specialist: str
    participants: list[str]
    specialist_queries: dict[str, str]
    answer: str
    evidence: list[EvidenceResponse]
    execution: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    app_name: str
    app_version: str
    environment: str
    conversation_store: str
    conversation_persistence: bool
    startup_warning: str | None = None


class SessionDeleteResponse(BaseModel):
    session_id: str
    cleared: bool


def create_app(
    *,
    runtime: ApiRuntime | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "runtime", None) is None:
            app.state.runtime = _build_default_runtime()
        yield

    app = FastAPI(
        title="Azure AI Foundry LangGraph Agents API",
        version="1.0.0",
        description=(
            "REST API for the multi-agent supply-chain copilot."
        ),
        lifespan=lifespan,
    )

    if runtime is not None:
        app.state.runtime = runtime

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["operations"],
    )
    def health() -> HealthResponse:
        current = _runtime(app)
        settings = current.settings
        store = current.conversation_store
        return HealthResponse(
            status="ok",
            app_name=settings.app_name,
            app_version=settings.app_version,
            environment=settings.app_environment,
            conversation_store=store.backend_name,
            conversation_persistence=store.is_persistent,
            startup_warning=store.startup_warning,
        )

    @app.post(
        "/copilot",
        response_model=CopilotResponse,
        tags=["copilot"],
    )
    def copilot(request: CopilotRequest) -> CopilotResponse:
        current = _runtime(app)
        message = request.message.strip()
        if not message:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="message cannot be empty.",
            )
        session_id = (
            request.session_id.strip()
            if request.session_id is not None
            else current.settings.default_session_id
        )
        if not session_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="session_id cannot be empty.",
            )

        try:
            result = current.graph.invoke(
                {
                    "user_input": message,
                    "intent": "general",
                    "agent": "general",
                    "answer": "",
                    "conversation_response_id": None,
                    "session_id": session_id,
                }
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "The copilot could not complete the request. "
                    f"{type(exc).__name__}: {exc}"
                ),
            ) from exc

        answer = str(result.get("answer", "")).strip()
        execution = result.get("execution")
        execution_payload = (
            execution.to_dict()
            if execution is not None and hasattr(execution, "to_dict")
            else {}
        )
        specialist_outputs = result.get("specialist_outputs") or {}
        participants = list(specialist_outputs.keys())
        structured_evidence = (
            list(getattr(execution, "evidence", []) or [])
            if execution is not None
            else []
        )
        parsed_evidence = list(extract_evidence(answer)) if not structured_evidence else []
        evidence = [
            EvidenceResponse(
                title=item.title,
                source=item.source,
                entity_id=item.entity_id,
            )
            for item in (structured_evidence or parsed_evidence)
        ]

        return CopilotResponse(
            session_id=str(result.get("session_id", session_id)),
            route=str(result.get("intent", "general")),
            specialist=str(result.get("agent", "general")),
            participants=participants,
            specialist_queries=dict(
                result.get("specialist_queries") or {}
            ),
            answer=answer,
            evidence=evidence,
            execution=execution_payload,
        )

    @app.delete(
        "/sessions/{session_id}",
        response_model=SessionDeleteResponse,
        tags=["sessions"],
    )
    def clear_session(session_id: str) -> SessionDeleteResponse:
        current = _runtime(app)
        normalized = session_id.strip()
        if not normalized:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="session_id cannot be empty.",
            )
        try:
            current.conversation_store.clear(normalized)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        return SessionDeleteResponse(
            session_id=normalized,
            cleared=True,
        )

    return app


def _runtime(app: FastAPI) -> ApiRuntime:
    current = getattr(app.state, "runtime", None)
    if current is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API runtime is not initialized.",
        )
    return current


def _build_default_runtime() -> ApiRuntime:
    # Lazy imports keep API unit tests independent from Azure credentials.
    from graphs.supervisor_graph import build_supervisor_graph
    from shared.conversation_store import build_conversation_store
    from shared.foundry_client import get_openai_client
    from shared.settings import load_settings

    settings = load_settings()
    conversation_store = build_conversation_store(settings)
    graph = build_supervisor_graph(
        settings=settings,
        client_factory=lambda: get_openai_client(settings),
        conversation_store=conversation_store,
    )
    return ApiRuntime(
        settings=settings,
        graph=graph,
        conversation_store=conversation_store,
    )


app = create_app()
