from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ConversationMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ConversationSession:
    session_id: str
    messages: tuple[ConversationMessage, ...] = ()
    last_entity: str | None = None
    previous_response_id: str | None = None


class ConversationStore(Protocol):
    backend_name: str
    is_persistent: bool
    startup_warning: str | None

    def load(self, session_id: str) -> ConversationSession:
        ...

    def append_turn(
        self,
        *,
        session_id: str,
        user_input: str,
        assistant_output: str,
        last_entity: str | None,
        previous_response_id: str | None,
    ) -> ConversationSession:
        ...

    def clear(self, session_id: str) -> None:
        ...

    def ttl(self, session_id: str) -> int | None:
        ...


class InMemoryConversationStore:
    backend_name = "memory"
    is_persistent = False

    def __init__(
        self,
        *,
        history_limit: int = 12,
        startup_warning: str | None = None,
    ) -> None:
        if history_limit < 1:
            raise ValueError("history_limit must be greater than zero.")
        self._history_limit = history_limit
        self.startup_warning = startup_warning
        self._sessions: dict[str, ConversationSession] = {}

    def load(self, session_id: str) -> ConversationSession:
        normalized = _normalize_session_id(session_id)
        return self._sessions.get(
            normalized,
            ConversationSession(session_id=normalized),
        )

    def append_turn(
        self,
        *,
        session_id: str,
        user_input: str,
        assistant_output: str,
        last_entity: str | None,
        previous_response_id: str | None,
    ) -> ConversationSession:
        current = self.load(session_id)
        messages = _trim_messages(
            (*current.messages,
             ConversationMessage(role="user", content=user_input.strip()),
             ConversationMessage(
                 role="assistant",
                 content=assistant_output.strip(),
             )),
            self._history_limit,
        )
        updated = ConversationSession(
            session_id=current.session_id,
            messages=messages,
            last_entity=(last_entity or current.last_entity),
            previous_response_id=previous_response_id,
        )
        self._sessions[current.session_id] = updated
        return updated

    def clear(self, session_id: str) -> None:
        self._sessions.pop(_normalize_session_id(session_id), None)

    def ttl(self, session_id: str) -> int | None:
        return None


class RedisConversationStore:
    backend_name = "redis"
    is_persistent = True
    startup_warning: str | None = None

    def __init__(
        self,
        *,
        redis_client: Any,
        ttl_seconds: int = 86400,
        history_limit: int = 12,
        key_prefix: str = "agent:conversation",
    ) -> None:
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be greater than zero.")
        if history_limit < 1:
            raise ValueError("history_limit must be greater than zero.")
        if not key_prefix.strip():
            raise ValueError("key_prefix cannot be empty.")

        self._redis = redis_client
        self._ttl_seconds = ttl_seconds
        self._history_limit = history_limit
        self._key_prefix = key_prefix.strip().rstrip(":")

    @classmethod
    def from_url(
        cls,
        *,
        redis_url: str,
        ttl_seconds: int = 86400,
        history_limit: int = 12,
        key_prefix: str = "agent:conversation",
    ) -> RedisConversationStore:
        if not redis_url.strip():
            raise ValueError("REDIS_URL is required for the Redis backend.")

        try:
            import redis
        except ImportError as exc:
            raise RuntimeError(
                "The redis package is required for Redis conversation memory. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        client = redis.Redis.from_url(
            redis_url.strip(),
            decode_responses=True,
        )
        client.ping()
        return cls(
            redis_client=client,
            ttl_seconds=ttl_seconds,
            history_limit=history_limit,
            key_prefix=key_prefix,
        )

    def load(self, session_id: str) -> ConversationSession:
        normalized = _normalize_session_id(session_id)
        key = self._key(normalized)
        raw = self._redis.get(key)
        if raw is None:
            return ConversationSession(session_id=normalized)

        payload = json.loads(raw)
        messages = tuple(
            ConversationMessage(
                role=str(item["role"]),
                content=str(item["content"]),
            )
            for item in payload.get("messages", [])
        )
        self._redis.expire(key, self._ttl_seconds)
        return ConversationSession(
            session_id=normalized,
            messages=messages,
            last_entity=_optional_text(payload.get("last_entity")),
            previous_response_id=_optional_text(
                payload.get("previous_response_id")
            ),
        )

    def append_turn(
        self,
        *,
        session_id: str,
        user_input: str,
        assistant_output: str,
        last_entity: str | None,
        previous_response_id: str | None,
    ) -> ConversationSession:
        current = self.load(session_id)
        messages = _trim_messages(
            (*current.messages,
             ConversationMessage(role="user", content=user_input.strip()),
             ConversationMessage(
                 role="assistant",
                 content=assistant_output.strip(),
             )),
            self._history_limit,
        )
        updated = ConversationSession(
            session_id=current.session_id,
            messages=messages,
            last_entity=(last_entity or current.last_entity),
            previous_response_id=previous_response_id,
        )
        payload = {
            "session_id": updated.session_id,
            "messages": [asdict(message) for message in updated.messages],
            "last_entity": updated.last_entity,
            "previous_response_id": updated.previous_response_id,
        }
        self._redis.set(
            self._key(updated.session_id),
            json.dumps(payload, ensure_ascii=False),
            ex=self._ttl_seconds,
        )
        return updated

    def clear(self, session_id: str) -> None:
        self._redis.delete(self._key(_normalize_session_id(session_id)))

    def ttl(self, session_id: str) -> int | None:
        value = int(self._redis.ttl(
            self._key(_normalize_session_id(session_id))
        ))
        return None if value < 0 else value

    def _key(self, session_id: str) -> str:
        return f"{self._key_prefix}:{session_id}"


def build_conversation_store(settings: Any) -> ConversationStore:
    backend = settings.conversation_store_backend.strip().lower()
    if backend == "memory":
        return InMemoryConversationStore(
            history_limit=settings.conversation_history_limit,
        )
    if backend == "redis":
        try:
            return RedisConversationStore.from_url(
                redis_url=settings.redis_url,
                ttl_seconds=settings.conversation_ttl_seconds,
                history_limit=settings.conversation_history_limit,
                key_prefix=settings.conversation_key_prefix,
            )
        except Exception as exc:
            if not settings.conversation_store_fallback_to_memory:
                raise RuntimeError(
                    "Redis conversation memory is configured, but the "
                    "connection could not be established. Start the Redis "
                    "service, configure REDIS_URL correctly, switch "
                    "CONVERSATION_STORE_BACKEND=memory, or explicitly enable "
                    "CONVERSATION_STORE_FALLBACK_TO_MEMORY=true."
                ) from exc

            warning = (
                "Redis is unavailable. Using process-local in-memory "
                "conversation storage. Conversation persistence is disabled. "
                f"Cause: {type(exc).__name__}: {exc}"
            )
            return InMemoryConversationStore(
                history_limit=settings.conversation_history_limit,
                startup_warning=warning,
            )
    raise ValueError(
        "CONVERSATION_STORE_BACKEND must be either 'memory' or 'redis'."
    )


def build_memory_context(
    session: ConversationSession,
    *,
    history_turns: int = 4,
) -> str:
    if not session.messages:
        return ""
    message_limit = max(1, history_turns) * 2
    recent = session.messages[-message_limit:]
    lines = ["Recent conversation context:"]
    for message in recent:
        label = "User" if message.role == "user" else "Assistant"
        lines.append(f"{label}: {message.content}")
    return "\n".join(lines)


def resolve_entity_reference(
    user_input: str,
    last_entity: str | None,
) -> str:
    normalized = user_input.strip()
    if not normalized or not last_entity:
        return normalized

    from shared.query_planner import extract_entity_id

    if extract_entity_id(normalized):
        return normalized

    reference_terms = (
        "esse item",
        "este item",
        "desse item",
        "deste item",
        "ele",
        "isso",
        "this item",
        "that item",
        "it",
    )
    lowered = normalized.lower()
    if any(term in lowered for term in reference_terms):
        return (
            f"{normalized}\n\nResolved conversation entity_id: "
            f"{last_entity}"
        )
    return normalized


def _normalize_session_id(session_id: str) -> str:
    normalized = session_id.strip()
    if not normalized:
        raise ValueError("session_id cannot be empty.")
    if len(normalized) > 128:
        raise ValueError("session_id cannot exceed 128 characters.")
    return normalized


def _trim_messages(
    messages: tuple[ConversationMessage, ...],
    history_limit: int,
) -> tuple[ConversationMessage, ...]:
    return messages[-(history_limit * 2):]


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
