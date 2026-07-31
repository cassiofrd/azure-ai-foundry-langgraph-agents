from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from shared.conversation_store import (
    ConversationMessage,
    InMemoryConversationStore,
    RedisConversationStore,
    build_conversation_store,
    build_memory_context,
    resolve_entity_reference,
)


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.expirations: dict[str, int] = {}
        self.ping_calls = 0

    def ping(self):
        self.ping_calls += 1
        return True

    def get(self, key: str):
        return self.values.get(key)

    def set(self, key: str, value: str, ex: int | None = None):
        self.values[key] = value
        if ex is not None:
            self.expirations[key] = ex
        return True

    def expire(self, key: str, seconds: int):
        if key not in self.values:
            return False
        self.expirations[key] = seconds
        return True

    def delete(self, key: str):
        self.values.pop(key, None)
        self.expirations.pop(key, None)
        return 1

    def ttl(self, key: str):
        if key not in self.values:
            return -2
        return self.expirations.get(key, -1)


def test_in_memory_store_persists_turn_and_entity():
    store = InMemoryConversationStore(history_limit=3)

    session = store.append_turn(
        session_id="demo",
        user_input="Qual é a política do M10?",
        assistant_output="A política foi encontrada.",
        last_entity="M10",
        previous_response_id="resp-1",
    )

    assert session.last_entity == "M10"
    assert session.previous_response_id == "resp-1"
    assert session.messages == (
        ConversationMessage(
            role="user",
            content="Qual é a política do M10?",
        ),
        ConversationMessage(
            role="assistant",
            content="A política foi encontrada.",
        ),
    )
    assert store.load("demo") == session


def test_in_memory_store_isolates_and_clears_sessions():
    store = InMemoryConversationStore()
    store.append_turn(
        session_id="one",
        user_input="M10",
        assistant_output="Resposta 1",
        last_entity="M10",
        previous_response_id="resp-1",
    )
    store.append_turn(
        session_id="two",
        user_input="A100",
        assistant_output="Resposta 2",
        last_entity="A100",
        previous_response_id="resp-2",
    )

    store.clear("one")

    assert store.load("one").messages == ()
    assert store.load("two").last_entity == "A100"


def test_in_memory_store_trims_history_by_turn():
    store = InMemoryConversationStore(history_limit=2)
    for index in range(3):
        store.append_turn(
            session_id="demo",
            user_input=f"question-{index}",
            assistant_output=f"answer-{index}",
            last_entity=None,
            previous_response_id=f"resp-{index}",
        )

    session = store.load("demo")
    assert [message.content for message in session.messages] == [
        "question-1",
        "answer-1",
        "question-2",
        "answer-2",
    ]


def test_redis_store_serializes_session_and_refreshes_ttl():
    redis = FakeRedis()
    store = RedisConversationStore(
        redis_client=redis,
        ttl_seconds=120,
        history_limit=2,
        key_prefix="test:conversation",
    )

    store.append_turn(
        session_id="demo",
        user_input="Quem fornece o M10?",
        assistant_output="Contoso Industrial.",
        last_entity="M10",
        previous_response_id="resp-final",
    )

    key = "test:conversation:demo"
    payload = json.loads(redis.values[key])
    assert payload["last_entity"] == "M10"
    assert redis.expirations[key] == 120

    loaded = store.load("demo")
    assert loaded.last_entity == "M10"
    assert loaded.previous_response_id == "resp-final"
    assert store.ttl("demo") == 120


def test_redis_store_clear_removes_session():
    redis = FakeRedis()
    store = RedisConversationStore(redis_client=redis)
    store.append_turn(
        session_id="demo",
        user_input="M10",
        assistant_output="Resposta",
        last_entity="M10",
        previous_response_id="resp",
    )

    store.clear("demo")

    assert store.load("demo").messages == ()
    assert store.ttl("demo") is None


def test_resolve_entity_reference_uses_last_entity():
    resolved = resolve_entity_reference(
        "E quem fornece esse item?",
        "M10",
    )

    assert resolved.endswith("Resolved conversation entity_id: M10")


def test_resolve_entity_reference_preserves_explicit_entity():
    assert resolve_entity_reference(
        "Quem fornece o A100?",
        "M10",
    ) == "Quem fornece o A100?"


def test_build_memory_context_uses_recent_messages():
    store = InMemoryConversationStore()
    session = store.append_turn(
        session_id="demo",
        user_input="Qual é a política do M10?",
        assistant_output="Estoque alvo: 900.",
        last_entity="M10",
        previous_response_id="resp",
    )

    context = build_memory_context(session)

    assert "User: Qual é a política do M10?" in context
    assert "Assistant: Estoque alvo: 900." in context


@pytest.mark.parametrize("session_id", ["", "   "])
def test_empty_session_id_is_rejected(session_id):
    store = InMemoryConversationStore()
    with pytest.raises(ValueError, match="session_id cannot be empty"):
        store.load(session_id)


def _store_settings(**overrides):
    values = {
        "conversation_store_backend": "memory",
        "conversation_store_fallback_to_memory": False,
        "redis_url": "redis://localhost:6379/0",
        "conversation_ttl_seconds": 86400,
        "conversation_history_limit": 12,
        "conversation_key_prefix": "agent:conversation",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_build_memory_store_reports_non_persistent_backend():
    store = build_conversation_store(_store_settings())

    assert store.backend_name == "memory"
    assert store.is_persistent is False
    assert store.startup_warning is None


def test_redis_failure_fails_fast_when_fallback_is_disabled(
    monkeypatch,
):
    def fail(**_kwargs):
        raise ConnectionError("redis unavailable")

    monkeypatch.setattr(
        RedisConversationStore,
        "from_url",
        fail,
    )

    with pytest.raises(
        RuntimeError,
        match="Redis conversation memory is configured",
    ):
        build_conversation_store(
            _store_settings(
                conversation_store_backend="redis",
            )
        )


def test_redis_failure_can_explicitly_fallback_to_memory(
    monkeypatch,
):
    def fail(**_kwargs):
        raise ConnectionError("redis unavailable")

    monkeypatch.setattr(
        RedisConversationStore,
        "from_url",
        fail,
    )

    store = build_conversation_store(
        _store_settings(
            conversation_store_backend="redis",
            conversation_store_fallback_to_memory=True,
        )
    )

    assert store.backend_name == "memory"
    assert store.is_persistent is False
    assert store.startup_warning is not None
    assert "Redis is unavailable" in store.startup_warning
    assert "persistence is disabled" in store.startup_warning
