from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from shared.embedding_service import EmbeddingService
from shared.settings import AppSettings


class FakeEmbeddingsResource:
    def __init__(
        self,
        *,
        embedding: list[float],
    ) -> None:
        self._embedding = embedding
        self.calls: list[dict[str, Any]] = []

    def create(
        self,
        *,
        model: str,
        input: str | list[str],
    ) -> Any:
        self.calls.append(
            {
                "model": model,
                "input": input,
            }
        )

        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    embedding=self._embedding,
                )
            ]
        )


class FakeEmbeddingClient:
    def __init__(
        self,
        *,
        embedding: list[float],
    ) -> None:
        self.embeddings = FakeEmbeddingsResource(
            embedding=embedding,
        )


def _create_settings() -> AppSettings:
    return AppSettings(
        foundry_project_endpoint=(
            "https://example.services.ai.azure.com/"
            "api/projects/example"
        ),
        foundry_model_deployment="gpt-test",
        foundry_embedding_deployment=(
            "text-embedding-3-small"
        ),
        app_name="test-app",
        app_environment="test",
        app_version="0.1.0",
        model_max_output_tokens=500,
        system_prompt="Test system prompt.",
        router_max_output_tokens=16,
        router_prompt="Test router prompt.",
        azure_search_endpoint="",
        azure_search_index_name="supply-chain-docs",
        azure_search_admin_key="",
        azure_search_top_k=3,
    )


def test_create_embedding_returns_normalized_result():
    client = FakeEmbeddingClient(
        embedding=[0.1, 0.2, 0.3],
    )

    service = EmbeddingService(
        settings=_create_settings(),
        client=client,
    )

    result = service.create_embedding(
        "supply chain inventory",
    )

    assert result.embedding == (
        0.1,
        0.2,
        0.3,
    )


def test_create_embedding_uses_configured_deployment():
    client = FakeEmbeddingClient(
        embedding=[0.4, 0.5],
    )

    service = EmbeddingService(
        settings=_create_settings(),
        client=client,
    )

    service.create_embedding(
        "inventory policy",
    )

    assert client.embeddings.calls == [
        {
            "model": "text-embedding-3-small",
            "input": "inventory policy",
        }
    ]


def test_create_embedding_strips_input_text():
    client = FakeEmbeddingClient(
        embedding=[0.6],
    )

    service = EmbeddingService(
        settings=_create_settings(),
        client=client,
    )

    service.create_embedding(
        "  supplier performance  ",
    )

    assert client.embeddings.calls[0]["input"] == (
        "supplier performance"
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "\n\t",
    ],
)
def test_create_embedding_rejects_empty_text(
    text: str,
):
    client = FakeEmbeddingClient(
        embedding=[0.1],
    )

    service = EmbeddingService(
        settings=_create_settings(),
        client=client,
    )

    with pytest.raises(
        ValueError,
        match="Embedding text cannot be empty",
    ):
        service.create_embedding(text)

    assert client.embeddings.calls == []


def test_create_embedding_converts_values_to_float():
    client = FakeEmbeddingClient(
        embedding=[1, 2, 3],
    )

    service = EmbeddingService(
        settings=_create_settings(),
        client=client,
    )

    result = service.create_embedding(
        "warehouse capacity",
    )

    assert result.embedding == (
        1.0,
        2.0,
        3.0,
    )

    assert all(
        isinstance(value, float)
        for value in result.embedding
    )