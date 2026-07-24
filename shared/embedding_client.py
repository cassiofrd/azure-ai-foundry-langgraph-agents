from __future__ import annotations

from typing import Any, Protocol

from openai import OpenAI


class EmbeddingsResourceProtocol(Protocol):
    def create(
        self,
        *,
        model: str,
        input: str | list[str],
    ) -> Any:
        ...


class EmbeddingClientProtocol(Protocol):
    embeddings: EmbeddingsResourceProtocol


def create_embedding_client(
    *,
    endpoint: str,
    api_key: str,
) -> EmbeddingClientProtocol:
    normalized_endpoint = endpoint.strip()

    if not normalized_endpoint:
        raise ValueError(
            "AZURE_OPENAI_ENDPOINT is required to create the embedding client."
        )

    normalized_api_key = api_key.strip()

    if not normalized_api_key:
        raise ValueError(
            "AZURE_OPENAI_API_KEY is required to create the embedding client."
        )

    return OpenAI(
        api_key=normalized_api_key,
        base_url=f"{normalized_endpoint.rstrip('/')}/openai/v1/",
    )