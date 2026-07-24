from __future__ import annotations

from dataclasses import dataclass

from shared.embedding_client import (
    EmbeddingClientProtocol,
    create_embedding_client,
)
from shared.settings import AppSettings


@dataclass(frozen=True)
class EmbeddingResult:
    embedding: tuple[float, ...]


class EmbeddingService:
    def __init__(
        self,
        *,
        settings: AppSettings,
        client: EmbeddingClientProtocol | None = None,
    ) -> None:
        self._settings = settings
        self._client = client or create_embedding_client(
            endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
        )

    def create_embedding(
        self,
        text: str,
    ) -> EmbeddingResult:
        normalized_text = text.strip()

        if not normalized_text:
            raise ValueError(
                "Embedding text cannot be empty."
            )

        response = self._client.embeddings.create(
            model=self._settings.foundry_embedding_deployment,
            input=normalized_text,
        )

        vector = tuple(
            float(value)
            for value in response.data[0].embedding
        )

        return EmbeddingResult(
            embedding=vector,
        )