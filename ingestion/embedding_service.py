from __future__ import annotations

from openai import OpenAI

from shared.settings import load_settings


class IngestionEmbeddingService:
    def __init__(self) -> None:
        settings = load_settings()

        self._deployment = settings.foundry_embedding_deployment
        self._client = OpenAI(
            api_key=settings.azure_openai_api_key,
            base_url=(
                f"{settings.azure_openai_endpoint.rstrip('/')}/openai/v1/"
            ),
        )

    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        response = self._client.embeddings.create(
            model=self._deployment,
            input=texts,
        )

        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]
