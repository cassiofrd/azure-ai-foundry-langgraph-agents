from __future__ import annotations

from typing import Any, Callable, Protocol

from azure.ai.projects import AIProjectClient
from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential


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


CredentialFactory = Callable[[], TokenCredential]

ProjectClientFactory = Callable[
    [str, TokenCredential],
    AIProjectClient,
]


def create_embedding_client(
    *,
    project_endpoint: str,
    credential_factory: CredentialFactory | None = None,
    project_client_factory: ProjectClientFactory | None = None,
) -> EmbeddingClientProtocol:
    normalized_endpoint = project_endpoint.strip()

    if not normalized_endpoint:
        raise ValueError(
            "FOUNDRY_PROJECT_ENDPOINT is required to create "
            "the embedding client."
        )

    credential_builder = (
        credential_factory or DefaultAzureCredential
    )

    project_client_builder = (
        project_client_factory or _create_project_client
    )

    credential = credential_builder()

    project_client = project_client_builder(
        normalized_endpoint,
        credential,
    )

    return project_client.get_openai_client()


def _create_project_client(
    endpoint: str,
    credential: TokenCredential,
) -> AIProjectClient:
    return AIProjectClient(
        endpoint=endpoint,
        credential=credential,
    )