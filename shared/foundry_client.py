from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from openai import OpenAI

from shared.settings import AppSettings


class ResponsesClient(Protocol):
    class Responses(Protocol):
        def create(self, **kwargs): ...

    responses: Responses


@lru_cache(maxsize=1)
def get_project_client(endpoint: str) -> AIProjectClient:
    return AIProjectClient(
        endpoint=endpoint,
        credential=DefaultAzureCredential(),
    )


def get_openai_client(settings: AppSettings) -> OpenAI:
    project_client = get_project_client(settings.foundry_project_endpoint)
    return project_client.get_openai_client()
