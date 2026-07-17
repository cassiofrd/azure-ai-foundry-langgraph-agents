from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class AppSettings:
    foundry_project_endpoint: str
    foundry_model_deployment: str
    app_name: str
    app_environment: str
    app_version: str
    model_max_output_tokens: int
    system_prompt: str
    router_max_output_tokens: int
    router_prompt: str

    azure_search_endpoint: str = ""
    azure_search_index_name: str = "supply-chain-docs"
    azure_search_admin_key: str = ""
    azure_search_top_k: int = 3


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()

    if not value:
        raise ValueError(
            f"{name} is required. "
            "Copy .env.example to .env and configure it."
        )

    return value


def _positive_int(
    name: str,
    default: int,
) -> int:
    raw = os.getenv(name, str(default)).strip()

    try:
        value = int(raw)

    except ValueError as exc:
        raise ValueError(
            f"{name} must be an integer."
        ) from exc

    if value < 1:
        raise ValueError(
            f"{name} must be greater than zero."
        )

    return value


def load_settings() -> AppSettings:
    return AppSettings(
        foundry_project_endpoint=_required(
            "FOUNDRY_PROJECT_ENDPOINT"
        ),
        foundry_model_deployment=_required(
            "FOUNDRY_MODEL_DEPLOYMENT"
        ),
        app_name=os.getenv(
            "APP_NAME",
            "azure-ai-foundry-langgraph-agents",
        ).strip(),
        app_environment=os.getenv(
            "APP_ENVIRONMENT",
            "local",
        ).strip(),
        app_version=os.getenv(
            "APP_VERSION",
            "0.1.0",
        ).strip(),
        model_max_output_tokens=_positive_int(
            "MODEL_MAX_OUTPUT_TOKENS",
            800,
        ),
        system_prompt=os.getenv(
            "SYSTEM_PROMPT",
            "You are a concise enterprise AI assistant.",
        ).strip(),
        router_max_output_tokens=_positive_int(
            "ROUTER_MAX_OUTPUT_TOKENS",
            16,
        ),
        router_prompt=os.getenv(
            "ROUTER_PROMPT",
            (
                "Classify the request as 'time' if it asks "
                "for the current time or UTC time; otherwise "
                "classify it as 'general'. Return only the "
                "route name."
            ),
        ).strip(),
        azure_search_endpoint=os.getenv(
            "AZURE_SEARCH_ENDPOINT",
            "",
        ).strip(),
        azure_search_index_name=os.getenv(
            "AZURE_SEARCH_INDEX_NAME",
            "supply-chain-docs",
        ).strip(),
        azure_search_admin_key=os.getenv(
            "AZURE_SEARCH_ADMIN_KEY",
            "",
        ).strip(),
        azure_search_top_k=_positive_int(
            "AZURE_SEARCH_TOP_K",
            3,
        ),
    )