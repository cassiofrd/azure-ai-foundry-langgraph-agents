from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class AppSettings:
    foundry_project_endpoint: str
    foundry_model_deployment: str
    foundry_embedding_deployment: str

    azure_openai_endpoint: str
    azure_openai_api_key: str

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
    azure_search_vector_field: str = "content_vector"
    azure_search_vector_dimensions: int = 1536

    telemetry_console_enabled: bool = True
    azure_monitor_enabled: bool = False
    applicationinsights_connection_string: str = ""
    azure_monitor_logger_name: str = "agent.telemetry"
    azure_monitor_force_flush: bool = False
    azure_monitor_flush_timeout_ms: int = 10000


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


def _boolean(
    name: str,
    default: bool,
) -> bool:
    default_text = "true" if default else "false"
    raw = os.getenv(name, default_text).strip().lower()

    if raw in {"1", "true", "yes", "on"}:
        return True

    if raw in {"0", "false", "no", "off"}:
        return False

    raise ValueError(
        f"{name} must be one of: "
        "true, false, 1, 0, yes, no, on, off."
    )


def load_settings() -> AppSettings:
    azure_monitor_enabled = _boolean(
        "AZURE_MONITOR_ENABLED",
        False,
    )
    applicationinsights_connection_string = os.getenv(
        "APPLICATIONINSIGHTS_CONNECTION_STRING",
        "",
    ).strip()

    if (
        azure_monitor_enabled
        and not applicationinsights_connection_string
    ):
        raise ValueError(
            "APPLICATIONINSIGHTS_CONNECTION_STRING is required "
            "when AZURE_MONITOR_ENABLED=true."
        )

    return AppSettings(
        foundry_project_endpoint=_required(
            "FOUNDRY_PROJECT_ENDPOINT"
        ),
        foundry_model_deployment=_required(
            "FOUNDRY_MODEL_DEPLOYMENT"
        ),
        foundry_embedding_deployment=_required(
            "FOUNDRY_EMBEDDING_DEPLOYMENT"
        ),
        azure_openai_endpoint=_required(
            "AZURE_OPENAI_ENDPOINT"
        ),
        azure_openai_api_key=_required(
            "AZURE_OPENAI_API_KEY"
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
        azure_search_vector_field=os.getenv(
            "AZURE_SEARCH_VECTOR_FIELD",
            "content_vector",
        ).strip(),
        azure_search_vector_dimensions=_positive_int(
            "AZURE_SEARCH_VECTOR_DIMENSIONS",
            1536,
        ),
        telemetry_console_enabled=_boolean(
            "TELEMETRY_CONSOLE_ENABLED",
            True,
        ),
        azure_monitor_enabled=azure_monitor_enabled,
        applicationinsights_connection_string=(
            applicationinsights_connection_string
        ),
        azure_monitor_logger_name=os.getenv(
            "AZURE_MONITOR_LOGGER_NAME",
            "agent.telemetry",
        ).strip(),
        azure_monitor_force_flush=_boolean(
            "AZURE_MONITOR_FORCE_FLUSH",
            False,
        ),
        azure_monitor_flush_timeout_ms=_positive_int(
            "AZURE_MONITOR_FLUSH_TIMEOUT_MS",
            10000,
        ),
    )
