from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


DEFAULT_SYSTEM_PROMPT = (
    "You are a concise enterprise AI assistant. "
    "When answering with information returned by the "
    "search_documents tool, use only the evidence contained "
    "in the returned documents. Always include an Evidence "
    "section that lists the title, source, and entity_id of "
    "each document used. Preserve these values exactly as "
    "returned by the tool. Never invent, infer, rename, or "
    "omit evidence metadata. When the tool returns count 0 "
    "or no documents, clearly state that no supporting "
    "document was found, do not provide values from similar "
    "entities, and do not include a fabricated Evidence "
    "section."
)


DEFAULT_ROUTER_PROMPT = (
    "Classify the user request into exactly one route: inventory, supplier, "
    "logistics, inventory_supplier, inventory_logistics, supplier_logistics, "
    "inventory_supplier_logistics, time, or general. Use inventory for stock levels, "
    "reorder points, replenishment quantities and shortage policies. "
    "Use supplier for approved suppliers, procurement sources, lead "
    "times and vendor information. Use inventory_supplier when a single "
    "request requires both inventory policy facts and supplier or lead-time "
    "facts to produce a complete answer. Use logistics for transportation modes, "
    "transit times, freight, dispatch, shipping constraints, urgency and logistics "
    "approvals. Use a combined route whenever two or three domains are required. "
    "Use time only for current UTC time. Use general for everything else. "
    "Return only the route name."
)

DEFAULT_INVENTORY_PROMPT = (
    DEFAULT_SYSTEM_PROMPT
    + " You are the Inventory Agent. Answer only inventory-domain "
    "questions. Use search_inventory_documents for enterprise facts. "
    "Do not answer supplier questions from general knowledge."
)

DEFAULT_SUPPLIER_PROMPT = (
    DEFAULT_SYSTEM_PROMPT
    + " You are the Supplier Agent. Answer only supplier and procurement "
    "questions. Use search_supplier_documents for enterprise facts. "
    "Do not answer inventory policy questions from general knowledge."
)

DEFAULT_LOGISTICS_PROMPT = (
    DEFAULT_SYSTEM_PROMPT
    + " You are the Logistics Agent. Answer only transportation and logistics "
    "questions. Use search_logistics_documents for enterprise facts. Do not "
    "answer inventory or supplier questions from general knowledge."
)

DEFAULT_MULTI_AGENT_PROMPT = (
    "You are the Supply Chain Supervisor. Synthesize one concise answer from "
    "the specialist outputs supplied by the application. Answer in the same "
    "language as the original user request. Use only facts present in the "
    "specialist outputs. Distinguish clearly between confirmed facts, a "
    "partially supported conclusion, and missing information. Preserve every "
    "Evidence item exactly as supplied, deduplicate repeated items, and include "
    "one Evidence section at the end. If a specialist found no supporting "
    "document, state only the limitation relevant to the answer and do not "
    "invent missing facts. Do not claim that a supplier or transportation plan "
    "can satisfy an inventory policy unless the supplied evidence supports it."
)

DEFAULT_TIME_PROMPT = (
    "You are the Time Agent. Use get_current_utc_time whenever the user "
    "asks for the current UTC time. Answer concisely."
)


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
    inventory_prompt: str = DEFAULT_INVENTORY_PROMPT
    supplier_prompt: str = DEFAULT_SUPPLIER_PROMPT
    logistics_prompt: str = DEFAULT_LOGISTICS_PROMPT
    time_prompt: str = DEFAULT_TIME_PROMPT
    multi_agent_prompt: str = DEFAULT_MULTI_AGENT_PROMPT

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
            DEFAULT_SYSTEM_PROMPT,
        ).strip(),
        router_max_output_tokens=_positive_int(
            "ROUTER_MAX_OUTPUT_TOKENS",
            16,
        ),
        router_prompt=os.getenv(
            "ROUTER_PROMPT",
            DEFAULT_ROUTER_PROMPT,
        ).strip(),
        inventory_prompt=os.getenv(
            "INVENTORY_PROMPT",
            DEFAULT_INVENTORY_PROMPT,
        ).strip(),
        supplier_prompt=os.getenv(
            "SUPPLIER_PROMPT",
            DEFAULT_SUPPLIER_PROMPT,
        ).strip(),
        logistics_prompt=os.getenv(
            "LOGISTICS_PROMPT",
            DEFAULT_LOGISTICS_PROMPT,
        ).strip(),
        time_prompt=os.getenv(
            "TIME_PROMPT",
            DEFAULT_TIME_PROMPT,
        ).strip(),
        multi_agent_prompt=os.getenv(
            "MULTI_AGENT_PROMPT",
            DEFAULT_MULTI_AGENT_PROMPT,
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
