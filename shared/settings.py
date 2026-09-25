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
    "inventory_supplier_logistics, time, or general. "
    "Use inventory for requests that can be answered using only stock levels, "
    "reorder points, safety stock, replenishment quantities, inventory policies, "
    "shortage status, open production orders, production demand, demand forecasts, "
    "replenishment procedures, shortage procedures, emergency procedures, or "
    "inventory-related workflows and decision rules. "
    "Questions asking what a replenishment or shortage procedure says, when it "
    "applies, or what steps it requires are inventory requests, even if the "
    "procedure mentions suppliers, transportation, approvals, or escalation. "
    "Use supplier for requests that can be answered using only approved suppliers, "
    "procurement sources, supplier contracts, lead times, MOQ, or vendor information. "
    "Use logistics for requests that can be answered using only transportation modes, "
    "transit times, freight rates, carrier restrictions, dispatch constraints, "
    "shipping urgency, or logistics approvals. "
    "Use a combined route whenever producing a complete answer requires facts from "
    "more than one specialist domain. "
    "Use inventory_supplier when inventory facts must be combined with supplier or "
    "lead-time facts. "
    "Use inventory_logistics when inventory facts must be combined with transportation "
    "or shipping facts. "
    "Use supplier_logistics when supplier facts must be combined with transportation "
    "or shipping facts. "
    "Distinguish a request to retrieve or explain a replenishment procedure from "
    "a request to create a replenishment plan. A procedure question asks what the "
    "documented process, rule, workflow, trigger, or required steps are and should "
    "normally route to inventory. A planning question asks what action should be "
    "taken for a concrete supply situation and may require multiple domains. "
    "Use inventory_supplier_logistics when the request asks for a complete, recommended, "
    "or end-to-end replenishment, sourcing, shortage-response, or supply plan that requires "
    "inventory status or policy, supplier selection or lead time, and transportation "
    "options or constraints. For example, 'What is the best replenishment plan for "
    "BOLT-M10 at PLANT-BH?' must route to inventory_supplier_logistics rather than "
    "inventory alone. "
    "Do not choose a single-domain route merely because one domain is mentioned first "
    "when the requested decision depends on multiple domains. "
    "Use time only for current UTC time. Use general for everything else. "
    "Return only the route name."
)

DEFAULT_INVENTORY_PROMPT = (
    DEFAULT_SYSTEM_PROMPT
    + " You are the Inventory Agent. Answer inventory, demand, and replenishment-"
    "planning questions assigned to you. Use get_current_inventory when exact "
    "current on-hand, reserved, available, and snapshot-date values are required. "
    "For replenishment planning, shortage-risk analysis, or projected inventory, "
    "use calculate_inventory_projection as the primary structured-data tool. Its output "
    "already contains the exact current inventory, complete open production orders, and "
    "exact demand forecast, so do not call get_current_inventory, "
    "get_open_production_orders, or get_demand_forecast again unless the projection is "
    "missing required data or the user explicitly asks for an underlying dataset. "
    "Do not perform inventory arithmetic yourself. Treat its committed-open-orders and "
    "forecast outputs as separate scenarios unless enterprise evidence explicitly "
    "establishes that they can be combined without double counting. "
    "Use search_inventory_documents for inventory policy and other relevance-oriented "
    "inventory knowledge, and search_demand_documents for relevance-oriented demand evidence. "
    "When the user explicitly asks which open production orders exist, or asks for the "
    "complete open-order set or exact total, use get_open_production_orders. "
    "Use search_procedure_documents for replenishment workflows, "
    "shortage escalation, emergency actions, approval requirements, and decision "
    "rules. For one planning request, prefer one sufficiently broad procedure search "
    "instead of repeating search_procedure_documents for the same item and location. "
    "Ground enterprise facts in tool evidence. Do not answer supplier "
    "questions from general knowledge."
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

    conversation_store_backend: str = "memory"
    conversation_store_fallback_to_memory: bool = False
    redis_url: str = "redis://localhost:6379/0"
    conversation_ttl_seconds: int = 86400
    conversation_history_limit: int = 12
    conversation_key_prefix: str = "agent:conversation"
    default_session_id: str = "local-demo"

    azure_search_endpoint: str = ""
    azure_search_index_name: str = "supply-chain-docs"
    azure_search_admin_key: str = ""
    azure_search_top_k: int = 3
    azure_search_vector_field: str = "content_vector"
    azure_search_vector_dimensions: int = 1536

    azure_storage_account_name: str = ""
    azure_storage_container_name: str = "rag-documents"

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

    conversation_store_backend = os.getenv(
        "CONVERSATION_STORE_BACKEND",
        "memory",
    ).strip().lower()
    if conversation_store_backend not in {"memory", "redis"}:
        raise ValueError(
            "CONVERSATION_STORE_BACKEND must be either "
            "'memory' or 'redis'."
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
            "1.0.0",
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
        conversation_store_backend=conversation_store_backend,
        conversation_store_fallback_to_memory=_boolean(
            "CONVERSATION_STORE_FALLBACK_TO_MEMORY",
            False,
        ),
        redis_url=os.getenv(
            "REDIS_URL",
            "redis://localhost:6379/0",
        ).strip(),
        conversation_ttl_seconds=_positive_int(
            "CONVERSATION_TTL_SECONDS",
            86400,
        ),
        conversation_history_limit=_positive_int(
            "CONVERSATION_HISTORY_LIMIT",
            12,
        ),
        conversation_key_prefix=os.getenv(
            "CONVERSATION_KEY_PREFIX",
            "agent:conversation",
        ).strip(),
        default_session_id=os.getenv(
            "DEFAULT_SESSION_ID",
            "local-demo",
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
        azure_storage_account_name=os.getenv(
            "AZURE_STORAGE_ACCOUNT_NAME",
            "",
        ).strip(),
        azure_storage_container_name=os.getenv(
            "AZURE_STORAGE_CONTAINER_NAME",
            "rag-documents",
        ).strip(),
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
