from __future__ import annotations

import json
import logging
from collections.abc import Callable
from threading import Lock
from typing import Any

from shared.execution_context import ExecutionContext
from shared.settings import AppSettings


OutputWriter = Callable[[str], Any]
TelemetrySink = Callable[[ExecutionContext], None]

_CONFIG_LOCK = Lock()
_CONFIGURED_LOGGERS: dict[str, logging.Logger] = {}


def build_execution_summary(
    execution_context: ExecutionContext,
) -> dict[str, Any]:
    return execution_context.to_dict()


def format_execution_summary(
    execution_context: ExecutionContext,
) -> str:
    summary = build_execution_summary(execution_context)
    metrics = summary["metrics"]

    duration = summary["duration_ms"]
    duration_text = (
        f"{duration:.2f} ms"
        if duration is not None
        else "not finished"
    )

    lines = [
        "",
        "Execution Summary",
        "-----------------",
        f"Execution ID: {summary['execution_id']}",
        f"Status: {summary['status']}",
        f"Duration: {duration_text}",
        "",
        (
            "LLM calls: "
            f"{metrics['llm_call_count']} "
            f"({metrics['successful_llm_calls']} successful)"
        ),
        f"LLM duration: {metrics['total_llm_duration_ms']:.2f} ms",
        f"Input tokens: {metrics['total_input_tokens']}",
        f"Output tokens: {metrics['total_output_tokens']}",
        f"Total tokens: {metrics['total_tokens']}",
        "",
        (
            "Tool calls: "
            f"{metrics['tool_call_count']} "
            f"({metrics['successful_tool_calls']} successful)"
        ),
    ]

    for tool in summary["tools"]:
        outcome = "success" if tool["error_type"] is None else "failed"
        lines.append(
            f"- {tool['tool_name']}: "
            f"{tool['duration_ms']:.2f} ms [{outcome}]"
        )

    lines.extend(
        [
            f"Tool duration: {metrics['total_tool_duration_ms']:.2f} ms",
            "",
            f"Errors: {metrics['error_count']}",
        ]
    )

    for error in summary["errors"]:
        lines.append(
            f"- {error['component']}.{error['operation']}: "
            f"{error['error_type']}: {error['error_message']}"
        )

    return "\n".join(lines)


def emit_execution_summary(
    execution_context: ExecutionContext,
    *,
    writer: OutputWriter = print,
) -> None:
    writer(format_execution_summary(execution_context))


def emit_execution_json(
    execution_context: ExecutionContext,
    *,
    writer: OutputWriter = print,
    indent: int | None = None,
) -> None:
    writer(
        json.dumps(
            build_execution_summary(execution_context),
            ensure_ascii=False,
            indent=indent,
        )
    )


def initialize_azure_monitor(
    settings: AppSettings,
) -> logging.Logger:
    logger_name = settings.azure_monitor_logger_name

    with _CONFIG_LOCK:
        existing = _CONFIGURED_LOGGERS.get(logger_name)
        if existing is not None:
            return existing

        from azure.monitor.opentelemetry import configure_azure_monitor

        configure_azure_monitor(
            connection_string=(
                settings.applicationinsights_connection_string
            ),
            logger_name=logger_name,
        )

        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)
        logger.propagate = True

        _CONFIGURED_LOGGERS[logger_name] = logger
        return logger


def force_flush_azure_monitor(
    settings: AppSettings,
) -> None:
    from opentelemetry._logs import get_logger_provider

    provider = get_logger_provider()
    force_flush = getattr(provider, "force_flush", None)

    if not callable(force_flush):
        raise RuntimeError(
            "The configured OpenTelemetry LoggerProvider "
            "does not support force_flush()."
        )

    completed = force_flush(
        timeout_millis=settings.azure_monitor_flush_timeout_ms
    )

    if completed is False:
        raise TimeoutError(
            "Azure Monitor flush exceeded "
            f"{settings.azure_monitor_flush_timeout_ms} ms."
        )


def _build_azure_monitor_attributes(
    execution_context: ExecutionContext,
    settings: AppSettings,
) -> dict[str, str | int | float | bool]:
    summary = build_execution_summary(execution_context)
    metrics = summary["metrics"]
    duration_ms = summary["duration_ms"]

    return {
        "microsoft.custom_event.name": "agent.execution.completed",
        "agent.execution_id": summary["execution_id"],
        "agent.status": summary["status"],
        "agent.duration_ms": (
            float(duration_ms)
            if duration_ms is not None
            else 0.0
        ),
        "agent.llm_call_count": metrics["llm_call_count"],
        "agent.successful_llm_calls": metrics[
            "successful_llm_calls"
        ],
        "agent.llm_duration_ms": metrics["total_llm_duration_ms"],
        "agent.input_tokens": metrics["total_input_tokens"],
        "agent.output_tokens": metrics["total_output_tokens"],
        "agent.total_tokens": metrics["total_tokens"],
        "agent.tool_call_count": metrics["tool_call_count"],
        "agent.successful_tool_calls": metrics[
            "successful_tool_calls"
        ],
        "agent.tool_duration_ms": metrics["total_tool_duration_ms"],
        "agent.error_count": metrics["error_count"],
        "agent.tool_names": ",".join(
            tool["tool_name"] for tool in summary["tools"]
        ),
        "agent.error_types": ",".join(
            error["error_type"] for error in summary["errors"]
        ),
        "service.name": settings.app_name,
        "service.version": settings.app_version,
        "deployment.environment.name": settings.app_environment,
    }


def emit_execution_to_azure_monitor(
    execution_context: ExecutionContext,
    *,
    settings: AppSettings,
    logger: logging.Logger | None = None,
) -> None:
    azure_logger = logger or initialize_azure_monitor(settings)

    azure_logger.warning(
        "Agent execution completed",
        extra=_build_azure_monitor_attributes(
            execution_context,
            settings,
        ),
    )

    if settings.azure_monitor_force_flush:
        force_flush_azure_monitor(settings)


def build_telemetry_sink(
    settings: AppSettings,
    *,
    writer: OutputWriter = print,
) -> TelemetrySink | None:
    if (
        not settings.telemetry_console_enabled
        and not settings.azure_monitor_enabled
    ):
        return None

    azure_logger = (
        initialize_azure_monitor(settings)
        if settings.azure_monitor_enabled
        else None
    )

    def sink(execution_context: ExecutionContext) -> None:
        failures: list[Exception] = []

        if settings.telemetry_console_enabled:
            try:
                emit_execution_summary(
                    execution_context,
                    writer=writer,
                )
            except Exception as exc:
                failures.append(exc)

        if settings.azure_monitor_enabled:
            try:
                emit_execution_to_azure_monitor(
                    execution_context,
                    settings=settings,
                    logger=azure_logger,
                )
            except Exception as exc:
                failures.append(exc)

        if failures:
            details = "; ".join(
                f"{type(error).__name__}: {error}"
                for error in failures
            )
            raise RuntimeError(
                f"One or more telemetry sinks failed: {details}"
            )

    return sink
