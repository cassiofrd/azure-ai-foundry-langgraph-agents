from __future__ import annotations

import pytest

from shared.settings import load_settings


def test_load_settings(monkeypatch):
    monkeypatch.setenv(
        "FOUNDRY_PROJECT_ENDPOINT",
        "https://example.services.ai.azure.com/api/projects/example",
    )
    monkeypatch.setenv(
        "FOUNDRY_MODEL_DEPLOYMENT",
        "gpt-test",
    )
    monkeypatch.setenv(
        "FOUNDRY_EMBEDDING_DEPLOYMENT",
        "text-embedding-3-small",
    )
    monkeypatch.setenv(
        "MODEL_MAX_OUTPUT_TOKENS",
        "500",
    )

    settings = load_settings()

    assert (
        settings.foundry_model_deployment
        == "gpt-test"
    )

    assert (
        settings.foundry_embedding_deployment
        == "text-embedding-3-small"
    )

    assert settings.model_max_output_tokens == 500


def test_missing_endpoint_fails_fast(monkeypatch):
    monkeypatch.delenv(
        "FOUNDRY_PROJECT_ENDPOINT",
        raising=False,
    )

    monkeypatch.setenv(
        "FOUNDRY_MODEL_DEPLOYMENT",
        "gpt-test",
    )

    monkeypatch.setenv(
        "FOUNDRY_EMBEDDING_DEPLOYMENT",
        "text-embedding-3-small",
    )

    with pytest.raises(
        ValueError,
        match="FOUNDRY_PROJECT_ENDPOINT",
    ):
        load_settings()


def test_missing_embedding_deployment_fails_fast(
    monkeypatch,
):
    monkeypatch.setenv(
        "FOUNDRY_PROJECT_ENDPOINT",
        "https://example.services.ai.azure.com/api/projects/example",
    )

    monkeypatch.setenv(
        "FOUNDRY_MODEL_DEPLOYMENT",
        "gpt-test",
    )

    monkeypatch.delenv(
        "FOUNDRY_EMBEDDING_DEPLOYMENT",
        raising=False,
    )

    with pytest.raises(
        ValueError,
        match="FOUNDRY_EMBEDDING_DEPLOYMENT",
    ):
        load_settings()