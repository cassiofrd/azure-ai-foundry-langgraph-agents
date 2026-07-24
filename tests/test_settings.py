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
    monkeypatch.setenv(
        "AZURE_SEARCH_VECTOR_FIELD",
        "document_vector",
    )
    monkeypatch.setenv(
        "AZURE_SEARCH_VECTOR_DIMENSIONS",
        "1536",
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

    assert (
        settings.azure_search_vector_field
        == "document_vector"
    )

    assert (
        settings.azure_search_vector_dimensions
        == 1536
    )


def test_vector_settings_use_defaults(monkeypatch):
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

    monkeypatch.delenv(
        "AZURE_SEARCH_VECTOR_FIELD",
        raising=False,
    )
    monkeypatch.delenv(
        "AZURE_SEARCH_VECTOR_DIMENSIONS",
        raising=False,
    )

    settings = load_settings()

    assert (
        settings.azure_search_vector_field
        == "content_vector"
    )
    assert (
        settings.azure_search_vector_dimensions
        == 1536
    )


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


@pytest.mark.parametrize(
    "value",
    [
        "0",
        "-1",
        "invalid",
    ],
)
def test_invalid_vector_dimensions_fail_fast(
    monkeypatch,
    value: str,
):
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
        "AZURE_SEARCH_VECTOR_DIMENSIONS",
        value,
    )

    with pytest.raises(
        ValueError,
        match="AZURE_SEARCH_VECTOR_DIMENSIONS",
    ):
        load_settings()