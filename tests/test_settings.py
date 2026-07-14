from __future__ import annotations

import pytest

from shared.settings import load_settings


def test_load_settings(monkeypatch):
    monkeypatch.setenv(
        "FOUNDRY_PROJECT_ENDPOINT",
        "https://example.services.ai.azure.com/api/projects/example",
    )
    monkeypatch.setenv("FOUNDRY_MODEL_DEPLOYMENT", "gpt-test")
    monkeypatch.setenv("MODEL_MAX_OUTPUT_TOKENS", "500")

    settings = load_settings()

    assert settings.foundry_model_deployment == "gpt-test"
    assert settings.model_max_output_tokens == 500


def test_missing_endpoint_fails_fast(monkeypatch):
    monkeypatch.delenv("FOUNDRY_PROJECT_ENDPOINT", raising=False)
    monkeypatch.setenv("FOUNDRY_MODEL_DEPLOYMENT", "gpt-test")

    with pytest.raises(ValueError, match="FOUNDRY_PROJECT_ENDPOINT"):
        load_settings()
