"""Tests for the semantic embedding provider factory (no real APIs)."""

import pytest

from ragwatch.semantic.providers import (
    SemanticProviderConfigError,
    create_semantic_embedding_provider_from_env,
)

ENV_VARS = [
    "RAGWATCH_SEMANTIC_PROVIDER",
    "RAGWATCH_LOCAL_EMBEDDING_MODEL",
    "RAGWATCH_OPENAI_API_KEY",
    "RAGWATCH_OPENAI_EMBEDDING_MODEL",
    "RAGWATCH_AZURE_OPENAI_API_KEY",
    "RAGWATCH_AZURE_OPENAI_ENDPOINT",
    "RAGWATCH_AZURE_OPENAI_API_VERSION",
    "RAGWATCH_AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Clear all semantic env vars and disable .env loading for each test."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        "ragwatch.semantic.providers.load_env", lambda: None
    )


class TestFactoryErrors:
    def test_openai_missing_api_key_raises(self, monkeypatch) -> None:
        monkeypatch.setenv("RAGWATCH_SEMANTIC_PROVIDER", "openai")
        with pytest.raises(SemanticProviderConfigError, match="OPENAI_API_KEY"):
            create_semantic_embedding_provider_from_env()

    def test_azure_missing_fields_raises(self, monkeypatch) -> None:
        monkeypatch.setenv("RAGWATCH_SEMANTIC_PROVIDER", "azure_openai")
        monkeypatch.setenv("RAGWATCH_AZURE_OPENAI_API_KEY", "key")
        # endpoint and deployment intentionally missing
        with pytest.raises(SemanticProviderConfigError) as excinfo:
            create_semantic_embedding_provider_from_env()
        message = str(excinfo.value)
        assert "RAGWATCH_AZURE_OPENAI_ENDPOINT" in message
        assert "RAGWATCH_AZURE_OPENAI_EMBEDDING_DEPLOYMENT" in message

    def test_unknown_provider_raises(self, monkeypatch) -> None:
        monkeypatch.setenv("RAGWATCH_SEMANTIC_PROVIDER", "made_up")
        with pytest.raises(SemanticProviderConfigError, match="Unknown"):
            create_semantic_embedding_provider_from_env()

    def test_openai_does_not_fall_back_to_local(self, monkeypatch) -> None:
        # Explicit openai with no key must error, not silently use local.
        monkeypatch.setenv("RAGWATCH_SEMANTIC_PROVIDER", "openai")
        with pytest.raises(SemanticProviderConfigError):
            create_semantic_embedding_provider_from_env()
