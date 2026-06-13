"""Semantic embedding providers for optional semantic KPIs.

Providers wrap a text-embedding backend behind a small interface so the
semantic metric layer does not depend on any specific SDK. Optional
dependencies (``sentence-transformers``, ``openai``) are imported lazily so
importing RAGWatch never requires them.

Provider selection is environment-driven. The relevant variables are:

    RAGWATCH_SEMANTIC_PROVIDER          local | openai | azure_openai

    # local (sentence-transformers)
    RAGWATCH_LOCAL_EMBEDDING_MODEL

    # direct OpenAI
    RAGWATCH_OPENAI_API_KEY
    RAGWATCH_OPENAI_EMBEDDING_MODEL

    # Azure OpenAI
    RAGWATCH_AZURE_OPENAI_API_KEY
    RAGWATCH_AZURE_OPENAI_ENDPOINT
    RAGWATCH_AZURE_OPENAI_API_VERSION
    RAGWATCH_AZURE_OPENAI_EMBEDDING_DEPLOYMENT
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ragwatch.config.env import get_env, load_env

# Centralized environment variable names (avoid scattering raw strings).
ENV_SEMANTIC_PROVIDER = "RAGWATCH_SEMANTIC_PROVIDER"

ENV_LOCAL_EMBEDDING_MODEL = "RAGWATCH_LOCAL_EMBEDDING_MODEL"

ENV_OPENAI_API_KEY = "RAGWATCH_OPENAI_API_KEY"
ENV_OPENAI_EMBEDDING_MODEL = "RAGWATCH_OPENAI_EMBEDDING_MODEL"

ENV_AZURE_OPENAI_API_KEY = "RAGWATCH_AZURE_OPENAI_API_KEY"
ENV_AZURE_OPENAI_ENDPOINT = "RAGWATCH_AZURE_OPENAI_ENDPOINT"
ENV_AZURE_OPENAI_API_VERSION = "RAGWATCH_AZURE_OPENAI_API_VERSION"
ENV_AZURE_OPENAI_EMBEDDING_DEPLOYMENT = "RAGWATCH_AZURE_OPENAI_EMBEDDING_DEPLOYMENT"

PROVIDER_LOCAL = "local"
PROVIDER_OPENAI = "openai"
PROVIDER_AZURE_OPENAI = "azure_openai"

DEFAULT_LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_AZURE_OPENAI_API_VERSION = "2024-02-01"


class SemanticProviderConfigError(RuntimeError):
    """Raised when a semantic provider is misconfigured (missing settings)."""


class BaseSemanticEmbeddingProvider(ABC):
    """Interface for text-embedding providers used by semantic KPIs."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts and return one vector per input text."""
        ...


class LocalSentenceTransformerProvider(BaseSemanticEmbeddingProvider):
    """Local provider backed by ``sentence-transformers``."""

    def __init__(self, model_name: str = DEFAULT_LOCAL_EMBEDDING_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SemanticProviderConfigError(
                "The 'sentence-transformers' package is required for the local "
                "semantic provider. Install it with: pip install -e \".[semantic]\""
            ) from exc

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [vec.tolist() for vec in embeddings]


class OpenAIEmbeddingProvider(BaseSemanticEmbeddingProvider):
    """Direct OpenAI embeddings provider.

    For direct OpenAI, ``model`` is an OpenAI *model name* (for example
    ``text-embedding-3-small``).
    """

    def __init__(self, api_key: str, model: str = DEFAULT_OPENAI_EMBEDDING_MODEL) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SemanticProviderConfigError(
                "The 'openai' package is required for the OpenAI semantic provider. "
                "Install it with: pip install -e \".[semantic]\""
            ) from exc

        # For direct OpenAI this is a model name, not a deployment name.
        self.model = model
        self._client = OpenAI(api_key=api_key)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


class AzureOpenAIEmbeddingProvider(BaseSemanticEmbeddingProvider):
    """Azure OpenAI embeddings provider.

    Important: on Azure OpenAI the embedding *deployment name* (not a model
    name) is what gets passed as the ``model`` argument of the embeddings call.
    We store it as ``deployment_name`` to avoid confusion with the direct
    OpenAI model name.
    """

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        deployment_name: str,
        api_version: str = DEFAULT_AZURE_OPENAI_API_VERSION,
    ) -> None:
        try:
            from openai import AzureOpenAI
        except ImportError as exc:
            raise SemanticProviderConfigError(
                "The 'openai' package is required for the Azure OpenAI semantic "
                "provider. Install it with: pip install -e \".[semantic]\""
            ) from exc

        # On Azure this is a deployment name; it is passed as the `model` arg.
        self.deployment_name = deployment_name
        self._client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # Azure passes the deployment name where OpenAI passes a model name.
        response = self._client.embeddings.create(
            model=self.deployment_name, input=texts
        )
        return [item.embedding for item in response.data]


def create_semantic_embedding_provider_from_env() -> BaseSemanticEmbeddingProvider:
    """Create a semantic embedding provider based on environment variables.

    Reads ``RAGWATCH_SEMANTIC_PROVIDER`` (default ``local``) and the relevant
    settings for the chosen provider. Raises ``SemanticProviderConfigError`` if
    required settings are missing. Does not silently fall back to ``local`` when
    ``openai`` or ``azure_openai`` is explicitly requested.
    """
    load_env()
    provider = (get_env(ENV_SEMANTIC_PROVIDER, default=PROVIDER_LOCAL) or "").strip().lower()

    if provider == PROVIDER_LOCAL:
        model_name = get_env(
            ENV_LOCAL_EMBEDDING_MODEL, default=DEFAULT_LOCAL_EMBEDDING_MODEL
        )
        return LocalSentenceTransformerProvider(model_name=model_name)

    if provider == PROVIDER_OPENAI:
        api_key = get_env(ENV_OPENAI_API_KEY)
        if not api_key:
            raise SemanticProviderConfigError(
                f"{ENV_OPENAI_API_KEY} is required when "
                f"{ENV_SEMANTIC_PROVIDER}={PROVIDER_OPENAI}."
            )
        model = get_env(
            ENV_OPENAI_EMBEDDING_MODEL, default=DEFAULT_OPENAI_EMBEDDING_MODEL
        )
        return OpenAIEmbeddingProvider(api_key=api_key, model=model)

    if provider == PROVIDER_AZURE_OPENAI:
        api_key = get_env(ENV_AZURE_OPENAI_API_KEY)
        endpoint = get_env(ENV_AZURE_OPENAI_ENDPOINT)
        deployment = get_env(ENV_AZURE_OPENAI_EMBEDDING_DEPLOYMENT)
        api_version = get_env(
            ENV_AZURE_OPENAI_API_VERSION, default=DEFAULT_AZURE_OPENAI_API_VERSION
        )
        missing = [
            name
            for name, value in (
                (ENV_AZURE_OPENAI_API_KEY, api_key),
                (ENV_AZURE_OPENAI_ENDPOINT, endpoint),
                (ENV_AZURE_OPENAI_EMBEDDING_DEPLOYMENT, deployment),
            )
            if not value
        ]
        if missing:
            raise SemanticProviderConfigError(
                f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} "
                f"required when {ENV_SEMANTIC_PROVIDER}={PROVIDER_AZURE_OPENAI}."
            )
        return AzureOpenAIEmbeddingProvider(
            api_key=api_key,  # type: ignore[arg-type]
            endpoint=endpoint,  # type: ignore[arg-type]
            deployment_name=deployment,  # type: ignore[arg-type]
            api_version=api_version,  # type: ignore[arg-type]
        )

    raise SemanticProviderConfigError(
        f"Unknown {ENV_SEMANTIC_PROVIDER}={provider!r}. "
        f"Expected one of: {PROVIDER_LOCAL}, {PROVIDER_OPENAI}, {PROVIDER_AZURE_OPENAI}."
    )
