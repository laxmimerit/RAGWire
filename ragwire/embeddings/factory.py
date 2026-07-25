"""
Embedding factory for multiple provider support.

Provides a unified interface for creating embedding models
from different providers (OpenAI, HuggingFace, Ollama, Google, etc.).
"""

import logging
import os
from typing import Optional, Any, List

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


def get_embedding(config: dict, **kwargs: Any) -> Any:
    """
    Create an embedding model instance from configuration.

    Supports multiple embedding providers with automatic fallback
    and error handling.

    Args:
        config: Configuration dictionary with 'provider' key
               and provider-specific settings
        **kwargs: Additional keyword arguments to pass to constructor

    Returns:
        Initialized embedding model instance

    Raises:
        ValueError: If provider is not supported or config is invalid

    Example:
        >>> # OpenAI embeddings
        >>> embedding = get_embedding({
        ...     "provider": "openai",
        ...     "model": "text-embedding-3-small"
        ... })

        >>> # HuggingFace embeddings
        >>> embedding = get_embedding({
        ...     "provider": "huggingface",
        ...     "model_name": "sentence-transformers/all-MiniLM-L6-v2"
        ... })

        >>> # Ollama embeddings
        >>> embedding = get_embedding({
        ...     "provider": "ollama",
        ...     "model": "nomic-embed-text",
        ...     "base_url": "http://localhost:11434"
        ... })
    """
    provider = config.get("provider", "").lower()

    try:
        if provider == "openai":
            return _get_openai_embeddings(config, **kwargs)

        elif provider == "huggingface":
            return _get_huggingface_embeddings(config, **kwargs)

        elif provider == "ollama":
            return _get_ollama_embeddings(config, **kwargs)

        elif provider == "google" or provider == "gemini":
            return _get_google_embeddings(config, **kwargs)

        elif provider == "fastembed":
            return _get_fastembed_embeddings(config, **kwargs)

        elif provider == "openrouter":
            return _get_openrouter_embeddings(config, **kwargs)

        else:
            valid = "ollama, openai, huggingface, google, fastembed, openrouter"
            raise ValueError(
                f"Unsupported embedding provider: '{provider}'. Valid options: {valid}\n"
                f"Example config:\n"
                f"  embeddings:\n"
                f"    provider: ollama\n"
                f"    model: nomic-embed-text"
            )

    except ImportError as e:
        logger.error(f"Missing dependency for {provider} embeddings: {e}")
        raise ImportError(
            f"Required package for '{provider}' embeddings is not installed.\n"
            f"Run: {get_install_command(provider)}"
        )


def _get_openai_embeddings(config: dict, **kwargs) -> Any:
    """Create OpenAI embedding model."""
    from langchain_openai import OpenAIEmbeddings

    model = config.get("model", "text-embedding-3-small")

    return OpenAIEmbeddings(
        model=model,
        openai_api_key=config.get("api_key"),
        openai_organization=config.get("organization"),
        **kwargs,
    )


def _get_huggingface_embeddings(config: dict, **kwargs) -> Any:
    """Create HuggingFace embedding model."""
    from langchain_huggingface import HuggingFaceEmbeddings

    model_name = config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=config.get("model_kwargs", {}),
        encode_kwargs=config.get("encode_kwargs", {}),
        **kwargs,
    )


def _get_ollama_embeddings(config: dict, **kwargs) -> Any:
    """Create Ollama embedding model."""
    from langchain_ollama import OllamaEmbeddings

    model = config.get("model", "nomic-embed-text")
    base_url = config.get("base_url", "http://localhost:11434")
    extra = {}
    if "num_ctx" in config:
        extra["num_ctx"] = config["num_ctx"]

    return OllamaEmbeddings(model=model, base_url=base_url, **extra, **kwargs)


def _get_google_embeddings(config: dict, **kwargs) -> Any:
    """Create Google/Gemini embedding model."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    model = config.get("model", "models/embedding-001")

    return GoogleGenerativeAIEmbeddings(
        model=model, google_api_key=config.get("api_key"), **kwargs
    )


def _get_fastembed_embeddings(config: dict, **kwargs) -> Any:
    """Create FastEmbed embedding model."""
    from langchain_community.embeddings import FastEmbedEmbeddings

    model_name = config.get("model_name", "BAAI/bge-small-en-v1.5")

    return FastEmbedEmbeddings(model_name=model_name, **kwargs)


class OpenRouterEmbeddings(Embeddings):
    """LangChain-compatible embeddings backed by the official OpenRouter SDK.

    Uses ``openrouter.OpenRouter().embeddings.generate`` directly, not the
    OpenAI client. This avoids the OpenAI client's default base64 encoding,
    which OpenRouter returns in a form the OpenAI parser rejects
    ("No embedding data received"). We always request ``encoding_format="float"``.

    Args:
        model: OpenRouter embedding model id (e.g. "nvidia/llama-nemotron-embed-vl-1b-v2:free")
        api_key: OpenRouter API key. Falls back to the OPENROUTER_API_KEY env var.
        dimensions: Optional output dimensionality (if the model supports it).
        batch_size: Max inputs per request when embedding many documents.
        server_url: Optional override for the OpenRouter API base URL.
    """

    def __init__(
        self,
        model: str = "nvidia/llama-nemotron-embed-vl-1b-v2:free",
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
        batch_size: int = 100,
        server_url: Optional[str] = None,
    ):
        try:
            from openrouter import OpenRouter
        except ImportError:
            raise ImportError(
                "openrouter is required for OpenRouter embeddings. "
                "Install with: pip install \"ragwire[openrouter]\""
            )

        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter API key not found. Set 'api_key' in the embeddings "
                "config or the OPENROUTER_API_KEY environment variable."
            )

        client_kwargs: dict = {"api_key": api_key}
        if server_url:
            client_kwargs["server_url"] = server_url

        self._client = OpenRouter(**client_kwargs)
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size

    def _generate(self, inputs: list) -> list:
        extra: dict = {}
        if self.dimensions is not None:
            extra["dimensions"] = self.dimensions
        resp = self._client.embeddings.generate(
            model=self.model,
            input=inputs,
            encoding_format="float",
            **extra,
        )
        # Sort by index to guarantee output order matches input order.
        return [item.embedding for item in sorted(resp.data, key=lambda d: d.index)]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(self._generate(texts[i : i + self.batch_size]))
        return vectors

    def embed_query(self, text: str) -> List[float]:
        return self._generate([text])[0]


def _get_openrouter_embeddings(config: dict, **kwargs) -> Any:
    """Create an OpenRouter embedding model using the official OpenRouter SDK.

    The API key is read from config (typically ``api_key: "${OPENROUTER_API_KEY}"``)
    or falls back to the OPENROUTER_API_KEY environment variable.
    """
    return OpenRouterEmbeddings(
        model=config.get("model", "nvidia/llama-nemotron-embed-vl-1b-v2:free"),
        api_key=config.get("api_key"),
        dimensions=config.get("dimensions"),
        batch_size=config.get("batch_size", 100),
        server_url=config.get("base_url"),
        **kwargs,
    )


def get_install_command(provider: str) -> str:
    """
    Get the pip install command for a provider.

    Args:
        provider: Provider name

    Returns:
        pip install command string
    """
    commands = {
        "openai": "pip install langchain-openai",
        "huggingface": "pip install langchain-huggingface",
        "ollama": "pip install langchain-ollama",
        "google": "pip install langchain-google-genai",
        "fastembed": "pip install langchain-community fastembed",
        "openrouter": "pip install \"ragwire[openrouter]\"",
    }
    return commands.get(provider, "pip install langchain-community")
