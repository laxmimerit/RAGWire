"""
Embedding factory for multiple provider support.

Provides a unified interface for creating embedding models
from different providers (OpenAI, HuggingFace, Ollama, Google, etc.).
"""

import logging
from typing import Optional, Any

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

        else:
            raise ValueError(
                f"Unsupported embedding provider: {provider}. "
                f"Supported providers: openai, huggingface, ollama, google, fastembed"
            )

    except ImportError as e:
        logger.error(f"Missing dependency for {provider} embeddings: {e}")
        raise ImportError(
            f"Required package for {provider} embeddings not installed. "
            f"Install with: {get_install_command(provider)}"
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
    }
    return commands.get(provider, "pip install langchain-community")
