"""
Configuration loader for RAG package.

Loads configuration from YAML files and environment variables.
Supports hierarchical config with defaults and overrides.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class Config:
    """
    Configuration loader supporting YAML files and environment variables.

    Attributes:
        config: Dictionary containing merged configuration
        config_path: Path to the configuration file

    Example:
        >>> config = Config("config.yaml")
        >>> qdrant_url = config.get("vectorstore.url")
        >>> embedding_model = config.get("embeddings.model", "default-model")
    """

    def __init__(self, path: str):
        """
        Initialize configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If YAML is not installed
        """
        load_dotenv()

        if not YAML_AVAILABLE:
            raise ValueError(
                "PyYAML is required for configuration loading. "
                "Install it with: pip install pyyaml"
            )

        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.config = self._resolve_env_vars(self.config)
        self.config_path = config_path

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key: Dot-separated key path (e.g., "vectorstore.url")
            default: Default value if key doesn't exist

        Returns:
            Configuration value or default if not found

        Example:
            >>> config.get("vectorstore.url")  # Returns value from vectorstore.url
            >>> config.get("embeddings.model", "default")  # Returns "default" if not found
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    @staticmethod
    def _resolve_env_vars(obj: Any) -> Any:
        """Recursively replace ${VAR} placeholders with environment variable values."""
        if isinstance(obj, dict):
            return {k: Config._resolve_env_vars(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [Config._resolve_env_vars(v) for v in obj]
        if isinstance(obj, str):
            def _replacer(match):
                var = match.group(1)
                value = os.getenv(var)
                if value is None:
                    logger.warning(f"Environment variable '{var}' referenced in config but not set")
                    return match.group(0)
                return value
            return re.sub(r"\$\{([^}]+)\}", _replacer, obj)
        return obj

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access with dot notation."""
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in configuration."""
        return self.get(key) is not None
