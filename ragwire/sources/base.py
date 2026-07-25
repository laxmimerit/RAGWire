"""
The source interface.

A source answers one question: which files should this collection contain
right now? Everything else about syncing follows from that answer, including
which files are new, which changed, and which have gone away.

Remote sources download to a local cache and return those paths, so the rest
of the pipeline never learns where a file came from. That keeps ingestion,
hashing and deduplication identical for a folder on disk and a bucket in
another region.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_EXTENSIONS = [".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".md"]


class Source:
    """
    Base class for document sources.

    Subclasses implement :meth:`list_files`. Everything else, including
    extension filtering, is provided here so every source filters the same way.

    Attributes:
        name: Identifies this source in logs and sync results
        extensions: File extensions to include. None means all of them.
    """

    #: Set by subclasses, used in the config registry
    type_name = "base"

    def __init__(self, name: str = "", extensions: Optional[List[str]] = None):
        self.name = name or self.type_name
        self.extensions = (
            [e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions]
            if extensions
            else None
        )

    def list_files(self) -> List[str]:
        """
        Return the local paths of every file this source currently holds.

        Returns:
            Local file paths, ready to pass to ``ingest_documents``

        Raises:
            NotImplementedError: On the base class
        """
        raise NotImplementedError

    def matches_extension(self, path: str) -> bool:
        """
        Whether a path passes this source's extension filter.

        Args:
            path: A file path or key

        Returns:
            True when the file should be ingested
        """
        if self.extensions is None:
            return True
        return Path(path).suffix.lower() in self.extensions

    def close(self) -> None:
        """Release anything the source holds open. A no-op by default."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


def build_source(config: Dict[str, Any]) -> Source:
    """
    Build a source from one entry of the ``sources`` config block.

    Args:
        config: A mapping with a ``type`` key plus that type's own settings

    Returns:
        The configured Source

    Raises:
        ValueError: If ``type`` is missing or unknown

    Example:
        >>> build_source({"type": "local", "path": "./documents"})
        <LocalSource name='local' ...>
    """
    if not isinstance(config, dict):
        raise ValueError(
            f"Each entry in 'sources' must be a mapping, got {type(config).__name__}"
        )

    settings = dict(config)
    source_type = settings.pop("type", None)
    if not source_type:
        raise ValueError(
            f"Source entry is missing a 'type' key: {config}. "
            f"Available types: {', '.join(sorted(REGISTRY))}"
        )

    if source_type not in REGISTRY:
        raise ValueError(
            f"Unknown source type: '{source_type}'. "
            f"Available: {', '.join(sorted(REGISTRY))}"
        )

    return REGISTRY[source_type](**settings)


def build_sources(config: Optional[List[Dict[str, Any]]]) -> List[Source]:
    """
    Build every source in a ``sources`` config block.

    Args:
        config: The list under the ``sources`` key, or None

    Returns:
        The configured sources, empty when nothing is configured
    """
    if not config:
        return []
    if not isinstance(config, list):
        raise ValueError(
            f"'sources' must be a list of source entries, got {type(config).__name__}"
        )
    return [build_source(entry) for entry in config]


def _registry() -> Dict[str, type]:
    """
    Map config type names to source classes.

    Imports are deferred to keep an optional dependency (boto3) out of the
    import path for anyone who is not using that source.
    """
    from .local import LocalSource
    from .s3 import S3Source

    return {
        LocalSource.type_name: LocalSource,
        S3Source.type_name: S3Source,
    }


class _LazyRegistry(dict):
    """Populates itself on first access so source modules load only when used."""

    def _load(self):
        if not super().__len__():
            self.update(_registry())
        return self

    def __getitem__(self, key):
        return dict.__getitem__(self._load(), key)

    def __contains__(self, key):
        return dict.__contains__(self._load(), key)

    def __iter__(self):
        return dict.__iter__(self._load())

    def __len__(self):
        return dict.__len__(self._load())

    def register(self, source_class: type) -> None:
        """
        Add a custom source type.

        Args:
            source_class: A Source subclass with a ``type_name``

        Example:
            >>> from ragwire.sources import REGISTRY, Source
            >>> class SharePointSource(Source):
            ...     type_name = "sharepoint"
            ...     def list_files(self): return []
            >>> REGISTRY.register(SharePointSource)
            >>> "sharepoint" in REGISTRY
            True
        """
        self._load()
        if not getattr(source_class, "type_name", None):
            raise ValueError(f"{source_class.__name__} needs a type_name to be registered")
        self[source_class.type_name] = source_class


REGISTRY = _LazyRegistry()
