"""
Document sources for :meth:`RAGWire.sync`.

A source answers one question: which files should this collection contain
right now? Sync takes that answer and reconciles the collection against it,
ingesting what is new, replacing what changed, and removing what is gone.

.. code-block:: yaml

    sources:
      - type: local
        path: ./documents
        recursive: true
      - type: s3
        bucket: filings
        prefix: 2026/

Adding your own takes one class and one line:

.. code-block:: python

    from ragwire.sources import REGISTRY, Source

    class SharePointSource(Source):
        type_name = "sharepoint"

        def list_files(self):
            return download_everything_to_a_local_cache()

    REGISTRY.register(SharePointSource)
"""

from .base import REGISTRY, Source, build_source, build_sources
from .local import LocalSource

__all__ = [
    "Source",
    "LocalSource",
    "S3Source",
    "REGISTRY",
    "build_source",
    "build_sources",
]


def __getattr__(name):
    """Load S3Source lazily so boto3 is only imported when it is used."""
    if name == "S3Source":
        from .s3 import S3Source

        return S3Source
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
