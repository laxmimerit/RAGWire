"""
A folder on disk as a source.

The simplest connector, and the one most syncs actually use. It exists mainly
so that local folders reconcile the same way remote buckets do: a file deleted
from the folder is removed from the collection rather than answering queries
forever.
"""

import logging
from pathlib import Path
from typing import List, Optional

from .base import Source

logger = logging.getLogger(__name__)


class LocalSource(Source):
    """
    Files in a local directory, or a single file.

    Attributes:
        path: Directory or file to read
        recursive: Whether to descend into subdirectories

    Example:
        >>> source = LocalSource(path="./documents", recursive=True)
        >>> source.list_files()  # doctest: +SKIP
        ['documents/a.pdf', 'documents/reports/b.pdf']
    """

    type_name = "local"

    def __init__(
        self,
        path: str,
        recursive: bool = False,
        extensions: Optional[List[str]] = None,
        name: str = "",
        **_ignored,
    ):
        super().__init__(name=name or f"local:{path}", extensions=extensions)
        self.path = Path(path)
        self.recursive = recursive

    def list_files(self) -> List[str]:
        """
        Return every matching file under the configured path.

        Returns:
            File paths in sorted order, so a sync run is reproducible

        Raises:
            FileNotFoundError: If the path does not exist. A missing folder is
                an error rather than an empty listing, because treating it as
                empty would delete every document the folder had contributed.
        """
        if not self.path.exists():
            raise FileNotFoundError(
                f"Source path does not exist: {self.path}. Fix the path or "
                f"remove the source from your config, since an unreadable "
                f"folder cannot be told apart from an emptied one."
            )

        if self.path.is_file():
            return [str(self.path)] if self.matches_extension(str(self.path)) else []

        pattern = "**/*" if self.recursive else "*"
        files = [
            str(p)
            for p in sorted(self.path.glob(pattern))
            if p.is_file() and self.matches_extension(str(p))
        ]

        logger.info(f"{self.name}: found {len(files)} file(s)")
        return files
