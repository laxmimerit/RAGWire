"""
An S3 bucket or prefix as a source.

Objects are downloaded into a local cache directory and their local paths
returned, so ingestion, hashing and deduplication work exactly as they do for
a folder on disk.

Downloads are skipped when the cached copy already matches the object's size
and modification time, which is what makes a repeated sync cheap. Content
changes still reach the collection, because ingestion hashes file contents and
replaces chunks whose hash moved.

Needs boto3: ``pip install ragwire[s3]``. Credentials are resolved by boto3's
normal chain, so environment variables, a shared credentials file, or an
instance role all work without RAGWire knowing about them.
"""

import logging
from pathlib import Path
from typing import List, Optional

from .base import Source

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = ".ragwire_cache"


class S3Source(Source):
    """
    Objects under an S3 bucket and prefix.

    Attributes:
        bucket: Bucket name
        prefix: Key prefix to list under. Empty means the whole bucket.
        cache_dir: Where downloaded objects are kept

    Example:
        >>> source = S3Source(bucket="filings", prefix="2026/")  # doctest: +SKIP
        >>> source.list_files()                                  # doctest: +SKIP
        ['.ragwire_cache/filings/2026/apple_10k.pdf']
    """

    type_name = "s3"

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        cache_dir: str = DEFAULT_CACHE_DIR,
        region: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        extensions: Optional[List[str]] = None,
        name: str = "",
        **_ignored,
    ):
        super().__init__(name=name or f"s3://{bucket}/{prefix}", extensions=extensions)

        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "The s3 source requires boto3. Install it with: pip install ragwire[s3]"
            ) from exc

        self.bucket = bucket
        self.prefix = prefix
        self.cache_dir = Path(cache_dir) / bucket

        client_kwargs = {}
        if region:
            client_kwargs["region_name"] = region
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url
        if aws_access_key_id and aws_secret_access_key:
            client_kwargs["aws_access_key_id"] = aws_access_key_id
            client_kwargs["aws_secret_access_key"] = aws_secret_access_key

        self.client = boto3.client("s3", **client_kwargs)

    def list_files(self) -> List[str]:
        """
        Download every matching object and return the local paths.

        Returns:
            Local paths of the cached objects, sorted by key

        Raises:
            RuntimeError: If the bucket cannot be listed. A listing failure is
                raised rather than returning nothing, because an empty listing
                would be read as "every object was deleted" and would empty the
                collection.
        """
        try:
            objects = self._list_objects()
        except Exception as exc:
            raise RuntimeError(
                f"Could not list s3://{self.bucket}/{self.prefix}: {exc}"
            ) from exc

        paths = []
        downloaded = 0

        for key, size, modified in objects:
            local_path = self.cache_dir / key
            if self._needs_download(local_path, size, modified):
                local_path.parent.mkdir(parents=True, exist_ok=True)
                self.client.download_file(self.bucket, key, str(local_path))
                downloaded += 1
            paths.append(str(local_path))

        logger.info(
            f"{self.name}: {len(paths)} object(s), {downloaded} downloaded, "
            f"{len(paths) - downloaded} already cached"
        )
        return paths

    def _list_objects(self):
        """List matching objects as (key, size, last_modified) tuples."""
        paginator = self.client.get_paginator("list_objects_v2")
        results = []

        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # A "folder" in S3 is a zero-byte key ending in a slash.
                if key.endswith("/"):
                    continue
                if not self.matches_extension(key):
                    continue
                results.append((key, obj["Size"], obj["LastModified"]))

        return sorted(results)

    @staticmethod
    def _needs_download(local_path: Path, size: int, modified) -> bool:
        """
        Whether the cached copy is stale.

        Size plus modification time is enough here. It can miss an edit that
        preserves both, but ingestion hashes file contents independently, so
        the worst case is a stale cache entry rather than stale chunks.
        """
        if not local_path.exists():
            return True

        stat = local_path.stat()
        if stat.st_size != size:
            return True

        return stat.st_mtime < modified.timestamp()
