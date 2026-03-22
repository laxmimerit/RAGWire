"""
Hashing utilities for document deduplication.

Provides SHA256 hashing functions for files and text content
to enable deduplication and change detection in the RAG pipeline.
"""

import hashlib
from pathlib import Path
from typing import Union


def sha256_text(text: str) -> str:
    """
    Calculate SHA256 hash of text content.

    Args:
        text: Text string to hash

    Returns:
        Hexadecimal SHA256 hash string (64 characters)

    Example:
        >>> hash = sha256_text("Hello, World!")
        >>> print(hash)
        '315f5bdb76d078c43b8ac0064e4a0164612b1fce77c869345bfc94c75894edd3'
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(content: bytes) -> str:
    """
    Calculate SHA256 hash of binary file content.

    Args:
        content: Binary content of the file

    Returns:
        Hexadecimal SHA256 hash string (64 characters)

    Example:
        >>> with open('document.pdf', 'rb') as f:
        ...     file_hash = sha256_file(f.read())
    """
    return hashlib.sha256(content).hexdigest()


def sha256_file_from_path(file_path: Union[str, Path]) -> str:
    """
    Calculate SHA256 hash of a file from its path.

    Reads the file in chunks to handle large files efficiently.

    Args:
        file_path: Path to the file to hash

    Returns:
        Hexadecimal SHA256 hash string (64 characters)

    Raises:
        FileNotFoundError: If file doesn't exist

    Example:
        >>> file_hash = sha256_file_from_path('/path/to/document.pdf')
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    sha256_hash = hashlib.sha256()

    # Read file in chunks for memory efficiency
    chunk_size = 8192  # 8KB chunks
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest()


def sha256_chunk(chunk_id: str, content: str) -> str:
    """
    Calculate SHA256 hash for a specific chunk.

    Combines chunk ID with content to create a unique hash
    that can detect both content changes and chunk reordering.

    Args:
        chunk_id: Unique identifier for the chunk
        content: Text content of the chunk

    Returns:
        Hexadecimal SHA256 hash string

    Example:
        >>> chunk_hash = sha256_chunk("chunk_001", "Document content here")
    """
    combined = f"{chunk_id}:{content}"
    return sha256_text(combined)


def compare_hashes(hash1: str, hash2: str) -> bool:
    """
    Compare two SHA256 hashes for equality.

    Uses constant-time comparison to prevent timing attacks.

    Args:
        hash1: First hash string
        hash2: Second hash string

    Returns:
        True if hashes are equal, False otherwise
    """
    return hashlib.compare_digest(hash1.encode("utf-8"), hash2.encode("utf-8"))
