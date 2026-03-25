"""
Text splitting utilities for RAG pipeline.

Provides configurable text splitters using RecursiveCharacterTextSplitter
from LangChain for chunking documents into appropriate sizes for
embedding and retrieval.

Reference: https://docs.langchain.com/oss/python/integrations/splitters
"""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_splitter(
    chunk_size: int = 10000,
    chunk_overlap: int = 2000,
    separators: List[str] = None,
) -> RecursiveCharacterTextSplitter:
    """
    Get a RecursiveCharacterTextSplitter with configured parameters.

    Uses LangChain's RecursiveCharacterTextSplitter which splits text
    by trying different separators in order until chunk size is achieved.

    Args:
        chunk_size: Maximum size of each chunk (default: 10000)
        chunk_overlap: Number of characters to overlap between chunks (default: 2000, 20%)
        separators: List of separators to try in order. If None, uses default:
                   ["\\n\\n", "\\n", " ", ""]

    Returns:
        Configured RecursiveCharacterTextSplitter instance

    Example:
        >>> splitter = get_splitter(chunk_size=500, chunk_overlap=100)
        >>> chunks = splitter.split_text(long_document)

        >>> # Custom separators
        >>> splitter = get_splitter(
        ...     separators=["\\n\\n", "\\n", " ", ""]
        ... )
    """
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        keep_separator=False,
        add_start_index=True,
        strip_whitespace=True,
    )


def get_markdown_splitter(
    chunk_size: int = 10000,
    chunk_overlap: int = 2000,
) -> RecursiveCharacterTextSplitter:
    """
    Get a RecursiveCharacterTextSplitter optimized for markdown documents.

    Splits on markdown headers and structural elements to preserve
    document hierarchy and context.

    Args:
        chunk_size: Maximum size of each chunk (default: 10000)
        chunk_overlap: Number of characters to overlap (default: 2000, 20%)

    Returns:
        Markdown-optimized RecursiveCharacterTextSplitter

    Example:
        >>> splitter = get_markdown_splitter(chunk_size=2000)
        >>> chunks = splitter.split_text(markdown_content)
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Markdown-specific separators (tries in order)
        separators=[
            "\n\n## ",  # Level 2 headers
            "\n\n### ",  # Level 3 headers
            "\n\n#### ",  # Level 4 headers
            "\n\n",  # Paragraph breaks
            "\n",  # Line breaks
            " ",  # Words
            "",  # Characters
        ],
        length_function=len,
        keep_separator=False,
        add_start_index=True,
        strip_whitespace=True,
    )


def get_code_splitter(
    chunk_size: int = 10000,
    chunk_overlap: int = 2000,
) -> RecursiveCharacterTextSplitter:
    """
    Get a RecursiveCharacterTextSplitter optimized for code documents.

    Splits on function definitions, class definitions, and comments
    to preserve code structure and context.

    Args:
        chunk_size: Maximum size of each chunk (default: 10000)
        chunk_overlap: Number of characters to overlap (default: 2000, 20%)

    Returns:
        Code-optimized RecursiveCharacterTextSplitter

    Example:
        >>> splitter = get_code_splitter(chunk_size=1000)
        >>> chunks = splitter.split_text(code_content)
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Code-specific separators (tries in order)
        separators=[
            "\n\nclass ",  # Class definitions
            "\n\ndef ",  # Function definitions
            "\n\n#",  # Comments
            "\n\n",  # Paragraph breaks
            "\n",  # Line breaks
            " ",  # Words
            "",  # Characters
        ],
        length_function=len,
        keep_separator=False,
        add_start_index=True,
        strip_whitespace=True,
    )
