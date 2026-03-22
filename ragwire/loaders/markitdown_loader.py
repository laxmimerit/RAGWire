"""
MarkItDown loader for document extraction.

Converts various document formats (PDF, DOCX, XLSX, etc.) to markdown
using the MarkItDown library for unified text extraction.
"""

import logging
from pathlib import Path
from typing import Union, Optional, List

logger = logging.getLogger(__name__)


class MarkItDownLoader:
    """
    Load and convert documents to markdown using MarkItDown.

    Supports multiple document formats including PDF, DOCX, XLSX, PPTX,
    and various image formats. Extracts text content in a flattened
    markdown format suitable for RAG processing.

    Attributes:
        md: MarkItDown instance for conversion

    Example:
        >>> loader = MarkItDownLoader()
        >>> result = loader.load("document.pdf")
        >>> print(result.text_content)
    """

    def __init__(self):
        """Initialize the MarkItDown loader."""
        try:
            from markitdown import MarkItDown

            self.md = MarkItDown()
            self.available = True
        except ImportError:
            self.md = None
            self.available = False
            logger.warning(
                "MarkItDown not installed. Install with: pip install markitdown"
            )

    def load(self, file_path: Union[str, Path]) -> dict:
        """
        Load and convert a single document to markdown.

        Args:
            file_path: Path to the document file

        Returns:
            Dictionary containing:
                - text_content: Extracted text as markdown
                - file_name: Original filename
                - file_type: File extension
                - success: Whether extraction succeeded

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If MarkItDown is not available

        Example:
            >>> result = loader.load("document.pdf")
            >>> print(result['text_content'])
            >>> print(result['file_name'])
        """
        if not self.available:
            raise ValueError(
                "MarkItDown is not available. Install it with: pip install markitdown"
            )

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            result = self.md.convert(str(file_path))

            return {
                "text_content": result.text_content,
                "file_name": file_path.name,
                "file_type": file_path.suffix.lower().lstrip("."),
                "success": True,
                "error": None,
            }
        except Exception as e:
            logger.error(f"Failed to convert {file_path}: {e}")
            return {
                "text_content": "",
                "file_name": file_path.name,
                "file_type": file_path.suffix.lower().lstrip("."),
                "success": False,
                "error": str(e),
            }

    def load_batch(self, file_paths: List[Union[str, Path]]) -> List[dict]:
        """
        Load and convert multiple documents.

        Args:
            file_paths: List of file paths to convert

        Returns:
            List of conversion results

        Example:
            >>> results = loader.load_batch(["doc1.pdf", "doc2.pdf"])
            >>> for result in results:
            ...     if result['success']:
            ...         process(result['text_content'])
        """
        results = []
        for file_path in file_paths:
            result = self.load(file_path)
            results.append(result)
        return results

    def load_directory(
        self,
        directory: Union[str, Path],
        extensions: Optional[List[str]] = None,
        recursive: bool = False,
    ) -> List[dict]:
        """
        Load all documents from a directory.

        Args:
            directory: Path to directory to scan
            extensions: List of file extensions to include (e.g., ['.pdf', '.docx'])
                       If None, loads all supported formats
            recursive: Whether to scan subdirectories

        Returns:
            List of conversion results

        Example:
            >>> results = loader.load_directory(
            ...     "data/documents",
            ...     extensions=['.pdf', '.docx'],
            ...     recursive=True
            ... )
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Default supported extensions
        if extensions is None:
            extensions = [".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".md", ".html"]

        # Find all matching files
        if recursive:
            file_paths = []
            for ext in extensions:
                file_paths.extend(directory.rglob(f"*{ext}"))
        else:
            file_paths = []
            for ext in extensions:
                file_paths.extend(directory.glob(f"*{ext}"))

        # Remove duplicates and sort
        file_paths = sorted(set(file_paths))

        # Load all files
        return self.load_batch(file_paths)
