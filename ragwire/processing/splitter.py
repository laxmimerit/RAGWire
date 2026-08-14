"""
Text splitting utilities for RAG pipeline.

Provides configurable text splitters using RecursiveCharacterTextSplitter
from LangChain for chunking documents into appropriate sizes for
embedding and retrieval, plus a PageSplitter that produces exactly one
chunk per page for page-wise ingestion.

Reference: https://docs.langchain.com/oss/python/integrations/splitters
"""

import re
from typing import Any, Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

#: Marker that separates pages in text-form files (.txt, .md) under the
#: "page" splitter strategy. Overridable via splitter.page_marker in config.
PAGE_MARKER_DEFAULT = "<!-- pagebreak -->"


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


class PageSplitter:
    """
    Split a document into exactly one chunk per page.

    Unlike the character splitters above, a page is never sub-split, merged
    or truncated: whatever a page holds becomes one chunk, however large or
    small, and there is no overlap across page boundaries. Retrieval results
    and citations therefore map one-to-one onto pages a reader can open.

    Where the pages come from depends on the input:

    - Loader-provided pages (PDF pages, PPTX slides) pass through unchanged.
    - Text containing the page marker is split on that marker.
    - Markdown/HTML without a marker is split on headings, and the heading
      text becomes the page label.
    - Anything else becomes a single page.

    Whitespace-only pages are dropped, since there is nothing to embed, but
    page numbering is preserved: a chunk's ``page_number`` always points at
    the real page position in the source document.

    Example:
        >>> splitter = PageSplitter()
        >>> pages = splitter.split("intro <!-- pagebreak --> details")
        >>> [(p["page_number"], p["text"]) for p in pages]
        [(1, 'intro'), (2, 'details')]
    """

    #: File types split on headings when the text contains no page marker
    HEADING_FILE_TYPES = {"md", "markdown", "html", "htm"}

    def __init__(
        self,
        page_marker: str = PAGE_MARKER_DEFAULT,
        max_heading_level: int = 2,
    ):
        """
        Args:
            page_marker: Marker string that separates pages in text-form
                files (default: "<!-- pagebreak -->")
            max_heading_level: Deepest markdown heading level that starts a
                new page when splitting by headings (default: 2, i.e. # and ##)
        """
        self.page_marker = page_marker
        self.max_heading_level = max_heading_level
        self._heading_re = re.compile(
            rf"^#{{1,{max_heading_level}}}\s+(.+?)\s*$", re.MULTILINE
        )

    def split(
        self,
        text: str,
        pages: Optional[List[Dict[str, Any]]] = None,
        file_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Split a document into page chunks.

        Args:
            text: Full document text (used when no loader pages are given)
            pages: Loader-provided pages as dicts with ``page_number``,
                ``page_label`` and ``text`` keys (PDF pages, PPTX slides)
            file_type: File extension without the dot; enables heading-wise
                splitting for markdown/HTML

        Returns:
            List of dicts with ``text``, ``page_number`` (1-based, position in
            the source document), ``page_label`` (heading, slide title or PDF
            page label, else None) and ``page_total`` (pages in the document,
            counting dropped empty ones).

        Example:
            >>> splitter = PageSplitter()
            >>> splitter.split("# A\\nbody", file_type="md")[0]["page_label"]
            'A'
        """
        if pages is not None:
            return self._normalize(pages, page_total=len(pages))

        if not text or not text.strip():
            return []

        if self.page_marker and self.page_marker in text:
            segments = text.split(self.page_marker)
            numbered = [
                {"page_number": i + 1, "page_label": None, "text": segment}
                for i, segment in enumerate(segments)
            ]
            return self._normalize(numbered, page_total=len(segments))

        if (file_type or "").lower() in self.HEADING_FILE_TYPES:
            sections = self._split_by_headings(text)
            if sections is not None:
                return sections

        return [
            {
                "page_number": 1,
                "page_label": None,
                "text": text.strip(),
                "page_total": 1,
            }
        ]

    @staticmethod
    def _normalize(
        pages: List[Dict[str, Any]], page_total: int
    ) -> List[Dict[str, Any]]:
        """Strip page texts, drop empty pages, stamp the page total."""
        normalized = []
        for page in pages:
            page_text = (page.get("text") or "").strip()
            if not page_text:
                continue
            normalized.append(
                {
                    "page_number": page.get("page_number"),
                    "page_label": page.get("page_label"),
                    "text": page_text,
                    "page_total": page_total,
                }
            )
        return normalized

    def _split_by_headings(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """
        Split markdown text into sections at headings.

        Each heading starts a new page whose label is the heading text; the
        heading line stays inside the page so embeddings keep that context.
        Content before the first heading becomes an unlabeled first page.

        Returns None when the text has no headings, so the caller can fall
        back to single-page behaviour.
        """
        matches = list(self._heading_re.finditer(text))
        if not matches:
            return None

        sections: List[Dict[str, Any]] = []
        if matches[0].start() > 0:
            sections.append({"page_label": None, "text": text[: matches[0].start()]})

        for index, match in enumerate(matches):
            end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(text)
            )
            sections.append(
                {"page_label": match.group(1).strip(), "text": text[match.start(): end]}
            )

        for number, section in enumerate(sections, start=1):
            section["page_number"] = number

        return self._normalize(sections, page_total=len(sections))
