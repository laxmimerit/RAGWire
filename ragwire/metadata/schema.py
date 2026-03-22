"""
Metadata schema definitions for RAG documents.

Defines the structure for document metadata using Pydantic models.
Supports finance-specific metadata and general document properties.
"""

from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """
    Metadata schema for RAG documents.

    Finance-specific fields:
        company_name: Company name (e.g., "apple", "microsoft")
        doc_type: Document type (10-K, 10-Q, 8-K)
        fiscal_quarter: Fiscal quarter (q1, q2, q3, q4)
        fiscal_year: Fiscal year(s) covered by the document

    File-level fields:
        source: Source path or URL of the document
        file_name: Original filename
        file_type: File extension/type
        file_hash: SHA256 hash of file content for deduplication

    Chunk-level fields:
        chunk_id: Unique identifier for this chunk
        chunk_hash: SHA256 hash of chunk content
        chunk_index: Position of this chunk within the document
        total_chunks: Total number of chunks in the document

    Timestamps:
        created_at: ISO format timestamp of ingestion
    """

    # Finance-specific metadata
    company_name: Optional[str] = Field(default=None, description="Company name (normalized to lowercase)")
    doc_type: Optional[str] = Field(default=None, description="Document type (10-K, 10-Q, 8-K)")
    fiscal_quarter: Optional[str] = Field(default=None, description="Fiscal quarter (q1-q4)")
    fiscal_year: Optional[List[int]] = Field(default=None, description="Fiscal year(s) covered")

    # File-level metadata
    source: str = Field(..., description="Source file path")
    file_name: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File extension/type")
    file_hash: str = Field(..., description="SHA256 hash of file content")

    # Chunk-level metadata
    chunk_id: str = Field(..., description="Unique chunk identifier")
    chunk_hash: str = Field(..., description="SHA256 hash of chunk content")
    chunk_index: int = Field(default=0, description="Position of chunk within the document")
    total_chunks: int = Field(default=1, description="Total chunks in the document")

    # Timestamps
    created_at: Optional[str] = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="ISO format timestamp of ingestion",
    )

    model_config = {
        "arbitrary_types_allowed": True,
        "json_schema_extra": {
            "example": {
                "company_name": "apple",
                "doc_type": "10-K",
                "fiscal_year": [2025],
                "source": "/data/Apple_10k_2025.pdf",
                "file_name": "Apple_10k_2025.pdf",
                "file_type": "pdf",
                "file_hash": "abc123...",
                "chunk_id": "abc123_0",
                "chunk_hash": "def456...",
                "chunk_index": 0,
                "total_chunks": 42,
            }
        },
    }
