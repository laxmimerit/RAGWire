"""Smoke tests — verify all public API symbols import correctly."""

import pytest


def test_package_imports():
    import ragwire
    assert ragwire.__version__
    assert ragwire.__author__ == "KGP Talkie Private Limited"


def test_public_api():
    from ragwire import (
        Config,
        RAGWire,
        DocumentMetadata,
        MetadataExtractor,
        MarkItDownLoader,
        get_splitter,
        get_markdown_splitter,
        get_code_splitter,
        sha256_text,
        sha256_file_from_path,
        sha256_chunk,
        get_embedding,
        QdrantStore,
        get_retriever,
        hybrid_search,
        mmr_search,
        setup_logging,
        get_logger,
    )


def test_splitters():
    from ragwire import get_splitter, get_markdown_splitter, get_code_splitter

    splitter = get_splitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split_text("word " * 100)
    assert len(chunks) > 1

    md_splitter = get_markdown_splitter(chunk_size=100, chunk_overlap=20)
    assert md_splitter is not None

    code_splitter = get_code_splitter(chunk_size=200, chunk_overlap=20)
    assert code_splitter is not None


def test_hashing():
    from ragwire import sha256_text, sha256_chunk

    h = sha256_text("hello world")
    assert len(h) == 64

    ch = sha256_chunk("chunk_0", "some content")
    assert len(ch) == 64


def test_document_metadata():
    from ragwire import DocumentMetadata

    meta = DocumentMetadata(
        source="/data/test.pdf",
        file_name="test.pdf",
        file_type="pdf",
        file_hash="abc" * 21 + "a",
        chunk_id="abc_0",
        chunk_hash="def" * 21 + "a",
    )
    assert meta.file_name == "test.pdf"
    assert meta.company_name is None
