"""
Regression tests for bugs found in the v1.3.2 audit.

Each test names the bug ID from PRODUCTION_READINESS.md so a failure points
straight at the original defect. These use an in-memory Qdrant client and stub
objects — no server, no LLM, no network.
"""

import logging

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from ragwire import RAGWire, DocumentMetadata, setup_logging
from ragwire.metadata.extractor import FinancialMetadata, MetadataExtractor
from ragwire.processing.hashing import compare_hashes
from ragwire.vectorstores.qdrant_store import QdrantStore


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def make_store(collection: str = "test_docs") -> QdrantStore:
    """Build a QdrantStore backed by an in-memory client, bypassing __init__."""
    store = object.__new__(QdrantStore)
    store.client = QdrantClient(location=":memory:")
    store.collection_name = collection
    store.embedding = None
    store.config = {}
    store.client.create_collection(
        collection_name=collection,
        vectors_config=rest.VectorParams(size=3, distance=rest.Distance.COSINE),
    )
    return store


def add_chunks(store: QdrantStore, file_hash: str, count: int, total_chunks: int):
    """Upsert `count` chunks that each claim the document has `total_chunks`."""
    store.client.upsert(
        collection_name=store.collection_name,
        points=[
            rest.PointStruct(
                id=abs(hash((file_hash, i))) % (10**12),
                vector=[0.1, 0.2, 0.3],
                payload={
                    "metadata": {
                        "file_hash": file_hash,
                        "chunk_index": i,
                        "total_chunks": total_chunks,
                    }
                },
            )
            for i in range(count)
        ],
    )


# --------------------------------------------------------------------------- #
# B1 — hashlib.compare_digest does not exist
# --------------------------------------------------------------------------- #

def test_b1_compare_hashes_does_not_raise():
    assert compare_hashes("a" * 64, "a" * 64) is True
    assert compare_hashes("a" * 64, "b" * 64) is False


# --------------------------------------------------------------------------- #
# B3 — partial ingest must not look complete
# --------------------------------------------------------------------------- #

def test_b3_absent_file_reports_absent():
    store = make_store()
    assert store.get_ingest_status("missing")[0] == "absent"


def test_b3_partial_ingest_is_not_mistaken_for_complete():
    store = make_store()
    add_chunks(store, "h1", count=150, total_chunks=400)

    status, stored, expected = store.get_ingest_status("h1")

    assert status == "partial"
    assert (stored, expected) == (150, 400)
    # The old code only asked "does any chunk exist?", which was True here and
    # caused the file to be skipped forever at 150/400 chunks.
    assert store.file_hash_exists("h1") is True


def test_b3_complete_ingest_reports_complete():
    store = make_store()
    add_chunks(store, "h2", count=4, total_chunks=4)
    assert store.get_ingest_status("h2")[0] == "complete"


def test_b3_delete_by_file_hash_clears_only_that_file():
    store = make_store()
    add_chunks(store, "keep", count=3, total_chunks=3)
    add_chunks(store, "drop", count=5, total_chunks=5)

    removed = store.delete_by_file_hash("drop")

    assert removed == 5
    assert store.count_by_file_hash("drop") == 0
    assert store.count_by_file_hash("keep") == 3


def test_b3_legacy_chunks_without_total_chunks_count_as_complete():
    """Data written by an older RAGWire has no marker — do not re-ingest it."""
    store = make_store()
    store.client.upsert(
        collection_name=store.collection_name,
        points=[
            rest.PointStruct(
                id=1, vector=[0.1, 0.2, 0.3],
                payload={"metadata": {"file_hash": "legacy"}},
            )
        ],
    )
    assert store.get_ingest_status("legacy")[0] == "complete"


# --------------------------------------------------------------------------- #
# B6 — payload index type must follow the schema, not a hardcoded name list
# --------------------------------------------------------------------------- #

def test_b6_builtin_schema_types():
    types = MetadataExtractor._infer_field_types(FinancialMetadata)
    assert types["fiscal_year"] == "integer"
    assert types["company_name"] == "keyword"


def test_b6_custom_integer_field_is_not_indexed_as_keyword():
    model = MetadataExtractor._build_schema_model([
        {"name": "publication_year", "description": "year", "type": "integer"},
        {"name": "authors", "description": "authors", "type": "list"},
        {"name": "topic", "description": "topic"},
    ])

    types = MetadataExtractor._infer_field_types(model)

    # This was the bug: publication_year fell outside the hardcoded
    # _INTEGER_FIELDS set and got a KEYWORD index, which does not index
    # numeric values — facets came back empty and filters matched nothing.
    assert types["publication_year"] == "integer"
    assert types["authors"] == "keyword"
    assert types["topic"] == "keyword"


class _RecordingClient:
    """Captures create_payload_index calls. Local Qdrant ignores payload
    indexes entirely, so assert on what we ask for rather than on the result."""

    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def create_payload_index(self, collection_name, field_name, field_schema):
        self.calls.append((field_name, field_schema))
        if self.error:
            raise self.error


def test_b6_indexes_are_created_with_the_right_schema_type():
    store = object.__new__(QdrantStore)
    store.collection_name = "c"
    store.client = _RecordingClient()

    store.create_payload_indexes(
        ["publication_year", "topic", "chunk_index"],
        field_types={"publication_year": "integer", "topic": "keyword"},
    )

    requested = dict(store.client.calls)
    assert requested["metadata.publication_year"] == rest.PayloadSchemaType.INTEGER
    assert requested["metadata.topic"] == rest.PayloadSchemaType.KEYWORD
    # System fields keep their known types without being passed in
    assert requested["metadata.chunk_index"] == rest.PayloadSchemaType.INTEGER


def test_b6_unknown_fields_default_to_keyword():
    store = object.__new__(QdrantStore)
    store.collection_name = "c"
    store.client = _RecordingClient()

    store.create_payload_indexes(["mystery"])

    assert dict(store.client.calls)["metadata.mystery"] == rest.PayloadSchemaType.KEYWORD


# --------------------------------------------------------------------------- #
# B7 — index failures must be logged, not swallowed
# --------------------------------------------------------------------------- #

def test_b7_real_failure_is_logged(caplog):
    store = object.__new__(QdrantStore)
    store.collection_name = "c"
    store.client = _RecordingClient(error=Exception("401 unauthorized"))

    with caplog.at_level(logging.WARNING, logger="ragwire.vectorstores.qdrant_store"):
        store.create_payload_indexes(["some_field"])

    # The old code assumed every exception meant "already exists" and dropped it,
    # so auth and connection failures left facets silently empty.
    assert any("Could not create payload index" in r.message for r in caplog.records)


def test_b7_already_exists_is_not_logged_as_a_warning(caplog):
    store = object.__new__(QdrantStore)
    store.collection_name = "c"
    store.client = _RecordingClient(error=Exception("Index already exists"))

    with caplog.at_level(logging.WARNING, logger="ragwire.vectorstores.qdrant_store"):
        store.create_payload_indexes(["some_field"])

    assert not caplog.records


def test_b7_already_exists_is_recognised():
    assert QdrantStore._is_already_exists(Exception("Index already exists")) is True
    assert QdrantStore._is_already_exists(Exception("connection refused")) is False


# --------------------------------------------------------------------------- #
# B8 — field discovery must not depend on a single sampled point
# --------------------------------------------------------------------------- #

def test_b8_metadata_keys_union_across_points():
    store = make_store()
    store.client.upsert(
        collection_name=store.collection_name,
        points=[
            # An 8-K: fiscal_quarter is null, so the key is absent entirely
            rest.PointStruct(
                id=1, vector=[0.1, 0.2, 0.3],
                payload={"metadata": {"file_hash": "a", "doc_type": "8-k"}},
            ),
            rest.PointStruct(
                id=2, vector=[0.1, 0.2, 0.3],
                payload={"metadata": {"file_hash": "b", "doc_type": "10-q",
                                      "fiscal_quarter": "q1"}},
            ),
        ],
    )

    keys = store.get_metadata_keys()

    # Sampling one point could return either document's keys; the union must
    # always include fiscal_quarter or it never gets an index.
    assert "fiscal_quarter" in keys
    assert "doc_type" in keys


def test_b8_empty_collection_returns_empty_list():
    assert make_store().get_metadata_keys() == []


# --------------------------------------------------------------------------- #
# B9 — caller-supplied filters must be normalized like extracted ones
# --------------------------------------------------------------------------- #

def test_b9_caller_filters_are_lowercased():
    normalized = RAGWire._normalize_filters({
        "company_name": "Apple Inc.",
        "fiscal_quarter": ["Q1", "Q2"],
        "fiscal_year": 2024,
    })

    # Stored values are lowercase and Qdrant MatchValue is exact — without this
    # an agent passing "Apple Inc." matched zero points.
    assert normalized == {
        "company_name": "apple inc.",
        "fiscal_quarter": ["q1", "q2"],
        "fiscal_year": 2024,
    }


def test_b9_normalization_trims_whitespace_and_keeps_non_strings():
    assert RAGWire._normalize_filters({"a": "  X  ", "b": 7, "c": None}) == {
        "a": "x", "b": 7, "c": None,
    }


# --------------------------------------------------------------------------- #
# B12 — console handler must honour the configured level
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("level", ["DEBUG", "WARNING"])
def test_b12_console_handler_uses_configured_level(level):
    logger = setup_logging(log_level=level, console_output=True)
    handler = logger.handlers[0]
    assert handler.level == getattr(logging, level)


# --------------------------------------------------------------------------- #
# B14 — DocumentMetadata must agree with what the LLM actually produces
# --------------------------------------------------------------------------- #

def test_b14_fiscal_year_type_matches_extraction_schema():
    doc_annotation = DocumentMetadata.model_fields["fiscal_year"].annotation
    llm_annotation = FinancialMetadata.model_fields["fiscal_year"].annotation
    assert doc_annotation == llm_annotation


def test_b14_metadata_status_defaults_to_ok():
    meta = DocumentMetadata(
        source="/data/t.pdf", file_name="t.pdf", file_type="pdf",
        file_hash="a" * 64, chunk_id="a_0", chunk_hash="b" * 64,
    )
    assert meta.metadata_status == "ok"


def test_b14_custom_schema_fields_pass_through():
    meta = DocumentMetadata(
        source="/data/t.pdf", file_name="t.pdf", file_type="pdf",
        file_hash="a" * 64, chunk_id="a_0", chunk_hash="b" * 64,
        publication_year=2023, authors=["ada"],
    )
    assert meta.publication_year == 2023


# --------------------------------------------------------------------------- #
# B13 — embedding dimension mismatch must fail at init with a clear message
# --------------------------------------------------------------------------- #

class _StubEmbedding:
    def __init__(self, size):
        self.size = size

    def embed_query(self, text):
        return [0.0] * self.size


def _pipeline_with(stored_size, model_size):
    rag = object.__new__(RAGWire)
    rag.config = {"embeddings": {"provider": "ollama", "model": "test-model"}}
    rag.embedding = _StubEmbedding(model_size)

    wrapper = object.__new__(QdrantStore)
    wrapper.get_vector_size = lambda *a, **k: stored_size
    rag.vectorstore_wrapper = wrapper
    return rag


def test_b13_dimension_mismatch_raises_actionable_error():
    rag = _pipeline_with(stored_size=768, model_size=1024)

    with pytest.raises(ValueError) as excinfo:
        rag._check_embedding_dimension("financial_docs")

    message = str(excinfo.value)
    assert "768" in message and "1024" in message
    assert "force_recreate" in message


def test_b13_matching_dimension_passes():
    _pipeline_with(stored_size=768, model_size=768)._check_embedding_dimension("c")


def test_b13_unknown_stored_size_is_not_an_error():
    _pipeline_with(stored_size=None, model_size=768)._check_embedding_dimension("c")


# --------------------------------------------------------------------------- #
# B15 — extraction limit must match what the documentation promises
# --------------------------------------------------------------------------- #

def test_b15_default_char_limit_is_10k():
    assert MetadataExtractor.DEFAULT_CHAR_LIMIT == 10000


# --------------------------------------------------------------------------- #
# B11 — the tracked config must not wipe the collection on startup
# --------------------------------------------------------------------------- #

def test_b11_shipped_config_does_not_force_recreate():
    import yaml
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for name in ("config.yaml", "config.example.yaml"):
        config = yaml.safe_load((root / name).read_text(encoding="utf-8"))
        assert config["vectorstore"]["force_recreate"] is False, (
            f"{name} would delete the user's collection on every startup"
        )
