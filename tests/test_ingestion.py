"""
Tests for Phase 2 ingestion features: retries, batching, concurrency,
change detection and chunk deduplication.

No server, no LLM, no network. The vector store is an in-memory Qdrant
client and writes go through a recording stub.
"""

import pytest
from qdrant_client.http import models as rest

from ragwire import RAGWire
from ragwire.utils.retry import retry_call
from ragwire.vectorstores.qdrant_store import QdrantStore

from .test_regressions import make_store


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Backoff delays make tests slow and prove nothing."""
    monkeypatch.setattr("ragwire.utils.retry.time.sleep", lambda s: None)


# --------------------------------------------------------------------------- #
# G3: retry with backoff
# --------------------------------------------------------------------------- #

def test_retry_returns_once_the_call_succeeds():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("boom")
        return "ok"

    assert retry_call(flaky, attempts=3) == "ok"
    assert calls["n"] == 3


def test_retry_reraises_after_exhausting_attempts():
    with pytest.raises(ConnectionError):
        retry_call(lambda: (_ for _ in ()).throw(ConnectionError("boom")), attempts=2)


def test_retry_does_not_retry_programming_errors():
    calls = {"n": 0}

    def broken():
        calls["n"] += 1
        raise TypeError("wrong argument type")

    with pytest.raises(TypeError):
        retry_call(broken, attempts=5)

    # Retrying a TypeError just delays the error the caller needs to see.
    assert calls["n"] == 1


def test_retry_single_attempt_disables_retrying():
    calls = {"n": 0}

    def always_fails():
        calls["n"] += 1
        raise ValueError("nope")

    with pytest.raises(ValueError):
        retry_call(always_fails, attempts=1)
    assert calls["n"] == 1


# --------------------------------------------------------------------------- #
# G13: batched writes
# --------------------------------------------------------------------------- #

class _RecordingVectorStore:
    """Records each add_documents batch; can fail the first N calls."""

    def __init__(self, fail_times=0):
        self.batches = []
        self.fail_times = fail_times
        self.calls = 0

    def add_documents(self, docs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError("qdrant unavailable")
        self.batches.append(list(docs))


def _pipeline_with_store(store, batch_size=10, retries=2):
    rag = object.__new__(RAGWire)
    rag.vectorstore = store
    rag._batch_size = batch_size
    rag._write_retries = retries
    return rag


def test_writes_are_split_into_batches():
    store = _RecordingVectorStore()
    rag = _pipeline_with_store(store, batch_size=10)

    rag._write_chunks([f"chunk{i}" for i in range(25)], "big.pdf")

    # A single add_documents call for a large document is one oversized request.
    assert [len(b) for b in store.batches] == [10, 10, 5]


def test_batch_size_larger_than_document_writes_once():
    store = _RecordingVectorStore()
    _pipeline_with_store(store, batch_size=64)._write_chunks(["a", "b"], "small.pdf")
    assert [len(b) for b in store.batches] == [2]


def test_a_transient_failure_retries_only_that_batch():
    store = _RecordingVectorStore(fail_times=1)
    rag = _pipeline_with_store(store, batch_size=10, retries=2)

    rag._write_chunks([f"chunk{i}" for i in range(15)], "doc.pdf")

    assert store.calls == 3           # first batch failed once, then both wrote
    assert [len(b) for b in store.batches] == [10, 5]


def test_write_raises_once_retries_are_exhausted():
    store = _RecordingVectorStore(fail_times=99)
    rag = _pipeline_with_store(store, batch_size=10, retries=1)

    with pytest.raises(ConnectionError):
        rag._write_chunks(["a"], "doc.pdf")
    assert store.calls == 2


# --------------------------------------------------------------------------- #
# G11: duplicate chunk removal
# --------------------------------------------------------------------------- #

def test_duplicate_chunks_are_dropped_keeping_first_occurrence():
    chunks = ["intro", "BOILERPLATE", "body", "BOILERPLATE", "tail"]
    assert RAGWire._drop_duplicate_chunks(chunks, "f.pdf") == [
        "intro", "BOILERPLATE", "body", "tail",
    ]


def test_dedup_ignores_surrounding_whitespace():
    assert RAGWire._drop_duplicate_chunks(["a", "  a  "], "f.pdf") == ["a"]


def test_dedup_keeps_distinct_chunks_untouched():
    chunks = ["a", "b", "c"]
    assert RAGWire._drop_duplicate_chunks(chunks, "f.pdf") == chunks


# --------------------------------------------------------------------------- #
# G4: replacing a changed document
# --------------------------------------------------------------------------- #

def _add(store, file_hash, source, count=2):
    store.client.upsert(
        collection_name=store.collection_name,
        points=[
            rest.PointStruct(
                id=abs(hash((file_hash, i))) % (10**12),
                vector=[0.1, 0.2, 0.3],
                payload={"metadata": {"file_hash": file_hash, "source": source,
                                      "total_chunks": count}},
            )
            for i in range(count)
        ],
    )


def test_delete_by_source_spares_the_new_version():
    store = make_store()
    _add(store, "old_hash", "/data/report.pdf")
    _add(store, "new_hash", "/data/report.pdf")
    _add(store, "other", "/data/untouched.pdf")

    removed = store.delete_by_source("/data/report.pdf", except_file_hash="new_hash")

    # Without this, an edited document is stored alongside its old version and
    # the stale text keeps surfacing in results.
    assert removed == 2
    assert store.count_by_file_hash("old_hash") == 0
    assert store.count_by_file_hash("new_hash") == 2
    assert store.count_by_file_hash("other") == 2


def test_delete_by_source_without_exception_removes_everything():
    store = make_store()
    _add(store, "h1", "/data/report.pdf")
    _add(store, "h2", "/data/report.pdf")

    assert store.delete_by_source("/data/report.pdf") == 4


def test_delete_by_source_on_unknown_source_is_a_noop():
    assert make_store().delete_by_source("/nope.pdf") == 0


# --------------------------------------------------------------------------- #
# G2: concurrent preparation
# --------------------------------------------------------------------------- #

def _pipeline_with_workers(workers):
    rag = object.__new__(RAGWire)
    rag._workers = workers
    rag._prepare_document = lambda path: {"file": path, "action": "write"}
    return rag


@pytest.mark.parametrize("workers", [1, 4])
def test_preparation_preserves_input_order(workers):
    rag = _pipeline_with_workers(workers)
    paths = [f"doc{i}.pdf" for i in range(20)]

    prepared = list(rag._prepare_documents(paths))

    # Results are written in input order regardless of completion order, so
    # stats and logs stay deterministic.
    assert [r["file"] for r in prepared] == paths


def test_single_worker_prepares_lazily():
    """With workers=1 progress should stream, not materialise up front."""
    rag = _pipeline_with_workers(1)
    seen = []
    rag._prepare_document = lambda path: (seen.append(path), {"file": path})[1]

    gen = rag._prepare_documents(["a.pdf", "b.pdf", "c.pdf"])
    next(gen)

    assert seen == ["a.pdf"]
