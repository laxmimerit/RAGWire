"""
Tests for source connectors and sync reconciliation.

No network, no vector store. The store is a recording stub and sources are
real LocalSource instances over tmp_path, since the interesting behaviour is
in what sync decides to delete rather than in how files are read.
"""

import pytest

from ragwire import RAGWire
from ragwire.cli import build_parser
from ragwire.sources import REGISTRY, LocalSource, Source, build_source, build_sources


# --------------------------------------------------------------------------- #
# LocalSource
# --------------------------------------------------------------------------- #

def _touch(directory, *names):
    paths = []
    for name in names:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("content", encoding="utf-8")
        paths.append(path)
    return paths


def test_local_source_lists_matching_files(tmp_path):
    _touch(tmp_path, "a.pdf", "b.md", "c.exe")

    files = LocalSource(path=str(tmp_path), extensions=[".pdf", ".md"]).list_files()

    assert [f.split("\\")[-1].split("/")[-1] for f in files] == ["a.pdf", "b.md"]


def test_extensions_may_be_given_without_a_leading_dot(tmp_path):
    _touch(tmp_path, "a.pdf", "b.md")

    files = LocalSource(path=str(tmp_path), extensions=["pdf"]).list_files()

    assert len(files) == 1


def test_no_extension_filter_takes_everything(tmp_path):
    _touch(tmp_path, "a.pdf", "b.exe")

    assert len(LocalSource(path=str(tmp_path)).list_files()) == 2


def test_subdirectories_are_skipped_unless_recursive(tmp_path):
    _touch(tmp_path, "a.pdf", "nested/b.pdf")

    assert len(LocalSource(path=str(tmp_path)).list_files()) == 1
    assert len(LocalSource(path=str(tmp_path), recursive=True).list_files()) == 2


def test_a_single_file_is_a_valid_source(tmp_path):
    (path,) = _touch(tmp_path, "only.pdf")

    assert LocalSource(path=str(path)).list_files() == [str(path)]


def test_listing_is_sorted_so_a_sync_is_reproducible(tmp_path):
    _touch(tmp_path, "c.pdf", "a.pdf", "b.pdf")

    files = LocalSource(path=str(tmp_path)).list_files()

    assert files == sorted(files)


def test_a_missing_path_raises_rather_than_listing_nothing(tmp_path):
    source = LocalSource(path=str(tmp_path / "gone"))

    # An empty listing would be read as "every document was deleted".
    with pytest.raises(FileNotFoundError) as excinfo:
        source.list_files()

    assert "does not exist" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Building sources from config
# --------------------------------------------------------------------------- #

def test_a_source_is_built_from_its_type(tmp_path):
    source = build_source({"type": "local", "path": str(tmp_path)})

    assert isinstance(source, LocalSource)


def test_a_missing_type_is_rejected_with_the_available_names():
    with pytest.raises(ValueError) as excinfo:
        build_source({"path": "./docs"})

    assert "'type'" in str(excinfo.value)
    assert "local" in str(excinfo.value)


def test_an_unknown_type_is_rejected():
    with pytest.raises(ValueError) as excinfo:
        build_source({"type": "dropbox", "path": "./docs"})

    assert "dropbox" in str(excinfo.value)


def test_a_non_mapping_source_entry_is_rejected():
    with pytest.raises(ValueError):
        build_source("./docs")


def test_no_sources_configured_is_not_an_error():
    assert build_sources(None) == []
    assert build_sources([]) == []


def test_a_sources_block_that_is_not_a_list_is_rejected():
    with pytest.raises(ValueError):
        build_sources({"type": "local"})


def test_a_custom_source_can_be_registered():
    class SharePointSource(Source):
        type_name = "sharepoint_test"

        def list_files(self):
            return ["a.docx"]

    REGISTRY.register(SharePointSource)
    try:
        source = build_source({"type": "sharepoint_test"})
        assert source.list_files() == ["a.docx"]
    finally:
        del REGISTRY["sharepoint_test"]


def test_registering_a_source_without_a_type_name_is_rejected():
    class Nameless(Source):
        type_name = ""

    with pytest.raises(ValueError):
        REGISTRY.register(Nameless)


# --------------------------------------------------------------------------- #
# Sync
# --------------------------------------------------------------------------- #

class _RecordingStore:
    """Reports stored sources and records deletions."""

    def __init__(self, sources=None):
        self._sources = list(sources or [])
        self.deleted = []

    def list_sources(self):
        return list(self._sources)

    def delete_by_source(self, source, except_file_hash=None):
        self.deleted.append(source)
        return 7


class _ExplodingSource(Source):
    type_name = "exploding"

    def list_files(self):
        raise ConnectionError("bucket unreachable")


class _EmptySource(Source):
    type_name = "empty"

    def list_files(self):
        return []


def _pipeline(store, ingested=None):
    rag = object.__new__(RAGWire)
    rag.vectorstore_wrapper = store
    rag.sources = []
    rag._stored_values_cache = "populated"

    def _ingest(paths):
        rag._ingested = list(paths)
        return ingested or {
            "total": len(paths), "processed": len(paths), "skipped": 0,
            "failed": 0, "chunks_created": len(paths) * 3,
            "metadata_failed": 0, "replaced": 0, "errors": [],
        }

    rag.ingest_documents = _ingest
    return rag


def test_sync_without_sources_says_what_to_do():
    rag = _pipeline(_RecordingStore())

    with pytest.raises(ValueError) as excinfo:
        rag.sync()

    assert "sources" in str(excinfo.value)


def test_sync_ingests_everything_the_sources_list(tmp_path):
    _touch(tmp_path, "a.pdf", "b.pdf")
    rag = _pipeline(_RecordingStore())

    stats = rag.sync(sources=[LocalSource(path=str(tmp_path))])

    assert stats["listed"] == 2
    assert stats["processed"] == 2
    assert len(rag._ingested) == 2


def test_a_document_missing_from_every_source_is_deleted(tmp_path):
    _touch(tmp_path, "a.pdf")
    store = _RecordingStore([str(tmp_path / "a.pdf"), str(tmp_path / "gone.pdf")])
    rag = _pipeline(store)

    stats = rag.sync(sources=[LocalSource(path=str(tmp_path))])

    assert store.deleted == [str(tmp_path / "gone.pdf")]
    assert stats["deleted"] == 1
    assert stats["deleted_chunks"] == 7


def test_a_stored_document_that_still_exists_is_left_alone(tmp_path):
    _touch(tmp_path, "a.pdf")
    store = _RecordingStore([str(tmp_path / "a.pdf")])
    rag = _pipeline(store)

    rag.sync(sources=[LocalSource(path=str(tmp_path))])

    assert store.deleted == []


def test_a_path_stored_in_a_different_form_is_not_treated_as_deleted(tmp_path):
    _touch(tmp_path, "a.pdf")
    # Stored with forward slashes, listed with the platform separator.
    store = _RecordingStore([str(tmp_path / "a.pdf").replace("\\", "/")])
    rag = _pipeline(store)

    rag.sync(sources=[LocalSource(path=str(tmp_path))])

    assert store.deleted == []


def test_delete_missing_false_makes_sync_additive(tmp_path):
    _touch(tmp_path, "a.pdf")
    store = _RecordingStore([str(tmp_path / "gone.pdf")])
    rag = _pipeline(store)

    stats = rag.sync(sources=[LocalSource(path=str(tmp_path))], delete_missing=False)

    assert store.deleted == []
    assert stats["deleted"] == 0


def test_a_source_that_fails_to_list_suppresses_all_deletions(tmp_path):
    _touch(tmp_path, "a.pdf")
    store = _RecordingStore([str(tmp_path / "gone.pdf")])
    rag = _pipeline(store)

    stats = rag.sync(sources=[LocalSource(path=str(tmp_path)), _ExplodingSource()])

    # A network blip is not evidence that every document was deleted.
    assert store.deleted == []
    assert stats["failed"] == 1
    assert any("held back" in w or "skipped" in w for w in stats["warnings"])


def test_a_source_listing_nothing_suppresses_deletions(tmp_path):
    store = _RecordingStore([str(tmp_path / "a.pdf")])
    rag = _pipeline(store)

    stats = rag.sync(sources=[_EmptySource()])

    assert store.deleted == []
    assert stats["warnings"]


def test_dry_run_reports_deletions_without_performing_them(tmp_path):
    _touch(tmp_path, "a.pdf")
    store = _RecordingStore([str(tmp_path / "a.pdf"), str(tmp_path / "gone.pdf")])
    rag = _pipeline(store)

    stats = rag.sync(sources=[LocalSource(path=str(tmp_path))], dry_run=True)

    assert stats["deleted"] == 1
    assert store.deleted == []


def test_dry_run_does_not_ingest(tmp_path):
    _touch(tmp_path, "a.pdf")
    rag = _pipeline(_RecordingStore())

    stats = rag.sync(sources=[LocalSource(path=str(tmp_path))], dry_run=True)

    assert stats["listed"] == 1
    assert stats["processed"] == 0
    assert not hasattr(rag, "_ingested")


def test_ingestion_counters_are_carried_through(tmp_path):
    _touch(tmp_path, "a.pdf", "b.pdf")
    rag = _pipeline(_RecordingStore(), ingested={
        "total": 2, "processed": 1, "skipped": 1, "failed": 0,
        "chunks_created": 4, "metadata_failed": 0, "replaced": 1,
        "errors": [{"file": "b.pdf", "error": "boom"}],
    })

    stats = rag.sync(sources=[LocalSource(path=str(tmp_path))])

    assert stats["skipped"] == 1
    assert stats["replaced"] == 1
    assert stats["chunks_created"] == 4
    assert stats["errors"] == [{"file": "b.pdf", "error": "boom"}]


def test_deleting_documents_clears_the_stored_value_cache(tmp_path):
    _touch(tmp_path, "a.pdf")
    rag = _pipeline(_RecordingStore([str(tmp_path / "gone.pdf")]))

    rag.sync(sources=[LocalSource(path=str(tmp_path))])

    # Filter values may have referenced only the removed document.
    assert rag._stored_values_cache is None


def test_multiple_sources_are_combined(tmp_path):
    first = tmp_path / "one"
    second = tmp_path / "two"
    _touch(first, "a.pdf")
    _touch(second, "b.pdf")
    rag = _pipeline(_RecordingStore())

    stats = rag.sync(sources=[
        LocalSource(path=str(first)), LocalSource(path=str(second))
    ])

    assert stats["listed"] == 2


def test_an_empty_collection_needs_no_deletion_pass(tmp_path):
    _touch(tmp_path, "a.pdf")
    store = _RecordingStore([])
    rag = _pipeline(store)

    rag.sync(sources=[LocalSource(path=str(tmp_path))])

    assert store.deleted == []


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def test_sync_parses_with_defaults():
    args = build_parser().parse_args(["sync"])

    assert args.command == "sync"
    assert args.dry_run is False
    assert args.no_delete is False


def test_sync_accepts_dry_run_and_no_delete():
    args = build_parser().parse_args(["sync", "--dry-run", "--no-delete"])

    assert args.dry_run is True
    assert args.no_delete is True
