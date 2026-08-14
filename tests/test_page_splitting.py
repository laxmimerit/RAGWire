"""
Tests for the "page" splitter strategy: PageSplitter, PageLoader and the
pipeline path that stamps page provenance onto chunk metadata.

No server, no LLM, no network. PDF reading is faked (generating a real
text-bearing PDF needs a renderer); PPTX files are generated for real with
python-pptx.
"""

import pytest

from ragwire import RAGWire
from ragwire.loaders.page_loader import PageLoader
from ragwire.processing.splitter import PAGE_MARKER_DEFAULT, PageSplitter


MARKER = PAGE_MARKER_DEFAULT


# --------------------------------------------------------------------------- #
# PageSplitter: marker-based splitting
# --------------------------------------------------------------------------- #

def test_marker_splits_text_into_numbered_pages():
    pages = PageSplitter().split(f"intro {MARKER} middle {MARKER} end")

    assert [(p["page_number"], p["text"]) for p in pages] == [
        (1, "intro"), (2, "middle"), (3, "end"),
    ]
    assert all(p["page_total"] == 3 for p in pages)
    assert all(p["page_label"] is None for p in pages)


def test_marker_preserves_numbering_when_blank_pages_are_dropped():
    # A leading marker makes segment 1 empty; the empty page is dropped but
    # the remaining pages keep their true positions.
    pages = PageSplitter().split(f"{MARKER} one {MARKER}   {MARKER} three")

    assert [(p["page_number"], p["text"]) for p in pages] == [
        (2, "one"), (4, "three"),
    ]
    assert all(p["page_total"] == 4 for p in pages)


def test_custom_marker_is_honoured():
    pages = PageSplitter(page_marker="=== PAGE ===").split("a === PAGE === b")
    assert [p["text"] for p in pages] == ["a", "b"]


def test_text_without_marker_is_a_single_page():
    pages = PageSplitter().split("just some plain text", file_type="txt")
    assert [(p["page_number"], p["page_total"]) for p in pages] == [(1, 1)]


def test_empty_text_yields_no_pages():
    assert PageSplitter().split("") == []
    assert PageSplitter().split("   \n  ") == []


# --------------------------------------------------------------------------- #
# PageSplitter: heading-based splitting for markdown/HTML
# --------------------------------------------------------------------------- #

def test_markdown_splits_on_headings_with_labels():
    text = "# Intro\nwelcome\n## Setup\ninstall it\n## Usage\nrun it"
    pages = PageSplitter().split(text, file_type="md")

    assert [(p["page_number"], p["page_label"]) for p in pages] == [
        (1, "Intro"), (2, "Setup"), (3, "Usage"),
    ]
    # The heading line stays inside its page for embedding context
    assert pages[1]["text"].startswith("## Setup")
    assert all(p["page_total"] == 3 for p in pages)


def test_deeper_headings_do_not_start_new_pages():
    text = "# Top\nbody\n### Subsection\nmore"
    pages = PageSplitter().split(text, file_type="md")

    assert len(pages) == 1
    assert "### Subsection" in pages[0]["text"]


def test_preamble_before_first_heading_becomes_unlabeled_first_page():
    text = "prologue text\n# One\nbody"
    pages = PageSplitter().split(text, file_type="md")

    assert [(p["page_number"], p["page_label"]) for p in pages] == [
        (1, None), (2, "One"),
    ]


def test_marker_wins_over_headings_in_markdown():
    # Explicit beats inferred: a document that carries the marker is split on
    # the marker even though it also has headings.
    text = f"# A\nfirst {MARKER} # B\nsecond"
    pages = PageSplitter().split(text, file_type="md")

    assert len(pages) == 2
    assert all(p["page_label"] is None for p in pages)


def test_markdown_without_headings_or_marker_is_a_single_page():
    pages = PageSplitter().split("no structure at all", file_type="md")
    assert len(pages) == 1


def test_non_markdown_types_never_split_on_headings():
    pages = PageSplitter().split("# looks like a heading\ntext", file_type="txt")
    assert len(pages) == 1


# --------------------------------------------------------------------------- #
# PageSplitter: loader-provided pages (PDF, PPTX)
# --------------------------------------------------------------------------- #

def test_loader_pages_pass_through_with_numbering_preserved():
    loader_pages = [
        {"page_number": 1, "page_label": None, "text": "first page"},
        {"page_number": 2, "page_label": None, "text": "   "},  # scanned/blank
        {"page_number": 3, "page_label": "iii", "text": "third page"},
    ]
    pages = PageSplitter().split("ignored", pages=loader_pages)

    assert [(p["page_number"], p["page_label"]) for p in pages] == [
        (1, None), (3, "iii"),
    ]
    # page_total counts the real document size, including the dropped page
    assert all(p["page_total"] == 3 for p in pages)


def test_loader_pages_win_over_marker_text():
    loader_pages = [{"page_number": 1, "page_label": None, "text": "real page"}]
    pages = PageSplitter().split(f"a {MARKER} b", pages=loader_pages)
    assert [p["text"] for p in pages] == ["real page"]


def test_all_blank_loader_pages_yield_no_chunks():
    loader_pages = [{"page_number": 1, "page_label": None, "text": " "}]
    assert PageSplitter().split("", pages=loader_pages) == []


# --------------------------------------------------------------------------- #
# PageLoader
# --------------------------------------------------------------------------- #

def test_text_files_are_read_raw_with_no_loader_pages(tmp_path):
    doc = tmp_path / "notes.md"
    doc.write_text(f"one {MARKER} two", encoding="utf-8")

    result = PageLoader().load(doc)

    assert result["success"] is True
    assert result["pages"] is None  # marker splitting is the splitter's job
    assert MARKER in result["text_content"]
    assert result["file_type"] == "md"


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        PageLoader().load("does_not_exist.pdf")


def test_pdf_is_loaded_page_by_page(tmp_path, monkeypatch):
    class FakePage:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class FakeReader:
        def __init__(self, path):
            self.pages = [FakePage("alpha"), FakePage(""), FakePage("gamma")]
            # Printed labels differing from the ordinals ("iv") are kept,
            # plain ordinals ("2", "3") are dropped as redundant.
            self.page_labels = ["iv", "2", "3"]

    import pypdf
    monkeypatch.setattr(pypdf, "PdfReader", FakeReader)

    doc = tmp_path / "report.pdf"
    doc.write_bytes(b"%PDF-1.4 fake")

    result = PageLoader().load(doc)

    assert result["success"] is True
    assert [(p["page_number"], p["page_label"], p["text"]) for p in result["pages"]] == [
        (1, "iv", "alpha"), (2, None, ""), (3, None, "gamma"),
    ]
    assert result["text_content"] == "alpha\n\n\n\ngamma"


def test_pptx_slides_become_pages_with_titles(tmp_path):
    pptx = pytest.importorskip("pptx")

    presentation = pptx.Presentation()
    layout = presentation.slide_layouts[1]  # title + content

    for title, body in [("First Slide", "point one"), ("Second Slide", "point two")]:
        slide = presentation.slides.add_slide(layout)
        slide.shapes.title.text = title
        slide.placeholders[1].text = body

    deck = tmp_path / "deck.pptx"
    presentation.save(str(deck))

    result = PageLoader().load(deck)

    assert result["success"] is True
    assert [(p["page_number"], p["page_label"]) for p in result["pages"]] == [
        (1, "First Slide"), (2, "Second Slide"),
    ]
    assert "point one" in result["pages"][0]["text"]


def test_unpaged_formats_fall_back_to_flat_text(tmp_path, monkeypatch):
    loader = PageLoader()
    monkeypatch.setattr(
        loader.fallback,
        "load",
        lambda path: {
            "text_content": "flat docx text",
            "file_name": "d.docx",
            "file_type": "docx",
            "success": True,
            "error": None,
        },
    )

    doc = tmp_path / "d.docx"
    doc.write_bytes(b"fake")

    result = loader.load(doc)
    assert result["pages"] is None
    assert result["text_content"] == "flat docx text"


# --------------------------------------------------------------------------- #
# Pipeline: page provenance on chunk metadata
# --------------------------------------------------------------------------- #

def _page_pipeline():
    rag = object.__new__(RAGWire)
    rag.splitter = PageSplitter()
    rag._dedup_chunks = False
    rag._extract_metadata_with_retry = lambda text, name: ({}, True)
    return rag


def test_page_chunks_carry_page_metadata():
    rag = _page_pipeline()

    docs, metadata_ok = rag._process_document(
        text=f"first {MARKER} second",
        file_path="d.txt",
        file_name="d.txt",
        file_type="txt",
        file_hash="hash",
    )

    assert metadata_ok is True
    assert [d.page_content for d in docs] == ["first", "second"]
    assert [d.metadata["page_number"] for d in docs] == [1, 2]
    assert all(d.metadata["page_total"] == 2 for d in docs)
    assert all(d.metadata["total_chunks"] == 2 for d in docs)
    # No label for marker pages: the field is omitted, not stored as null
    assert all("page_label" not in d.metadata for d in docs)


def test_loader_pages_flow_through_to_metadata():
    rag = _page_pipeline()
    loader_pages = [
        {"page_number": 1, "page_label": "Overview", "text": "slide one"},
        {"page_number": 2, "page_label": None, "text": "slide two"},
    ]

    docs, _ = rag._process_document(
        text="slide one\n\nslide two",
        file_path="deck.pptx",
        file_name="deck.pptx",
        file_type="pptx",
        file_hash="hash",
        pages=loader_pages,
    )

    assert docs[0].metadata["page_label"] == "Overview"
    assert "page_label" not in docs[1].metadata
    assert [d.metadata["page_number"] for d in docs] == [1, 2]


def test_dedup_keeps_page_metadata_aligned():
    rag = _page_pipeline()
    rag._dedup_chunks = True

    docs, _ = rag._process_document(
        text=f"same {MARKER} same {MARKER} other",
        file_path="d.txt",
        file_name="d.txt",
        file_type="txt",
        file_hash="hash",
    )

    # The duplicate page is dropped; the survivors keep their true pages
    assert [(d.page_content, d.metadata["page_number"]) for d in docs] == [
        ("same", 1), ("other", 3),
    ]


def test_non_page_strategies_add_no_page_fields():
    from ragwire.processing.splitter import get_markdown_splitter

    rag = object.__new__(RAGWire)
    rag.splitter = get_markdown_splitter(chunk_size=100, chunk_overlap=0)
    rag._dedup_chunks = False
    rag._extract_metadata_with_retry = lambda text, name: ({}, True)

    docs, _ = rag._process_document(
        text="plain content",
        file_path="d.txt",
        file_name="d.txt",
        file_type="txt",
        file_hash="hash",
    )

    assert docs
    assert all("page_number" not in d.metadata for d in docs)
    assert all("page_total" not in d.metadata for d in docs)
