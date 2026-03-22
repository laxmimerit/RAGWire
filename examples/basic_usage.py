"""
Example: Basic RAG Pipeline Usage

Demonstrates the full RAG pipeline:
1. Initialize with config.yaml
2. Ingest PDF documents from examples/data/
3. Retrieve relevant chunks — plain and with metadata filters
4. Display results with full metadata
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragwire import RAGPipeline, setup_logging

# Setup logging
logger = setup_logging(log_level="INFO", console_output=True)

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
CONFIG_PATH = SCRIPT_DIR.parent / "config.yaml"


def print_results(results, label=""):
    if label:
        logger.info(f"\n  [{label}]")
    if not results:
        logger.info("  No results found.")
        return
    for i, doc in enumerate(results, 1):
        logger.info(f"\n  Result {i}:")
        logger.info(f"    File         : {doc.metadata.get('file_name', 'Unknown')}")
        logger.info(f"    Company      : {doc.metadata.get('company_name', 'Unknown')}")
        logger.info(f"    Doc Type     : {doc.metadata.get('doc_type', 'Unknown')}")
        logger.info(f"    Fiscal Year  : {doc.metadata.get('fiscal_year', 'Unknown')}")
        logger.info(f"    Fiscal Qtr   : {doc.metadata.get('fiscal_quarter', 'Unknown')}")
        logger.info(f"    Chunk        : {doc.metadata.get('chunk_index', '?')} / {doc.metadata.get('total_chunks', '?')}")
        logger.info(f"    Content      : {doc.page_content[:300]}...")


def main():
    """Run the basic RAG pipeline example."""

    # Check config
    if not CONFIG_PATH.exists():
        logger.error(f"Config file not found: {CONFIG_PATH}")
        return

    # Discover PDFs in examples/data/
    pdf_files = sorted(DATA_DIR.glob("*.pdf")) if DATA_DIR.exists() else []

    if not pdf_files:
        logger.error(f"No PDF files found in {DATA_DIR}")
        logger.info("Place PDF files in the examples/data/ directory and re-run.")
        return

    logger.info(f"Found {len(pdf_files)} PDF(s) to process:")
    for f in pdf_files:
        logger.info(f"  - {f.name}")

    # ------------------------------------------------------------------ #
    # 1. Initialize pipeline
    # ------------------------------------------------------------------ #
    logger.info("\nInitializing RAG pipeline...")
    pipeline = RAGPipeline(str(CONFIG_PATH))

    # ------------------------------------------------------------------ #
    # 2. Ingest documents
    # ------------------------------------------------------------------ #
    logger.info("\nIngesting documents...")
    stats = pipeline.ingest_documents([str(f) for f in pdf_files])

    logger.info("\nIngestion complete:")
    logger.info(f"  - Processed : {stats['processed']}/{stats['total']}")
    logger.info(f"  - Skipped   : {stats['skipped']} (already ingested)")
    logger.info(f"  - Chunks    : {stats['chunks_created']}")
    if stats["errors"]:
        logger.warning(f"  - Errors    : {len(stats['errors'])}")
        for err in stats["errors"][:3]:
            logger.warning(f"    * {err['file']}: {err['error']}")

    # Pipeline stats
    pipeline_stats = pipeline.get_stats()
    logger.info(f"\nCollection  : {pipeline_stats['collection_name']}")
    logger.info(f"Total chunks: {pipeline_stats['total_documents']}")
    logger.info(f"Vector size : {pipeline_stats['vector_size']}")

    # ------------------------------------------------------------------ #
    # 3. Basic retrieval (no filters)
    # ------------------------------------------------------------------ #
    logger.info("\n" + "=" * 60)
    logger.info("SECTION 1 — Basic Retrieval (no filters)")
    logger.info("=" * 60)

    queries = [
        "What is the total revenue?",
        "What are the main product categories and their revenue?",
        "What are the key risk factors?",
        "What is the net income and earnings per share?",
        "Describe the research and development expenses.",
    ]

    for query in queries:
        logger.info(f"\nQuery: {query}")
        logger.info("-" * 60)
        results = pipeline.retrieve(query, top_k=3)
        print_results(results)

    # ------------------------------------------------------------------ #
    # 4. Retrieval with metadata filters
    # ------------------------------------------------------------------ #
    logger.info("\n" + "=" * 60)
    logger.info("SECTION 2 — Retrieval with Metadata Filters")
    logger.info("=" * 60)

    # Get company name and year from the first ingested doc's metadata
    sample = pipeline.retrieve("revenue", top_k=1)
    company = sample[0].metadata.get("company_name") if sample else None
    year = sample[0].metadata.get("fiscal_year") if sample else None
    year_val = year[0] if isinstance(year, list) and year else year

    logger.info(f"\nDetected — company: {company}, fiscal_year: {year_val}")

    # Filter by company
    logger.info(f"\nQuery: 'total revenue'  |  filter: company_name={company}")
    logger.info("-" * 60)
    results = pipeline.retrieve("total revenue", top_k=3, filters={"company_name": company})
    print_results(results, label=f"company_name={company}")

    # Filter by fiscal year
    logger.info(f"\nQuery: 'net income'  |  filter: fiscal_year={year_val}")
    logger.info("-" * 60)
    results = pipeline.retrieve("net income", top_k=3, filters={"fiscal_year": year_val})
    print_results(results, label=f"fiscal_year={year_val}")

    # Filter by company + year combined
    logger.info(f"\nQuery: 'risk factors'  |  filter: company={company}, year={year_val}")
    logger.info("-" * 60)
    results = pipeline.retrieve(
        "risk factors",
        top_k=3,
        filters={"company_name": company, "fiscal_year": year_val},
    )
    print_results(results, label=f"company={company} + fiscal_year={year_val}")

    # ------------------------------------------------------------------ #
    # 5. Hybrid search
    # ------------------------------------------------------------------ #
    logger.info("\n" + "=" * 60)
    logger.info("SECTION 3 — Hybrid Search")
    logger.info("=" * 60)

    hybrid_query = f"{company} revenue fiscal {year_val}" if company else "revenue fiscal year"
    logger.info(f"\nQuery: '{hybrid_query}'")
    logger.info("-" * 60)
    results = pipeline.hybrid_search(hybrid_query, k=3)
    print_results(results)

    # ------------------------------------------------------------------ #
    # 6. Metadata inspection
    # ------------------------------------------------------------------ #
    logger.info("\n" + "=" * 60)
    logger.info("SECTION 4 — Full Metadata Inspection")
    logger.info("=" * 60)

    results = pipeline.retrieve("revenue", top_k=1)
    if results:
        logger.info("\nFull metadata on first retrieved chunk:")
        for key, value in results[0].metadata.items():
            logger.info(f"  {key:<20}: {value}")

    logger.info("\n" + "=" * 60)
    logger.info("All tests completed successfully.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
