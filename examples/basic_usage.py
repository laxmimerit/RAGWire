"""
Example: Basic RAG Pipeline Usage

Demonstrates the full RAG pipeline:
1. Initialize with config.yaml (Ollama embeddings + local Qdrant)
2. Ingest PDF documents from examples/data/
3. Retrieve relevant chunks for test queries
4. Display results with metadata
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

    # Initialize pipeline
    logger.info("\nInitializing RAG pipeline...")
    pipeline = RAGPipeline(str(CONFIG_PATH))

    # Ingest
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

    # Test queries — Apple 10-K 2025
    queries = [
        "What is Apple's total revenue for fiscal year 2025?",
        "What are Apple's main product categories and their revenue?",
        "What are the key risk factors mentioned in Apple's 10-K?",
        "What is Apple's net income and earnings per share?",
        "Describe Apple's research and development expenses.",
    ]

    logger.info("\n" + "=" * 60)
    logger.info("Test Queries")
    logger.info("=" * 60)

    for query in queries:
        logger.info(f"\nQuery: {query}")
        logger.info("-" * 60)

        results = pipeline.retrieve(query, top_k=3)

        if results:
            for i, doc in enumerate(results, 1):
                logger.info(f"\n  Result {i}:")
                logger.info(f"    File    : {doc.metadata.get('file_name', 'Unknown')}")
                logger.info(f"    Company : {doc.metadata.get('company_name', 'Unknown')}")
                logger.info(f"    Doc Type: {doc.metadata.get('doc_type', 'Unknown')}")
                logger.info(f"    Year    : {doc.metadata.get('fiscal_year', 'Unknown')}")
                logger.info(f"    Content : {doc.page_content[:300]}...")
        else:
            logger.info("  No results found.")

    logger.info("\n" + "=" * 60)
    logger.info("Test completed successfully.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
