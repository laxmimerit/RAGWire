"""
Custom Metadata Example — RAGWire
==================================
Demonstrates custom metadata extraction using a domain-specific metadata.yaml.

Setup:
  1. pip install "ragwire[openai]" fastembed
  2. export OPENAI_API_KEY="sk-..."
  3. Add to config.yaml:
       metadata:
         config_file: "metadata.yaml"
  4. Place documents in examples/data/
  5. Run: python examples/custom_metadata_usage.py

The metadata.yaml used here defines:
  - organization: organization name
  - doc_type:     contract | policy | report | memo
  - effective_year: year the document is effective
  - jurisdiction:  country or region
"""

from ragwire import RAGWire, setup_logging

logger = setup_logging(log_level="INFO")


# ------------------------------------------------------------------ #
# 1. Initialize
# ------------------------------------------------------------------ #
rag = RAGWire("config.yaml")


# ------------------------------------------------------------------ #
# 2. Ingest
# ------------------------------------------------------------------ #
logger.info("\nIngesting documents...")
stats = rag.ingest_directory("data/")
logger.info(f"  Processed : {stats['processed']}/{stats['total']}")
logger.info(f"  Skipped   : {stats['skipped']} (already ingested)")
logger.info(f"  Chunks    : {stats['chunks_created']}")
for err in stats.get("errors", []):
    logger.warning(f"  Error: {err['file']}: {err['error']}")


# ------------------------------------------------------------------ #
# 3. Inspect extracted metadata
# ------------------------------------------------------------------ #
logger.info("\n" + "=" * 60)
logger.info("SECTION 1 — Extracted Custom Metadata")
logger.info("=" * 60)

results = rag.retrieve("organization policy", top_k=1)
if results:
    logger.info("\nMetadata on first retrieved chunk:")
    for key, value in results[0].metadata.items():
        logger.info(f"  {key:<20}: {value}")


# ------------------------------------------------------------------ #
# 4. Retrieve with explicit filters
# ------------------------------------------------------------------ #
logger.info("\n" + "=" * 60)
logger.info("SECTION 2 — Explicit Custom Filters")
logger.info("=" * 60)

logger.info("\nFilter: doc_type=policy")
results = rag.retrieve(
    "employee responsibilities", top_k=3, filters={"doc_type": "policy"}
)
for i, doc in enumerate(results, 1):
    logger.info(f"\n  Result {i}:")
    logger.info(f"    Organization : {doc.metadata.get('organization', 'N/A')}")
    logger.info(f"    Doc Type     : {doc.metadata.get('doc_type', 'N/A')}")
    logger.info(f"    Content      : {doc.page_content[:200]}...")

logger.info("\nFilter: jurisdiction=eu, doc_type=contract")
results = rag.retrieve(
    "termination clauses",
    top_k=3,
    filters={"jurisdiction": "eu", "doc_type": "contract"},
)
for i, doc in enumerate(results, 1):
    logger.info(f"\n  Result {i}:")
    logger.info(f"    Organization : {doc.metadata.get('organization', 'N/A')}")
    logger.info(f"    Jurisdiction : {doc.metadata.get('jurisdiction', 'N/A')}")
    logger.info(f"    Content      : {doc.page_content[:200]}...")

logger.info("\n" + "=" * 60)
logger.info("Custom metadata example completed.")
logger.info("=" * 60)
