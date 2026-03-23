"""
Custom Metadata Example — RAGWire
==================================
Demonstrates how to use a custom metadata.yaml to extract domain-specific
fields (legal/HR in this example) instead of the default financial fields.

Setup:
  1. pip install "ragwire[openai]" fastembed
  2. export OPENAI_API_KEY="sk-..."
  3. Place documents in examples/data/
  4. Run: python examples/custom_metadata_usage.py

The metadata.yaml used here defines:
  - organization: organization name
  - doc_type:     contract | policy | report | memo | other
  - effective_year: year the document is effective
  - jurisdiction:  country or region
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ragwire import RAGWire, setup_logging

logger = setup_logging(log_level="INFO")

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
METADATA_PATH = Path(__file__).parent.parent / "metadata.example.yaml"
DATA_DIR = Path(__file__).parent / "data"


# ------------------------------------------------------------------ #
# Verify prerequisites
# ------------------------------------------------------------------ #
if not CONFIG_PATH.exists():
    logger.error(f"Config not found: {CONFIG_PATH}")
    sys.exit(1)

if not METADATA_PATH.exists():
    logger.error(f"metadata.example.yaml not found: {METADATA_PATH}")
    logger.info("Copy metadata.example.yaml to metadata.yaml and edit for your domain.")
    sys.exit(1)

if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
    logger.error(f"No files found in {DATA_DIR}")
    logger.info("Place documents in examples/data/ and re-run.")
    sys.exit(1)


# ------------------------------------------------------------------ #
# 1. Patch config at runtime to use custom metadata YAML
#    (alternatively, set metadata: config_file: in config.yaml directly)
# ------------------------------------------------------------------ #
import yaml

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

config["metadata"] = {"config_file": str(METADATA_PATH)}

import tempfile, os

with tempfile.NamedTemporaryFile(
    mode="w", suffix=".yaml", delete=False, dir=CONFIG_PATH.parent
) as tmp:
    yaml.dump(config, tmp)
    tmp_config_path = tmp.name

try:
    # ------------------------------------------------------------------ #
    # 2. Initialize pipeline with custom metadata config
    # ------------------------------------------------------------------ #
    logger.info(f"\nUsing metadata config: {METADATA_PATH.name}")
    rag = RAGWire(tmp_config_path)

    # ------------------------------------------------------------------ #
    # 3. Ingest documents
    # ------------------------------------------------------------------ #
    logger.info("\nIngesting documents...")
    stats = rag.ingest_directory(str(DATA_DIR))

    logger.info(f"\nIngestion complete:")
    logger.info(f"  Processed : {stats['processed']}/{stats['total']}")
    logger.info(f"  Skipped   : {stats['skipped']} (already ingested)")
    logger.info(f"  Chunks    : {stats['chunks_created']}")
    if stats["errors"]:
        for err in stats["errors"]:
            logger.warning(f"  Error: {err['file']}: {err['error']}")

    # ------------------------------------------------------------------ #
    # 4. Inspect extracted metadata on a retrieved chunk
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
    # 5. Retrieve without filters — LLM auto-extracts from query
    # ------------------------------------------------------------------ #
    logger.info("\n" + "=" * 60)
    logger.info("SECTION 2 — Auto-filter from Query")
    logger.info("=" * 60)

    query = "What are the data protection policies for EU jurisdiction?"
    logger.info(f"\nQuery: {query}")
    results = rag.retrieve(query, top_k=3)
    for i, doc in enumerate(results, 1):
        logger.info(f"\n  Result {i}:")
        logger.info(f"    Organization : {doc.metadata.get('organization', 'N/A')}")
        logger.info(f"    Doc Type     : {doc.metadata.get('doc_type', 'N/A')}")
        logger.info(f"    Jurisdiction : {doc.metadata.get('jurisdiction', 'N/A')}")
        logger.info(f"    Content      : {doc.page_content[:200]}...")

    # ------------------------------------------------------------------ #
    # 6. Retrieve with explicit custom filters
    # ------------------------------------------------------------------ #
    logger.info("\n" + "=" * 60)
    logger.info("SECTION 3 — Explicit Custom Filters")
    logger.info("=" * 60)

    logger.info("\nFilter: doc_type=policy")
    results = rag.retrieve("employee responsibilities", top_k=3,
                           filters={"doc_type": "policy"})
    for i, doc in enumerate(results, 1):
        logger.info(f"\n  Result {i}:")
        logger.info(f"    Organization : {doc.metadata.get('organization', 'N/A')}")
        logger.info(f"    Doc Type     : {doc.metadata.get('doc_type', 'N/A')}")
        logger.info(f"    Content      : {doc.page_content[:200]}...")

    logger.info("\n" + "=" * 60)
    logger.info("Custom metadata example completed.")
    logger.info("=" * 60)

finally:
    os.unlink(tmp_config_path)
