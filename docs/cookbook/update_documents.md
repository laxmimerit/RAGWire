# Update a Document Without Duplicates

RAGWire deduplicates by SHA256 file hash — re-ingesting the same file is a no-op. To update a document, the file content must change (even a single byte triggers re-ingestion).

```python
from ragwire import RAGWire

rag = RAGWire("config.yaml")

# First ingest
stats = rag.ingest_documents(["reports/Q1_2025.pdf"])
print(stats["processed"])   # → 1

# Re-run with same file — skipped automatically
stats = rag.ingest_documents(["reports/Q1_2025.pdf"])
print(stats["skipped"])     # → 1

# Update the file, re-run — new version ingested
# (old chunks remain; to remove them, set force_recreate: true and re-ingest all)
stats = rag.ingest_documents(["reports/Q1_2025.pdf"])
print(stats["processed"])   # → 1
```

!!! note "Full replacement"
    RAGWire does not delete old chunks when a file is updated. For a full replacement, set `force_recreate: true` in config and re-ingest all documents.

## Scheduled Re-ingestion

For a folder that receives new or updated files regularly:

```python
import schedule, time
from ragwire import RAGWire

rag = RAGWire("config.yaml")

def sync():
    stats = rag.ingest_directory("data/")
    print(f"Processed: {stats['processed']}, Skipped: {stats['skipped']}, Failed: {stats['failed']}")

schedule.every(1).hours.do(sync)

while True:
    schedule.run_pending()
    time.sleep(60)
```

Only changed files are re-ingested — unchanged files are skipped automatically.
