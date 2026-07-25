# Syncing Sources

Ingestion only ever adds. `sync()` reconciles.

## The problem it solves

`ingest_directory()` walks a folder and stores what it finds. Run it again after deleting a file and nothing happens: the file is gone from disk, but its chunks are still in Qdrant, still matching queries, still being cited in answers. There is no error, no warning, and nothing in the collection that says the document was retired.

`sync()` compares the collection against what the sources actually hold right now:

| Situation | What sync does |
|---|---|
| File is new | Ingests it |
| File is unchanged | Skips it |
| File content changed | Replaces the old version's chunks |
| File no longer exists at any source | **Deletes its chunks** |

That last row is the one that turns RAGWire from a script you re-run into a pipeline you can schedule.

## Setup

```yaml title="config.yaml"
sources:
  - type: local
    path: "./documents"
    recursive: true
```

```python
from ragwire import RAGWire

rag = RAGWire("config.yaml")
stats = rag.sync()

print(stats)
```

```python
{'listed': 42, 'processed': 3, 'skipped': 38, 'replaced': 1,
 'deleted': 2, 'deleted_chunks': 57, 'failed': 0,
 'chunks_created': 12, 'warnings': [], 'errors': []}
```

Three new documents, one edited, two removed from the folder and now removed from the collection.

Or from the command line:

```bash
ragwire sync --config config.yaml
```

## Look before you delete

The first time you point sync at an existing collection, check what it intends to do:

```bash
ragwire sync --dry-run
```

Nothing is written and nothing is deleted. `deleted` reports what would go, and every doomed document is named in the log. This is worth doing whenever you change `path`, `prefix` or `extensions`, because narrowing a filter makes previously-ingested documents look deleted.

To make sync purely additive, permanently:

```bash
ragwire sync --no-delete
```

```python
rag.sync(delete_missing=False)
```

## The safety rule

Deletion is the destructive half, so it has a guard rail worth understanding.

**If any source fails to list, or lists zero files, sync deletes nothing at all that run.**

The reason is that "the bucket returned no objects" and "every object was deleted" look identical from the outside. Acting on the wrong reading empties your collection. So an S3 timeout, an expired credential, or a mistyped prefix produces a warning and a skipped deletion pass rather than a catastrophe:

```python
stats = rag.sync()

for warning in stats["warnings"]:
    print(warning)
```

```
Deletions skipped: at least one source failed to list or was empty
```

Ingestion still runs normally. Only deletion is held back.

A local source goes further and raises `FileNotFoundError` if its path does not exist, for the same reason: a typo in `path` should not be indistinguishable from an emptied folder.

## S3

```bash
pip install ragwire[s3]
```

```yaml title="config.yaml"
sources:
  - type: s3
    bucket: "my-filings"
    prefix: "2026/"
    cache_dir: ".ragwire_cache"
```

Credentials come from boto3's normal chain, so environment variables, `~/.aws/credentials` or an instance role all work without RAGWire being told about them. To be explicit:

```yaml
  - type: s3
    bucket: "my-filings"
    region: "us-east-1"
    aws_access_key_id: "${AWS_ACCESS_KEY_ID}"
    aws_secret_access_key: "${AWS_SECRET_ACCESS_KEY}"
```

Objects are downloaded into `cache_dir` and ingested from there, so hashing and deduplication behave exactly as they do for local files. A repeated sync only re-downloads objects whose size or modification time moved, so the second run is cheap.

Add `.ragwire_cache/` to your `.gitignore`.

!!! note "S3-compatible storage"
    `endpoint_url` points the client somewhere else, so MinIO, Cloudflare R2, Backblaze B2 and similar all work through the same connector.

## Several sources at once

```yaml
sources:
  - type: local
    path: "./internal_docs"
    recursive: true
  - type: s3
    bucket: "public-filings"
    prefix: "2026/"
```

Sources are combined into one listing before reconciliation, so a document is only deleted when *no* source holds it. Moving a file from local disk into a bucket is therefore not a delete followed by a re-ingest; it stays put.

## Scheduling it

Sync is idempotent, so running it on a timer is safe:

```bash title="crontab"
0 * * * * cd /srv/rag && /srv/rag/venv/bin/ragwire sync --config config.yaml >> sync.log 2>&1
```

`ragwire sync` exits non-zero when every file failed, so a monitoring system can alert on it.

## Writing your own source

A source answers one question: which files should this collection contain right now? Implement `list_files()` and return local paths.

```python
from ragwire.sources import REGISTRY, Source

class SharePointSource(Source):
    type_name = "sharepoint"

    def __init__(self, site, folder, **kwargs):
        super().__init__(name=f"sharepoint:{site}", **kwargs)
        self.site = site
        self.folder = folder

    def list_files(self):
        # Download to a local cache and return those paths.
        return download_folder(self.site, self.folder, into=".ragwire_cache")

REGISTRY.register(SharePointSource)
```

It is then usable from config like any built-in type:

```yaml
sources:
  - type: sharepoint
    site: "https://contoso.sharepoint.com/sites/finance"
    folder: "Shared Documents/Filings"
```

Two rules matter, and both are about deletion safety:

1. **Raise on failure, never return an empty list.** An empty listing means "everything was deleted here", and sync will act on it if no other source objects.
2. **Return stable paths.** A cache path that changes between runs makes every document look new on one run and deleted on the next.

`self.matches_extension(path)` applies the configured extension filter, so your source gets the same filtering behaviour as the built-in ones for free.

!!! note "Google Drive is not included"
    A Drive connector needs an interactive OAuth flow and token storage, and a half-built one is worse than none. The extension point above is the supported way to add it, and a service account plus `google-api-python-client` is about thirty lines.

## See also

- [Update Documents](update_documents.md) for how change detection works underneath
- [API Reference](../api_reference.md#sources-section) for every config key
