# Preview Documents Before Ingesting

Load and inspect extracted text before committing to the vector store — useful for debugging loader issues or verifying OCR quality.

## Single File

```python
from ragwire import MarkItDownLoader

loader = MarkItDownLoader()

result = loader.load("data/Apple_10k_2025.pdf")
if result["success"]:
    print(result["text_content"][:500])
else:
    print(f"Error: {result['error']}")
```

## Batch of Files

```python
results = loader.load_batch(["data/Apple_10k_2025.pdf", "data/Q1_report.docx"])
for r in results:
    status = "OK" if r["success"] else f"FAILED: {r['error']}"
    print(f"{r['file_name']}: {status} ({len(r.get('text_content', ''))} chars)")
```

## Whole Directory

```python
results = loader.load_directory("data/", extensions=[".pdf", ".docx"])
for r in results:
    status = "OK" if r["success"] else f"FAILED: {r['error']}"
    print(f"{r['file_name']}: {status}")
```

`extensions` is optional — omit it to load all supported formats (`.pdf`, `.docx`, `.xlsx`, `.pptx`, `.txt`, `.md`).

## Result Object

Each result dict contains:

| Key | Type | Description |
|---|---|---|
| `success` | `bool` | Whether loading succeeded |
| `file_path` | `str` | Absolute path to the file |
| `file_name` | `str` | Filename only |
| `text_content` | `str` | Extracted text (empty string on failure) |
| `error` | `str \| None` | Error message if `success` is `False` |

## Common Use Cases

- **Verify OCR quality** on scanned PDFs before ingesting
- **Debug missing content** — check if a section appears in the extracted text
- **Estimate chunk counts** — `len(text) / chunk_size` gives a rough estimate
- **Filter files** before ingestion based on content patterns
