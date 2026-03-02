# Core: Dataloaders

## 1. What This Feature Is

Dataloaders provide a unified interface for loading benchmark datasets from HuggingFace, normalizing them into a consistent internal format (`DatasetRecord`), and converting them to framework-specific document objects (Haystack or LangChain `Document`).

This module handles:

- **Dataset loading**: Streaming or full loading from HuggingFace datasets with configurable splits and limits.
- **Schema normalization**: Different datasets have different schemas; all are normalized to `DatasetRecord(text, metadata)`.
- **Framework conversion**: Same loaded dataset can convert to Haystack or LangChain documents interchangeably.
- **Evaluation query extraction**: Deduplicated queries with ground-truth answers for benchmarking retrieval quality.
- **Extensibility**: New datasets can be added by subclassing `BaseDatasetLoader` and registering in `DataloaderCatalog`.

## 2. Why It Exists in Retrieval/RAG

Retrieval benchmarks require consistent inputs to compare retrieval quality fairly. Without normalization:

- **Schema inconsistency**: TriviaQA has `question` + `search_results`, ARC has `question` + `choices`, PopQA has `entity` + `paragraph`.
- **Answer format variance**: Some datasets store answers as lists, others as single strings, some nested in JSON.
- **Document granularity**: One TriviaQA question maps to multiple evidence documents; ARC questions map to single passages.
- **Evaluation complexity**: The same question appearing with multiple evidence documents would inflate metrics if not deduplicated.

This module exists to:

- **Ensure fair comparisons**: Same dataset limits, same normalization, same evaluation queries across all retrieval experiments.
- **Enable framework switching**: Load once, convert to Haystack or LangChain documents as needed.
- **Simplify benchmark setup**: One-line dataset loading with sensible defaults for splits and limits.
- **Support reproducibility**: Deterministic record limits and query deduplication ensure experiments are repeatable.

## 3. Indexing Pipeline: Step-by-Step

```mermaid
flowchart TD
    A[DataloaderCatalog.create] --> B[BaseDatasetLoader._load_dataset_iterable]
    B --> C[HuggingFace Dataset Streaming]
    C --> D[Row-by-Row Parsing]
    D --> E[_parse_row → List[DatasetRecord]]
    E --> F[Accumulate Until Limit]
    F --> G[LoadedDataset Wrapper]
    G --> H[to_haystack or to_langchain]
    H --> I[Framework Documents Ready for Indexing]
```

### Step-by-Step Flow (TriviaQA Example)

1. **Catalog creation**: `DataloaderCatalog.create("triviaqa", split="test", limit=500)` returns `TriviaQALoader`.
2. **Dataset loading**: `loader.load()` calls `_load_dataset_iterable()` which invokes `hf_load_dataset("trivia_qa", "rc", split="test", streaming=True)`.
3. **Row iteration**: Streams rows one at a time (memory efficient for large datasets).
4. **Row parsing**: Each row passes through `_parse_row(row)`:
   - Extracts `question` and `search_results` from row.
   - Normalizes answers from `row["answer"]` (list or string) to list.
   - Iterates over `search_results` (multiple evidence documents per question).
   - Creates one `DatasetRecord` per evidence document with metadata (`question`, `answers`, `title`, `rank`, `evidence_index`).
5. **Limit enforcement**: Stops after `limit` records (not questions — 500 records may represent fewer unique questions).
6. **Wrapper creation**: Returns `LoadedDataset(dataset_type="triviaqa", records=...)`.
7. **Document conversion**: `dataset.to_haystack()` converts all records to Haystack `Document(content=record.text, meta=record.metadata)`.

### Row Expansion Pattern (TriviaQA Specific)

One TriviaQA question row:

```python
{
    "question": "What is RAG?",
    "answer": ["Retrieval-Augmented Generation", "RAG"],
    "search_results": {
        "title": ["Doc A", "Doc B"],
        "search_context": ["Context A", "Context B"],
        "rank": [1, 2],
    }
}
```

Expands to **two** `DatasetRecord` objects:

```python
[
    DatasetRecord(
        text="Context A",
        metadata={"question": "What is RAG?", "answers": ["RAG"], "title": "Doc A", "rank": 1, "evidence_index": 0}
    ),
    DatasetRecord(
        text="Context B",
        metadata={"question": "What is RAG?", "answers": ["RAG"], "title": "Doc B", "rank": 2, "evidence_index": 1}
    ),
]
```

This expansion is critical for evaluation: the same query appears multiple times but should only be evaluated once (handled by `evaluation_queries()` deduplication).

## 4. Search Pipeline: Step-by-Step

Dataloaders don't directly participate in search, but they provide the **evaluation queries** used to benchmark search quality:

```mermaid
flowchart TD
    A[LoadedDataset.evaluation_queries] --> B[EvaluationExtractor.extract]
    B --> C[Iterate Records]
    C --> D[Normalize Query Text]
    D --> E[Deduplication Check]
    E --> F[Extract Answers]
    F --> G[Build EvaluationQuery]
    G --> H[Return List[EvaluationQuery]]
```

### Evaluation Query Extraction Flow

1. **Record iteration**: Iterates through all `DatasetRecord` objects in loaded dataset.
2. **Query extraction**: Reads query text from `metadata["question"]` (most datasets) or `metadata["entity"]` (PopQA).
3. **Normalization**: Collapses whitespace and lowercases query: `"  What Is RAG?  "` → `"what is rag"`.
4. **Deduplication**: Checks if normalized query already seen; skips duplicates.
5. **Answer extraction**: Reads from `metadata["answers"]` (list) or `metadata["answer"]` (string), normalizes to list.
6. **ID extraction**: If `metadata["id"]` exists, adds to `relevant_doc_ids`.
7. **Metadata filtering**: Removes extracted fields (`question`, `answers`, `answer`, `entity`) from retained metadata.
8. **Limit application**: Stops after `limit` unique queries.

### Using Evaluation Queries for Search Benchmarking

```python
from vectordb.dataloaders.catalog import DataloaderCatalog

# Load dataset
loader = DataloaderCatalog.create("triviaqa", split="test", limit=500)
dataset = loader.load()

# Extract evaluation queries (deduplicated)
queries = dataset.evaluation_queries(limit=100)

# Benchmark search
correct = 0
for q in queries:
    results = search_pipeline.search(q.query, top_k=5)
    # Check if any result contains an answer
    for doc in results:
        if any(ans.lower() in doc.content.lower() for ans in q.answers):
            correct += 1
            break

recall = correct / len(queries)
```

## 5. When to Use It

Use dataloaders when:

- **Benchmarking retrieval quality**: Need consistent datasets with ground-truth answers for evaluation.
- **Comparing features**: Running the same dataset across different retrieval features (metadata filtering, hybrid search, reranking) to measure improvements.
- **Framework-agnostic experiments**: Want to compare Haystack vs. LangChain pipelines on identical data.
- **Reproducible research**: Need deterministic dataset limits and query deduplication for repeatable experiments.
- **Multi-dataset evaluation**: Testing retrieval robustness across different domains (science QA, finance, open-domain QA).

## 6. When Not to Use It

Avoid dataloaders when:

- **Using custom datasets**: Your data isn't on HuggingFace or has a completely custom schema (implement custom loader instead).
- **Production deployments**: Dataloaders are for benchmarking; production systems should use live data pipelines.
- **Need real-time updates**: Dataloaders are batch-oriented; they don't support streaming updates to existing datasets.
- **Domain-specific preprocessing needed**: Some domains require custom text cleaning, chunking, or enrichment before indexing (extend base loader with custom `_parse_row`).

## 7. What This Codebase Provides

### Core Classes (all in `src/vectordb/dataloaders/`)

```python
from vectordb.dataloaders import (
    DataloaderCatalog,
    LoadedDataset,
    DatasetRecord,
    EvaluationQuery,
    DocumentConverter,
)
from vectordb.dataloaders.datasets import (
    TriviaQALoader,
    ARCLoader,
    PopQALoader,
    FactScoreLoader,
    EarningsCallsLoader,
)
```

### Type Definitions

```python
from vectordb.dataloaders.types import (
    DatasetRecord,        # text: str, metadata: dict
    EvaluationQuery,      # query: str, answers: list[str], relevant_doc_ids: list[str]
    DatasetType,          # Literal["triviaqa", "arc", "popqa", "factscore", "earnings_calls"]
    DataloaderError,      # Base exception
    UnsupportedDatasetError,  # Unknown dataset name
    DatasetLoadError,     # Loading failed
    DatasetValidationError,   # Schema mismatch
)
```

### Catalog API

```python
DataloaderCatalog.create(
    name="triviaqa",      # Dataset identifier
    split="test",         # Dataset split
    limit=500,            # Record limit (None = unlimited)
    dataset_id=None,      # Override HuggingFace dataset ID
)
```

### LoadedDataset API

```python
dataset = loader.load()

dataset.records()              # List[DatasetRecord]
dataset.to_dict_items()        # List[dict] for custom conversion
dataset.to_haystack()          # List[Haystack Document]
dataset.to_langchain()         # List[LangChain Document]
dataset.evaluation_queries(limit=100)  # List[EvaluationQuery]
```

### Document Converter

```python
from vectordb.dataloaders.converters import DocumentConverter, records_to_items

# Convert records to dict items
items = records_to_items(records)  # [{"text": ..., "metadata": ...}, ...]

# Convert to framework documents
haystack_docs = DocumentConverter.to_haystack(items)
langchain_docs = DocumentConverter.to_langchain(items)
```

## 8. Backend-Specific Behavior Differences

Dataloaders are framework-agnostic, but conversion behavior differs slightly:

### Haystack Conversion

```python
HaystackDocument(content=item["text"], meta=item["metadata"])
```

- Metadata stored in `doc.meta` dict.
- Embeddings added separately via `doc.embedding` after embedding step.
- ID auto-generated from content hash unless explicitly provided.

### LangChain Conversion

```python
LangChainDocument(page_content=item["text"], metadata=item["metadata"])
```

- Metadata stored in `doc.metadata` dict.
- Embeddings added separately via embedding models.
- ID management depends on vector store (some auto-generate, some require explicit IDs).

### Dataset-Specific Metadata Fields

| Dataset | Key Metadata Fields |
|---------|---------------------|
| **TriviaQA** | `question`, `answers`, `title`, `rank`, `evidence_index`, `id` |
| **ARC** | `question`, `choices`, `answerKey`, `id` |
| **PopQA** | `entity`, `paragraph`, `question`, `answers` |
| **FActScore** | `question`, `answers`, `source` |
| **Earnings Calls** | `question`, `answer`, `quarter`, `year`, `company` |

## 9. Configuration Semantics

### DataloaderCatalog.create Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | Literal | Required | Dataset identifier (`"triviaqa"`, `"arc"`, etc.) |
| `split` | str | `"test"` | Dataset split (`"train"`, `"test"`, `"validation"`) |
| `limit` | int | `None` | Max records to load (None = all) |
| `dataset_id` | str | `None` | Override HuggingFace dataset ID |

### BaseDatasetLoader Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_name` | str | Required | HuggingFace dataset identifier |
| `split` | str | Required | Dataset split |
| `limit` | int | `None` | Record limit |
| `streaming` | bool | `True` | Use streaming mode (memory efficient) |

### Default Dataset Limits (from `utils/config.py`)

| Dataset | Default Index Limit | Default Eval Limit | Typical Use Case |
|---------|---------------------|-------------------|------------------|
| **TriviaQA** | 500 | 100 | Open-domain QA benchmarking |
| **ARC** | 1000 | 200 | Science QA (multiple choice) |
| **PopQA** | 500 | 100 | Entity-centric QA |
| **FActScore** | 500 | 100 | Factuality evaluation |
| **Earnings Calls** | 300 | 50 | Long-form financial QA |

Access via:

```python
from vectordb.utils.config import get_dataset_limits
limits = get_dataset_limits("triviaqa")  # {"index_limit": 500, "eval_limit": 100}
```

### HuggingFace Dataset Configuration

Some datasets require specific configurations:

```python
# TriviaQA requires "rc" config for retrieval corpus
loader = DataloaderCatalog.create("triviaqa", split="test", limit=500, dataset_id="trivia_qa")
# Internally uses: load_dataset("trivia_qa", "rc", split="test")

# ARC uses "AI2-ARC" config
loader = DataloaderCatalog.create("arc", split="test", limit=1000)
# Internally uses: load_dataset("ai2_arc", split="test")
```

## 10. Failure Modes and Edge Cases

### Common Failure Modes

| Failure | Cause | Mitigation |
|---------|-------|------------|
| **UnsupportedDatasetError** | Invalid dataset name in `DataloaderCatalog.create` | Use `DataloaderCatalog.supported_datasets()` to check valid names |
| **DatasetLoadError** | HuggingFace dataset unavailable, network error, auth required | Check dataset access, network connectivity, use streaming mode |
| **DatasetValidationError** | Missing required fields in row schema | Verify dataset config matches expected schema; check HuggingFace dataset docs |
| **Empty results** | Limit too small, split doesn't exist, streaming timeout | Increase limit, verify split name, use non-streaming mode |
| **Answer extraction fails** | Answers stored in unexpected field | Check dataset schema; some use `answer` (singular), others `answers` (plural) |
| **Query deduplication too aggressive** | Different questions normalize to same text | Review normalization logic; case-folding may merge distinct queries |

### Dataset-Specific Edge Cases

**TriviaQA**:

- One question → multiple evidence documents → limit counts records, not questions.
- 500 records may represent only ~50-100 unique questions.
- `evaluation_queries()` deduplication is critical for fair metrics.

**ARC**:

- Multiple-choice format requires special handling for answer extraction.
- `answerKey` field contains letter (A/B/C/D), not full answer text.
- Choices must be mapped to answer key for evaluation.

**PopQA**:

- Uses `entity` field instead of `question` for query text.
- `EvaluationExtractor` checks both `question` and `entity` fields.
- Some entities may have empty paragraphs → skipped during parsing.

**Earnings Calls**:

- Long documents (transcript excerpts) may exceed embedding model limits.
- Requires chunking strategy before indexing (not handled by dataloader).
- Financial terminology may need domain-specific embedding models.

### Streaming vs. Non-Streaming

**Streaming mode** (`streaming=True`):

- Memory efficient for large datasets.
- Cannot shuffle or random access.
- May be slower due to network latency per row.

**Non-streaming mode** (`streaming=False`):

- Loads entire dataset into memory.
- Enables shuffling, random access.
- Faster for small datasets (<10k records).

## 11. Practical Usage Examples

### Example 1: Basic TriviaQA Benchmarking

```python
from vectordb.dataloaders.catalog import DataloaderCatalog
from vectordb.haystack.semantic_search import ChromaSemanticIndexingPipeline, ChromaSemanticSearchPipeline

# Load dataset
loader = DataloaderCatalog.create("triviaqa", split="test", limit=500)
dataset = loader.load()

# Convert to Haystack documents
docs = dataset.to_haystack()

# Index documents
indexer = ChromaSemanticIndexingPipeline(config_path="configs/chroma_triviaqa.yaml")
indexer.run(documents=docs)

# Extract evaluation queries
queries = dataset.evaluation_queries(limit=100)

# Benchmark retrieval
searcher = ChromaSemanticSearchPipeline(config_path="configs/chroma_triviaqa.yaml")
correct = 0
for q in queries:
    results = searcher.search(q.query, top_k=5)
    # Check if any result contains an answer
    for doc in results:
        if any(ans.lower() in doc.content.lower() for ans in q.answers):
            correct += 1
            break

recall = correct / len(queries)
print(f"Recall@5: {recall:.2%}")
```

### Example 2: Framework Comparison (Haystack vs. LangChain)

```python
from vectordb.dataloaders.catalog import DataloaderCatalog

# Load once
loader = DataloaderCatalog.create("arc", split="test", limit=1000)
dataset = loader.load()

# Convert to both frameworks
haystack_docs = dataset.to_haystack()
langchain_docs = dataset.to_langchain()

# Index with Haystack pipeline
haystack_pipeline.run(documents=haystack_docs)

# Index with LangChain pipeline
langchain_pipeline.add_documents(langchain_docs)

# Same evaluation queries for both
queries = dataset.evaluation_queries(limit=200)
```

### Example 3: Custom Dataset Limit

```python
from vectordb.dataloaders.catalog import DataloaderCatalog

# Override default limits
loader = DataloaderCatalog.create(
    "popqa",
    split="test",
    limit=250,  # Custom limit
)
dataset = loader.load()

# Get dataset limits from config
from vectordb.utils.config import get_dataset_limits
limits = get_dataset_limits("popqa")
print(f"Default index limit: {limits['index_limit']}, eval limit: {limits['eval_limit']}")
```

### Example 4: Implementing a Custom Loader

```python
from vectordb.dataloaders.base import BaseDatasetLoader
from vectordb.dataloaders.types import DatasetRecord, DatasetType
from datasets import load_dataset

class CustomQALoader(BaseDatasetLoader):
    """Load custom QA dataset."""

    def __init__(self, split="test", limit=None):
        super().__init__(
            dataset_name="my-org/custom-qa",
            split=split,
            limit=limit,
            streaming=True,
        )

    @property
    def dataset_type(self) -> DatasetType:
        # Add "custom_qa" to DatasetType Literal first
        return "triviaqa"  # Reuse existing type or extend

    def _load_dataset_iterable(self):
        return load_dataset(self.dataset_name, split=self.split, streaming=self.streaming)

    def _parse_row(self, row) -> list[DatasetRecord]:
        return [
            DatasetRecord(
                text=row["passage"],
                metadata={
                    "question": row["question"],
                    "answers": [row["answer"]],
                    "source": row["source"],
                }
            )
        ]

# Register in catalog
DataloaderCatalog._REGISTRY["custom_qa"] = CustomQALoader

# Use it
loader = DataloaderCatalog.create("custom_qa", split="test", limit=500)
```

### Example 5: Evaluation Query Analysis

```python
from vectordb.dataloaders.catalog import DataloaderCatalog

loader = DataloaderCatalog.create("triviaqa", split="test", limit=500)
dataset = loader.load()

# Analyze query distribution
queries = dataset.evaluation_queries()
print(f"Total unique queries: {len(queries)}")
print(f"Total records: {len(dataset.records())}")
print(f"Expansion ratio: {len(dataset.records()) / len(queries):.1f}x")

# Inspect query structure
q = queries[0]
print(f"Query: {q.query}")
print(f"Answers: {q.answers}")
print(f"Relevant doc IDs: {q.relevant_doc_ids}")
print(f"Metadata keys: {q.metadata.keys()}")
```

## 12. Source Walkthrough Map

### Core Module Files

| File | Lines | Key Classes/Functions |
|------|-------|----------------------|
| `src/vectordb/dataloaders/base.py` | ~100 | `BaseDatasetLoader` (abstract base) |
| `src/vectordb/dataloaders/catalog.py` | ~40 | `DataloaderCatalog` |
| `src/vectordb/dataloaders/dataset.py` | ~50 | `LoadedDataset` |
| `src/vectordb/dataloaders/types.py` | ~50 | `DatasetRecord`, `EvaluationQuery`, exceptions |
| `src/vectordb/dataloaders/converters.py` | ~60 | `DocumentConverter`, `records_to_items` |
| `src/vectordb/dataloaders/evaluation.py` | ~80 | `EvaluationExtractor` |

### Dataset Loader Implementations

| File | Dataset | HuggingFace ID | Key Features |
|------|---------|----------------|--------------|
| `src/vectordb/dataloaders/datasets/triviaqa.py` | TriviaQA | `trivia_qa` (config: `rc`) | Multi-evidence expansion, ranked results |
| `src/vectordb/dataloaders/datasets/arc.py` | ARC | `ai2_arc` | Multiple-choice, answer key mapping |
| `src/vectordb/dataloaders/datasets/popqa.py` | PopQA | `akariasai/PopQA` | Entity-centric, paragraph-based |
| `src/vectordb/dataloaders/datasets/factscore.py` | FActScore | `dskar/FActScore` | Factuality evaluation |
| `src/vectordb/dataloaders/datasets/earnings_calls.py` | Earnings Calls | `lamini/earnings-calls-qa` | Long-form financial QA |

### Supporting Files

| File | Purpose |
|------|---------|
| `src/vectordb/dataloaders/__init__.py` | Module exports |
| `src/vectordb/dataloaders/README.md` | Architecture overview |
| `src/vectordb/utils/config.py` | Dataset limit defaults |

### Test Files

| File | Coverage |
|------|----------|
| `tests/dataloaders/test_catalog.py` | Catalog creation, unsupported datasets |
| `tests/dataloaders/test_dataset.py` | LoadedDataset conversions |
| `tests/dataloaders/test_converters.py` | Framework document conversion |
| `tests/dataloaders/test_evaluation.py` | Query extraction, deduplication |
| `tests/dataloaders/datasets/` | Per-dataset loader tests |

---

**Next Steps**: After understanding dataloaders, proceed to:

- **Shared Utils** (`docs/core/shared-utils.md`) for configuration, logging, and evaluation utilities.
- **Framework feature modules** (`docs/haystack/` or `docs/langchain/`) for retrieval pipeline implementations.
