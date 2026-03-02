# Dataloaders

The dataloaders module converts raw benchmark datasets from HuggingFace into a consistent internal format so that retrieval pipelines can be benchmarked fairly across backends and features.

## Why This Matters

Different QA datasets have different schemas, answer formats, and evidence document structures. Without normalization, comparing retrieval quality across features becomes meaningless because the inputs differ. This module defines a single normalized record type — `DatasetRecord(text, metadata)` — and a single evaluation query type — `EvaluationQuery(query, answers, relevant_doc_ids, metadata)` — that all datasets produce.

The same loaded dataset can be converted to either Haystack `Document` objects or LangChain `Document` objects with a single method call, making it straightforward to run the same benchmark against both frameworks.

## Architecture

```
DataloaderCatalog.create(name, split, limit)
        │
        ▼
BaseDatasetLoader (abstract)
        │  _load_dataset_iterable()  →  raw HuggingFace rows
        │  _parse_row()             →  List[DatasetRecord]
        │  load()                   →  LoadedDataset
        ▼
LoadedDataset
        │  .records()              →  List[DatasetRecord]
        │  .to_haystack()          →  List[haystack.Document]
        │  .to_langchain()         →  List[langchain.Document]
        │  .evaluation_queries()   →  List[EvaluationQuery]
```

## Supported Datasets

| Loader | HuggingFace ID | Description |
|---|---|---|
| `TriviaQALoader` | `trivia_qa` (config: `rc`) | Open-domain QA. Each row contains a question and multiple ranked evidence documents from web search. Each evidence document becomes a separate `DatasetRecord`. |
| `ARCLoader` | `ai2_arc` | Science QA (AI2 Reasoning Challenge). Multiple-choice questions with answer choices. Each question with its choices becomes one record. |
| `PopQALoader` | `akariasai/PopQA` | Entity-centric QA derived from Wikipedia. Each row produces one record with the entity and associated passage. |
| `FactScoreLoader` | `dskar/FActScore` | Factuality-focused QA for evaluating answer grounding. |
| `EarningsCallsLoader` | `lamini/earnings-calls-qa` | Financial QA from earnings call transcripts. Longer documents suitable for testing chunking and compression. |

## Using the Catalog

The `DataloaderCatalog` is the primary entry point. It maps dataset names to loader classes and instantiates them with consistent parameters.

```python
from vectordb.dataloaders.catalog import DataloaderCatalog

# Load 500 TriviaQA records from the test split
loader = DataloaderCatalog.create("triviaqa", split="test", limit=500)
dataset = loader.load()

# Convert to Haystack documents for indexing
haystack_docs = dataset.to_haystack()

# Convert to LangChain documents for indexing
langchain_docs = dataset.to_langchain()

# Extract evaluation queries for benchmarking (deduplicated by query text)
queries = dataset.evaluation_queries(limit=100)
for q in queries:
    print(q.query, q.answers, q.relevant_doc_ids)
```

Supported dataset names: `"triviaqa"`, `"arc"`, `"popqa"`, `"factscore"`, `"earnings_calls"`.

## Normalized Records

Every dataset is normalized into `DatasetRecord` objects:

```python
@dataclass(frozen=True, slots=True)
class DatasetRecord:
    text: str                  # Document content to index
    metadata: dict[str, Any]   # Dataset-specific metadata (question, answers, etc.)
```

And every evaluation query is normalized into `EvaluationQuery` objects:

```python
@dataclass(frozen=True, slots=True)
class EvaluationQuery:
    query: str                   # The question text
    answers: list[str]           # Ground-truth answer strings
    relevant_doc_ids: list[str]  # IDs of known relevant documents (when available)
    metadata: dict[str, Any]     # Additional metadata
```

## Document Conversion

The `DocumentConverter` class handles the final transformation from normalized records to framework objects:

```python
from vectordb.dataloaders.converters import DocumentConverter, records_to_items

# Convert to Haystack Documents
items = records_to_items(records)   # [{text: ..., metadata: ...}, ...]
haystack_docs = DocumentConverter.to_haystack(items)
# Each item becomes: HaystackDocument(content=item["text"], meta=item["metadata"])

# Convert to LangChain Documents
langchain_docs = DocumentConverter.to_langchain(items)
# Each item becomes: LangChainDocument(page_content=item["text"], metadata=item["metadata"])
```

## Evaluation Query Extraction

`EvaluationExtractor.extract()` deduplicates queries by normalizing whitespace and case before comparing. This is necessary because TriviaQA can contain the same question paired with multiple evidence documents — each becomes a separate record, but the query should only be evaluated once.

The extractor reads query text from either `metadata["question"]` (most datasets) or `metadata["entity"]` (PopQA). Answers are read from `metadata["answers"]` (list) or `metadata["answer"]` (string) and normalized to a list.

## Default Dataset Limits

Predefined indexing and evaluation limits are stored in `utils/config.py`:

| Dataset | Default Index Limit | Default Eval Limit |
|---|---|---|
| TriviaQA | 500 | 100 |
| ARC | 1000 | 200 |
| PopQA | 500 | 100 |
| FActScore | 500 | 100 |
| Earnings Calls | 300 | 50 |

These defaults are available via `get_dataset_limits(dataset_name)` and are used by pipeline configs to ensure reproducible experiments.

## Implementing a Custom Loader

To add a new dataset, subclass `BaseDatasetLoader` and implement three methods:

```python
from vectordb.dataloaders.base import BaseDatasetLoader
from vectordb.dataloaders.types import DatasetRecord, DatasetType

class MyLoader(BaseDatasetLoader):
    def __init__(self, split="test", limit=None):
        super().__init__("my/hf-dataset", split=split, limit=limit)

    @property
    def dataset_type(self) -> DatasetType:
        return "my_dataset"  # add to DatasetType Literal first

    def _load_dataset_iterable(self):
        from datasets import load_dataset
        return load_dataset(self.dataset_name, split=self.split, streaming=self.streaming)

    def _parse_row(self, row) -> list[DatasetRecord]:
        return [DatasetRecord(
            text=row["passage"],
            metadata={"question": row["question"], "answer": row["answer"]}
        )]
```

Then register it in `DataloaderCatalog._REGISTRY`.

## Common Pitfalls

- Mixing differently normalized datasets in the same benchmark without documenting which metadata fields are available. Some features (such as metadata filtering) require specific metadata fields to be present.
- Comparing retrieval quality between feature runs that used different split sizes or random subsets.
- Ignoring deduplication: TriviaQA expands one question into many records (one per evidence document). The `limit` parameter counts records, not questions, so 500 records may represent far fewer unique questions.
