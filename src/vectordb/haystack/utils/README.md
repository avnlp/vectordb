# Haystack Utilities

Shared helper modules used across all Haystack feature pipelines. These utilities encapsulate common concerns such as configuration loading, embedding creation, filtering, and result processing, so that individual feature modules remain focused on their specific pipeline logic.

## Modules

### Configuration Loading

`config.py` - Loads YAML configuration files with support for environment variable resolution. Recognizes both `${VAR}` and `${VAR:-default}` syntax for injecting secrets and connection parameters at runtime. Validates that all required database-specific settings are present and resolves model name aliases to their canonical identifiers.

### Embedder Factory

`embeddings.py` -- Creates dense and sparse embedding components from configuration. Accepts a model specification and returns the appropriate Haystack-compatible embedder, supporting multiple model backends. Used by both indexing pipelines (for document embedding) and search pipelines (for query embedding).

### Document Filtering

`filters.py` -- Applies metadata-based filters to documents using a declarative filter specification. Supports operators for equality, comparison, set membership, and substring matching. Allows pipelines to narrow retrieval results based on structured metadata fields.

### RAG Generation Helper

`rag.py` -- Creates large language model generators using Groq-compatible API endpoints. Handles prompt formatting for retrieval-augmented generation, combining retrieved context with the user query into a structured prompt suitable for answer generation.

### Reranker Factory

`reranker.py` -- Creates cross-encoder rerankers and diversity-based rankers from configuration. Supports local cross-encoder models and API-based reranking services. Returns a configured reranker component ready for insertion into a Haystack pipeline.

### Dataset Loading Helper

`dataloader.py` -- Integrates with the project's dataloader system to load and prepare documents for any supported dataset. Provides a uniform interface so that feature pipelines can load documents without knowing the details of each dataset's format.

### Result Fusion

`fusion.py` -- Implements strategies for merging results from multiple retrieval sources. Supports reciprocal rank fusion and weighted score merging, enabling hybrid search pipelines to combine dense and sparse retrieval results into a single ranked list.

### Result Diversification

`diversification.py` -- Applies semantic diversity filtering to reduce redundancy in search results. Uses similarity-based clustering to ensure that the final result set covers a broad range of relevant information rather than concentrating on a single topic.

## Directory Structure

```
utils/
    __init__.py
    config.py
    dataloader.py
    diversification.py
    embeddings.py
    filters.py
    fusion.py
    rag.py
    reranker.py
```
