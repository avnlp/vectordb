"""Qdrant indexing pipeline for diversity filtering feature.

Indexes documents from standard datasets into Qdrant for diversity-aware
retrieval. Supports TriviaQA, ARC, PopQA, FactScore, and Earnings Calls.

Pipeline Flow:
1. Load dataset via DatasetRegistry with configurable split and limit
2. Convert to Haystack Document objects with metadata preservation
3. Embed documents in batches using SentenceTransformersDocumentEmbedder
4. Index into Qdrant with specified collection name and dimension

Embedding Configuration:
- Model: Configurable sentence-transformers model (default: all-MiniLM-L6-v2)
- Batch size: Configurable for memory efficiency (default: 32)
- Device: Auto-detected (cuda/cpu/mps) or explicitly specified

Memory Optimization:
Documents are processed in batches to prevent high memory consumption or
OutOfMemory errors for large datasets. Each batch is embedded sequentially
before being added to the full document list.

Index Management:
Performs incremental indexing by default, upserting documents into existing
collections without data loss. Set `recreate=true` to replace the entire
collection. Configure via YAML:

```yaml
index:
  name: my_collection
  recreate: false  # Default: incremental/upsert (preserves existing data)
  # recreate: true  # Destructive: deletes and recreates collection
```
"""

from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder

from vectordb.databases.qdrant import QdrantVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.diversity_filtering.utils.config_loader import (
    ConfigLoader,
    DiversityFilteringConfig,
)


def load_documents(config: DiversityFilteringConfig) -> list[Document]:
    """Load documents from dataset using DatasetRegistry.

    Args:
        config: Diversity filtering configuration.

    Returns:
        List of Haystack Document objects.
    """
    dataset_config = config.dataset
    loader = DataloaderCatalog.create(
        dataset_config.name,
        split=dataset_config.split,
        limit=dataset_config.max_documents,
    )

    return loader.load().to_haystack()


def run_indexing(config_path: str) -> dict:
    """Run Qdrant indexing pipeline.

    Args:
        config_path: Path to configuration YAML file.

    Returns:
        Dictionary with indexing results including number of documents indexed.

    Raises:
        FileNotFoundError: If config file not found.
        ValueError: If configuration invalid.
    """
    config = ConfigLoader.load(config_path)

    documents = load_documents(config)
    if not documents:
        return {"documents_indexed": 0, "error": "No documents loaded"}

    embedder = SentenceTransformersDocumentEmbedder(
        model=config.embedding.model,
        batch_size=config.embedding.batch_size,
        device=config.embedding.device,
    )
    embedder.warm_up()

    # Process documents in batches to reduce memory consumption
    batch_size = config.embedding.batch_size or 32
    embedded_docs: list[Document] = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        embedded_batch = embedder.run(documents=batch)["documents"]
        embedded_docs.extend(embedded_batch)

    db = QdrantVectorDB(
        url=config.vectordb.qdrant.url,
        api_key=config.vectordb.qdrant.api_key,
        index=config.index.name,
        embedding_dim=config.embedding.dimension,
        recreate_index=config.index.recreate,
    )

    db.index_documents(embedded_docs)

    return {
        "documents_indexed": len(embedded_docs),
        "index_name": config.index.name,
        "embedding_model": config.embedding.model,
        "embedding_dimension": config.embedding.dimension,
        "batches_processed": (len(documents) + batch_size - 1) // batch_size,
    }
