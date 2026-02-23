"""Pinecone indexing pipeline for diversity filtering feature.

Indexes documents from standard datasets into Pinecone for diversity-aware
retrieval. Supports TriviaQA, ARC, PopQA, FactScore, and Earnings Calls.

Pipeline Flow:
1. Load dataset via DatasetRegistry with configurable split and limit
2. Convert to Haystack Document objects with metadata preservation
3. Embed documents using SentenceTransformersDocumentEmbedder
4. Index into Pinecone with specified index name

Embedding Configuration:
- Model: Configurable sentence-transformers model (default: all-MiniLM-L6-v2)
- Batch size: Configurable for throughput optimization
- Device: Auto-detected (cuda/cpu/mps) or explicitly specified

Index Management:
Indexes documents into existing Pinecone index. Pinecone indexes must be
created beforehand via Pinecone console or API with appropriate dimension.
Documents are indexed with full embedding vectors for similarity search.
"""

from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder

from vectordb.databases.pinecone import PineconeVectorDB
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
    """Run Pinecone indexing pipeline.

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

    embedded_docs = embedder.run(documents=documents)["documents"]

    db = PineconeVectorDB(
        api_key=config.vectordb.pinecone.api_key,
        index_name=config.vectordb.pinecone.index_name,
        embedding_dim=config.embedding.dimension,
    )

    db.index_documents(embedded_docs)

    return {
        "documents_indexed": len(embedded_docs),
        "index_name": config.index.name,
        "embedding_model": config.embedding.model,
        "embedding_dimension": config.embedding.dimension,
    }
