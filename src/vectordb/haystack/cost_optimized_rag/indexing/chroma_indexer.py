"""Chroma indexing pipeline for cost-optimized RAG.

Lightweight vector indexing with Chroma's simple persistent storage.
Optimized for development, prototyping, and small-to-medium scale deployments
where operational simplicity outweighs advanced features.

Cost Optimization Strategies:

    Zero Infrastructure Overhead:
        - Single-node persistent storage (SQLite + local files)
        - No distributed coordination or cluster management
        - Runs on existing application servers
        - Eliminates dedicated vector DB infrastructure costs

    Metadata Flattening:
        - Converts nested metadata to flat key-value pairs
        - Avoids complex schema migrations
        - JSON serialization for non-primitive types
        - Reduces storage overhead vs full document stores

    Local Computation:
        - sentence-transformers for local embedding
        - No API costs for embedding generation
        - CPU-based inference on indexing host

Chroma Cost Characteristics:

    Storage Costs:
        - Raw vectors: 4 bytes per dimension
        - Metadata: Depends on document size
        - HNSW index: ~50% overhead on vectors
        - Typical: ~200MB per 100k 768-dim vectors

    Operational Costs:
        - Minimal: Runs alongside application
        - Backup: Simple file copy
        - Monitoring: Basic file system checks

    Scaling Limits:
        - Single-node only (no horizontal scaling)
        - Memory-bound for active collections
        - Suitable for <10M vectors typically

When to Use Chroma:
    - Prototyping and development
    - Small-to-medium scale (<1M documents)
    - Want minimal dependencies and setup
    - Single-node deployment acceptable

When to Consider Alternatives:
    - Production high-availability requirements
    - Large scale (>10M vectors)
    - Need distributed querying
    - Advanced filtering and hybrid search
"""

import json
from pathlib import Path

import chromadb
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder

from vectordb.haystack.cost_optimized_rag.base.config import load_config
from vectordb.haystack.cost_optimized_rag.utils.common import (
    create_logger,
    load_documents_from_config,
)


class ChromaIndexingPipeline:
    """Chroma indexing pipeline with persistent local storage.

    Simplified vector indexing for cost-conscious deployments.
    Uses local sentence-transformers for embeddings and Chroma's
    persistent client for zero-infrastructure storage.

    Cost Architecture:
        - Embedding: $0 (local CPU inference)
        - Storage: Local disk only (no cloud DB costs)
        - Memory: HNSW index loaded in RAM during queries
        - Compute: Minimal overhead beyond embedding

    Performance Characteristics:
        - Indexing: 100-300 docs/sec (CPU-bound embedding)
        - Query latency: 10-50ms for typical collections
        - Memory: O(n) for HNSW index in active state

    Metadata Handling:
        - Flattens nested metadata for Chroma compatibility
        - JSON serialization for complex types
        - String/int/float/bool stored natively

    Example:
        >>> pipeline = ChromaIndexingPipeline("config.yaml")
        >>> pipeline.run()  # Creates local Chroma DB
        # Storage: Single SQLite file + vector segments
    """

    def __init__(self, config_path: str | Path) -> None:
        """Initialize pipeline from config file.

        Args:
            config_path: Path to YAML configuration with Chroma path,
                embedding model, and collection settings.
        """
        self.config = load_config(str(config_path))
        self.logger = create_logger(self.config)

        # Local embedding - zero API costs
        self.embedder = SentenceTransformersDocumentEmbedder(
            model=self.config.embeddings.model,
            batch_size=self.config.embeddings.batch_size,
        )
        self.embedder.warm_up()

        self._connect()

    def _connect(self) -> None:
        """Initialize Chroma persistent client."""
        if self.config.chroma is None:
            msg = "Chroma configuration is missing"
            raise ValueError(msg)

        self.client = chromadb.PersistentClient(path=self.config.chroma.path)
        self.logger.info(
            "Initialized Chroma client (path: %s)",
            self.config.chroma.path,
        )
        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        """Create or retrieve collection with minimal metadata."""
        collection_name = self.config.collection.name

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": self.config.collection.description,
            },
        )
        self.logger.info("Created/retrieved collection: %s", collection_name)

    def run(self) -> None:
        """Execute indexing pipeline with batch processing.

        Pipeline flow:
            1. Load documents from dataset
            2. Compute embeddings locally
            3. Batch add to Chroma with flattened metadata
        """
        self.logger.info("Starting Chroma indexing pipeline")

        documents = load_documents_from_config(self.config)
        self.logger.info("Loaded %d documents", len(documents))

        # Batch embedding
        self.logger.info("Embedding documents")
        result = self.embedder.run(documents=documents)
        embedded_docs: list[Document] = result["documents"]

        self._add_documents(embedded_docs)
        self.logger.info("Indexing complete")

    def _add_documents(self, documents: list[Document]) -> None:
        """Add documents to Chroma with metadata flattening.

        Chroma requires flat metadata (no nested objects).
        Complex types are JSON-serialized for storage.

        Args:
            documents: Documents with pre-computed embeddings.
        """
        batch_size = self.config.embeddings.batch_size

        ids = []
        embeddings = []
        metadatas = []
        documents_list = []

        for doc in documents:
            if doc.embedding is None:
                continue

            ids.append(doc.id)
            embeddings.append(doc.embedding)

            # Flatten metadata for Chroma compatibility
            flat_metadata = {}
            for key, value in doc.meta.items():
                if isinstance(value, (str, int, float, bool)):
                    flat_metadata[key] = value
                else:
                    flat_metadata[key] = json.dumps(value)

            metadatas.append(flat_metadata)
            documents_list.append(doc.content)

        # Batch processing reduces SQLite transaction overhead
        total = len(ids)
        for i in range(0, total, batch_size):
            end = min(i + batch_size, total)
            self.collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end],
                documents=documents_list[i:end],
            )
            self.logger.info("Added batch %d-%d", i, end)
