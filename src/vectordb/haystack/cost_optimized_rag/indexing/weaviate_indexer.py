"""Weaviate indexing pipeline for cost-optimized RAG.

Leverages Weaviate's modular architecture for efficient vector storage with
configurable cost-performance trade-offs. Local embedding computation minimizes
API costs while Weaviate handles scalable vector indexing.

Cost Optimization Strategies:

    Local Embedding Computation:
        - Uses sentence-transformers models running locally
        - Eliminates per-token embedding API costs entirely
        - One-time model download vs ongoing API charges
        - Typical savings: $0.10-0.50 per 1M documents vs OpenAI embeddings

    Batch Processing:
        - Configurable batch sizes minimize network round-trips
        - Reduces connection overhead and latency
        - Default 100-document batches balance memory and throughput
        - Larger batches (500-1000) for high-volume indexing

    Schema Design:
        - Minimal property indexing reduces storage overhead
        - Content stored as TEXT for full-text search fallback
        - Metadata serialized as JSON (flexible, queryable)
        - No vectorization module (we provide pre-computed vectors)

Weaviate Cost Characteristics:

    Storage Costs:
        - Self-hosted: Infrastructure cost only (EC2, GKE, etc.)
        - Weaviate Cloud: ~$0.25 per GB/month for vector storage
        - Dimension count directly impacts memory requirements

    Performance Trade-offs:
        - HNSW index: Fast queries, higher memory (default)
        - Flat index: Slower queries, lower memory
        - Dynamic index updates add overhead during indexing

When to Use Weaviate:
    - Multi-modal search requirements (text + other modalities)
    - Need GraphQL interface for complex queries
    - Self-hosted deployment preferred over managed
    - Module ecosystem (reranking, Q&A) valuable

When to Consider Alternatives:
    - Simple vector-only use cases (consider Qdrant/Chroma)
    - Ultra-low latency requirements (<10ms p99)
    - Minimal operational overhead (managed Pinecone)
"""

import json
from pathlib import Path

import weaviate
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from weaviate.classes.config import DataType

from vectordb.haystack.cost_optimized_rag.base.config import load_config
from vectordb.haystack.cost_optimized_rag.utils.common import (
    create_logger,
    load_documents_from_config,
)


class WeaviateIndexingPipeline:
    """Weaviate indexing pipeline using local embeddings and batch upserts.

    Optimized for cost-efficient bulk indexing with configurable throughput.
    Computes embeddings locally using sentence-transformers to eliminate
    per-document API charges.

    Cost Architecture:
        - Embedding: $0 (local computation)
        - Storage: Self-hosted infrastructure or Weaviate Cloud
        - Network: Batch writes minimize API calls
        - Compute: One-time model warmup, reused across batches

    Performance Characteristics:
        - Indexing throughput: 100-500 docs/sec (depends on model size)
        - Memory: O(batch_size × embedding_dim × 4 bytes)
        - Network: One round-trip per batch

    Resource Optimization:
        - Batch size tunes memory vs throughput trade-off
        - Warm-up phase pre-loads model to avoid per-batch loading
        - Schema recreation ensures clean state (drops existing data)

    Example:
        >>> pipeline = WeaviateIndexingPipeline("config.yaml")
        >>> pipeline.run()  # Indexes all documents from config
        # Cost: ~$0 for embeddings on 100k documents
    """

    def __init__(self, config_path: str | Path) -> None:
        """Initialize pipeline from config file.

        Args:
            config_path: Path to YAML configuration with Weaviate and embedding
                settings. Controls batch size, model selection, and connection
                parameters for cost optimization.
        """
        self.config = load_config(str(config_path))
        self.logger = create_logger(self.config)

        # Local embedding model - eliminates API costs
        self.embedder = SentenceTransformersDocumentEmbedder(
            model=self.config.embeddings.model,
            batch_size=self.config.embeddings.batch_size,
        )
        self.embedder.warm_up()

        self._connect()

    def _connect(self) -> None:
        """Establish Weaviate connection with configured authentication."""
        if self.config.weaviate is None:
            msg = "Weaviate configuration is missing"
            raise ValueError(msg)

        self.client = weaviate.Client(
            url=f"http://{self.config.weaviate.host}:{self.config.weaviate.port}",
            auth_client_secret=weaviate.auth.AuthApiKey(self.config.weaviate.api_key)
            if self.config.weaviate.api_key
            else None,
        )
        self.logger.info(
            "Connected to Weaviate at %s:%d",
            self.config.weaviate.host,
            self.config.weaviate.port,
        )
        self._ensure_class_exists()

    def _ensure_class_exists(self) -> None:
        """Create Weaviate class schema optimized for vector storage.

        Schema uses minimal indexed properties to reduce storage overhead.
        Vectorizer disabled since embeddings are computed locally.
        """
        class_name = self.config.collection.name

        # Recreate for clean state - drops existing to avoid migration costs
        try:
            self.client.schema.delete_class(class_name)
            self.logger.info("Deleted existing class '%s'", class_name)
        except Exception as e:
            self.logger.debug(
                "Class '%s' does not exist, skipping delete: %s",
                class_name,
                str(e),
            )

        # Minimal property schema - reduces indexing overhead
        properties = [
            {
                "name": "content",
                "dataType": [DataType.TEXT],
                "description": "Document content",
            },
            {
                "name": "metadata",
                "dataType": [DataType.TEXT],
                "description": "Document metadata as JSON",
            },
        ]

        self.client.schema.create_class(
            {
                "class": class_name,
                "description": self.config.collection.description,
                "properties": properties,
                "vectorizer": "none",  # Pre-computed vectors reduce compute costs
            }
        )
        self.logger.info("Created class '%s'", class_name)

    def run(self) -> None:
        """Execute full indexing pipeline with batch processing.

        Pipeline flow:
            1. Load documents from configured dataset
            2. Compute embeddings locally (batch processing)
            3. Batch upsert to Weaviate

        Cost optimized through batch sizes configured in YAML.
        """
        self.logger.info("Starting Weaviate indexing pipeline")

        documents = load_documents_from_config(self.config)
        self.logger.info("Loaded %d documents", len(documents))

        # Batch embedding - processes all docs in memory-efficient chunks
        self.logger.info("Embedding documents")
        result = self.embedder.run(documents=documents)
        embedded_docs: list[Document] = result["documents"]

        self._insert_documents(embedded_docs)
        self.logger.info("Indexing complete")

    def _insert_documents(self, documents: list[Document]) -> None:
        """Batch insert documents with configurable batch sizing.

        Batch processing minimizes network round-trips and connection overhead.
        Weaviate's batch API handles partial failures gracefully.

        Args:
            documents: Documents with pre-computed embeddings.
        """
        class_name = self.config.collection.name
        batch_size = self.config.embeddings.batch_size

        # Batch upsert minimizes API calls and connection overhead
        with self.client.batch as batch:
            batch.batch_size = batch_size

            for i, doc in enumerate(documents):
                if doc.embedding is None:
                    continue

                properties = {
                    "content": doc.content,
                    "metadata": json.dumps(doc.meta),
                }

                batch.add_data_object(
                    data_object=properties,
                    class_name=class_name,
                    uuid=doc.id,
                    vector=doc.embedding,
                )

                if (i + 1) % batch_size == 0:
                    self.logger.info("Processed %d documents", i + 1)
