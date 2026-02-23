"""Qdrant indexing pipeline for cost-optimized RAG.

Implements efficient vector indexing with Qdrant's payload-based filtering
and HNSW indexing. Optimized for applications requiring fast hybrid search
(vector similarity + metadata filtering) with minimal operational complexity.

Cost Optimization Strategies:

    Payload Indexing:
        - Creates secondary indexes on metadata fields
        - Enables pre-filtering before vector search
        - Reduces HNSW search space = faster queries
        - Cost: Small storage overhead per indexed field

    Batch Upserts:
        - Uses Qdrant's batch upsert API
        - Single RPC for multiple points
        - Reduces network round-trips significantly
        - Configurable batch sizes for throughput tuning

    Distance Metric Selection:
        - COSINE: Best for normalized embeddings (default)
        - EUCLID: Good for unnormalized vectors
        - DOT: Fastest computation, good for unit vectors
        - Selection impacts query performance

    Local Embeddings:
        - sentence-transformers on indexing nodes
        - Zero per-document API costs
        - Distributed across workers if needed

Qdrant Cost Characteristics:

    Storage Costs (Self-hosted):
        - Raw vectors: 4 bytes × dimension × count
        - HNSW index: ~100% overhead on vectors
        - Payload data: Depends on metadata size
        - Indexed fields: Small btree overhead

    Qdrant Cloud:
        - ~$0.05 per GB/month for storage
        - Query costs based on CPU usage
        - Free tier for small deployments

When to Use Qdrant:
    - Need hybrid search (vector + filter)
    - Want efficient payload filtering
    - Real-time index updates required
    - Prefer simpler ops than Milvus

When to Consider Alternatives:
    - Massive scale (billions) - consider Milvus
    - Ultra-simple use case - Chroma simpler
    - Need managed service - Pinecone easier
"""

from pathlib import Path

from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams

from vectordb.haystack.cost_optimized_rag.base.config import load_config
from vectordb.haystack.cost_optimized_rag.utils.common import (
    create_logger,
    load_documents_from_config,
)


class QdrantIndexingPipeline:
    """Qdrant indexing pipeline with payload indexes and HNSW optimization.

    Optimized for hybrid search scenarios combining vector similarity
    with metadata filtering. Creates secondary indexes on configured
    payload fields for efficient pre-filtering.

    Cost Architecture:
        - Embedding: $0 (local sentence-transformers)
        - Storage: Vectors + HNSW + payload indexes
        - Payload indexes: Enable faster filtered queries
        - Network: Batch upserts minimize RPCs

    Performance Characteristics:
        - Indexing: 200-800 docs/sec
        - Query latency: 5-20ms with HNSW
        - Filtered queries: Benefit from payload indexes

    Payload Index Strategy:
        - Configurable field indexes in YAML
        - Keyword indexes for exact matching
        - Text indexes for full-text search
        - Numeric indexes for range queries

    Example:
        >>> pipeline = QdrantIndexingPipeline("config.yaml")
        >>> pipeline.run()  # Creates collection with payload indexes
    """

    def __init__(self, config_path: str | Path) -> None:
        """Initialize pipeline from config file.

        Args:
            config_path: Path to YAML with Qdrant connection, embedding
                settings, and payload index configuration.
        """
        self.config = load_config(str(config_path))
        self.logger = create_logger(self.config)

        # Local embedding model
        self.embedder = SentenceTransformersDocumentEmbedder(
            model=self.config.embeddings.model,
            batch_size=self.config.embeddings.batch_size,
        )
        self.embedder.warm_up()

        self._connect()

    def _connect(self) -> None:
        """Connect to Qdrant with configured authentication and TLS."""
        if self.config.qdrant is None:
            msg = "Qdrant configuration is missing"
            raise ValueError(msg)

        self.client = QdrantClient(
            host=self.config.qdrant.host,
            port=self.config.qdrant.port,
            api_key=self.config.qdrant.api_key or None,
            https=self.config.qdrant.https,
        )
        self.logger.info(
            "Connected to Qdrant at %s:%d",
            self.config.qdrant.host,
            self.config.qdrant.port,
        )
        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        """Create collection with HNSW index and payload indexes.

        Creates:
            - Vector collection with configurable distance metric
            - HNSW index for fast ANN search
            - Secondary payload indexes for filtering
        """
        collection_name = self.config.collection.name
        vec_config = self.config.indexing.vector_config

        # Recreate for clean state
        try:
            self.client.get_collection(collection_name)
            self.logger.warning(
                "Collection '%s' already exists, recreating",
                collection_name,
            )
            self.client.delete_collection(collection_name)
        except Exception as e:
            self.logger.debug(
                "Collection '%s' does not exist, creating new: %s",
                collection_name,
                str(e),
            )

        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclid": Distance.EUCLID,
            "Dot": Distance.DOT,
        }
        distance = distance_map.get(vec_config.distance, Distance.COSINE)

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vec_config.size, distance=distance),
        )
        self.logger.info(
            "Created collection '%s' (size=%d, distance=%s)",
            collection_name,
            vec_config.size,
            vec_config.distance,
        )

        # Payload indexes enable efficient filtering
        payload_indexes = self.config.indexing.payload_indexes
        schema_type_map = {
            "keyword": PayloadSchemaType.KEYWORD,
            "text": PayloadSchemaType.TEXT,
            "integer": PayloadSchemaType.INTEGER,
            "float": PayloadSchemaType.FLOAT,
            "bool": PayloadSchemaType.BOOL,
        }

        for index_config in payload_indexes:
            field = index_config.get("field")
            schema_type_str = index_config.get("schema_type", "keyword")
            schema_type = schema_type_map.get(
                schema_type_str, PayloadSchemaType.KEYWORD
            )

            if field:
                try:
                    self.client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field,
                        field_schema=schema_type,
                    )
                    self.logger.info(
                        "Created payload index: %s (%s)",
                        field,
                        schema_type_str,
                    )
                except Exception as e:
                    self.logger.warning(
                        "Failed to create payload index %s: %s",
                        field,
                        str(e),
                    )

    def run(self) -> None:
        """Execute indexing pipeline with batch upserts.

        Pipeline flow:
            1. Load documents from dataset
            2. Compute embeddings locally
            3. Batch upsert to Qdrant with payload
        """
        self.logger.info("Starting Qdrant indexing pipeline")

        documents = load_documents_from_config(self.config)
        self.logger.info("Loaded %d documents", len(documents))

        # Batch embedding
        self.logger.info("Embedding documents")
        result = self.embedder.run(documents=documents)
        embedded_docs: list[Document] = result["documents"]

        self._upsert_documents(embedded_docs)
        self.logger.info("Indexing complete")

    def _upsert_documents(self, documents: list[Document]) -> None:
        """Batch upsert documents to Qdrant.

        Uses Qdrant's upsert API for idempotent writes.
        Batch processing minimizes gRPC overhead.

        Args:
            documents: Documents with pre-computed embeddings.
        """
        collection_name = self.config.collection.name
        batch_size = self.config.embeddings.batch_size

        points = []
        for doc in documents:
            if doc.embedding is None:
                continue
            points.append(
                PointStruct(
                    id=doc.id,
                    vector=doc.embedding,
                    payload={
                        "content": doc.content,
                        **doc.meta,
                    },
                )
            )

        # Batch upsert minimizes RPC round-trips
        total = len(points)
        for i in range(0, total, batch_size):
            end = min(i + batch_size, total)
            batch = points[i:end]
            self.client.upsert(collection_name=collection_name, points=batch)
            self.logger.info("Inserted batch %d-%d", i, end)
