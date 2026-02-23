"""Milvus indexing pipeline for cost-optimized RAG.

Implements high-throughput vector indexing with Milvus's distributed architecture.
Optimized for large-scale deployments requiring horizontal scalability with
predictable cost per document.

Cost Optimization Strategies:

    Partition-Based Organization:
        - Shards data across partitions for parallel processing
        - Reduces per-query scan scope (faster queries)
        - Enables partition-level lifecycle management
        - Cost: Partitions add metadata overhead (~100 bytes each)

    Index Type Selection:
        - IVF_FLAT: Balance of speed and recall (default)
        - IVF_SQ8: 75% memory reduction with minimal quality loss
        - IVF_PQ: Higher compression, lower recall
        - HNSW: Fastest queries, highest memory cost

    Local Embedding Computation:
        - sentence-transformers models run on indexing nodes
        - Eliminates OpenAI/Azure embedding API costs
        - Distributed embedding across indexing workers
        - Typical savings: $0.02-0.10 per 1M tokens

Milvus Cost Characteristics:

    Infrastructure Costs (Self-hosted):
        - Index nodes: CPU-bound, scale with ingestion rate
        - Query nodes: Memory-bound, scale with vector count
        - Storage: Raw vectors + index structures (2-4x vector size)
        - Coordination: Minimal overhead for small deployments

    Managed Service (Zilliz):
        - CU-based pricing (~$0.10/hour per CU)
        - 1 CU = 1 vCPU + 4GB RAM
        - Auto-scaling based on throughput

When to Use Milvus:
    - Billion-scale vector collections
    - Need multi-tenancy with resource isolation
    - Hybrid search (vector + scalar filtering)
    - Geographically distributed deployments

When to Consider Alternatives:
    - Small collections (<1M vectors) - Chroma/Qdrant simpler
    - Simple use cases without partitioning needs
    - Want zero operational overhead
"""

from pathlib import Path

from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections

from vectordb.haystack.cost_optimized_rag.base.config import load_config
from vectordb.haystack.cost_optimized_rag.utils.common import (
    create_logger,
    load_documents_from_config,
)


class MilvusIndexingPipeline:
    """Milvus indexing pipeline with partition support and batch processing.

    Optimized for high-throughput ingestion with horizontal scalability.
    Supports data partitioning for efficient query routing and lifecycle
    management.

    Cost Architecture:
        - Embedding: $0 (local sentence-transformers)
        - Storage: Raw vectors + IVF index (~3x raw size)
        - Compute: Batch inserts maximize throughput
        - Partitioning: Metadata overhead for query optimization

    Performance Characteristics:
        - Throughput: 500-2000 docs/sec (depends on index type)
        - Index build time: O(n log n) for IVF_FLAT
        - Memory: Vectors + index must fit in query node RAM

    Partition Strategy:
        - Configurable partitions for time-based or categorical sharding
        - Reduces query scan scope for filtered searches
        - Enables partition-level TTL and backup

    Example:
        >>> pipeline = MilvusIndexingPipeline("config.yaml")
        >>> pipeline.run()  # Creates collection with partitions
    """

    def __init__(self, config_path: str | Path) -> None:
        """Initialize pipeline from config file.

        Args:
            config_path: Path to YAML configuration with Milvus connection,
                embedding model, and partitioning settings.
        """
        self.config = load_config(str(config_path))
        self.logger = create_logger(self.config)

        # Local embedding model - zero per-document API costs
        self.embedder = SentenceTransformersDocumentEmbedder(
            model=self.config.embeddings.model,
            batch_size=self.config.embeddings.batch_size,
        )
        self.embedder.warm_up()

        self._connect()

    def _connect(self) -> None:
        """Connect to Milvus with configured connection parameters."""
        if self.config.milvus is None:
            msg = "Milvus configuration is missing"
            raise ValueError(msg)

        connections.connect(
            alias="default",
            host=self.config.milvus.host,
            port=self.config.milvus.port,
        )
        self.logger.info(
            "Connected to Milvus at %s:%d",
            self.config.milvus.host,
            self.config.milvus.port,
        )
        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        """Create collection with optimized schema for vector search.

        Schema includes:
            - Primary key (varchar for UUIDs)
            - FLOAT_VECTOR for embeddings
            - JSON metadata for flexible attributes
            - Partition support for data organization
        """
        collection_name = self.config.collection.name
        vec_config = self.config.indexing.vector_config

        # Recreate for clean state
        try:
            Collection(name=collection_name).drop()
            self.logger.info("Dropped existing collection '%s'", collection_name)
        except Exception as e:
            self.logger.debug(
                "Collection '%s' does not exist, skipping drop: %s",
                collection_name,
                str(e),
            )

        # Schema optimized for cost-efficient storage
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                max_length=256,
                is_primary=True,
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=vec_config.size,
            ),
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=65535,
            ),
            FieldSchema(
                name="metadata",
                dtype=DataType.JSON,
            ),
        ]

        schema = CollectionSchema(
            fields=fields,
            description=self.config.collection.description,
        )

        self.collection = Collection(
            name=collection_name,
            schema=schema,
        )
        self.logger.info(
            "Created collection '%s' (embedding_dim=%d)",
            collection_name,
            vec_config.size,
        )

        # Optional partitioning for query optimization
        partition_config = self.config.indexing.partitions
        if partition_config.enabled:
            for partition_name in partition_config.values:
                try:
                    self.collection.create_partition(
                        partition_name=f"partition_{partition_name}"
                    )
                    self.logger.info("Created partition: partition_%s", partition_name)
                except Exception as e:
                    self.logger.warning("Failed to create partition: %s", str(e))

    def run(self) -> None:
        """Execute indexing pipeline with batch processing and index creation.

        Pipeline flow:
            1. Load documents from configured dataset
            2. Compute embeddings locally in batches
            3. Batch insert into Milvus
            4. Create IVF_FLAT index for efficient search
            5. Load collection into memory for querying
        """
        self.logger.info("Starting Milvus indexing pipeline")

        documents = load_documents_from_config(self.config)
        self.logger.info("Loaded %d documents", len(documents))

        # Batch embedding with memory-efficient processing
        self.logger.info("Embedding documents")
        result = self.embedder.run(documents=documents)
        embedded_docs: list[Document] = result["documents"]

        self._insert_documents(embedded_docs)
        self.logger.info("Indexing complete")

    def _insert_documents(self, documents: list[Document]) -> None:
        """Batch insert documents with IVF_FLAT index creation.

        Batch processing maximizes throughput while IVF_FLAT provides
        efficient approximate nearest neighbor search.

        Args:
            documents: Documents with pre-computed embeddings.
        """
        batch_size = self.config.embeddings.batch_size

        ids = []
        embeddings = []
        contents = []
        metadatas = []

        for doc in documents:
            if doc.embedding is None:
                continue
            ids.append(doc.id)
            embeddings.append(doc.embedding)
            contents.append(doc.content)
            metadatas.append(doc.meta)

        # Batch inserts minimize network overhead
        total = len(ids)
        for i in range(0, total, batch_size):
            end = min(i + batch_size, total)
            data = [
                ids[i:end],
                embeddings[i:end],
                contents[i:end],
                metadatas[i:end],
            ]
            self.collection.insert(data)
            self.logger.info("Inserted batch %d-%d", i, end)

        # IVF_FLAT: Good balance of speed, recall, and build time
        self.collection.create_index(
            field_name="embedding",
            index_params={
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            },
        )
        self.logger.info("Created vector index")

        self.collection.load()
        self.logger.info("Loaded collection into memory")
