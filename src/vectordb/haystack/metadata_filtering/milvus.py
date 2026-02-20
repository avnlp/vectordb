"""Milvus metadata filtering pipeline for Haystack.

Implements BaseMetadataFilteringPipeline for Milvus vector database with
pre-filtering and vector search capabilities.
"""

from typing import Any

import numpy as np
from haystack import Document
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections

from vectordb.haystack.metadata_filtering.base import (
    BaseMetadataFilteringPipeline,
    Timer,
)
from vectordb.haystack.metadata_filtering.vectordb_pipeline_type import (
    FilteredQueryResult,
    FilterSpec,
    MilvusFilterExpressionBuilder,
    TimingMetrics,
    parse_filter_from_config,
)


__all__ = ["MilvusMetadataFilteringPipeline"]


class MilvusMetadataFilteringPipeline(BaseMetadataFilteringPipeline):
    """Milvus metadata filtering pipeline with pre-filter and vector search.

    Connects to Milvus server, indexes documents with metadata, applies
    pre-filtering, then performs vector search on matching candidates.

    Attributes:
        collection: Milvus collection instance.
        dimension: Embedding vector dimension.
    """

    def __init__(self, config_path: str) -> None:
        """Initialize Milvus pipeline from configuration.

        Args:
            config_path: Path to YAML configuration file.
        """
        self.collection: Collection | None = None
        self.dimension: int = 384  # Default for all-MiniLM-L6-v2
        super().__init__(config_path)

    def _connect(self) -> None:
        """Establish connection to Milvus server.

        Reads connection parameters from config and connects using pymilvus.
        """
        milvus_config = self.config.get("milvus", {})
        host = milvus_config.get("host", "localhost")
        port = milvus_config.get("port", "19530")

        connections.connect(alias="default", host=host, port=port)
        self.logger.info("Connected to Milvus at %s:%s", host, port)

    def _create_collection(self, collection_name: str, dimension: int) -> Collection:
        """Create Milvus collection with metadata schema.

        Args:
            collection_name: Name for the collection.
            dimension: Vector embedding dimension.

        Returns:
            Created or existing Milvus Collection.
        """
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]
        schema = CollectionSchema(fields, description="Metadata filtering collection")

        collection = Collection(name=collection_name, schema=schema)
        collection.create_index(
            field_name="embedding",
            index_params={
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            },
        )
        collection.load()

        self.logger.info("Created collection: %s", collection_name)
        return collection

    def _index_documents(self, documents: list[Document]) -> int:
        """Index documents with embeddings and metadata into Milvus.

        Args:
            documents: List of Haystack documents with embeddings.

        Returns:
            Number of documents indexed.
        """
        if not self.collection:
            raise ValueError("Collection not initialized")

        embeddings = [doc.embedding for doc in documents if doc.embedding is not None]
        contents = [doc.content or "" for doc in documents]
        metadatas = [doc.meta for doc in documents]

        self.collection.insert([embeddings, contents, metadatas])
        self.collection.flush()

        self.logger.info("Indexed %d documents", len(documents))
        return len(documents)

    def _pre_filter(self, filter_expr: str) -> int:
        """Apply pre-filter and count matching candidates.

        Args:
            filter_expr: Milvus filter expression string.

        Returns:
            Number of documents matching the filter.
        """
        if not self.collection or not filter_expr:
            return self.collection.num_entities if self.collection else 0

        # Query to count matching documents
        results = self.collection.query(expr=filter_expr, output_fields=["id"])
        return len(results)

    def _vector_search(
        self,
        query_embedding: list[float],
        filter_expr: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Execute vector search with filter expression.

        Args:
            query_embedding: Query vector embedding.
            filter_expr: Milvus filter expression.
            top_k: Number of results to return.

        Returns:
            List of search results with id, score, content, metadata.
        """
        if not self.collection:
            raise ValueError("Collection not initialized")

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=filter_expr if filter_expr else None,
            output_fields=["content", "metadata"],
        )

        search_results = []
        for hits in results:
            for hit in hits:
                search_results.append(
                    {
                        "id": hit.id,
                        "score": hit.score,
                        "content": hit.entity.get("content", ""),
                        "metadata": hit.entity.get("metadata", {}),
                    }
                )

        return search_results

    def run(self) -> list[FilteredQueryResult]:
        """Execute complete metadata filtering pipeline.

        Orchestrates:
        1. Initialize embedder
        2. Create/connect to collection
        3. Load and index documents
        4. Parse filter from config
        5. Build Milvus filter expression
        6. Pre-filter to count candidates
        7. Run vector search on candidates
        8. Return ranked results with timing

        Returns:
            List of FilteredQueryResult objects with timing metrics.
        """
        self._init_embedder()

        collection_config = self.config.get("collection", {})
        collection_name = collection_config.get("name", "metadata_filtered")
        embeddings_config = self.config.get("embeddings", {})
        self.dimension = embeddings_config.get("dimension", 384)

        self.collection = self._create_collection(collection_name, self.dimension)

        # In production, use dataloader from config
        documents: list[Document] = []
        dataloader_config = self.config.get("dataloader", {})
        limit = dataloader_config.get("limit", 50)
        self.logger.info("Dataloader configured with limit: %d", limit)

        # Skip indexing if no documents
        if documents:
            # Embed documents
            if self.embedder:
                embedded = self.embedder.run(documents)
                documents = embedded.get("documents", documents)
            self._index_documents(documents)

        filter_spec: FilterSpec = parse_filter_from_config(self.config)
        schema = self._get_metadata_schema()

        builder = MilvusFilterExpressionBuilder(schema)
        filter_expr = builder.build(filter_spec)
        self.logger.info("Built filter expression: %s", filter_expr)

        metadata_config = self.config.get("metadata_filtering", {})
        test_query = metadata_config.get("test_query", "test query")

        if self.embedder:
            # Use text embedder for query
            from haystack.components.embedders import SentenceTransformersTextEmbedder

            embeddings_config = self.config.get("embeddings", {})
            model_name = embeddings_config.get(
                "model", "sentence-transformers/all-MiniLM-L6-v2"
            )
            text_embedder = SentenceTransformersTextEmbedder(model=model_name)
            text_embedder.warm_up()
            query_result = text_embedder.run(text=test_query)
            query_embedding = query_result["embedding"]
        else:
            query_embedding = list(np.zeros(self.dimension))

        # Pre-filter timing
        with Timer() as pre_filter_timer:
            num_candidates = self._pre_filter(filter_expr)

        # Vector search timing
        with Timer() as search_timer:
            search_results = self._vector_search(query_embedding, filter_expr)

        total_docs = self.collection.num_entities if self.collection else 0

        results: list[FilteredQueryResult] = []
        timing = TimingMetrics(
            pre_filter_ms=pre_filter_timer.elapsed_ms,
            vector_search_ms=search_timer.elapsed_ms,
            total_ms=pre_filter_timer.elapsed_ms + search_timer.elapsed_ms,
            num_candidates=num_candidates,
            num_total_docs=total_docs,
        )

        for rank, result in enumerate(search_results, start=1):
            doc = Document(
                content=result["content"],
                meta=result["metadata"],
            )
            results.append(
                FilteredQueryResult(
                    document=doc,
                    relevance_score=result["score"],
                    rank=rank,
                    filter_matched=True,
                    timing=timing if rank == 1 else None,
                )
            )

        self.logger.info(
            "Pipeline complete: %d results, pre_filter=%.2fms, search=%.2fms",
            len(results),
            timing.pre_filter_ms,
            timing.vector_search_ms,
        )

        return results
