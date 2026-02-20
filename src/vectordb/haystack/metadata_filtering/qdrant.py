"""Qdrant metadata filtering pipeline for Haystack.

Implements BaseMetadataFilteringPipeline for Qdrant vector database with
pre-filtering and vector search capabilities.
"""

from typing import Any

import numpy as np
from haystack import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from vectordb.haystack.metadata_filtering.base import (
    BaseMetadataFilteringPipeline,
    Timer,
)
from vectordb.haystack.metadata_filtering.vectordb_pipeline_type import (
    FilteredQueryResult,
    FilterSpec,
    QdrantFilterExpressionBuilder,
    TimingMetrics,
    parse_filter_from_config,
)


__all__ = ["QdrantMetadataFilteringPipeline"]


class QdrantMetadataFilteringPipeline(BaseMetadataFilteringPipeline):
    """Qdrant metadata filtering pipeline with pre-filter and vector search.

    Connects to Qdrant server, indexes documents with metadata, applies
    pre-filtering, then performs vector search on matching candidates.

    Attributes:
        client: Qdrant client instance.
        collection_name: Name of the collection.
        dimension: Embedding vector dimension.
    """

    def __init__(self, config_path: str) -> None:
        """Initialize Qdrant pipeline from configuration.

        Args:
            config_path: Path to YAML configuration file.
        """
        self.client: QdrantClient | None = None
        self.collection_name: str = ""
        self.dimension: int = 384  # Default for all-MiniLM-L6-v2
        super().__init__(config_path)

    def _connect(self) -> None:
        """Establish connection to Qdrant server.

        Reads connection parameters from config and creates QdrantClient.
        """
        qdrant_config = self.config.get("qdrant", {})
        host = qdrant_config.get("host", "localhost")
        port = int(qdrant_config.get("port", 6333))
        api_key = qdrant_config.get("api_key")

        self.client = QdrantClient(host=host, port=port, api_key=api_key or None)
        self.logger.info("Connected to Qdrant at %s:%s", host, port)

    def _create_collection(self, collection_name: str, dimension: int) -> None:
        """Create Qdrant collection with vector configuration.

        Args:
            collection_name: Name for the collection.
            dimension: Vector embedding dimension.
        """
        if not self.client:
            raise ValueError("Client not initialized")

        self.collection_name = collection_name

        collections = self.client.get_collections().collections
        if any(c.name == collection_name for c in collections):
            self.logger.info("Collection %s already exists", collection_name)
            return

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )
        self.logger.info("Created collection: %s", collection_name)

    def _index_documents(self, documents: list[Document]) -> int:
        """Index documents with embeddings and metadata into Qdrant.

        Args:
            documents: List of Haystack documents with embeddings.

        Returns:
            Number of documents indexed.
        """
        if not self.client:
            raise ValueError("Client not initialized")

        points = []
        for idx, doc in enumerate(documents):
            if doc.embedding is None:
                continue
            points.append(
                PointStruct(
                    id=idx,
                    vector=doc.embedding,
                    payload={"content": doc.content or "", **doc.meta},
                )
            )

        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

        self.logger.info("Indexed %d documents", len(points))
        return len(points)

    def _pre_filter(self, filter_obj: Any) -> int:
        """Apply pre-filter and count matching candidates.

        Args:
            filter_obj: Qdrant Filter object.

        Returns:
            Number of documents matching the filter.
        """
        if not self.client:
            return 0

        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=filter_obj,
            limit=10000,
            with_payload=False,
            with_vectors=False,
        )
        return len(points)

    def _vector_search(
        self,
        query_embedding: list[float],
        filter_obj: Any,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Execute vector search with filter.

        Args:
            query_embedding: Query vector embedding.
            filter_obj: Qdrant Filter object.
            top_k: Number of results to return.

        Returns:
            List of search results with id, score, content, metadata.
        """
        if not self.client:
            raise ValueError("Client not initialized")

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=filter_obj,
            limit=top_k,
            with_payload=True,
        )

        search_results = []
        for hit in results:
            payload = hit.payload or {}
            content = payload.pop("content", "")
            search_results.append(
                {
                    "id": hit.id,
                    "score": hit.score,
                    "content": content,
                    "metadata": payload,
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
        5. Build Qdrant filter object
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

        self._create_collection(collection_name, self.dimension)

        documents: list[Document] = []
        dataloader_config = self.config.get("dataloader", {})
        limit = dataloader_config.get("limit", 50)
        self.logger.info("Dataloader configured with limit: %d", limit)

        # Skip indexing if no documents
        if documents:
            if self.embedder:
                embedded = self.embedder.run(documents)
                documents = embedded.get("documents", documents)
            self._index_documents(documents)

        filter_spec: FilterSpec = parse_filter_from_config(self.config)
        schema = self._get_metadata_schema()

        builder = QdrantFilterExpressionBuilder(schema)
        filter_obj = builder.build(filter_spec)
        self.logger.info("Built filter object: %s", filter_obj)

        metadata_config = self.config.get("metadata_filtering", {})
        test_query = metadata_config.get("test_query", "test query")

        if self.embedder:
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
            num_candidates = self._pre_filter(filter_obj)

        # Vector search timing
        with Timer() as search_timer:
            search_results = self._vector_search(query_embedding, filter_obj)

        if self.client:
            collection_info = self.client.get_collection(self.collection_name)
            total_docs = collection_info.points_count or 0
        else:
            total_docs = 0

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
