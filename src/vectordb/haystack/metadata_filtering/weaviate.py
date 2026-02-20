"""Weaviate metadata filtering pipeline for Haystack.

Implements BaseMetadataFilteringPipeline for Weaviate vector database with
pre-filtering and vector search capabilities.
"""

import os
from typing import Any

import numpy as np
import weaviate
from haystack import Document
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery

from vectordb.haystack.metadata_filtering.base import (
    BaseMetadataFilteringPipeline,
    Timer,
)
from vectordb.haystack.metadata_filtering.vectordb_pipeline_type import (
    FilteredQueryResult,
    FilterSpec,
    TimingMetrics,
    WeaviateFilterExpressionBuilder,
    parse_filter_from_config,
)


__all__ = ["WeaviateMetadataFilteringPipeline"]


class WeaviateMetadataFilteringPipeline(BaseMetadataFilteringPipeline):
    """Weaviate metadata filtering pipeline with pre-filter and vector search.

    Connects to Weaviate, indexes documents with metadata, applies
    pre-filtering, then performs vector search on matching candidates.

    Attributes:
        client: Weaviate client instance.
        collection: Weaviate collection instance.
        collection_name: Name of the collection.
        dimension: Embedding vector dimension.
    """

    def __init__(self, config_path: str) -> None:
        """Initialize Weaviate pipeline from configuration.

        Args:
            config_path: Path to YAML configuration file.
        """
        self.client: weaviate.WeaviateClient | None = None
        self.collection: Any = None
        self.collection_name: str = ""
        self.dimension: int = 384  # Default for all-MiniLM-L6-v2
        super().__init__(config_path)

    def _connect(self) -> None:
        """Establish connection to Weaviate server.

        Reads connection parameters from config and creates Weaviate client.
        """
        weaviate_config = self.config.get("weaviate", {})
        url = weaviate_config.get("url") or os.getenv(
            "WEAVIATE_URL", "http://localhost:8080"
        )
        api_key = weaviate_config.get("api_key") or os.getenv("WEAVIATE_API_KEY")

        if api_key:
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=url,
                auth_credentials=Auth.api_key(api_key=api_key),
            )
        else:
            self.client = weaviate.connect_to_local(
                host=url.replace("http://", "").split(":")[0]
            )

        self.logger.info("Connected to Weaviate at %s", url)

    def _create_collection(self, collection_name: str) -> None:
        """Create Weaviate collection with schema.

        Args:
            collection_name: Name for the collection (PascalCase).
        """
        if not self.client:
            raise ValueError("Client not initialized")

        self.collection_name = collection_name

        if self.client.collections.exists(collection_name):
            self.collection = self.client.collections.get(collection_name)
            self.logger.info("Collection %s already exists", collection_name)
            return

        properties = [
            Property(name="content", data_type=DataType.TEXT),
        ]

        schema = self._get_metadata_schema()
        for field_name, field_def in schema.items():
            data_type = DataType.TEXT
            if field_def.type == "integer":
                data_type = DataType.INT
            elif field_def.type == "float":
                data_type = DataType.NUMBER
            elif field_def.type == "boolean":
                data_type = DataType.BOOL

            properties.append(Property(name=field_name, data_type=data_type))

        self.collection = self.client.collections.create(
            name=collection_name,
            properties=properties,
            vectorizer_config=Configure.Vectorizer.none(),
        )
        self.logger.info("Created collection: %s", collection_name)

    def _index_documents(self, documents: list[Document]) -> int:
        """Index documents with embeddings and metadata into Weaviate.

        Args:
            documents: List of Haystack documents with embeddings.

        Returns:
            Number of documents indexed.
        """
        if not self.collection:
            raise ValueError("Collection not initialized")

        count = 0
        with self.collection.batch.dynamic() as batch:
            for doc in documents:
                if doc.embedding is None:
                    continue

                properties = {"content": doc.content or "", **doc.meta}
                batch.add_object(properties=properties, vector=doc.embedding)
                count += 1

        self.logger.info("Indexed %d documents", count)
        return count

    def _pre_filter(self, filter_obj: Any) -> int:
        """Apply pre-filter and count matching candidates.

        Args:
            filter_obj: Weaviate Filter object.

        Returns:
            Number of documents matching the filter.
        """
        if not self.collection:
            return 0

        # Use aggregate to count matching objects
        try:
            if filter_obj:
                result = self.collection.aggregate.over_all(
                    filters=filter_obj, total_count=True
                )
            else:
                result = self.collection.aggregate.over_all(total_count=True)
            return result.total_count or 0
        except Exception:
            return 0

    def _vector_search(
        self,
        query_embedding: list[float],
        filter_obj: Any,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Execute vector search with filter.

        Args:
            query_embedding: Query vector embedding.
            filter_obj: Weaviate Filter object.
            top_k: Number of results to return.

        Returns:
            List of search results with id, score, content, metadata.
        """
        if not self.collection:
            raise ValueError("Collection not initialized")

        query_params: dict[str, Any] = {
            "near_vector": query_embedding,
            "limit": top_k,
            "return_metadata": MetadataQuery(distance=True, score=True),
        }

        if filter_obj:
            query_params["filters"] = filter_obj

        results = self.collection.query.near_vector(**query_params)

        search_results = []
        for obj in results.objects:
            properties = obj.properties or {}
            content = properties.pop("content", "")

            score = 1.0 - (obj.metadata.distance or 0.0) if obj.metadata else 0.0

            search_results.append(
                {
                    "id": str(obj.uuid),
                    "score": score,
                    "content": content,
                    "metadata": properties,
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
        5. Build Weaviate filter object
        6. Pre-filter to count candidates
        7. Run vector search on candidates
        8. Return ranked results with timing

        Returns:
            List of FilteredQueryResult objects with timing metrics.
        """
        self._init_embedder()

        collection_config = self.config.get("collection", {})
        collection_name = collection_config.get("name", "MetadataFiltered")
        embeddings_config = self.config.get("embeddings", {})
        self.dimension = embeddings_config.get("dimension", 384)

        self._create_collection(collection_name)

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

        builder = WeaviateFilterExpressionBuilder(schema)
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

        if self.collection:
            try:
                total_result = self.collection.aggregate.over_all(total_count=True)
                total_docs = total_result.total_count or 0
            except Exception:
                total_docs = 0
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

    def __del__(self) -> None:
        """Close Weaviate client connection on cleanup."""
        import contextlib

        if self.client:
            with contextlib.suppress(Exception):
                self.client.close()
