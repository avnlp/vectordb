"""Chroma metadata filtering pipeline for Haystack.

Implements BaseMetadataFilteringPipeline for Chroma vector database with
pre-filtering and vector search capabilities.
"""

import os
import sys
from typing import Any

import numpy as np
from haystack import Document


# pysqlite3 workaround for Chroma
try:
    import pysqlite3  # noqa: F401

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb  # noqa: E402
from chromadb import Collection  # noqa: E402

from vectordb.haystack.metadata_filtering.base import (  # noqa: E402
    BaseMetadataFilteringPipeline,
    Timer,
)
from vectordb.haystack.metadata_filtering.vectordb_pipeline_type import (  # noqa: E402
    ChromaFilterExpressionBuilder,
    FilteredQueryResult,
    FilterSpec,
    TimingMetrics,
    parse_filter_from_config,
)


__all__ = ["ChromaMetadataFilteringPipeline"]


class ChromaMetadataFilteringPipeline(BaseMetadataFilteringPipeline):
    """Chroma metadata filtering pipeline with pre-filter and vector search.

    Connects to Chroma, indexes documents with metadata, applies
    pre-filtering, then performs vector search on matching candidates.

    Attributes:
        client: Chroma client instance.
        collection: Chroma collection instance.
        collection_name: Name of the collection.
        dimension: Embedding vector dimension.
    """

    def __init__(self, config_path: str) -> None:
        """Initialize Chroma pipeline from configuration.

        Args:
            config_path: Path to YAML configuration file.
        """
        self.client: chromadb.Client | chromadb.PersistentClient | None = None
        self.collection: Collection | None = None
        self.collection_name: str = ""
        self.dimension: int = 384  # Default for all-MiniLM-L6-v2
        super().__init__(config_path)

    def _connect(self) -> None:
        """Initialize Chroma client.

        Reads connection parameters from config and creates Chroma client.
        """
        chroma_config = self.config.get("chroma", {})
        persist_dir = chroma_config.get("persist_directory") or os.getenv(
            "CHROMA_PERSIST_DIR", "./chroma_data"
        )

        # Use persistent client if directory specified
        if persist_dir:
            self.client = chromadb.PersistentClient(path=persist_dir)
        else:
            self.client = chromadb.Client()

        self.logger.info("Initialized Chroma client, persist_dir: %s", persist_dir)

    def _create_collection(self, collection_name: str) -> Collection:
        """Create or get Chroma collection.

        Args:
            collection_name: Name for the collection.

        Returns:
            Chroma Collection instance.
        """
        if not self.client:
            raise ValueError("Client not initialized")

        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.logger.info("Created/retrieved collection: %s", collection_name)
        return self.collection

    def _index_documents(self, documents: list[Document]) -> int:
        """Index documents with embeddings and metadata into Chroma.

        Args:
            documents: List of Haystack documents with embeddings.

        Returns:
            Number of documents indexed.
        """
        if not self.collection:
            raise ValueError("Collection not initialized")

        ids = []
        embeddings = []
        documents_text = []
        metadatas = []

        for idx, doc in enumerate(documents):
            if doc.embedding is None:
                continue
            ids.append(str(idx))
            embeddings.append(doc.embedding)
            documents_text.append(doc.content or "")
            metadatas.append(doc.meta)

        if ids:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents_text,
                metadatas=metadatas,
            )

        self.logger.info("Indexed %d documents", len(ids))
        return len(ids)

    def _pre_filter(self, filter_dict: dict[str, Any]) -> int:
        """Apply pre-filter and count matching candidates.

        Args:
            filter_dict: Chroma filter dictionary (where clause).

        Returns:
            Number of documents matching the filter.
        """
        if not self.collection:
            return 0

        try:
            if filter_dict:
                results = self.collection.get(where=filter_dict, include=[])
            else:
                results = self.collection.get(include=[])
            return len(results.get("ids", []))
        except Exception:
            return 0

    def _vector_search(
        self,
        query_embedding: list[float],
        filter_dict: dict[str, Any],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Execute vector search with filter.

        Args:
            query_embedding: Query vector embedding.
            filter_dict: Chroma filter dictionary (where clause).
            top_k: Number of results to return.

        Returns:
            List of search results with id, score, content, metadata.
        """
        if not self.collection:
            raise ValueError("Collection not initialized")

        query_params: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }

        if filter_dict:
            query_params["where"] = filter_dict

        results = self.collection.query(**query_params)

        search_results = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for idx, doc_id in enumerate(ids):
            # For cosine distance configured in collection: similarity = 1 - distance
            distance = distances[idx] if idx < len(distances) else 0.0
            score = 1.0 - distance

            search_results.append(
                {
                    "id": doc_id,
                    "score": score,
                    "content": documents[idx] if idx < len(documents) else "",
                    "metadata": metadatas[idx] if idx < len(metadatas) else {},
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
        5. Build Chroma filter dict
        6. Pre-filter to count candidates
        7. Run vector search on candidates
        8. Return ranked results with timing

        Returns:
            List of FilteredQueryResult objects with timing metrics.
        """
        self._init_embedder()

        collection_config = self.config.get("collection", {})
        chroma_config = self.config.get("chroma", {})
        collection_name = chroma_config.get("collection_name") or collection_config.get(
            "name", "metadata_filtered"
        )
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

        builder = ChromaFilterExpressionBuilder(schema)
        filter_dict = builder.build(filter_spec)
        self.logger.info("Built filter dict: %s", filter_dict)

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
            num_candidates = self._pre_filter(filter_dict)

        # Vector search timing
        with Timer() as search_timer:
            search_results = self._vector_search(query_embedding, filter_dict)

        total_docs = self.collection.count() if self.collection else 0

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
