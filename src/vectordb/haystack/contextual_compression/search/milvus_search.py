"""Milvus search pipeline with contextual compression.

Retrieves documents from Milvus vector store using IVF_FLAT index and applies
contextual compression to return only relevant chunks.

Retrieval Strategy:
    1. Generate query embedding with dense embedder
    2. Search Milvus collection using IVF_FLAT index with nprobe=10
    3. Metric: Inner Product (IP) for similarity scoring
    4. Deserialize JSON metadata stored during indexing

Compression Integration:
    Documents are retrieved and passed to compressor (reranker or LLM extractor)
    via BaseContextualCompressionPipeline.run().

Milvus-Specific Notes:
    - Requires collection.load() before search (handled in _ensure_collection_ready)
    - IVF_FLAT index with 128 clusters (nlist=128)
    - Metadata stored as JSON strings, deserialized with json.loads

Compression Benefits with Milvus:
    - Can retrieve large top_k (e.g., 50-100) for better coverage
    - Compression filters to optimal top_k (e.g., 5-10) for LLM context
    - Reduces token usage while maintaining retrieval quality

Example:
    >>> pipeline = MilvusCompressionSearch("configs/milvus/arc/llm_extraction.yaml")
    >>> results = pipeline.run("Explain quantum computing", top_k=10)
"""

import ast
import contextlib
import json

from haystack import Document
from pymilvus import Collection, connections

from vectordb.haystack.contextual_compression.base import (
    BaseContextualCompressionPipeline,
)


class MilvusCompressionSearch(BaseContextualCompressionPipeline):
    """Milvus search pipeline with contextual compression.

    Retrieves documents via dense search and applies compression
    (reranking or LLM extraction).
    """

    def _connect(self) -> None:
        """Establish connection to Milvus."""
        milvus_config = self.config.get("milvus", {})
        host = milvus_config.get("host", "localhost")
        port = milvus_config.get("port", 19530)

        connections.connect(alias="default", host=host, port=int(port))
        self.logger.info("Connected to Milvus at %s:%s", host, port)

    def _ensure_collection_ready(self) -> None:
        """Verify Milvus collection exists and is loaded."""
        milvus_config = self.config.get("milvus", {})
        collection_name = milvus_config.get("collection_name", "compression")

        collection = Collection(collection_name, using="default")
        collection.load()
        self.collection_name = collection_name
        self.logger.info("Collection '%s' loaded", collection_name)

    def _retrieve_base_results(self, query: str, top_k: int) -> list[Document]:
        """Retrieve documents from Milvus via dense search.

        Args:
            query: Search query text.
            top_k: Number of results to retrieve.

        Returns:
            List of Document objects with similarity scores in metadata.
        """
        # Embed query
        embedding_result = self.dense_embedder.run(text=query)
        query_embedding = embedding_result["embedding"]

        # Search
        collection = Collection(self.collection_name, using="default")
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["content", "metadata"],
        )

        documents = []
        for hit in results[0]:
            metadata = {}
            metadata_str = hit.entity.get("metadata")
            if metadata_str:
                try:
                    metadata = json.loads(metadata_str)
                except json.JSONDecodeError:
                    # Fallback to ast.literal_eval for Python dict strings
                    with contextlib.suppress(ValueError, SyntaxError):
                        metadata = ast.literal_eval(metadata_str)

            doc = Document(
                content=hit.entity.get("content", ""),
                meta={
                    "distance": hit.distance,
                    "milvus_id": hit.id,
                    **metadata,
                },
            )
            documents.append(doc)

        self.logger.debug("Retrieved %d documents from Milvus", len(documents))
        return documents
