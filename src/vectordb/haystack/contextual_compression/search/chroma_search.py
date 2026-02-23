"""Chroma search pipeline with contextual compression.

Retrieves documents from Chroma vector store and applies compression to filter
irrelevant chunks before returning results to the user or LLM.

Retrieval Strategy:
    1. Embed query using configured dense embedder (e.g., Qwen3-Embedding-0.6B)
    2. Query Chroma collection using cosine similarity
    3. Convert Chroma distances to similarity scores (1 - distance)
    4. Return documents with metadata including chroma_distance

Compression Integration:
    Base class (BaseContextualCompressionPipeline) handles compression via:
    - Reranking: Cross-encoder or API-based scoring
    - LLM Extraction: Extract relevant passages with LLM

Chroma-Specific Notes:
    - Uses PersistentClient for local/embedded deployment
    - Cosine distance metric (converted to similarity for scoring)
    - Metadata includes deserialized JSON from indexing

Example:
    >>> pipeline = ChromaCompressionSearch("configs/chroma/triviaqa/reranking.yaml")
    >>> results = pipeline.run("What is Python?", top_k=5)
    >>> for doc in results["documents"]:
    ...     print(f"Score: {doc.meta['score']:.3f}, Content: {doc.content[:100]}...")
"""

import ast

import chromadb
from haystack import Document

from vectordb.haystack.contextual_compression.base import (
    BaseContextualCompressionPipeline,
)


class ChromaCompressionSearch(BaseContextualCompressionPipeline):
    """Chroma search pipeline with contextual compression.

    Retrieves documents via dense search and applies compression
    (reranking or LLM extraction).
    """

    def _connect(self) -> None:
        """Establish connection to Chroma."""
        import tempfile

        chroma_config = self.config.get("chroma", {})
        path = chroma_config.get("path", tempfile.gettempdir() + "/chroma")

        self.client = chromadb.PersistentClient(path=path)
        self.logger.info("Connected to Chroma at %s", path)

    def _ensure_collection_ready(self) -> None:
        """Verify Chroma collection exists."""
        chroma_config = self.config.get("chroma", {})
        collection_name = chroma_config.get("collection_name", "compression")

        try:
            self.collection = self.client.get_collection(name=collection_name)
            self.logger.info("Collection '%s' ready", collection_name)
        except Exception as e:
            self.logger.error("Collection '%s' not found: %s", collection_name, str(e))
            raise

    def _retrieve_base_results(self, query: str, top_k: int) -> list[Document]:
        """Retrieve documents from Chroma via dense search.

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
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = []
        for i, doc_text in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else 0.0
            # Chroma returns distances; convert to similarity
            similarity = 1 - distance

            doc = Document(
                content=doc_text,
                meta={
                    "score": similarity,
                    "chroma_distance": distance,
                    **(
                        ast.literal_eval(metadata.pop("metadata", "{}"))
                        if "metadata" in metadata
                        else {}
                    ),
                    **metadata,
                },
            )
            documents.append(doc)

        self.logger.debug("Retrieved %d documents from Chroma", len(documents))
        return documents
