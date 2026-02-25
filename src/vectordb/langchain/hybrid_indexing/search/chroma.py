"""Chroma hybrid search pipeline (LangChain).

Implements search using Chroma's vector similarity capabilities. Chroma has
limited native hybrid search support compared to other vector databases, so
this pipeline focuses on dense vector search while maintaining API consistency
with other hybrid pipelines.

Hybrid search limitations:
    - Chroma does not support native sparse vector storage or search
    - Sparse embeddings may be stored in document metadata for custom ranking
    - Primary search relies on dense vector cosine similarity

When to use Chroma:
    - Local development and prototyping
    - Smaller-scale deployments (<1M documents)
    - When simplicity and ease of setup are priorities
    - When semantic search alone is sufficient

Note:
    For true hybrid search with Chroma, consider post-processing retrieved
    results with custom keyword matching or reranking using the sparse
    embeddings stored in metadata.
"""

import logging
from typing import Any

from vectordb.databases.chroma import ChromaVectorDB
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    RAGHelper,
    SparseEmbedder,
)
from vectordb.utils.chroma_document_converter import ChromaDocumentConverter


logger = logging.getLogger(__name__)


class ChromaHybridSearchPipeline:
    """Chroma search pipeline with hybrid API compatibility.

    Performs dense vector similarity search using Chroma. Maintains the same
    interface as other hybrid pipelines for code portability, though Chroma's
    native hybrid capabilities are limited to dense vectors only.

    Attributes:
        config: Validated configuration dictionary.
        dense_embedder: Embedder for semantic vector generation.
        sparse_embedder: SparseEmbedder instance (initializes sparse models
            but not used for Chroma search).
        db: ChromaVectorDB instance for vector operations.
        collection_name: Target Chroma collection name.
        llm: Optional language model for RAG answer generation.

    Example:
        >>> pipeline = ChromaHybridSearchPipeline("config.yaml")
        >>> results = pipeline.search("vector database comparison", top_k=5)
        >>> for doc in results["documents"]:
        ...     print(doc.page_content[:100])

    Note:
        This pipeline uses dense embeddings only. For custom hybrid search,
        retrieve with top_k*2 results and apply custom sparse reranking.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize search pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain chroma section with connection details.

        Raises:
            ValueError: If required configuration keys are missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.

        Note:
            Sparse embedder is initialized for API consistency but not used
            in Chroma search operations. Sparse embeddings can be accessed
            from document metadata if stored during indexing.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "chroma")

        self.dense_embedder = EmbedderHelper.create_embedder(self.config)
        self.sparse_embedder = SparseEmbedder()

        chroma_config = self.config["chroma"]
        self.db = ChromaVectorDB(
            host=chroma_config.get("host", "localhost"),
            port=chroma_config.get("port", 8000),
            collection_name=chroma_config.get("collection_name"),
        )

        self.collection_name = chroma_config.get("collection_name")
        self.llm = RAGHelper.create_llm(self.config)

        logger.info("Initialized Chroma search pipeline (LangChain)")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute vector similarity search.

        Generates dense embedding for the query and executes Chroma vector
        search. Optionally generates RAG answer if LLM configured.

        Args:
            query: Search query text to embed and search.
            top_k: Maximum number of results to return. Defaults to 10.
            filters: Optional metadata filters as dictionary for pre-filtering.
                Chroma supports simple equality filters on metadata fields.

        Returns:
            Dictionary containing:
                - documents: List of retrieved Document objects
                - query: Original query string
                - answer: Generated RAG answer (only if LLM configured)

        Raises:
            RuntimeError: If database connection fails during search.

        Note:
            This method performs dense vector search only. The sparse_embedder
            attribute exists for API consistency with other hybrid pipelines.
        """
        dense_embedding = EmbedderHelper.embed_query(self.dense_embedder, query)
        logger.info("Generated dense embedding for query: %s...", query[:50])

        self.db._get_collection(self.collection_name)
        raw_results = self.db.query(
            query_embedding=dense_embedding,
            n_results=top_k,
            where=filters,
        )
        documents = (
            ChromaDocumentConverter.convert_query_results_to_langchain_documents(
                raw_results
            )
        )
        logger.info("Retrieved %d documents from Chroma", len(documents))

        result = {
            "documents": documents,
            "query": query,
        }

        if self.llm is not None:
            answer = RAGHelper.generate(self.llm, query, documents)
            result["answer"] = answer
            logger.info("Generated RAG answer")

        return result
