"""Qdrant cost-optimized RAG indexing pipeline for LangChain.

This module implements a cost-optimized document indexing pipeline for Qdrant
vector database. It generates both dense and sparse embeddings to enable
hybrid search while minimizing API costs through local sparse embedding generation.

Cost Optimization:
    - Dense embeddings: API-based for semantic understanding
    - Sparse embeddings: Local TF-IDF generation (zero cost)
    - Batch processing: Efficient document batching for API calls
    - Qdrant sparse vectors: Native support for efficient hybrid queries

Example:
    >>> pipeline = QdrantCostOptimizedRAGIndexingPipeline(
    ...     {
    ...         "qdrant": {
    ...             "url": "http://localhost:6333",
    ...             "collection_name": "my-docs",
    ...             "dimension": 1536,
    ...         },
    ...         "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
    ...     }
    ... )
    >>> result = pipeline.run()
    >>> print(f"Indexed {result['chunks_created']} chunks")
"""

import logging
import uuid
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from vectordb.databases.qdrant import QdrantVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    SparseEmbedder,
)


logger = logging.getLogger(__name__)


class QdrantCostOptimizedRAGIndexingPipeline:
    """Qdrant indexing pipeline for cost-optimized RAG using LangChain.

    This pipeline provides document ingestion for Qdrant with hybrid search support.
    It optimizes costs by generating sparse embeddings locally while using API
    services only for dense semantic embeddings.

    Attributes:
        config: Pipeline configuration dictionary.
        dense_embedder: API-based dense embedding model.
        sparse_embedder: Local sparse embedding generator (TF-IDF based).
        db: QdrantVectorDB client instance.
        collection_name: Name of the Qdrant collection.
        dimension: Vector dimension matching the embedding model.
        text_splitter: LangChain document chunker.

    Example:
        >>> pipeline = QdrantCostOptimizedRAGIndexingPipeline("config.yaml")
        >>> result = pipeline.run()
        >>> print(f"Created {result['chunks_created']} chunks")
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Qdrant indexing pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
                Must contain qdrant section with url/api_key and collection_name.
                Must contain embedding section with provider and model settings.

        Raises:
            ValueError: If required configuration is missing or invalid.
            FileNotFoundError: If config_or_path is a file path that does not exist.

        Note:
            The pipeline validates configuration immediately on initialization
            to ensure all required parameters are present before processing.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "qdrant")

        self.dense_embedder = EmbedderHelper.create_embedder(self.config)
        self.sparse_embedder = SparseEmbedder()

        qdrant_config = self.config["qdrant"]
        self.db = QdrantVectorDB(
            url=qdrant_config.get("url", "http://localhost:6333"),
            api_key=qdrant_config.get("api_key"),
        )

        self.collection_name = qdrant_config.get("collection_name")
        self.dimension = qdrant_config.get("dimension", 384)

        chunking_config = self.config.get("chunking", {})
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunking_config.get("chunk_size", 1000),
            chunk_overlap=chunking_config.get("chunk_overlap", 200),
            separators=chunking_config.get(
                "separators",
                ["\n\n", "\n", " ", ""],
            ),
        )

        logger.info(
            "Initialized Qdrant cost-optimized RAG indexing pipeline (LangChain)"
        )

    def run(self) -> dict[str, Any]:
        """Execute the complete indexing pipeline for Qdrant.

        Orchestrates document loading, chunking, dual embedding generation,
        and vector storage in Qdrant with hybrid search support.

        Returns:
            Dictionary containing indexing statistics:
            - documents_indexed: Number of source documents processed
            - chunks_created: Number of text chunks generated and indexed

        Raises:
            RuntimeError: If indexing fails due to API errors or database issues.

        Example:
            >>> pipeline = QdrantCostOptimizedRAGIndexingPipeline(config)
            >>> result = pipeline.run()
            >>> print(f"Success: {result['chunks_created']} chunks indexed")
        """
        # Load documents with optional limit from configuration
        limit = self.config.get("dataloader", {}).get("limit")
        dl_config = self.config.get("dataloader", {})
        loader = DataloaderCatalog.create(
            dl_config.get("type", "triviaqa"),
            split=dl_config.get("split", "test"),
            limit=limit,
        )
        dataset = loader.load()
        documents = dataset.to_langchain()
        logger.info("Loaded %d documents", len(documents))

        if not documents:
            logger.warning("No documents to index")
            return {"documents_indexed": 0, "chunks_created": 0}

        chunks = self.text_splitter.split_documents(documents)
        logger.info("Created %d chunks from documents", len(chunks))

        if not chunks:
            logger.warning("No chunks created")
            return {"documents_indexed": 0, "chunks_created": 0}

        _, dense_embeddings = EmbedderHelper.embed_documents(
            self.dense_embedder, chunks
        )
        logger.info("Generated dense embeddings for %d chunks", len(chunks))

        texts = [chunk.page_content for chunk in chunks]
        sparse_embeddings = self.sparse_embedder.embed_documents(texts)
        logger.info("Generated sparse embeddings for %d chunks", len(chunks))

        recreate = self.config.get("qdrant", {}).get("recreate", False)
        self.db.create_collection(
            collection_name=self.collection_name,
            vector_size=self.dimension,
            recreate=recreate,
        )

        upsert_data = []
        for i, (chunk, dense_emb, sparse_emb) in enumerate(
            zip(chunks, dense_embeddings, sparse_embeddings)
        ):
            chunk_id = str(uuid.uuid4())
            upsert_data.append(
                {
                    "id": chunk_id,
                    "vector": dense_emb,
                    "sparse_vector": sparse_emb,
                    "payload": {
                        "text": chunk.page_content,
                        "chunk_index": i,
                        **(chunk.metadata or {}),
                    },
                }
            )

        num_indexed = self.db.upsert(
            collection_name=self.collection_name,
            documents=upsert_data,
        )
        logger.info("Indexed %d chunks with hybrid embeddings to Qdrant", num_indexed)

        return {
            "documents_indexed": len(documents),
            "chunks_created": len(chunks),
        }
