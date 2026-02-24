"""Milvus cost-optimized RAG indexing pipeline for LangChain.

This module implements a cost-optimized document indexing pipeline for Milvus
vector database with support for hybrid dense and sparse vector search.

Cost Optimization:
    - Dense embeddings: API-based for semantic understanding
    - Sparse embeddings: Local TF-IDF generation (zero cost)
    - Batch processing: Efficient bulk operations
    - Native sparse support: Milvus 2.3+ supports sparse vectors natively

Milvus Features:
    Milvus provides high-performance vector search with native support for
    both dense and sparse vectors, making it ideal for hybrid retrieval
    at scale.

Example:
    >>> pipeline = MilvusCostOptimizedRAGIndexingPipeline(
    ...     {
    ...         "milvus": {
    ...             "uri": "http://localhost:19530",
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

from vectordb.databases.milvus import MilvusVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    SparseEmbedder,
)


logger = logging.getLogger(__name__)


class MilvusCostOptimizedRAGIndexingPipeline:
    """Milvus indexing pipeline for cost-optimized RAG using LangChain.

    This pipeline provides document ingestion for Milvus with hybrid search
    support. Milvus's native sparse vector support enables efficient
    lexical search alongside semantic dense search.

    Attributes:
        config: Pipeline configuration dictionary.
        dense_embedder: API-based dense embedding model.
        sparse_embedder: Local sparse embedding generator.
        db: MilvusVectorDB client instance.
        collection_name: Name of the Milvus collection.
        dimension: Vector dimension.
        text_splitter: LangChain document chunker.

    Example:
        >>> pipeline = MilvusCostOptimizedRAGIndexingPipeline("config.yaml")
        >>> result = pipeline.run()
        >>> print(f"Created {result['chunks_created']} chunks")
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Milvus indexing pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
                Must contain milvus section with connection parameters
                (uri, db_name) and collection_name.
                Must contain embedding section with provider and model.

        Raises:
            ValueError: If required configuration is missing or invalid.
            FileNotFoundError: If config_or_path is a file path that does not exist.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "milvus")

        self.dense_embedder = EmbedderHelper.create_embedder(self.config)
        self.sparse_embedder = SparseEmbedder()

        milvus_config = self.config["milvus"]
        self.db = MilvusVectorDB(
            uri=milvus_config.get("uri", "http://localhost:19530"),
            db_name=milvus_config.get("db_name", "default"),
        )

        self.collection_name = milvus_config.get("collection_name")
        self.dimension = milvus_config.get("dimension", 384)

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
            "Initialized Milvus cost-optimized RAG indexing pipeline (LangChain)"
        )

    def run(self) -> dict[str, Any]:
        """Execute the complete indexing pipeline for Milvus.

        Processes documents through loading, chunking, dual embedding generation,
        and storage in Milvus with hybrid vector support.

        Returns:
            Dictionary containing indexing statistics:
            - documents_indexed: Number of source documents processed
            - chunks_created: Number of text chunks indexed

        Raises:
            RuntimeError: If indexing fails due to API or database errors.
        """
        # Load documents with optional limit
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

        recreate = self.config.get("milvus", {}).get("recreate", False)
        self.db.create_collection(
            collection_name=self.collection_name,
            dimension=self.dimension,
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
                    "text": chunk.page_content,
                    "chunk_index": i,
                    **(chunk.metadata or {}),
                }
            )

        num_indexed = self.db.upsert(
            collection_name=self.collection_name,
            documents=upsert_data,
        )
        logger.info("Indexed %d chunks with hybrid embeddings to Milvus", num_indexed)

        return {
            "documents_indexed": len(documents),
            "chunks_created": len(chunks),
        }
