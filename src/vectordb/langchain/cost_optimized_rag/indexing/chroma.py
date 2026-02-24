"""Chroma cost-optimized RAG indexing pipeline for LangChain.

This module implements a cost-optimized document indexing pipeline for Chroma
vector database. It generates dense embeddings for semantic search and stores
sparse embeddings in metadata for hybrid retrieval capabilities.

Cost Optimization:
    - Dense embeddings: API-based for semantic understanding
    - Sparse embeddings: Local TF-IDF generation stored as metadata
    - Local storage: Chroma runs locally, eliminating hosting costs
    - Metadata storage: Sparse vectors stored as strings to work with
      Chroma's metadata filtering

Note:
    Chroma has limited native sparse vector support. This pipeline stores
    sparse embeddings as metadata strings for later retrieval and fusion.

Example:
    >>> pipeline = ChromaCostOptimizedRAGIndexingPipeline(
    ...     {
    ...         "chroma": {
    ...             "persist_directory": "./chroma_db",
    ...             "collection_name": "my-docs",
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

from vectordb.databases.chroma import ChromaVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    SparseEmbedder,
)


logger = logging.getLogger(__name__)


class ChromaCostOptimizedRAGIndexingPipeline:
    """Chroma indexing pipeline for cost-optimized RAG using LangChain.

    This pipeline provides document ingestion for Chroma with hybrid search
    support via metadata storage of sparse embeddings. Chroma stores dense
    vectors natively while sparse vectors are serialized in metadata for
    later retrieval and fusion operations.

    Attributes:
        config: Pipeline configuration dictionary.
        dense_embedder: API-based dense embedding model.
        sparse_embedder: Local sparse embedding generator.
        db: ChromaVectorDB client instance.
        collection_name: Name of the Chroma collection.
        text_splitter: LangChain document chunker.

    Note:
        Chroma stores sparse embeddings as serialized strings in metadata
        since it lacks native sparse vector support. The search pipeline
        deserializes these for RRF fusion.

    Example:
        >>> pipeline = ChromaCostOptimizedRAGIndexingPipeline("config.yaml")
        >>> result = pipeline.run()
        >>> print(f"Created {result['chunks_created']} chunks")
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Chroma indexing pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
                Must contain chroma section with persist_directory and
                collection_name settings.
                Must contain embedding section with provider and model.

        Raises:
            ValueError: If required configuration is missing or invalid.
            FileNotFoundError: If config_or_path is a file path that does not exist.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "chroma")

        self.dense_embedder = EmbedderHelper.create_embedder(self.config)
        self.sparse_embedder = SparseEmbedder()

        chroma_config = self.config["chroma"]
        self.db = ChromaVectorDB(
            persist_directory=chroma_config.get("persist_directory", "./chroma_db"),
        )

        self.collection_name = chroma_config.get("collection_name")

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
            "Initialized Chroma cost-optimized RAG indexing pipeline (LangChain)"
        )

    def run(self) -> dict[str, Any]:
        """Execute the complete indexing pipeline for Chroma.

        Processes documents through loading, chunking, embedding generation,
        and storage in Chroma with sparse vectors stored as metadata.

        Returns:
            Dictionary containing indexing statistics:
            - documents_indexed: Number of source documents processed
            - chunks_created: Number of text chunks indexed

        Raises:
            RuntimeError: If indexing fails due to API or storage errors.
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

        recreate = self.config.get("chroma", {}).get("recreate", False)
        self.db.create_collection(
            name=self.collection_name,
            recreate=recreate,
        )

        upsert_data = []
        ids = []
        for i, (chunk, dense_emb, sparse_emb) in enumerate(
            zip(chunks, dense_embeddings, sparse_embeddings)
        ):
            chunk_id = str(uuid.uuid4())
            ids.append(chunk_id)
            upsert_data.append(
                {
                    "text": chunk.page_content,
                    "embedding": dense_emb,
                    "chunk_index": i,
                    "sparse_embedding": str(sparse_emb),
                    **(chunk.metadata or {}),
                }
            )

        num_indexed = self.db.upsert(
            collection_name=self.collection_name,
            documents=upsert_data,
            ids=ids,
        )
        logger.info("Indexed %d chunks with embeddings to Chroma", num_indexed)

        return {
            "documents_indexed": len(documents),
            "chunks_created": len(chunks),
        }
