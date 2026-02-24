"""Weaviate cost-optimized RAG indexing pipeline for LangChain.

This module implements a cost-optimized document indexing pipeline for Weaviate
vector database with native hybrid search support.

Cost Optimization:
    - Dense embeddings: API-based for semantic understanding
    - Sparse embeddings: Local TF-IDF generation (zero cost)
    - Native hybrid: Weaviate has built-in BM25 for lexical search
    - Efficient storage: Both vector types stored in single object

Weaviate Features:
    Weaviate provides native hybrid search combining vector similarity with
    BM25 keyword matching, making it ideal for cost-optimized RAG without
    requiring external fusion algorithms.

Example:
    >>> pipeline = WeaviateCostOptimizedRAGIndexingPipeline(
    ...     {
    ...         "weaviate": {
    ...             "url": "http://localhost:8080",
    ...             "collection_name": "Documents",
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

from vectordb.databases.weaviate import WeaviateVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    SparseEmbedder,
)


logger = logging.getLogger(__name__)


class WeaviateCostOptimizedRAGIndexingPipeline:
    """Weaviate indexing pipeline for cost-optimized RAG using LangChain.

    This pipeline provides document ingestion for Weaviate with hybrid search
    support. Weaviate's native hybrid capabilities combine dense vectors with
    BM25 lexical search for optimal retrieval quality at reduced cost.

    Attributes:
        config: Pipeline configuration dictionary.
        dense_embedder: API-based dense embedding model.
        sparse_embedder: Local sparse embedding generator.
        db: WeaviateVectorDB client instance.
        collection_name: Name of the Weaviate collection.
        dimension: Vector dimension.
        text_splitter: LangChain document chunker.

    Example:
        >>> pipeline = WeaviateCostOptimizedRAGIndexingPipeline("config.yaml")
        >>> result = pipeline.run()
        >>> print(f"Created {result['chunks_created']} chunks")
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Weaviate indexing pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
                Must contain weaviate section with url and optionally
                api_key, collection_name, and dimension.
                Must contain embedding section with provider and model.

        Raises:
            ValueError: If required configuration is missing or invalid.
            FileNotFoundError: If config_or_path is a file path that does not exist.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "weaviate")

        self.dense_embedder = EmbedderHelper.create_embedder(self.config)
        self.sparse_embedder = SparseEmbedder()

        weaviate_config = self.config["weaviate"]
        self.db = WeaviateVectorDB(
            url=weaviate_config["url"],
            api_key=weaviate_config.get("api_key"),
        )

        self.collection_name = weaviate_config.get("collection_name")
        self.dimension = weaviate_config.get("dimension", 384)

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
            "Initialized Weaviate cost-optimized RAG indexing pipeline (LangChain)"
        )

    def run(self) -> dict[str, Any]:
        """Execute the complete indexing pipeline for Weaviate.

        Processes documents through loading, chunking, dual embedding generation,
        and storage in Weaviate with native hybrid search support.

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

        recreate = self.config.get("weaviate", {}).get("recreate", False)
        self.db.create_collection(
            name=self.collection_name,
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
                    "metadata": {
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
        logger.info("Indexed %d chunks with hybrid embeddings to Weaviate", num_indexed)

        return {
            "documents_indexed": len(documents),
            "chunks_created": len(chunks),
        }
