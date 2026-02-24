"""Pinecone cost-optimized RAG indexing pipeline for LangChain.

This module implements a cost-optimized document indexing pipeline for Pinecone
vector database. It generates both dense and sparse embeddings to enable
hybrid search while minimizing API costs.

Cost Optimization:
    - Dense embeddings: Generated via API for semantic understanding
    - Sparse embeddings: Generated locally using TF-IDF (zero API cost)
    - Efficient batching: Embeds documents in optimized batch sizes
    - Pinecone native sparse: Leverages Pinecone's native sparse vector support

Example:
    >>> pipeline = PineconeCostOptimizedRAGIndexingPipeline(
    ...     {
    ...         "pinecone": {
    ...             "api_key": "...",
    ...             "index_name": "my-docs",
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

from vectordb.databases.pinecone import PineconeVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    SparseEmbedder,
)


logger = logging.getLogger(__name__)


class PineconeCostOptimizedRAGIndexingPipeline:
    """Pinecone indexing pipeline for cost-optimized RAG using LangChain.

    This pipeline provides a complete document ingestion workflow that prepares
    data for hybrid semantic+lexical search in Pinecone. It implements cost
    optimization by generating sparse embeddings locally while using API-based
    dense embeddings for semantic understanding.

    Attributes:
        config: Pipeline configuration dictionary.
        dense_embedder: API-based dense embedding model.
        sparse_embedder: Local sparse embedding generator (TF-IDF based).
        db: PineconeVectorDB client instance.
        index_name: Name of the Pinecone index.
        namespace: Namespace for document organization.
        dimension: Vector dimension (must match embedding model).
        text_splitter: LangChain text splitter for document chunking.

    Example:
        >>> pipeline = PineconeCostOptimizedRAGIndexingPipeline("config.yaml")
        >>> result = pipeline.run()
        >>> print(
        ...     f"Created {result['chunks_created']} chunks from "
        ...     f"{result['documents_indexed']} documents"
        ... )
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Pinecone indexing pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
                Must contain pinecone section with api_key and optionally
                index_name, namespace, dimension, and metric settings.
                Must contain embedding section with provider and model.

        Raises:
            ValueError: If required configuration is missing or invalid.
            FileNotFoundError: If config_or_path is a file path that does not exist.

        Note:
            The pipeline validates the configuration on initialization to fail
            fast if required parameters are missing.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "pinecone")

        self.dense_embedder = EmbedderHelper.create_embedder(self.config)
        self.sparse_embedder = SparseEmbedder()

        pinecone_config = self.config["pinecone"]
        self.db = PineconeVectorDB(
            api_key=pinecone_config["api_key"],
            index_name=pinecone_config.get("index_name"),
        )

        self.index_name = pinecone_config.get("index_name")
        self.namespace = pinecone_config.get("namespace", "")
        self.dimension = pinecone_config.get("dimension", 384)

        self.use_text_splitter = self.config.get("dataloader", {}).get(
            "use_text_splitter", True
        )
        if self.use_text_splitter:
            chunking_config = self.config.get("chunking", {})
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunking_config.get("chunk_size", 1000),
                chunk_overlap=chunking_config.get("chunk_overlap", 200),
                separators=chunking_config.get(
                    "separators",
                    ["\n\n", "\n", " ", ""],
                ),
            )
        else:
            self.text_splitter = None

        logger.info(
            "Initialized Pinecone cost-optimized RAG indexing pipeline (LangChain)"
        )

    def run(self) -> dict[str, Any]:
        """Execute the complete indexing pipeline.

        This method orchestrates the full document indexing workflow:
        1. Load documents from configured source
        2. Chunk documents into optimal segments
        3. Generate dual embeddings (dense API + sparse local)
        4. Create/update Pinecone index
        5. Upsert documents with hybrid vectors

        Returns:
            Dictionary containing indexing statistics:
            - documents_indexed: Number of source documents processed
            - chunks_created: Number of text chunks generated and indexed

        Raises:
            RuntimeError: If indexing fails due to API errors or database issues.

        Example:
            >>> pipeline = PineconeCostOptimizedRAGIndexingPipeline(config)
            >>> result = pipeline.run()
            >>> print(f"Success: {result['chunks_created']} chunks indexed")
        """
        # Load documents from configured data source with optional limit
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

        # Early exit if no documents to process
        if not documents:
            logger.warning("No documents to index")
            return {"documents_indexed": 0, "chunks_created": 0}

        if self.use_text_splitter:
            chunks = self.text_splitter.split_documents(documents)
            logger.info("Created %d chunks from documents", len(chunks))
        else:
            chunks = documents
            logger.info("Using %d documents without splitting", len(chunks))

        if not chunks:
            logger.warning("No chunks created")
            return {"documents_indexed": len(documents), "chunks_created": 0}

        _, dense_embeddings = EmbedderHelper.embed_documents(
            self.dense_embedder, chunks
        )
        logger.info("Generated dense embeddings for %d chunks", len(chunks))

        texts = [chunk.page_content for chunk in chunks]
        sparse_embeddings = self.sparse_embedder.embed_documents(texts)
        logger.info("Generated sparse embeddings for %d chunks", len(chunks))

        recreate = self.config.get("pinecone", {}).get("recreate", False)
        self.db.create_index(
            index_name=self.index_name,
            dimension=self.dimension,
            metric=self.config.get("pinecone", {}).get("metric", "cosine"),
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
                    "values": dense_emb,
                    "sparse_values": sparse_emb,
                    "metadata": {
                        "text": chunk.page_content,
                        "chunk_index": i,
                        **(chunk.metadata or {}),
                    },
                }
            )

        num_indexed = self.db.upsert(
            data=upsert_data,
            namespace=self.namespace,
        )
        logger.info("Indexed %d chunks with hybrid embeddings to Pinecone", num_indexed)

        return {
            "documents_indexed": len(documents),
            "chunks_created": len(chunks),
        }
