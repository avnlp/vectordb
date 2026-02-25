"""Weaviate parent document retrieval indexing pipeline (LangChain).

This module implements the indexing pipeline for parent document retrieval
using Weaviate as the vector database. It chunks documents while maintaining
parent references and stores the parent documents in a ParentDocumentStore.

Pipeline Flow:
    1. Load configuration and initialize WeaviateVectorDB
    2. Create RecursiveCharacterTextSplitter for document chunking
    3. For each document:
        a. Generate unique parent_id (UUID)
        b. Store parent document in ParentDocumentStore
        c. Split document into chunks
        d. For each chunk, create chunk document with parent_id metadata
    4. Generate embeddings for all chunks
    5. Create Weaviate collection (optionally recreating)
    6. Upsert chunks with parent_id, chunk_index, and metadata
    7. Save ParentDocumentStore to disk (if cache_dir configured)

Metadata Structure:
    Each chunk stored in Weaviate includes:
        - text: Chunk text content
        - vector: Vector embedding
        - parent_id: UUID of parent document
        - chunk_index: Position of chunk within parent
        - All original document metadata

Configuration Requirements:
    Required Weaviate config:
        - url: Weaviate server URL
        - api_key: Optional API key for authentication
        - collection_name: Name of Weaviate collection
        - recreate: Whether to recreate collection (default: False)

Example:
    >>> pipeline = WeaviateParentDocumentRetrievalIndexingPipeline("config.yaml")
    >>> result = pipeline.run()
    >>> print(f"Indexed {result['documents_indexed']} documents")
    >>> print(f"Created {result['chunks_created']} chunks")
    >>> print(f"Parent store: {result['parent_store_path']}")
"""

import logging
import uuid
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vectordb.databases.weaviate import WeaviateVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.parent_document_retrieval.parent_store import (
    ParentDocumentStore,
)
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
)


logger = logging.getLogger(__name__)


class WeaviateParentDocumentRetrievalIndexingPipeline:
    """Weaviate indexing pipeline for parent document retrieval (LangChain).

    Chunks documents with RecursiveCharacterTextSplitter while maintaining
    parent references in metadata. Stores chunks in Weaviate and maintains
    a ParentDocumentStore for retrieval of full documents.

    Attributes:
        config: Loaded configuration dictionary
        embedder: Embedding model instance from EmbedderHelper
        db: WeaviateVectorDB instance for vector storage
        collection_name: Name of Weaviate collection
        text_splitter: RecursiveCharacterTextSplitter instance
        parent_store: ParentDocumentStore for parent-child mappings

    Example:
        >>> pipeline = WeaviateParentDocumentRetrievalIndexingPipeline(
        ...     {
        ...         "weaviate": {
        ...             "url": "http://localhost:8080",
        ...             "collection_name": "my-collection",
        ...         },
        ...         "embedder": {
        ...             "type": "sentence_transformers",
        ...             "model": "all-MiniLM-L6-v2",
        ...         },
        ...         "chunking": {"chunk_size": 1000, "chunk_overlap": 200},
        ...         "parent_store": {"cache_dir": "./cache"},
        ...     }
        ... )
        >>> result = pipeline.run()
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration.

        Loads configuration, initializes Weaviate connection, sets up text
        splitter, and creates the parent document store.

        Args:
            config_or_path: Config dict or path to YAML configuration file.
                Must contain weaviate, embedder, and chunking sections.

        Raises:
            ValueError: If required config keys are missing or invalid.

        Example:
            >>> pipeline = WeaviateParentDocumentRetrievalIndexingPipeline(
            ...     "/path/to/config.yaml"
            ... )
            >>> pipeline = WeaviateParentDocumentRetrievalIndexingPipeline(
            ...     {
            ...         "weaviate": {
            ...             "url": "http://localhost:8080",
            ...             "collection_name": "docs",
            ...         },
            ...         "embedder": {
            ...             "type": "sentence_transformers",
            ...             "model": "all-MiniLM-L6-v2",
            ...         },
            ...         "chunking": {"chunk_size": 500},
            ...     }
            ... )
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "weaviate")

        # Initialize embedding model
        self.embedder = EmbedderHelper.create_embedder(self.config)

        # Initialize Weaviate connection
        weaviate_config = self.config["weaviate"]
        self.db = WeaviateVectorDB(
            url=weaviate_config["url"],
            api_key=weaviate_config.get("api_key"),
        )

        # Store collection name
        self.collection_name = weaviate_config.get("collection_name")

        # Configure text splitter for chunking
        chunking_config = self.config.get("chunking", {})
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunking_config.get("chunk_size", 1000),
            chunk_overlap=chunking_config.get("chunk_overlap", 200),
            separators=chunking_config.get(
                "separators",
                ["\n\n", "\n", " ", ""],
            ),
        )

        # Initialize parent document store with optional persistence
        cache_dir = self.config.get("parent_store", {}).get("cache_dir")
        self.parent_store = ParentDocumentStore(cache_dir=cache_dir)

        logger.info(
            "Initialized Weaviate parent document retrieval indexing pipeline (LangChain)"
        )

    def run(self) -> dict[str, Any]:
        """Execute indexing pipeline.

        Runs the complete indexing workflow:
            1. Load documents from configured dataloader
            2. Chunk documents and store parent references
            3. Generate embeddings for chunks
            4. Create Weaviate collection
            5. Upsert chunks with parent metadata
            6. Save parent store (if configured)

        Returns:
            Dictionary containing:
                - documents_indexed: Number of parent documents processed
                - chunks_created: Total number of chunks indexed
                - parent_store_path: Path to saved parent store file (or None)

        Example:
            >>> result = pipeline.run()
            >>> print(f"Indexed: {result['documents_indexed']}")
            >>> print(f"Chunks: {result['chunks_created']}")
            >>> print(f"Store: {result['parent_store_path']}")
        """
        # Load documents from configured dataloader
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

        # Handle empty document set
        if not documents:
            logger.warning("No documents to index")
            return {
                "documents_indexed": 0,
                "chunks_created": 0,
                "parent_store_path": None,
            }

        all_chunks = []

        # Process each document: store parent and create chunks
        for doc_idx, doc in enumerate(documents):
            # Generate unique parent ID for this document
            parent_id = str(uuid.uuid4())

            # Store complete parent document in ParentDocumentStore
            # This enables retrieval of full document text during search
            parent_data = {
                "text": doc.page_content,
                "metadata": doc.metadata or {},
                "source_index": doc_idx,
            }
            self.parent_store.add_parent(parent_id, parent_data)

            # Split document into chunks using configured splitter
            chunks = self.text_splitter.split_text(doc.page_content)
            logger.debug("Created %d chunks from document %d", len(chunks), doc_idx)

            # Create chunk documents with parent_id metadata
            for chunk_idx, chunk_text in enumerate(chunks):
                chunk_doc = {
                    "text": chunk_text,
                    "parent_id": parent_id,
                    "chunk_index": chunk_idx,
                    "metadata": doc.metadata or {},
                }
                all_chunks.append(chunk_doc)

        logger.info("Created %d chunks from documents", len(all_chunks))

        # Handle case where no chunks were created
        if not all_chunks:
            logger.warning("No chunks created")
            return {
                "documents_indexed": 0,
                "chunks_created": 0,
                "parent_store_path": None,
            }

        # Generate embeddings for all chunks
        chunk_texts = [chunk["text"] for chunk in all_chunks]
        chunk_documents = [Document(page_content=text) for text in chunk_texts]
        _, embeddings = EmbedderHelper.embed_documents(self.embedder, chunk_documents)
        logger.info("Generated embeddings for %d chunks", len(all_chunks))

        # Create or recreate Weaviate collection
        recreate = self.config.get("weaviate", {}).get("recreate", False)
        self.db.create_collection(
            name=self.collection_name,
            recreate=recreate,
        )

        # Prepare data for Weaviate upsert
        upsert_data = []
        ids = []

        for chunk, embedding in zip(all_chunks, embeddings):
            # Generate unique chunk ID for vector database
            chunk_id = str(uuid.uuid4())
            ids.append(chunk_id)

            # Build document with parent reference and vector
            upsert_data.append(
                {
                    "uuid": chunk_id,
                    "id": chunk_id,
                    "text": chunk["text"],
                    "vector": embedding,
                    "parent_id": chunk["parent_id"],
                    "chunk_index": chunk["chunk_index"],
                    **(chunk["metadata"]),
                }
            )

            # Update parent store mapping chunk_id -> parent_id
            self.parent_store.add_chunk_mapping(chunk_id, chunk["parent_id"])

        # Upsert chunks to Weaviate
        num_indexed = self.db.upsert(
            collection_name=self.collection_name,
            documents=upsert_data,
        )
        logger.info("Indexed %d chunks to Weaviate", num_indexed)

        # Save parent store to disk if cache_dir is configured
        parent_store_path = None
        if self.config.get("parent_store", {}).get("cache_dir"):
            parent_store_path = self.parent_store.save(
                f"{self.collection_name}_parent_store.pkl"
            )
            logger.info("Saved parent store to %s", parent_store_path)

        return {
            "documents_indexed": len(documents),
            "chunks_created": len(all_chunks),
            "parent_store_path": parent_store_path,
        }
