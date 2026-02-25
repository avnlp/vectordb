"""Pinecone parent document retrieval indexing pipeline (LangChain).

This module implements the indexing pipeline for parent document retrieval
using Pinecone as the vector database. It chunks documents while maintaining
parent references and stores the parent documents in a ParentDocumentStore.

Pipeline Flow:
    1. Load configuration and initialize PineconeVectorDB
    2. Create RecursiveCharacterTextSplitter for document chunking
    3. For each document:
        a. Generate unique parent_id (UUID)
        b. Store parent document in ParentDocumentStore
        c. Split document into chunks
        d. For each chunk, create chunk document with parent_id metadata
    4. Generate embeddings for all chunks
    5. Create Pinecone index (optionally recreating)
    6. Upsert chunks with parent_id, chunk_index, and metadata
    7. Save ParentDocumentStore to disk (if cache_dir configured)

Metadata Structure:
    Each chunk stored in Pinecone includes:
        - text: Chunk text content
        - parent_id: UUID of parent document
        - chunk_index: Position of chunk within parent
        - All original document metadata

Configuration Requirements:
    Required Pinecone config:
        - api_key: Pinecone API key
        - index_name: Name of Pinecone index
        - namespace: Optional namespace (default: "")
        - dimension: Vector dimension (default: 384)
        - metric: Distance metric (default: "cosine")
        - recreate: Whether to recreate index (default: False)

Example:
    >>> pipeline = PineconeParentDocumentRetrievalIndexingPipeline("config.yaml")
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

from vectordb.databases.pinecone import PineconeVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.parent_document_retrieval.parent_store import (
    ParentDocumentStore,
)
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
)


logger = logging.getLogger(__name__)


class PineconeParentDocumentRetrievalIndexingPipeline:
    """Pinecone indexing pipeline for parent document retrieval (LangChain).

    Chunks documents with RecursiveCharacterTextSplitter while maintaining
    parent references in metadata. Stores chunks in Pinecone and maintains
    a ParentDocumentStore for retrieval of full documents.

    Attributes:
        config: Loaded configuration dictionary
        embedder: Embedding model instance from EmbedderHelper
        db: PineconeVectorDB instance for vector storage
        index_name: Name of Pinecone index
        namespace: Pinecone namespace for isolation
        dimension: Vector dimension for embeddings
        text_splitter: RecursiveCharacterTextSplitter instance
        parent_store: ParentDocumentStore for parent-child mappings

    Example:
        >>> pipeline = PineconeParentDocumentRetrievalIndexingPipeline(
        ...     {
        ...         "pinecone": {
        ...             "api_key": "your-key",
        ...             "index_name": "my-index",
        ...             "dimension": 384,
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

        Loads configuration, initializes Pinecone connection, sets up text
        splitter, and creates the parent document store.

        Args:
            config_or_path: Config dict or path to YAML configuration file.
                Must contain pinecone, embedder, and chunking sections.

        Raises:
            ValueError: If required config keys are missing or invalid.

        Example:
            >>> pipeline = PineconeParentDocumentRetrievalIndexingPipeline(
            ...     "/path/to/config.yaml"
            ... )
            >>> pipeline = PineconeParentDocumentRetrievalIndexingPipeline(
            ...     {
            ...         "pinecone": {"api_key": "key", "index_name": "idx"},
            ...         "embedder": {
            ...             "type": "sentence_transformers",
            ...             "model": "all-MiniLM-L6-v2",
            ...         },
            ...         "chunking": {"chunk_size": 500},
            ...     }
            ... )
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "pinecone")

        # Initialize embedding model
        self.embedder = EmbedderHelper.create_embedder(self.config)

        # Initialize Pinecone connection
        pinecone_config = self.config["pinecone"]
        self.db = PineconeVectorDB(
            api_key=pinecone_config["api_key"],
            index_name=pinecone_config.get("index_name"),
        )

        # Store Pinecone settings
        self.index_name = pinecone_config.get("index_name")
        self.namespace = pinecone_config.get("namespace", "")
        self.dimension = pinecone_config.get("dimension", 384)

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
            "Initialized Pinecone parent document retrieval indexing pipeline (LangChain)"
        )

    def run(self) -> dict[str, Any]:
        """Execute indexing pipeline.

        Runs the complete indexing workflow:
            1. Load documents from configured dataloader
            2. Chunk documents and store parent references
            3. Generate embeddings for chunks
            4. Create Pinecone index
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

        # Map to track parent_id -> document index mapping
        parent_id_map = {}
        all_chunks = []

        # Process each document: store parent and create chunks
        for doc_idx, doc in enumerate(documents):
            # Generate unique parent ID for this document
            parent_id = str(uuid.uuid4())
            parent_id_map[parent_id] = doc_idx

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

        # Create or recreate Pinecone index
        recreate = self.config.get("pinecone", {}).get("recreate", False)
        self.db.create_index(
            index_name=self.index_name,
            dimension=self.dimension,
            metric=self.config.get("pinecone", {}).get("metric", "cosine"),
            recreate=recreate,
        )

        # Prepare data for Pinecone upsert
        upsert_data = []
        chunk_ids = []

        for chunk, embedding in zip(all_chunks, embeddings):
            # Generate unique chunk ID for vector database
            chunk_id = str(uuid.uuid4())
            chunk_ids.append(chunk_id)

            # Build metadata with parent reference and chunk position
            upsert_data.append(
                {
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": {
                        "id": chunk_id,
                        "text": chunk["text"],
                        "parent_id": chunk["parent_id"],
                        "chunk_index": chunk["chunk_index"],
                        **(chunk["metadata"]),
                    },
                }
            )

            # Update parent store mapping chunk_id -> parent_id
            self.parent_store.add_chunk_mapping(chunk_id, chunk["parent_id"])

        # Upsert chunks to Pinecone
        num_indexed = self.db.upsert(
            data=upsert_data,
            namespace=self.namespace,
        )
        logger.info("Indexed %d chunks to Pinecone", num_indexed)

        # Save parent store to disk if cache_dir is configured
        parent_store_path = None
        if self.config.get("parent_store", {}).get("cache_dir"):
            parent_store_path = self.parent_store.save(
                f"{self.index_name}_parent_store.pkl"
            )
            logger.info("Saved parent store to %s", parent_store_path)

        return {
            "documents_indexed": len(documents),
            "chunks_created": len(all_chunks),
            "parent_store_path": parent_store_path,
        }
