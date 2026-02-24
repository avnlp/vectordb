"""Chroma Parent Document Indexing Pipeline.

This pipeline implements parent document retrieval by creating hierarchical document
structures. Documents are split into parent and child chunks, where children are
indexed for retrieval but parents are returned.

Parent Document Retrieval Strategy:
    The key concept is chunk-to-parent mapping: small child chunks enable precise
    retrieval matching, while parent documents provide the broader context needed
    for meaningful results. This balances retrieval accuracy with result quality.

Hierarchical Splitting Strategy:
    - Parent chunks: Larger blocks (e.g., 100 words) containing multiple concepts
    - Child chunks: Smaller blocks (e.g., 25 words) for precise retrieval matching
    - Overlap: Small overlap between chunks to prevent context loss at boundaries

Pipeline Flow:
    1. Load documents from configured dataset
    2. Split hierarchically using HierarchicalDocumentSplitter:
       - Creates parent chunks with level=1 metadata
       - Creates child chunks (leaves) that reference their parents
    3. Store parent documents in InMemoryDocumentStore
    4. Embed child documents using SentenceTransformers
    5. Index child embeddings in Chroma vector database
    6. Return statistics about indexed documents

Metadata Management:
    The HierarchicalDocumentSplitter manages parent-child relationships through
    metadata:
    - Parents have level=1 in metadata and children_ids listing their child IDs
    - Children (leaves) have no children_ids, indicating they are leaf nodes
    - Parent_id in child metadata links children to their parent documents

Configuration (YAML):
    Required sections:
        - database.chroma: Chroma connection and collection settings
        - embeddings.model: SentenceTransformers model name
        - dataloader: Dataset type, name, split, and limit settings

    Optional chunking settings:
        - chunking.parent_chunk_size_words: Size of parent chunks (default: 100)
        - chunking.child_chunk_size_words: Size of child chunks (default: 25)
        - chunking.split_overlap: Overlap between chunks (default: 5)

    Example config:
        database:
          chroma:
            collection_name: "parent_doc_leaves"
            persist_directory: "./chroma_db"
        embeddings:
          model: "Qwen/Qwen3-Embedding-0.6B"
        dataloader:
          type: "triviaqa"
          dataset_name: "triviaqa"
          split: "test"
          index_limit: 100
        chunking:
          parent_chunk_size_words: 100
          child_chunk_size_words: 25
          split_overlap: 5

Usage:
    >>> pipeline = ChromaParentDocIndexingPipeline("config.yaml")
    >>> stats = pipeline.run(limit=100)
    >>> print(f"Documents: {stats['num_documents']}")
    >>> print(f"Parents: {stats['num_parents']}")
    >>> print(f"Leaves: {stats['num_leaves']}")

Attributes:
    config: Loaded and validated configuration dictionary
    logger: Configured logger instance
    doc_embedder: SentenceTransformersDocumentEmbedder instance
    splitter: HierarchicalDocumentSplitter instance
    vector_db: ChromaVectorDB instance for indexing leaves
    parent_store: InMemoryDocumentStore for storing parents
    index_name: Name of the Chroma collection for leaves
"""

from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors import HierarchicalDocumentSplitter
from haystack.document_stores.in_memory import InMemoryDocumentStore

from vectordb.databases.chroma import ChromaVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.parent_document_retrieval.utils.config import (
    load_parent_doc_config,
)
from vectordb.utils.config import setup_logger


class ChromaParentDocIndexingPipeline:
    """Chroma indexing pipeline for parent document retrieval.

    This pipeline builds a hierarchical document index where child chunks are
    stored in Chroma for retrieval and parent documents are stored in a
    separate document store. The parent-child relationship enables returning
    broader context documents instead of small matching chunks.

    Attributes:
        config: Validated configuration dictionary loaded from YAML or dict
        logger: Logger instance for pipeline operations
        doc_embedder: SentenceTransformers embedder for document embeddings
        splitter: HierarchicalDocumentSplitter for parent/child chunking
        vector_db: ChromaVectorDB connection for indexing leaves
        parent_store: InMemoryDocumentStore for storing parent documents
        index_name: Name of the Chroma collection
    """

    def __init__(self, config_path: str) -> None:
        """Initialize the indexing pipeline from configuration.

        Loads configuration, initializes all components including the embedder,
        hierarchical splitter, Chroma vector database, and parent document store.

        Args:
            config_path: Path to YAML configuration file

        Raises:
            ValueError: If configuration is missing required sections
        """
        self.config = load_parent_doc_config(config_path)
        self.logger = setup_logger(self.config)

        self._init_embedder()
        self._init_splitter()
        self._init_vector_db()
        self.parent_store = InMemoryDocumentStore()

    def _init_embedder(self) -> None:
        """Initialize the document embedder component.

        Creates a SentenceTransformersDocumentEmbedder with the model specified
        in configuration. The embedder is warmed up to load the model into memory.

        Configuration:
            embeddings.model: Model name (default: "Qwen/Qwen3-Embedding-0.6B")

        The embedder converts text documents into dense vector representations
        for storage in the vector database.
        """
        model = self.config.get("embeddings", {}).get(
            "model", "Qwen/Qwen3-Embedding-0.6B"
        )
        self.doc_embedder = SentenceTransformersDocumentEmbedder(model=model)
        self.doc_embedder.warm_up()

    def _init_splitter(self) -> None:
        """Initialize the hierarchical document splitter.

        Creates a HierarchicalDocumentSplitter configured with parent and child
        chunk sizes. The splitter creates a two-level hierarchy:
        - Level 1: Parent chunks (larger, broader context)
        - Level 2+: Child chunks (smaller, precise matching)

        Configuration:
            chunking.parent_chunk_size_words: Parent chunk size (default: 100)
            chunking.child_chunk_size_words: Child chunk size (default: 25)
            chunking.split_overlap: Overlap in words (default: 5)
        """
        chunking = self.config.get("chunking", {})
        parent_size = chunking.get("parent_chunk_size_words", 100)
        child_size = chunking.get("child_chunk_size_words", 25)
        overlap = chunking.get("split_overlap", 5)

        self.splitter = HierarchicalDocumentSplitter(
            block_sizes={parent_size, child_size},
            split_overlap=overlap,
            split_by="word",
        )

    def _init_vector_db(self) -> None:
        """Initialize the Chroma vector database connection.

        Creates a ChromaVectorDB instance and ensures the collection exists
        for storing child document embeddings. The collection is created with
        the appropriate dimension for the embedding model.

        Configuration:
            database.chroma: Chroma connection settings
            database.chroma.collection_name: Collection name
                (default: "parent_doc_leaves")

        Note:
            Uses 1024 dimensions, which is the output size of the default
            Qwen/Qwen3-Embedding-0.6B model.
        """
        db_config = self.config.get("database", {})
        self.vector_db = ChromaVectorDB(config={"chroma": db_config.get("chroma", {})})

        index_name = db_config.get("chroma", {}).get(
            "collection_name", "parent_doc_leaves"
        )
        embedding_dim = 1024  # Qwen3-Embedding-0.6B dimension
        self.vector_db.create_collection(
            collection_name=index_name,
            dimension=embedding_dim,
        )
        self.index_name = index_name

    def run(self, limit: int | None = None) -> dict:
        """Execute the indexing pipeline.

        Loads documents from the configured dataset, splits them hierarchically,
        stores parents in the document store, and indexes children in Chroma.

        Args:
            limit: Maximum number of documents to index. If None, uses the
                limit from configuration or indexes all available documents.

        Returns:
            Dictionary with indexing statistics:
                - num_documents: Total number of source documents indexed
                - num_parents: Number of parent chunks created
                - num_leaves: Number of child chunks (leaves) indexed

        Pipeline Steps:
            1. Load documents using DatasetRegistry
            2. Convert to Haystack Document format
            3. Split hierarchically into parents and children
            4. Store parents in InMemoryDocumentStore
            5. Embed and store leaves in Chroma vector database
        """
        dataloader_config = self.config.get("dataloader", {})
        loader = DataloaderCatalog.create(
            dataloader_config.get("type", "triviaqa"),
            split=dataloader_config.get("split", "test"),
            limit=limit or dataloader_config.get("index_limit"),
            dataset_id=dataloader_config.get("dataset_name"),
        )
        documents = loader.load().to_haystack()

        # Split documents into hierarchical parent-child structure
        all_docs = self.splitter.run(documents)["documents"]

        # Parents have level=1 metadata assigned by HierarchicalDocumentSplitter
        parents = [d for d in all_docs if d.meta.get("level") == 1]

        # Leaves have no children_ids, indicating they are leaf nodes
        leaves = [d for d in all_docs if not d.meta.get("children_ids")]

        # Store parents in document store for later retrieval
        self.parent_store.write_documents(parents)

        # Embed leaves and index in Chroma for similarity search
        embedded_leaves = self.doc_embedder.run(documents=leaves)["documents"]
        self.vector_db.upsert(data=embedded_leaves)

        return {
            "num_documents": len(documents),
            "num_parents": len(parents),
            "num_leaves": len(leaves),
        }
