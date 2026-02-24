"""Weaviate Parent Document Search Pipeline.

This pipeline implements parent document retrieval by searching child chunks
in Weaviate and resolving matches to their parent documents.

Parent Document Retrieval Strategy:
    1. Embed the search query using the same model as indexing
    2. Search child chunks in Weaviate vector database (oversample for better recall)
    3. Use Haystack's AutoMergingRetriever to resolve child matches to parents
    4. Return parent documents with broader context

The key insight is that small child chunks enable precise semantic matching,
while parent documents provide the comprehensive context needed for LLM prompts.

Search Flow:
    1. Embed query using SentenceTransformersTextEmbedder
    2. Query Weaviate for top child matches (top_k * 3 for oversampling)
    3. Use AutoMergingRetriever to resolve children to parents
    4. Return parent documents and metadata

Auto-Merging Strategy:
    The AutoMergingRetriever groups matched children by their parent documents
    and returns the parent when enough children from the same parent match.
    The merge_threshold controls how many children must match before returning
    the parent.

Configuration (YAML):
    Required sections:
        - database.weaviate: Weaviate connection and collection settings
        - embeddings.model: Must match the model used during indexing
        - retrieval.merge_threshold: Threshold for auto-merging (default: 0.5)

    Example config:
        database:
          weaviate:
            collection_name: "ParentDocLeaves"
        embeddings:
          model: "Qwen/Qwen3-Embedding-0.6B"
        retrieval:
          merge_threshold: 0.5

Usage:
    >>> from vectordb.haystack.parent_document_retrieval import (
    ...     WeaviateParentDocIndexingPipeline,
    ...     WeaviateParentDocSearchPipeline,
    ... )
    >>> # First, index documents
    >>> indexing = WeaviateParentDocIndexingPipeline("config.yaml")
    >>> stats = indexing.run(limit=100)
    >>> # Then, search
    >>> search = WeaviateParentDocSearchPipeline(
    ...     "config.yaml", parent_store=indexing.parent_store
    ... )
    >>> results = search.search("What is machine learning?", top_k=5)
    >>> for doc in results["documents"]:
    ...     print(doc.content)

Attributes:
    config: Loaded and validated configuration dictionary
    logger: Configured logger instance
    text_embedder: SentenceTransformersTextEmbedder for query embedding
    vector_db: WeaviateVectorDB connection for searching leaves
    parent_store: InMemoryDocumentStore containing parent documents
    auto_merger: AutoMergingRetriever for resolving children to parents
    index_name: Name of the Weaviate collection for leaves
"""

from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.auto_merging_retriever import AutoMergingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from vectordb.databases.weaviate import WeaviateVectorDB
from vectordb.haystack.parent_document_retrieval.utils.config import (
    load_parent_doc_config,
)
from vectordb.utils.config import setup_logger


class WeaviateParentDocSearchPipeline:
    """Weaviate search pipeline for parent document retrieval.

    This pipeline searches child chunks in Weaviate and uses the parent_id
    metadata to retrieve the corresponding parent documents. The parent
    documents provide broader context than the individual child chunks.

    The search process:
    1. Embed the query using SentenceTransformers
    2. Search Weaviate for matching child chunks
    3. Use AutoMergingRetriever to group children and return their parents

    Attributes:
        config: Validated configuration dictionary
        logger: Logger instance for pipeline operations
        text_embedder: SentenceTransformers embedder for query embeddings
        vector_db: WeaviateVectorDB connection for searching leaves
        parent_store: InMemoryDocumentStore containing parent documents
        auto_merger: AutoMergingRetriever for resolving children to parents
        index_name: Name of the Weaviate collection
    """

    def __init__(
        self,
        config_path: str,
        parent_store: InMemoryDocumentStore,
    ) -> None:
        """Initialize the search pipeline.

        Loads configuration, initializes the query embedder, Weaviate connection,
        and AutoMergingRetriever. Requires the parent_store from the indexing
        pipeline to resolve child matches to parent documents.

        Args:
            config_path: Path to YAML configuration file
            parent_store: InMemoryDocumentStore from indexing pipeline containing
                the parent documents to be returned in search results

        Raises:
            ValueError: If configuration is missing required sections
        """
        self.config = load_parent_doc_config(config_path)
        self.logger = setup_logger(self.config)
        self.parent_store = parent_store

        self._init_embedder()
        self._init_vector_db()
        self._init_auto_merger()

    def _init_embedder(self) -> None:
        """Initialize the text embedder component.

        Creates a SentenceTransformersTextEmbedder with the model specified
        in configuration. The embedder is warmed up to load the model into memory.

        Configuration:
            embeddings.model: Model name (default: "Qwen/Qwen3-Embedding-0.6B")

        Important:
            Must use the same model as during indexing for consistent embeddings.

        The embedder converts the search query into a dense vector for
        similarity search in the vector database.
        """
        model = self.config.get("embeddings", {}).get(
            "model", "Qwen/Qwen3-Embedding-0.6B"
        )
        self.text_embedder = SentenceTransformersTextEmbedder(model=model)
        self.text_embedder.warm_up()

    def _init_vector_db(self) -> None:
        """Initialize the Weaviate vector database connection.

        Creates a WeaviateVectorDB instance connected to the same collection
        used during indexing. This enables searching the child chunks that
        were indexed.

        Configuration:
            database.weaviate: Weaviate connection settings
            database.weaviate.collection_name: Collection name
                (default: "ParentDocLeaves")
        """
        db_config = self.config.get("database", {})
        self.vector_db = WeaviateVectorDB(
            config={"weaviate": db_config.get("weaviate", {})}
        )
        self.index_name = db_config.get("weaviate", {}).get(
            "collection_name", "ParentDocLeaves"
        )

    def _init_auto_merger(self) -> None:
        """Initialize the auto-merging retriever.

        Creates an AutoMergingRetriever that groups matched child chunks by
        their parent documents. When multiple children from the same parent
        match (based on the threshold), the parent document is returned.

        Configuration:
            retrieval.merge_threshold: Threshold for auto-merging (default: 0.5)

        The merge_threshold determines what fraction of a parent's children
        must match before the parent is returned. A lower threshold returns
        more parents; a higher threshold requires stronger matches.
        """
        threshold = self.config.get("retrieval", {}).get("merge_threshold", 0.5)
        self.auto_merger = AutoMergingRetriever(
            document_store=self.parent_store,
            threshold=threshold,
        )

    def search(self, query: str, top_k: int = 5) -> dict:
        """Search for parent documents matching the query.

        Executes the parent document retrieval search: embeds the query,
        searches child chunks in Weaviate, and resolves matches to parent
        documents using the AutoMergingRetriever.

        Args:
            query: Search query string
            top_k: Number of parent documents to return (default: 5)

        Returns:
            Dictionary containing:
                - query: The search query
                - documents: List of parent Document objects
                - num_leaves_matched: Number of child chunks matched
                - num_parents_returned: Number of parent documents returned

        Search Strategy:
            1. Embed the query using SentenceTransformers
            2. Search Weaviate for top_k * 3 child matches (oversample for recall)
            3. Use AutoMergingRetriever to resolve children to parents
            4. Return up to top_k parent documents
        """
        # Convert query to embedding vector for similarity search
        query_embedding = self.text_embedder.run(text=query)["embedding"]

        # Search for matching child chunks in Weaviate
        # Oversample (top_k * 3) to ensure good recall before merging
        leaves = self.vector_db.query(
            vector=query_embedding,
            limit=top_k * 3,  # Oversample leaves for better recall before merging
            return_documents=True,
        )

        # Resolve child matches to their parent documents
        merged = self.auto_merger.run(documents=leaves)["documents"]

        return {
            "query": query,
            "documents": merged[:top_k],
            "num_leaves_matched": len(leaves),
            "num_parents_returned": len(merged[:top_k]),
        }
