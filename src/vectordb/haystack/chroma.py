"""Chroma VectorDB implementation for Haystack integration.

This module provides a comprehensive interface to Chroma vector database,
supporting both local persistent storage and cloud/remote deployments.
It enables RAG (Retrieval Augmented Generation) workflows through Haystack
document integration, with support for multi-tenancy, hybrid search,
and advanced metadata filtering.

Key features:
    - Multi-tenancy support via tenant/database isolation
    - Hybrid search capabilities (dense + sparse vectors)
    - Automatic metadata flattening for Chroma compatibility
    - Lazy client initialization for connection efficiency
    - Both persistent and ephemeral client modes

The module handles the pysqlite3 compatibility workaround required for
Chroma to work with older sqlite3 versions.
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Union

import pysqlite3  # noqa: F401


# pysqlite3 workaround: Replace sqlite3 module before importing chromadb.
# Chroma requires sqlite3 features that may not be available in system versions.
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb  # noqa: E402, I001
from chromadb import Collection  # noqa: E402, I001
from chromadb.api.configuration import CollectionConfiguration  # noqa: E402, I001
from chromadb.api.types import (  # noqa: E402, I001
    CollectionMetadata,
    Embeddable,
    EmbeddingFunction,
)
from chromadb.config import Settings  # noqa: E402, I001
from chromadb.utils import embedding_functions  # noqa: E402, I001
from haystack import Document  # noqa: E402, I001

from vectordb.utils.chroma_document_converter import ChromaDocumentConverter  # noqa: E402, I001
from vectordb.utils.config import load_config, resolve_env_vars  # noqa: E402, I001
from vectordb.utils.logging import LoggerFactory  # noqa: E402, I001

logger_factory = LoggerFactory(logger_name=__name__, log_level=logging.INFO)
logger = logger_factory.get_logger()


class ChromaVectorDB:
    """Comprehensive Chroma Vector Database interface for Haystack integration.

    Provides a unified interface to Chroma vector database with support for
    cloud deployments (Chroma Cloud), local persistent storage, and ephemeral
    in-memory storage. Designed for RAG workflows with Haystack document
    processing and retrieval.

    Attributes:
        host: Chroma server hostname for remote connections.
        port: Chroma server port number.
        api_key: Authentication key for Chroma Cloud or secured instances.
        tenant: Tenant name for multi-tenant deployments.
        database: Database name within the tenant.
        path: Local filesystem path for persistent client storage.
        persistent: Whether to use persistent storage when running locally.
        collection_name: Default collection for operations.
        ssl: Whether to use SSL/TLS for remote connections.
        tracing_project_name: Project name for tracing/logging purposes.
        client: Lazy-initialized Chroma client instance.
        collection: Lazy-initialized Chroma collection instance.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        tenant: Optional[str] = None,
        database: Optional[str] = None,
        path: Optional[str] = None,
        persistent: bool = True,
        collection_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize ChromaVectorDB with configuration parameters.

        Configuration precedence: constructor parameters > config dict/file >
        environment variables > defaults.

        Args:
            host: Chroma server hostname for HttpClient connections.
            port: Chroma server port number (default: 8000).
            api_key: API key for authenticated Chroma instances.
            tenant: Tenant name for multi-tenant mode (default: "default_tenant").
            database: Database name within tenant (default: "default_database").
            path: Local storage path for PersistentClient (default: "./chroma").
            persistent: Use persistent storage when no host provided (default: True).
            collection_name: Default collection name for operations.
            config: Direct configuration dictionary to load settings from.
            config_path: Path to YAML configuration file.
            **kwargs: Additional parameters including:
                - ssl: Enable SSL for remote connections (default: True)
                - tracing_project_name: Project identifier for traces

        Raises:
            ValueError: If configuration is invalid or conflicting.
        """
        if config_path:
            self.config = load_config(config_path)
        elif config:
            self.config = resolve_env_vars(config)
        else:
            self.config = {}

        db_config = self.config.get("chroma", {})

        # Priority: Parameters > Config > Environment Variables
        self.host = host or db_config.get("host") or os.environ.get("CHROMA_HOST")
        self.port = (
            port or db_config.get("port") or int(os.environ.get("CHROMA_PORT", "8000"))
        )
        self.api_key = (
            api_key or db_config.get("api_key") or os.environ.get("CHROMA_API_KEY")
        )
        self.tenant = (
            tenant
            or db_config.get("tenant")
            or os.environ.get("CHROMA_TENANT", "default_tenant")
        )
        self.database = (
            database
            or db_config.get("database")
            or os.environ.get("CHROMA_DATABASE", "default_database")
        )

        self.path = path or db_config.get("path", "./chroma")
        self.persistent = (
            persistent if persistent is not None else db_config.get("persistent", True)
        )
        self.collection_name = collection_name or db_config.get("collection_name")

        self.ssl = kwargs.get("ssl", db_config.get("ssl", True))
        self.tracing_project_name = kwargs.get("tracing_project_name", "chroma")

        self.client: Optional[chromadb.ClientAPI] = None
        self.collection: Optional[Collection] = None

        logger.info(f"Initialized ChromaVectorDB. Collection: {self.collection_name}")

    def _get_client(self) -> chromadb.ClientAPI:
        """Lazy-load and return the Chroma client instance.

        Creates the appropriate client type based on configuration:
        - HttpClient for remote/cloud deployments
        - PersistentClient for local file storage
        - EphemeralClient for in-memory only

        Returns:
            Initialized Chroma ClientAPI instance.

        Raises:
            ConnectionError: If unable to connect to remote Chroma server.
        """
        if self.client is None:
            if self.host:
                logger.info(
                    f"Connecting to Chroma HttpClient at {self.host}:{self.port}"
                )
                self.client = chromadb.HttpClient(
                    host=self.host,
                    port=self.port,
                    ssl=self.ssl,
                    settings=Settings(
                        chroma_api_key=self.api_key,
                        chroma_tenant=self.tenant,
                        chroma_database=self.database,
                    ),
                )
            elif self.persistent:
                logger.info(f"Initializing Chroma PersistentClient at {self.path}")
                self.client = chromadb.PersistentClient(path=self.path)
            else:
                logger.info("Initializing Chroma EphemeralClient")
                self.client = chromadb.EphemeralClient()

            logger.info("Chroma client initialized successfully.")
        return self.client

    def _get_collection(self, name: Optional[str] = None) -> Collection:
        """Lazy-load and return the Chroma collection instance.

        Updates internal collection_name if a new name is provided.
        Reuses cached collection if name matches current collection.

        Args:
            name: Collection name to load. Uses default collection_name if None.

        Returns:
            Chroma Collection instance ready for operations.

        Raises:
            ValueError: If no collection name is available.
            CollectionNotFoundError: If collection does not exist in database.
        """
        if name:
            self.collection_name = name

        if self.collection is None or (name and self.collection.name != name):
            if not self.collection_name:
                msg = "Collection name is required."
                raise ValueError(msg)
            client = self._get_client()
            self.collection = client.get_collection(name=self.collection_name)
        return self.collection

    def create_collection(
        self,
        name: str,
        configuration: Optional[CollectionConfiguration] = None,
        metadata: Optional[CollectionMetadata] = None,
        embedding_function: Optional[EmbeddingFunction[Embeddable]] = None,
        get_or_create: bool = True,
        **kwargs: Any,
    ) -> None:
        """Create a new collection in the Chroma database.

        Args:
            name: Unique identifier for the collection.
            configuration: Collection-specific configuration object.
            metadata: Optional metadata to associate with the collection.
            embedding_function: Function to generate embeddings. Uses
                DefaultEmbeddingFunction if not provided.
            get_or_create: If True, retrieves existing collection if it exists.
                If False, raises error if collection exists.
            **kwargs: Additional parameters passed to Chroma collection creation.

        Raises:
            ValueError: If collection name is invalid.
            CollectionAlreadyExistsError: If get_or_create=False and collection exists.
        """
        client = self._get_client()
        self.collection_name = name

        if embedding_function is None:
            embedding_function = embedding_functions.DefaultEmbeddingFunction()

        if get_or_create:
            self.collection = client.get_or_create_collection(
                name=name,
                configuration=configuration,
                metadata=metadata,
                embedding_function=embedding_function,
                **kwargs,
            )
        else:
            self.collection = client.create_collection(
                name=name,
                configuration=configuration,
                metadata=metadata,
                embedding_function=embedding_function,
                **kwargs,
            )
        logger.info(f"Collection '{name}' created/retrieved.")

    def upsert(
        self,
        data: Union[List[Any], Dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        """Insert or update documents in the current collection.

        Handles both Haystack Document objects and raw dictionary data.
        Automatically flattens nested metadata for Chroma compatibility.

        Args:
            data: Either a list of Haystack Documents or a dictionary with
                keys: ids, documents/texts, metadatas, embeddings.
            **kwargs: Additional parameters passed to Chroma collection.upsert().

        Raises:
            ValueError: If data format is invalid or missing required fields.
            CollectionNotFoundError: If no collection is available.
        """
        collection = self._get_collection()

        if isinstance(data, list):
            # Haystack Documents need metadata flattening before conversion
            for doc in data:
                if hasattr(doc, "meta") and doc.meta:
                    doc.meta = self.flatten_metadata(doc.meta)
            formatted_data = (
                ChromaDocumentConverter.prepare_haystack_documents_for_upsert(data)
            )
        else:
            # Direct dictionary format requires manual metadata flattening
            formatted_data = data.copy()
            if "metadatas" in formatted_data:
                formatted_data["metadatas"] = [
                    self.flatten_metadata(m) for m in formatted_data["metadatas"]
                ]

        # Map 'texts' from converter to 'documents' for Chroma API
        documents = formatted_data.get("texts") or formatted_data.get("documents")
        metadatas = formatted_data.get("metadatas")

        # Chroma 1.4+ requires non-empty metadata dicts or None
        if metadatas:
            if all(not m for m in metadatas):
                metadatas = None
            else:
                # Add dummy field to avoid empty dict issues while preserving data
                metadatas = [m if m else {"_": "_"} for m in metadatas]

        collection.upsert(
            ids=formatted_data["ids"],
            embeddings=formatted_data.get("embeddings"),
            documents=documents,
            metadatas=metadatas,
            **kwargs,
        )
        logger.info(
            f"Upserted {len(formatted_data['ids'])} records to collection '{self.collection_name}'"
        )

    def query(
        self,
        query_embedding: Optional[list[float]] = None,
        query_text: Optional[str] = None,
        n_results: int = 10,
        where: Optional[dict[str, Any]] = None,
        where_document: Optional[dict[str, Any]] = None,
        include: List[str] = None,
        include_vectors: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Query the collection for similar vectors.

        Performs similarity search using either embeddings or text queries.
        Supports metadata filtering and document content filtering.

        Args:
            query_embedding: Vector embedding to search for (optional).
            query_text: Raw text to embed and search for (optional).
            n_results: Maximum number of results to return (default: 10).
            where: Metadata filter conditions (Chroma filter syntax).
            where_document: Document content filter conditions.
            include: List of fields to include in results. Defaults to
                ["metadatas", "documents", "distances"].
            include_vectors: Whether to include embeddings in results.
            **kwargs: Additional parameters passed to Chroma query.

        Returns:
            Dictionary containing query results with keys:
            - ids: List of document IDs
            - documents: List of document contents
            - metadatas: List of metadata dictionaries
            - distances: List of distance scores
            - embeddings: List of vectors (if include_vectors=True)

        Raises:
            ValueError: If neither query_embedding nor query_text is provided.
            CollectionNotFoundError: If collection does not exist.
        """
        if include is None:
            include = ["metadatas", "documents", "distances"]
            if include_vectors:
                include.append("embeddings")

        collection = self._get_collection()

        return collection.query(
            query_embeddings=[query_embedding] if query_embedding else None,
            query_texts=[query_text] if query_text else None,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include,
            **kwargs,
        )

    def query_to_documents(
        self,
        results: Dict[str, Any],
    ) -> List[Document]:
        """Convert Chroma query results to Haystack Document objects.

        Transforms raw Chroma query results into Haystack Document format,
        normalizing distance scores to similarity scores for Haystack compatibility.

        Args:
            results: Chroma query results dictionary from query() or search().
                Expected keys: ids, documents, metadatas, distances, embeddings.

        Returns:
            List of Haystack Document objects with:
            - id: Document identifier
            - content: Document text
            - meta: Metadata dictionary
            - score: Normalized similarity score (1.0 - distance)
            - embedding: Vector if available in results
        """
        documents = []
        ids = results.get("ids", [[]])[0]
        contents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        embeddings = (
            results.get("embeddings", [[]])[0] if results.get("embeddings") else []
        )

        for i, doc_id in enumerate(ids):
            doc = Document(
                id=doc_id,
                content=contents[i] if contents and i < len(contents) else "",
                meta=metadatas[i] if metadatas and i < len(metadatas) else {},
            )

            # Convert Chroma distance (0-2 for cosine) to Haystack similarity (0-1)
            # Lower distance means higher similarity in Chroma
            # Haystack expects higher score = better match
            if distances and i < len(distances):
                doc.score = 1.0 - distances[i]

            if embeddings and i < len(embeddings):
                doc.embedding = embeddings[i]

            documents.append(doc)

        return documents

    def search(
        self,
        query_text: Optional[str] = None,
        query_embeddings: Optional[Union[List[float], List[List[float]]]] = None,
        n_results: int = 10,
        where: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Advanced search supporting hybrid and sparse retrieval.

        Uses Chroma's experimental Search API when available (Chroma 0.6.0+,
        hosted/cloud environments). Falls back to standard query() for local
        deployments or unsupported configurations.

        Args:
            query_text: Text query for hybrid/full-text search.
            query_embeddings: Pre-computed embedding(s) for dense search.
                Can be single embedding or list of embeddings.
            n_results: Maximum number of results to return (default: 10).
            where: Metadata filter conditions.
            **kwargs: Additional search parameters including:
                - searches: Pre-built Search object for complex queries

        Returns:
            Dictionary containing search results in Chroma format.

        Raises:
            ValueError: If neither query_text nor query_embeddings provided.
            CollectionNotFoundError: If collection unavailable.
        """
        collection = self._get_collection()

        if hasattr(collection, "search") and self.host:
            try:
                from chromadb.execution.expression import Knn, Search

                # Build Search object if not provided directly
                if "searches" not in kwargs:
                    s = Search().limit(n_results)
                    if where:
                        s = s.where(where)

                    if query_embeddings:
                        # Normalize single vs multiple embeddings to single vector
                        emb = (
                            query_embeddings[0]
                            if isinstance(query_embeddings[0], list)
                            else query_embeddings
                        )
                        s = s.rank(Knn(query=emb))
                    elif query_text:
                        # Text-only search requires native embedding support
                        # Fall back to query() since Search API needs embeddings
                        logger.debug("Falling back to query() for text-only search.")
                        return self.query(
                            query_text=query_text,
                            n_results=n_results,
                            where=where,
                            **kwargs,
                        )

                    return collection.search(s, **kwargs)
                return collection.search(kwargs["searches"], **kwargs)
            except (ImportError, NotImplementedError) as e:
                logger.warning(
                    f"Native search() failed or not available: {e}. Falling back to query()."
                )

        # Standard query fallback for local/non-hosted deployments
        return self.query(
            query_text=query_text,
            query_embedding=query_embeddings[0]
            if isinstance(query_embeddings, list)
            and isinstance(query_embeddings[0], list)
            else query_embeddings,
            n_results=n_results,
            where=where,
            **kwargs,
        )

    def delete_collection(self, name: Optional[str] = None) -> None:
        """Delete a collection from the Chroma database.

        Args:
            name: Name of collection to delete. Uses default collection_name
                if not provided.

        Raises:
            ValueError: If no collection name is specified.
            CollectionNotFoundError: If collection does not exist.
        """
        client = self._get_client()
        target_name = name or self.collection_name
        if not target_name:
            msg = "Collection name is required."
            raise ValueError(msg)
        client.delete_collection(name=target_name)
        if target_name == self.collection_name:
            self.collection = None
        logger.info(f"Collection '{target_name}' deleted.")

    def list_collections(self) -> List[str]:
        """List all collections in the current database.

        Returns:
            List of collection names as strings.
        """
        client = self._get_client()
        return [c.name for c in client.list_collections()]

    def delete_documents(
        self, ids: Optional[List[str]] = None, where: Optional[Dict[str, Any]] = None
    ) -> None:
        """Delete documents from the current collection.

        Args:
            ids: List of document IDs to delete. Deletes all if None and
                where filter provided.
            where: Metadata filter to select documents for deletion.
                Deletes documents matching filter criteria.

        Raises:
            ValueError: If neither ids nor where filter is provided.
            CollectionNotFoundError: If collection unavailable.
        """
        collection = self._get_collection()
        collection.delete(ids=ids, where=where)
        logger.info(f"Deleted documents from collection '{self.collection_name}'")

    def with_tenant(
        self, tenant: str, database: Optional[str] = None
    ) -> "ChromaVectorDB":
        """Create a new instance with different tenant/database context.

        Enables multi-tenant workflows by cloning the current configuration
        with updated tenant and optional database settings.

        Args:
            tenant: Tenant name for the new instance.
            database: Database name for the new instance. Uses current
                database if not specified.

        Returns:
            New ChromaVectorDB instance configured for the specified tenant.
        """
        return ChromaVectorDB(
            host=self.host,
            port=self.port,
            api_key=self.api_key,
            tenant=tenant,
            database=database or self.database,
            path=self.path,
            persistent=self.persistent,
            collection_name=self.collection_name,
            ssl=self.ssl,
            tracing_project_name=self.tracing_project_name,
        )

    def flatten_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively flatten nested metadata for Chroma compatibility.

        Chroma's metadata indexing only supports flat key-value pairs with
        scalar values (str, int, float, bool) or uniform lists. This method
        converts nested dictionaries to dot-notation keys and handles
        complex types through string conversion.

        Args:
            metadata: Dictionary potentially containing nested dictionaries,
                lists, or complex types.

        Returns:
            Flattened dictionary compatible with Chroma metadata storage.
            Nested keys use dot notation (e.g., "parent.child.key").
            Complex types are converted to strings.
            None values are omitted.
        """
        flat = {}
        for k, v in metadata.items():
            if isinstance(v, dict):
                # Recursively flatten nested dicts with dot notation
                inner_flat = self.flatten_metadata(v)
                for ik, iv in inner_flat.items():
                    flat[f"{k}.{ik}"] = iv
            elif isinstance(v, (str, int, float, bool)):
                flat[k] = v
            elif isinstance(v, list):
                # Chroma supports lists but filtering works best with scalar types
                if all(isinstance(item, (str, int, float, bool)) for item in v):
                    flat[k] = v
                else:
                    # Convert complex list contents to string representation
                    flat[k] = str(v)
            elif v is None:
                continue
            else:
                flat[k] = str(v)
        return flat
