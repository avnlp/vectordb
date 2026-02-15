"""Chroma vector database interface.

This module provides an interface for interacting with Chroma vector databases,
including collection management, vector upserts, and similarity search queries.
"""

import logging
import sys
from typing import Any, Optional

import pysqlite3  # noqa: F401
import weave  # type: ignore[attr-defined]
from weave import Model  # type: ignore[misc, attr-defined]


sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb  # noqa: E402
from chromadb import Client, Collection  # noqa: E402
from chromadb.api.configuration import CollectionConfiguration  # noqa: E402
from chromadb.api.types import (  # noqa: E402
    CollectionMetadata,
    Embeddable,
    EmbeddingFunction,
)
from chromadb.utils import embedding_functions  # noqa: E402

from vectordb.utils.logging import LoggerFactory  # noqa: E402


logger_factory = LoggerFactory(logger_name=__name__, log_level=logging.INFO)
logger = logger_factory.get_logger()


class ChromaVectorDB(Model):
    """Interface for interacting with Chroma vector databases.

    This class provides functionalities to manage Chroma collections, upsert vectors,
    and query vectors for similarity search and retrieval tasks.
    """

    path: str = "./chroma"
    persistent: bool = True
    collection_name: Optional[str] = None
    tracing_project_name: str = "pinecone"
    weave_params: Optional[dict[str, Any]] = None
    client: Optional[Client] = None
    collection: Optional[Collection] = None

    def __init__(
        self,
        path: str = "./chroma",
        persistent: bool = True,
        collection_name: Optional[str] = None,
        tracing_project_name: str = "chroma",
        weave_params: Optional[dict[str, Any]] = None,
    ):
        """Initialize the ChromaVectorDB client.

        Args:
            path (str): Path to the database for persistence. Defaults to "./chroma".
            persistent (bool): Whether to use a persistent database. Defaults to True.
            collection_name (Optional[str]): Name of an existing collection to use.
                Defaults to None.
            tracing_project_name (str): The name of the Weave project for tracing.
                Defaults to "chroma".
            weave_params (Optional[dict[str, Any]]): Additional parameters for
                initializing Weave.

        Raises:
            ValueError: If the client initialization fails.
        """
        super().__init__(
            path=path,
            persistent=persistent,
            collection_name=collection_name,
            tracing_project_name=tracing_project_name,
            weave_params=weave_params,
        )

        self.client = chromadb.Client()
        if persistent:
            self.client = chromadb.PersistentClient(path)

        logger.info("Chroma client initialized successfully.")

        if collection_name is not None:
            self.collection = self.client.get_collection(collection_name)

        self.tracing_project_name = tracing_project_name
        self.weave_params = weave_params

        self._initialize_weave(**(weave_params or {}))

    def _initialize_weave(self, **weave_params: Any) -> None:
        """Initialize Weave with the specified tracing project name.

        Sets up the Weave environment and creates a tracer for monitoring pipeline
        execution.

        Args:
            weave_params (dict[str, Any]): Additional parameters for configuring Weave.
        """
        weave.init(self.tracing_project_name, **weave_params)

    @weave.op()  # type: ignore[no-untyped-dec]
    def create_collection(  # type: ignore[no-untyped-def]
        self,
        name: str,
        configuration: Optional[CollectionConfiguration] = None,
        metadata: Optional[CollectionMetadata] = None,
        embedding_function: Optional[EmbeddingFunction[Embeddable]] = None,
        **kwargs: Any,
    ) -> None:
        """Create a new collection.

        Args:
            name (str): Name of the collection to create or retrieve.
            configuration (Optional[CollectionConfiguration]): Configuration for the
                collection.
            metadata (Optional[CollectionMetadata]): Metadata associated with the
                collection.
            embedding_function (Optional[EmbeddingFunction[Embeddable]]): Function for
                embedding data. Defaults to DefaultEmbeddingFunction if not provided.
            **kwargs: Additional arguments for the collection creation.

        Returns:
            None
        """
        if embedding_function is None:
            embedding_function = embedding_functions.DefaultEmbeddingFunction()

        self.collection = self.client.get_or_create_collection(
            name=name,
            configuration=configuration,
            metadata=metadata,
            embedding_function=embedding_function,
            **kwargs,
        )

    @weave.op()  # type: ignore[no-untyped-dec]
    def upsert(  # type: ignore[no-untyped-def]
        self,
        data: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Upserts (inserts or updates) vectors into the current collection.

        Args:
            data (Dict[str, Any]): Data containing embeddings, texts, metadatas, and
                ids.
                - embeddings: A list of embedding vectors.
                - texts: A list of corresponding text documents.
                - metadatas: A list of metadata dictionaries for the documents.
                - ids: A list of unique identifiers for the embeddings.
            **kwargs: Additional arguments for the upsert operation.

        Raises:
            ValueError: If the collection is not initialized.

        Returns:
            None
        """
        if not hasattr(self, "collection"):
            msg = "No collection initialized. Use `create_collection` first."
            raise ValueError(msg)

        self.collection.add(
            embeddings=data["embeddings"],
            documents=data["texts"],
            metadatas=data["metadatas"],
            ids=data["ids"],
            **kwargs,
        )

    @weave.op()  # type: ignore[no-untyped-dec]
    def query(  # type: ignore[no-untyped-def]
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: Optional[dict[str, Any]] = None,
        where_document: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Query the collection for similar vectors.

        Args:
            query_embedding (List[float]): The embedding vector to query against the
                collection.
            n_results (int): The number of results to retrieve. Defaults to 10.
            where (Optional[Dict[str, Any]]): Filter conditions for metadata.
            where_document (Optional[Dict[str, Any]]): Filter conditions for documents.
            **kwargs: Additional arguments for the query operation.

        Raises:
            ValueError: If the collection is not initialized.

        Returns:
            Dict[str, Any]: A dictionary containing the query results.
        """
        if not hasattr(self, "collection"):
            msg = "No collection initialized. Use `create_collection` first."
            raise ValueError(msg)

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            where_document=where_document,
            **kwargs,
        )
