"""Weaviate vector database integration module.

This module provides a WeaviateVectorDB class for interacting with Weaviate
vector databases, including collection management, vector operations, and
querying capabilities with Weave tracing integration.
"""

import logging
from typing import Any, Optional, Union

import weave
import weaviate
from weave import Model
from weaviate import WeaviateClient
from weaviate.classes.config import Configure, Property
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.collections import Collection

from vectordb.utils.logging import LoggerFactory


logger_factory = LoggerFactory(logger_name=__name__, log_level=logging.INFO)
logger = logger_factory.get_logger()


class WeaviateVectorDB(Model):
    """Interface for interacting with Weaviate vector databases.

    Provides functionalities for creating collections, upserting vectors,
    querying vectors, and managing collections.
    """

    cluster_url: str
    api_key: str
    headers: Optional[dict[str, str]] = None
    collection_name: Optional[str] = None
    tracing_project_name: str
    weave_params: Optional[dict[str, Any]] = None
    client: Optional[WeaviateClient] = None
    collection: Optional[Collection] = None

    def __init__(
        self,
        cluster_url: str,
        api_key: str,
        headers: Optional[dict[str, str]] = None,
        collection_name: Optional[str] = None,
        tracing_project_name: str = "weaviate",
        weave_params: Optional[dict[str, Any]] = None,
    ):
        """Initialize the WeaviateVectorDB client.

        Args:
            cluster_url (str): URL of the Weaviate cluster.
            api_key (Optional[AuthCredentials]): Authentication credentials
                for Weaviate.
            headers (Optional[dict[str, str]]): Additional headers for the
                HTTP requests.
            collection_name (Optional[str]): Name of the collection to
                interact with.
            tracing_project_name (str): The name of the Weave project for
                tracing. Defaults to "weaviate".
            weave_params (Optional[dict[str, Any]]): Additional parameters
                for initializing Weave.
        """
        super().__init__(
            cluster_url=cluster_url,
            api_key=api_key,
            headers=headers,
            collection_name=collection_name,
            tracing_project_name=tracing_project_name,
            weave_params=weave_params,
        )

        self.cluster_url = cluster_url
        self.api_key = api_key
        self.headers = headers or {}
        self.collection_name = collection_name

        self.client = None
        self.collection = None

        logger.info("Initializing Weaviate client.")
        try:
            self._initialize_client()
            if self.collection_name:
                self._select_collection(self.collection_name)
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

        self.tracing_project_name = tracing_project_name
        self.weave_params = weave_params

        self._initialize_weave(**(weave_params or {}))

    def _initialize_weave(self, **weave_params) -> None:
        """Initialize Weave with the specified tracing project name.

        Sets up the Weave environment and creates a tracer for monitoring
        pipeline execution.

        Args:
            weave_params (dict[str, Any]): Additional parameters for
                configuring Weave.
        """
        weave.init(self.tracing_project_name, **weave_params)

    def _initialize_client(self) -> None:
        """Initialize the Weaviate client."""
        try:
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self.cluster_url,
                auth_credentials=Auth.api_key(api_key=self.api_key),
                headers=self.headers,
            )
            logger.info("Weaviate client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate client: {e}")
            raise

    def _select_collection(self, collection_name: str) -> bool:
        """Select an existing collection for operations.

        Args:
            collection_name (str): Name of the collection to select.

        Returns:
            bool: True if the collection exists and is selected; False otherwise.
        """
        if self.client.collections.exists(collection_name):
            self.collection = self.client.collections.get(collection_name)
            logger.info(f"Selected existing collection: {collection_name}")
            return True
        logger.warning(f"Collection {collection_name} does not exist.")
        return False

    @weave.op()
    def create_collection(
        self,
        collection_name: str,
        properties: Optional[list[Property]] = None,
        vectorizer_config: Optional[Configure.Vectorizer] = None,
        generative_config: Optional[Configure.Generative] = None,
    ) -> None:
        """Create a new Weaviate collection.

        Args:
            collection_name (str): Name of the collection to create.
            properties (Optional[list[Property]]): Properties of the
                collection schema.
            vectorizer_config (Optional[Configure.Vectorizer]): Vectorizer
                configuration.
            generative_config (Optional[Configure.Generative]): Generative
                AI configuration.

        Raises:
            ValueError: If the Weaviate client is not initialized.
        """
        if not self.client:
            msg = "Weaviate client is not initialized."
            logger.error(msg)
            raise ValueError(msg)

        self.collection_name = collection_name
        if self._select_collection(collection_name):
            return

        logger.info(f"Creating collection: {collection_name}")
        self.client.collections.create(
            name=collection_name,
            properties=properties,
            vectorizer_config=vectorizer_config,
            generative_config=generative_config,
        )
        self._select_collection(collection_name)
        logger.info(f"Collection {collection_name} created and selected.")

    @weave.op()
    def upsert(self, data: list[dict[str, Any]]) -> None:
        """Upsert vectors into the selected Weaviate collection.

        Args:
            data (list[dict[str, Any]]): List of records with vector and
                properties to upsert.

        Raises:
            RuntimeError: If upsertion fails for any objects.
        """
        with self.collection.batch.dynamic() as batch:
            for row in data:
                vector = row.pop("vector", None)
                uuid = row.pop("uuid", None)
                logger.debug(f"Upserting row with UUID: {uuid}")
                batch.add_object(properties=row, vector=vector, uuid=uuid)

        if len(self.collection.batch.failed_objects) > 0:
            failed_count = len(self.collection.batch.failed_objects)
            msg = f"Upsert failed for {failed_count} objects."
            logger.error(msg)
            raise RuntimeError(msg)

    @weave.op()
    def query(
        self,
        vector: list[float],
        limit: int = 10,
        filters: Optional[Union[Filter, tuple[Filter, ...]]] = None,
        hybrid: bool = False,
        alpha: float = 0.5,
        query_string: Optional[str] = None,
    ) -> Any:
        """Query vectors from the selected Weaviate collection.

        Args:
            vector (list[float]): Vector embedding of the query.
            limit (int): Maximum number of results to return. Default is 10.
            filters (Optional[Union[Filter, tuple[Filter, ...]]]): Query filters.
            hybrid (bool): Whether to use a hybrid query. Default is False.
            alpha (float): Weight for hybrid query results. Default is 0.5.
            query_string (Optional[str]): Query string for hybrid search.

        Returns:
            Any: Query response from Weaviate.
        """
        query_params = {
            "vector": vector,
            "limit": limit,
            "filters": filters,
            "return_metadata": MetadataQuery(distance=True, score=True),
        }

        if hybrid:
            query_response = self.collection.query.hybrid(
                query=query_string,
                alpha=alpha,
                **query_params,
            )
        else:
            query_response = self.collection.query.near_vector(**query_params)
        return query_response
