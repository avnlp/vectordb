"""Pinecone Vector Database.

This module provides an interface for interacting with Pinecone, enabling functionalities such as
index creation, vector upsertion, querying, and deletion. It simplifies the management of vector
databases with Pinecone's API.

Classes:
    - PineconeVectorDB: Encapsulates methods for managing Pinecone vector databases.

Methods:
    - create_index: Creates a new Pinecone index with the specified configuration.
    - upsert: Inserts or updates vectors in the selected Pinecone index.
    - query: Performs a nearest-neighbor search on the index for a given vector.
    - delete_index: Deletes the current Pinecone index.
"""

import logging
import time
from typing import Any, Optional, Union

import weave
from pinecone import ServerlessSpec
from pinecone.data import Index
from pinecone.grpc import PineconeGRPC as Pinecone
from weave import Model

from vectordb.utils.logging import LoggerFactory

logger_factory = LoggerFactory(logger_name=__name__, log_level=logging.INFO)
logger = logger_factory.get_logger()


class PineconeVectorDB(Model):
    """Interface for interacting with Pinecone vector databases.

    Provides functionalities for creating indexes, upserting vectors, querying vectors, and
    deleting indexes.
    """

    api_key: Optional[str] = None
    host: Optional[str] = None
    proxy_url: Optional[str] = None
    proxy_headers: Optional[dict[str, str]] = None
    ssl_ca_certs: Optional[str] = None
    ssl_verify: Optional[bool] = True
    additional_headers: Optional[dict[str, str]] = None
    pool_threads: int = 1
    index_name: Optional[str] = None
    tracing_project_name: str = "pinecone"
    weave_params: Optional[dict[str, Any]] = None
    client: Optional[Pinecone] = None
    index: Optional[Index] = None

    def __init__(
        self,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        proxy_url: Optional[str] = None,
        proxy_headers: Optional[dict[str, str]] = None,
        ssl_ca_certs: Optional[str] = None,
        ssl_verify: Optional[bool] = True,
        additional_headers: Optional[dict[str, str]] = None,
        pool_threads: int = 1,
        index_name: Optional[str] = None,
        tracing_project_name: str = "pinecone",
        weave_params: Optional[dict[str, Any]] = None,
    ):
        """Initialize the Pinecone client with the given parameters.

        Args:
            api_key (Optional[str]): API key for Pinecone authentication.
            host (Optional[str]): Control plane host for Pinecone.
            proxy_url (Optional[str]): Proxy URL for the connection.
            proxy_headers (Optional[dict[str, str]]): Headers for proxy authentication.
            ssl_ca_certs (Optional[str]): Path to SSL CA certificate bundle in PEM format.
            ssl_verify (Optional[bool]): Flag for SSL verification (default: True).
            additional_headers (Optional[dict[str, str]]): Additional headers for API requests.
            pool_threads (int): Number of threads for the connection pool (default: 1).
            index_name (Optional[str]): Name of the Pinecone index to use (if existing).
            tracing_project_name (str): The name of the Weave project for tracing. Defaults to "pinecone".
            weave_params (Optional[dict[str, Any]]): Additional parameters for initializing Weave.
        """
        super().__init__(
            api_key=api_key,
            host=host,
            proxy_url=proxy_url,
            proxy_headers=proxy_headers,
            ssl_ca_certs=ssl_ca_certs,
            ssl_verify=ssl_verify,
            additional_headers=additional_headers,
            pool_threads=pool_threads,
            index_name=index_name,
            tracing_project_name=tracing_project_name,
            weave_params=weave_params,
        )

        self.api_key = api_key
        self.host = host
        self.proxy_url = proxy_url
        self.proxy_headers = proxy_headers
        self.ssl_ca_certs = ssl_ca_certs
        self.ssl_verify = ssl_verify
        self.additional_headers = additional_headers or {}
        self.pool_threads = pool_threads
        self.index_name = index_name
        self.tracing_project_name = tracing_project_name
        self.weave_params = weave_params

        self.client = None
        self.index = None

        logger.info("Initializing Pinecone client.")
        self._initialize_client()

        if self.index_name:
            self._select_index(self.index_name)

        self._initialize_weave(**(weave_params or {}))

    def _initialize_weave(self, **weave_params) -> None:
        """Initialize Weave with the specified tracing project name.

        Sets up the Weave environment and creates a tracer for monitoring pipeline execution.

        Args:
            weave_params (dict[str, Any]): Additional parameters for configuring Weave.
        """
        weave.init(self.tracing_project_name, **weave_params)

    def _initialize_client(self):
        """Initialize the Pinecone client."""
        self.client = Pinecone(
            api_key=self.api_key,
            environment=self.host,
            proxy=self.proxy_url,
            proxy_headers=self.proxy_headers,
            ssl_ca_certs=self.ssl_ca_certs,
            verify_ssl=self.ssl_verify,
            additional_headers=self.additional_headers,
            pool_threads=self.pool_threads,
        )
        logger.info("Pinecone client initialized successfully.")

    def _select_index(self, index_name: str) -> bool:
        """Select an existing index for operations.

        Args:
            index_name (str): Name of the index to select.

        Returns:
            bool: True if the index exists and is selected, False otherwise.
        """
        existing_indexes = [index_info["name"] for index_info in self.client.list_indexes()]
        if index_name in existing_indexes:
            self.index = self.client.Index(index_name)
            logger.info(f"Selected existing index: {index_name}")
            return True
        logger.warning(f"Index {index_name} does not exist.")
        return False

    @weave.op()
    def create_index(
        self,
        index_name: str,
        dimension: int,
        metric: str = "cosine",
        spec: ServerlessSpec = ServerlessSpec(cloud="aws", region="us-east-1"),
        deletion_protection: str = "disabled",
    ):
        """Create a Pinecone index with the specified configuration.

        Args:
            index_name (str): Name of the new index.
            dimension (int): Dimensionality of the vectors.
            metric (str): Distance metric for similarity search (default: "cosine").
            spec (ServerlessSpec): Serverless specifications for the index.
            deletion_protection (str): Index deletion protection setting (default: "disabled").
        """
        self.index_name = index_name

        # Select an existing index
        if self._select_index(index_name):
            logger.info(f"Using existing index: {index_name}")
            return

        # Create a new index
        self.client.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=spec,
            deletion_protection=deletion_protection,
        )

        # Wait until the new index is ready
        while not self.client.describe_index(index_name).status["ready"]:
            logger.debug(f"Waiting for index '{index_name}' to become ready...")
            time.sleep(1)

        # Select the newly created index
        self._select_index(index_name)
        logger.info(f"New index '{index_name}' created and selected.")

    @weave.op()
    def upsert(
        self,
        data: list[dict[str, Any]],
        namespace: Optional[str] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ):
        """Upsert vectors into the selected Pinecone index.

        Args:
            data (List[Dict[str, Any]]): List of vectors to upsert.
            namespace (Optional[str]): Namespace for the vectors.
            batch_size (Optional[int]): Number of vectors per batch.
            show_progress (bool): Whether to show a progress bar (default: True).

        Raises:
            ValueError: If no index is selected.
        """
        if not self.index:
            msg = "No index selected. Create or specify an existing index."
            logger.error(msg)
            raise ValueError(msg)

        logger.info(f"Upserting {len(data)} vectors into index {self.index_name}.")
        self.index.upsert(
            vectors=data,
            namespace=namespace,
            batch_size=batch_size,
            show_progress=show_progress,
        )

    @weave.op()
    def query(
        self,
        namespace: str,
        vector: list[float],
        top_k: int = 5,
        sparse_vector: Optional[dict[str, Union[list[float], list[int]]]] = None,
        filter: Optional[dict[str, Union[str, float, int, bool, list, dict]]] = None,
        include_values: bool = False,
        include_metadata: bool = True,
    ) -> Any:
        """Query the Pinecone index.

        Args:
            namespace (str): Namespace for the query.
            vector (List[float]): Query vector.
            top_k (int): Number of nearest neighbors to return (default: 5).
            sparse_vector (Optional[Dict[str, Union[List[float], List[int]]]]): Sparse vector.
            filter (Optional[Dict]): Query filter.
            include_values (bool): Whether to include vector values in the response (default: False).
            include_metadata (bool): Whether to include metadata in the response (default: True).

        Returns:
            Any: Query response from Pinecone.
        """
        if not self.index:
            logger.warning("No index selected. Cannot perform query.")
            return None

        logger.info(f"Querying index {self.index_name} with top_k={top_k}.")

        query_response = self.index.query(
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            sparse_vector=sparse_vector,
            filter=filter,
            include_values=include_values,
            include_metadata=include_metadata,
        )
        return query_response

    @weave.op()
    def delete_index(self):
        """Delete the currently selected Pinecone index.

        Raises:
            ValueError: If no index is selected.
        """
        if not self.index_name:
            logger.error("No index specified for deletion.")
            msg = "No index selected. Specify or create an index before deletion."
            raise ValueError(msg)

        logger.info(f"Deleting index {self.index_name}.")
        self.client.delete_index(self.index_name)
        self.index_name = None
        self.index = None
        logger.info("Index deleted successfully.")
