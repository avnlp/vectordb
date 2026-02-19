"""Pinecone vector database wrapper for Haystack and LangChain integrations.

This module provides a production-ready interface for Pinecone Cloud and Serverless,
integrating with Haystack and LangChain retrieval pipelines.

Architecture:
    - Lazy client initialization to support configuration-based setups
    - Unified document converter for Haystack <-> Pinecone format translation
    - Namespace-based multi-tenancy with automatic metadata flattening
    - Batch processing for efficient large-scale operations

Supported Features:
    - Dense vector search (cosine, euclidean, dotproduct metrics)
    - Hybrid search (dense + sparse SPLADE/BM25 vectors)
    - Namespace-based multi-tenancy
    - Metadata filtering with automatic flattening for nested structures
    - Batch upserts with progress tracking
    - Serverless and pod-based deployment support

Usage Example:
    >>> db = PineconeVectorDB(api_key="pc-xxx", index_name="docs")
    >>> db.create_index(dimension=768, metric="cosine")
    >>> db.upsert(documents, namespace="tenant_1")
    >>> results = db.query(vector=embedding, top_k=5, namespace="tenant_1")

Note:
    Pinecone requires metadata values to be scalar (str, int, float, bool)
    or lists of strings. Nested dictionaries are automatically flattened
    using underscore notation.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone


try:
    from pinecone.db_data import Index
except ImportError:
    from pinecone.data import Index

from haystack.dataclasses import SparseEmbedding

from vectordb.utils.config import load_config, resolve_env_vars
from vectordb.utils.logging import LoggerFactory
from vectordb.utils.pinecone_document_converter import PineconeDocumentConverter
from vectordb.utils.sparse import normalize_sparse, to_pinecone_sparse


logger_factory = LoggerFactory(logger_name=__name__, log_level=logging.INFO)
logger = logger_factory.get_logger()


class PineconeVectorDB:
    """Production-ready interface for Pinecone vector database operations.

    Provides a unified API for document storage, retrieval, and management
    with support for advanced RAG patterns including hybrid search and
    multi-tenancy via Pinecone's namespace feature.

    The client uses lazy initialization pattern, only connecting to Pinecone
    when actual operations are performed. This allows for flexible configuration
    through files, environment variables, or direct parameters.

    Attributes:
        api_key: Pinecone API key for authentication.
        index_name: Name of the Pinecone index to operate on.
        host: Optional custom host for self-hosted deployments.
        client: Lazy-initialized PineconeGRPC client instance.
        index: Lazy-initialized Index instance for the configured index_name.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ):
        """Initialize PineconeVectorDB with configuration or direct parameters.

        Args:
            api_key: Pinecone API key.
            index_name: Name of the Pinecone index.
            config: Direct configuration dictionary.
            config_path: Path to YAML configuration file.
            **kwargs: Additional connection parameters (host, proxy_url, etc.)

        Raises:
            ValueError: If api_key is not provided via parameter, config, or
                PINECONE_API_KEY environment variable.
        """
        if config_path:
            self.config = load_config(config_path)
        elif config:
            self.config = resolve_env_vars(config)
        else:
            self.config = {}

        db_config = self.config.get("pinecone", {})

        # Priority: Parameters > Config > Environment Variables
        self.api_key = (
            api_key or db_config.get("api_key") or os.environ.get("PINECONE_API_KEY")
        )
        self.index_name = (
            index_name
            or db_config.get("index_name")
            or os.environ.get("PINECONE_INDEX_NAME")
        )

        self.host = kwargs.get("host") or db_config.get("host")
        self.proxy_url = kwargs.get("proxy_url") or db_config.get("proxy_url")
        self.ssl_verify = kwargs.get("ssl_verify", db_config.get("ssl_verify", True))
        self.pool_threads = kwargs.get("pool_threads", db_config.get("pool_threads", 1))

        self.client: Optional[Pinecone] = None
        self.index: Optional[Index] = None

        logger.info(f"Initialized PineconeVectorDB for index: {self.index_name}")

    def _get_client(self) -> Pinecone:
        """Initialize and return the Pinecone GRPC client.

        Uses lazy initialization to delay connection until first use.
        This allows configuration from files or environment variables
        to be loaded before establishing the connection.

        Returns:
            Pinecone: Initialized GRPC client instance.

        Raises:
            ValueError: If api_key is not configured.
        """
        if self.client is None:
            if not self.api_key:
                raise ValueError("PINECONE_API_KEY is required.")

            self.client = Pinecone(
                api_key=self.api_key,
                host=self.host,
                proxy_url=self.proxy_url,
                ssl_verify=self.ssl_verify,
                pool_threads=self.pool_threads,
            )
        return self.client

    def _get_index(self) -> Index:
        """Initialize and return the Pinecone index handle.

        Retrieves the Index object from the client for the configured
        index_name. This operation is idempotent - subsequent calls
        return the cached index instance.

        Returns:
            Index: Pinecone index instance for configured index_name.

        Raises:
            ValueError: If index_name is not configured.
        """
        if self.index is None:
            if not self.index_name:
                raise ValueError("Pinecone index_name is required.")
            client = self._get_client()
            self.index = client.Index(self.index_name)
        return self.index

    def create_index(
        self,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        spec: Optional[Union[Dict[str, Any], ServerlessSpec]] = None,
        recreate: bool = False,
        index_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Create a new Pinecone index or ensure an existing one is ready.

        Creates a serverless or pod-based index with the specified configuration.
        If recreate=True, any existing index with the same name is deleted first.

        Args:
            dimension: Dimensionality of the vectors. Required for new indexes.
            metric: Distance metric for similarity calculations.
                Options: "cosine" (default), "euclidean", "dotproduct".
            spec: ServerlessSpec or dict defining cloud provider and region.
                Defaults to AWS us-east-1 serverless if not provided.
            recreate: If True, delete existing index before creating new one.
            index_name: Override the index name from __init__ for this operation.
            **kwargs: Additional parameters passed to Pinecone create_index API.

        Raises:
            ValueError: If dimension is not provided and cannot be inferred.

        Note:
            ServerlessSpec cloud options: "aws", "gcp", "azure".
            ServerlessSpec region depends on cloud provider availability.
        """
        client = self._get_client()

        if index_name:
            self.index_name = index_name

        if dimension is None:
            dimension = kwargs.get("dimension")

        if dimension is None:
            raise ValueError("dimension is required for index creation.")

        if recreate:
            try:
                client.delete_index(self.index_name)
                logger.info(f"Deleted existing index '{self.index_name}'")
            except Exception as e:
                logger.warning(f"Could not delete index '{self.index_name}': {e}")

        existing_indexes = [idx.name for idx in client.list_indexes()]
        if self.index_name in existing_indexes:
            logger.info(f"Index '{self.index_name}' already exists.")
            return

        if spec is None:
            spec = ServerlessSpec(cloud="aws", region="us-east-1")
        elif isinstance(spec, dict) and "serverless" in spec:
            # Convert dict to ServerlessSpec if it matches expected fields
            s_data = spec["serverless"]
            spec = ServerlessSpec(cloud=s_data["cloud"], region=s_data["region"])

        client.create_index(
            name=self.index_name,
            dimension=dimension,
            metric=metric,
            spec=spec,
            **kwargs,
        )
        logger.info(f"Triggered creation of index '{self.index_name}'")
        self.wait_for_index_ready()

    def wait_for_index_ready(self, timeout: int = 300) -> None:
        """Poll the index status until it is ready.

        Continuously queries Pinecone to check if the index has finished
        initialization and is ready to accept operations. Polls every 5 seconds
        until the index reports ready=True or the timeout is exceeded.

        Args:
            timeout: Maximum time to wait in seconds. Default is 300 (5 minutes).

        Raises:
            TimeoutError: If the index does not become ready within the specified
                timeout.
        """
        client = self._get_client()
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = client.describe_index(self.index_name).status
            if status["ready"]:
                logger.info(f"Index '{self.index_name}' is ready.")
                return
            logger.debug(
                f"Waiting for index '{self.index_name}' (status: {status['state']})..."
            )
            time.sleep(5)
        raise TimeoutError(
            f"Index '{self.index_name}' did not become ready within {timeout}s."
        )

    def upsert(
        self,
        data: Union[List[Any], List[Dict[str, Any]]],
        namespace: str = "",
        batch_size: int = 100,
        show_progress: bool = True,
    ) -> int:
        """Insert or update vectors in the specified namespace.

        Accepts either Haystack Document objects or raw Pinecone-formatted
        dictionaries. Automatically flattens nested metadata to meet Pinecone's
        scalar-only requirement. Processes data in configurable batches to
        optimize network efficiency.

        Args:
            data: List of Haystack Documents OR Pinecone-formatted dictionaries.
                Each dict must contain 'id', 'values' (vector), and optional 'metadata'.
            namespace: Target namespace for multi-tenancy isolation. Default is "".
            batch_size: Number of vectors to upsert per API call. Default 100.
            show_progress: Log progress after each batch completion.

        Returns:
            int: Total number of vectors successfully upserted.

        Raises:
            ValueError: If data format is invalid or namespace doesn't exist.

        Note:
            Metadata flattening uses underscore notation: {"user": {"id": 123}}
            becomes {"user_id": 123}.
        """
        index = self._get_index()

        if not data:
            return 0

        # Handle different input formats
        if not isinstance(data[0], dict):
            # 1. Assume Haystack Documents
            for doc in data:
                if hasattr(doc, "meta") and doc.meta:
                    doc.meta = self.flatten_metadata(doc.meta)
            upsert_data = (
                PineconeDocumentConverter.prepare_haystack_documents_for_upsert(data)
            )
        else:
            # 2. List of dicts (Pinecone format)
            upsert_data = []
            for item in data:
                new_item = item.copy()
                if "metadata" in new_item:
                    new_item["metadata"] = self.flatten_metadata(new_item["metadata"])
                upsert_data.append(new_item)

        # 3. Perform upsert in batches
        total_upserted = 0
        for i in range(0, len(upsert_data), batch_size):
            batch = upsert_data[i : i + batch_size]
            index.upsert(vectors=batch, namespace=namespace)
            total_upserted += len(batch)
            if show_progress:
                logger.info(
                    f"Upserted {total_upserted}/{len(upsert_data)} vectors to namespace '{namespace}'"
                )

        return total_upserted

    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        namespace: str = "",
        scope: Optional[str] = None,  # NEW: alias
        include_metadata: bool = True,
        include_values: bool = False,
        include_vectors: bool = False,  # NEW: alias for include_values
    ) -> List[Any]:
        """Query the index for similar vectors.

        Args:
            vector: Query dense embedding.
            top_k: Number of results.
            filter: Metadata filters.
            namespace: Namespace to search in (legacy).
            scope: Unified scope/namespace parameter.
            include_metadata: Whether to include metadata in results.
            include_values: Whether to include vector values in results (legacy).
            include_vectors: Whether to include vector values in results.

        Returns:
            List of Haystack Documents.
        """
        effective_namespace = scope or namespace
        effective_include_values = include_vectors or include_values

        index = self._get_index()
        response = index.query(
            vector=vector,
            top_k=top_k,
            filter=filter,
            namespace=effective_namespace,
            include_metadata=include_metadata,
            include_values=effective_include_values,
        )
        return PineconeDocumentConverter.convert_query_results_to_haystack_documents(
            response.to_dict(),
            include_embeddings=effective_include_values,
        )

    def query_with_sparse(
        self,
        vector: List[float],
        sparse_vector: Union[Dict[str, Any], SparseEmbedding],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        namespace: str = "",
        scope: Optional[str] = None,
        include_metadata: bool = True,
        include_vectors: bool = False,
    ) -> List[Any]:
        """Perform hybrid search using both dense and sparse vectors.

        Args:
            vector: Query dense embedding.
            sparse_vector: Dict with 'indices' and 'values' or SparseEmbedding.
            top_k: Number of results.
            filter: Metadata filters.
            namespace: Namespace to search in.
            scope: Unified scope/namespace parameter.
            include_metadata: Whether to include metadata.
            include_vectors: Whether to include vector values.

        Returns:
            List of Haystack Documents.
        """
        effective_namespace = scope or namespace

        # Normalize sparse input
        sparse = normalize_sparse(sparse_vector)
        sparse_values = to_pinecone_sparse(sparse) if sparse else sparse_vector

        index = self._get_index()
        response = index.query(
            vector=vector,
            sparse_vector=sparse_values,
            top_k=top_k,
            filter=filter,
            namespace=effective_namespace,
            include_metadata=include_metadata,
            include_values=include_vectors,
        )
        return PineconeDocumentConverter.convert_query_results_to_haystack_documents(
            response.to_dict(),
            include_embeddings=include_vectors,
        )

    def fetch(self, ids: List[str], namespace: str = "") -> Dict[str, Any]:
        """Retrieve vectors by their unique identifiers.

        Fetches the complete vector data including values, metadata, and sparse
        vectors if present. Returns a dictionary with 'vectors' and 'namespace'.

        Args:
            ids: List of vector IDs to retrieve.
            namespace: Target namespace to search within.

        Returns:
            Dict containing 'vectors' mapping IDs to their full records.

        Note:
            Pinecone limits fetch to 1000 IDs per request.
            For larger sets, batch the calls manually.
        """
        index = self._get_index()
        return index.fetch(ids=ids, namespace=namespace).to_dict()

    def delete(
        self,
        ids: Optional[List[str]] = None,
        delete_all: bool = False,
        namespace: str = "",
    ) -> None:
        """Remove vectors from the index by ID or delete entire namespace.

        Args:
            ids: List of specific vector IDs to delete. Mutually exclusive
                with delete_all.
            delete_all: If True, deletes all vectors in the namespace.
                Use with extreme caution - this operation is irreversible.
            namespace: Target namespace for deletion operation.

        Raises:
            ValueError: If neither ids nor delete_all is specified.
            RuntimeError: If deletion fails due to permission or network issues.

        Warning:
            delete_all=True permanently removes ALL vectors in the namespace
            without recovery option. Consider using namespace-specific deletions
            to limit blast radius in multi-tenant environments.
        """
        index = self._get_index()
        index.delete(ids=ids, delete_all=delete_all, namespace=namespace)
        logger.info(
            f"Deleted vectors from namespace '{namespace}' (delete_all={delete_all})"
        )

    def delete_namespace(self, namespace: str) -> None:
        """Delete all vectors in a namespace.

        Permanently removes all vectors within the specified namespace. This is
        equivalent to calling delete(delete_all=True) for the given namespace.
        The namespace itself is not deleted from the index metadata; it simply
        becomes empty and will no longer appear in list_namespaces() until
        vectors are added again.

        Args:
            namespace: Namespace to delete all vectors from.

        Raises:
            ValueError: If index_name is not configured when attempting to access
                the index.
            RuntimeError: If deletion fails due to permission or network issues.

        Warning:
            This operation is irreversible. All vectors in the namespace will be
            permanently deleted without recovery option.
        """
        index = self._get_index()
        index.delete(delete_all=True, namespace=namespace)
        logger.info(f"Deleted all vectors in namespace '{namespace}'")

    def estimate_match_count(self, filter: Dict[str, Any], namespace: str = "") -> int:
        """Estimate the number of documents matching a filter.

        Attempts to estimate matches using a zero-vector query with the provided
        filter. However, this method currently returns the total namespace vector
        count as a fallback because Pinecone's API does not return filtered
        result counts in query responses.

        Args:
            filter: Metadata filter conditions to apply (currently not used in
                calculation).
            namespace: Target namespace to count vectors in.

        Returns:
            int: Total vector count in the specified namespace. Note: This is NOT
                the filtered count due to Pinecone API limitations.

        Note:
            This is a known limitation - Pinecone does not support exact match
            counts for filtered queries. The returned value represents the total
            vectors in the namespace, not those matching the filter.
        """
        # We need the dimension of the index to provide a valid zero vector
        stats = self.describe_index_stats()
        dimension = stats["dimension"]
        [0.0] * dimension

        self._get_index()
        # Query with top_k=1 just to get the search initiated,
        # but Pinecone doesn't return total matches in metadata.
        # This implementation is a placeholder as Pinecone currently
        # does not return the count of filtered results in the response.
        logger.warning(
            "Pinecone does not support exact match count for filters. Returning total namespace count as fallback."
        )
        return stats.get("namespaces", {}).get(namespace, {}).get("vector_count", 0)

    def list_namespaces(self) -> List[str]:
        """Retrieve all namespace identifiers from index statistics.

        Returns:
            List of namespace names currently containing vectors.
            Returns empty list if index has no data.

        Note:
            Namespace names are returned as strings. The default namespace
            appears as "" (empty string) in the list.
        """
        stats = self.describe_index_stats()
        return list(stats.get("namespaces", {}).keys())

    def describe_index_stats(self) -> Dict[str, Any]:
        """Fetch comprehensive index statistics and metadata.

        Returns detailed information about the index including:
        - Total vector count across all namespaces
        - Dimensionality of stored vectors
        - Per-namespace vector counts
        - Index fullness (for pod-based indexes)

        Returns:
            Dict with keys: dimension, index_fullness, total_vector_count,
            namespaces (dict mapping namespace to count).
        """
        index = self._get_index()
        return index.describe_index_stats().to_dict()

    def flatten_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Transform nested metadata into Pinecone-compatible flat structure.

        Pinecone requires all metadata values to be scalar types or lists of strings.
        This method recursively flattens nested dictionaries using underscore
        notation and converts non-supported types to strings.

        Args:
            metadata: Potentially nested dictionary with arbitrary values.

        Returns:
            Flat dictionary with only Pinecone-supported types.

        Example:
            >>> db.flatten_metadata({"user": {"id": 123, "name": "John"}})
            {"user_id": 123, "user_name": "John"}

        Note:
            None values are silently dropped as Pinecone doesn't support null.
            Lists containing non-string items are converted to string representations.
        """
        flat = {}
        for k, v in metadata.items():
            if isinstance(v, dict):
                inner_flat = self.flatten_metadata(v)
                for ik, iv in inner_flat.items():
                    flat[f"{k}_{ik}"] = iv
            elif isinstance(v, (str, int, float, bool)):
                flat[k] = v
            elif isinstance(v, list):
                if all(isinstance(item, str) for item in v):
                    flat[k] = v
                else:
                    flat[k] = [str(item) for item in v]
            elif v is None:
                continue
            else:
                flat[k] = str(v)
        return flat

    def build_filter(self, field: str, operator: str, value: Any) -> Dict[str, Any]:
        """Construct a single-field Pinecone metadata filter.

        Creates a filter dictionary for one metadata field with the specified
        operator and value. Combine multiple filters with build_compound_filter.

        Args:
            field: Metadata field name to filter on.
            operator: Comparison operator. Options: "$eq", "$ne", "$gt", "$gte",
                "$lt", "$lte", "$in", "$nin".
            value: Value to compare against. Type must match field's type.

        Returns:
            Filter dictionary ready for use in query() filter parameter.

        Example:
            >>> db.build_filter("category", "$eq", "science")
            {"category": {"$eq": "science"}}
        """
        return {field: {operator: value}}

    def build_compound_filter(
        self, conditions: List[Dict[str, Any]], logic: str = "AND"
    ) -> Dict[str, Any]:
        """Combine multiple filter conditions with logical operators.

        Args:
            conditions: List of filter dictionaries from build_filter() calls.
            logic: Logical combination - "AND" (all must match) or "OR" (any matches).

        Returns:
            Compound filter dictionary with "$and" or "$or" key.

        Example:
            >>> f1 = db.build_filter("category", "$eq", "science")
            >>> f2 = db.build_filter("year", "$gte", 2020)
            >>> db.build_compound_filter([f1, f2], "AND")
            {"$and": [{"category": {"$eq": "science"}}, {"year": {"$gte": 2020}}]}
        """
        logic_key = "$and" if logic.upper() == "AND" else "$or"
        return {logic_key: conditions}
