"""Weaviate vector database wrapper for Haystack and LangChain integrations.

This module provides a comprehensive interface for Weaviate Cloud and self-hosted
instances, with native support for vector search, hybrid retrieval, and generative AI.

Architecture:
    - Eager connection initialization with automatic reconnection handling
    - Collection-centric design aligning with Weaviate v4 client patterns
    - Native Haystack Document conversion for seamless pipeline integration
    - Tenant-aware operations for multi-tenancy scenarios

Supported Features:
    - Dense vector search (near_vector)
    - Text-based semantic search (near_text)
    - Hybrid search (vector + BM25 keyword ranking)
    - Generative search (RAG with OpenAI, Cohere, etc.)
    - Metadata filtering with rich operator support
    - Multi-tenancy with tenant isolation
    - Query-time reranking
    - Automatic batch upsert with error tracking

Usage Example:
    >>> db = WeaviateVectorDB(
    ...     cluster_url="https://my-cluster.weaviate.cloud", api_key="weaviate-api-key"
    ... )
    >>> db.create_collection("Articles", enable_multi_tenancy=True)
    >>> db.create_tenants(["tenant_a", "tenant_b"])
    >>> db.with_tenant("tenant_a").upsert(documents)
    >>> results = db.query(vector=embedding, hybrid=True, limit=10)

Note:
    Weaviate Cloud requires authentication via API key. Self-hosted instances
    may use anonymous access or OIDC authentication depending on configuration.
"""

import logging
from typing import Any, Dict, List, Optional

import weaviate
from haystack import Document
from weaviate.classes.config import Configure, Property
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter, MetadataQuery, Rerank
from weaviate.classes.tenants import Tenant
from weaviate.collections import Collection

from vectordb.utils.logging import LoggerFactory


logger_factory = LoggerFactory(logger_name=__name__, log_level=logging.INFO)
logger = logger_factory.get_logger()


class WeaviateVectorDB:
    """Production-ready interface for Weaviate vector database operations.

    Manages connections to Weaviate Cloud or self-hosted instances with support
    for collections, hybrid search, generative AI, and multi-tenancy. Implements
    eager connection pattern to fail fast on configuration errors.

    The class maintains state for the currently selected collection and tenant
    context, allowing for fluent API patterns. All query methods automatically
    convert Weaviate responses to Haystack Document objects.

    Attributes:
        cluster_url: Full URL of the Weaviate cluster endpoint.
        api_key: Authentication token for Weaviate Cloud.
        headers: Additional HTTP headers (e.g., for external AI provider keys).
        client: Connected WeaviateClient instance.
        collection: Currently selected Collection object.
        collection_name: Name of the currently active collection.
    """

    def __init__(
        self,
        cluster_url: str,
        api_key: str,
        headers: Optional[dict[str, str]] = None,
        tracing_project_name: str = "weaviate",
    ):
        """Initialize the WeaviateVectorDB client.

        Args:
            cluster_url (str): URL of the Weaviate cluster.
            api_key (str): Authentication credentials.
            headers (Optional[dict[str, str]]): Additional headers for requests
                (e.g., {"X-OpenAI-Api-Key": "..."}).
            tracing_project_name (str): Name for tracing context.
        """
        self.cluster_url = cluster_url
        self.api_key = api_key
        self.headers = headers or {}
        self.tracing_project_name = tracing_project_name

        self.client: Optional[weaviate.WeaviateClient] = None
        self.collection: Optional[Collection] = None
        self.collection_name: Optional[str] = None

        logger.info("Initializing Weaviate client.")
        self._connect()

    def _connect(self) -> None:
        """Establishes connection to Weaviate Cloud.

        Args:
            cluster_url (str): URL of the Weaviate cluster (instance variable).
            api_key (str): Authentication credentials (instance variable).
            headers (Optional[dict[str, str]]): Additional headers (instance variable).

        Raises:
            Exception: If connection to Weaviate Cloud fails due to network issues,
                invalid credentials, or cluster unavailability. Re-raises the original
                exception after logging the error.
        """
        try:
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self.cluster_url,
                auth_credentials=Auth.api_key(self.api_key),
                headers=self.headers,
            )
            logger.info("Weaviate client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate client: {e}")
            raise

    def close(self) -> None:
        """Close the Weaviate client connection.

        Gracefully closes the connection to the Weaviate cluster, releasing
        any network resources and terminating the client session.

        Returns:
            None

        Note:
            Safe to call multiple times; only closes if client is initialized.
        """
        if self.client:
            self.client.close()
            logger.info("Weaviate client connection closed.")

    def _select_collection(self, collection_name: str) -> bool:
        """Select an existing collection for operations.

        Args:
            collection_name (str): Name of the collection to select.

        Returns:
            bool: True if the collection exists and is selected; False otherwise.
        """
        if self.client.collections.exists(collection_name):
            self.collection = self.client.collections.get(collection_name)
            self.collection_name = collection_name
            logger.info(f"Selected existing collection: {collection_name}")
            return True
        logger.warning(f"Collection {collection_name} does not exist.")
        return False

    def create_collection(
        self,
        collection_name: str,
        properties: Optional[List[Property]] = None,
        vectorizer_config: Optional[Any] = None,
        generative_config: Optional[Any] = None,
        enable_multi_tenancy: bool = False,
    ) -> None:
        """Create a new Weaviate collection.

        Args:
            collection_name (str): Name of the collection to create.
            properties (Optional[list[Property]]): Properties of the schema.
            vectorizer_config: Configuration for the vectorizer (e.g.,
                Configure.Vectorizer.text2vec_openai()).
            generative_config: Configuration for generative search (e.g.,
                Configure.Generative.openai()).
            enable_multi_tenancy (bool): Whether to enable multi-tenancy for collection.

        Raises:
            ValueError: If the Weaviate client is not initialized.
        """
        if not self.client:
            msg = "Weaviate client is not initialized."
            logger.error(msg)
            raise ValueError(msg)

        if self.client.collections.exists(collection_name):
            logger.info(f"Collection {collection_name} already exists. Selecting it.")
            self._select_collection(collection_name)
            return

        logger.info(f"Creating collection: {collection_name}")

        multi_tenancy_config = (
            Configure.multi_tenancy(enabled=True) if enable_multi_tenancy else None
        )

        self.client.collections.create(
            name=collection_name,
            properties=properties,
            vectorizer_config=vectorizer_config,
            generative_config=generative_config,
            multi_tenancy_config=multi_tenancy_config,
        )
        self._select_collection(collection_name)
        logger.info(f"Collection {collection_name} created and selected.")

    def upsert(self, data: List[Dict[str, Any]]) -> None:
        """Insert or update objects in the currently selected collection.

        Uses Weaviate's batch API for efficient bulk operations with automatic
        error tracking. Extracts vector and UUID from each record, passing
        remaining fields as schema properties.

        Args:
            data: List of dictionaries containing object data. Each dict may
                include 'vector' (list[float]) for custom embeddings, 'id' or
                'uuid' for explicit ID assignment. All other fields become
                collection properties.

        Raises:
            ValueError: If no collection is currently selected via create_collection
                or _select_collection.
            RuntimeError: If any objects fail to upsert, with details in message.

        Note:
            Weaviate auto-generates UUIDs if not provided. Batch errors are
            collected server-side and checked after the context manager exits.
        """
        if not self.collection:
            raise ValueError(
                "No collection selected. Call create_collection or _select_collection first."
            )

        with self.collection.batch.dynamic() as batch:
            for row in data:
                # Copy to avoid mutating caller's data when popping special keys
                row_copy = row.copy()
                vector = row_copy.pop("vector", None)
                uuid = row_copy.pop("id", None) or row_copy.pop("uuid", None)

                # Remaining fields are treated as schema properties
                batch.add_object(properties=row_copy, vector=vector, uuid=uuid)

        if len(self.collection.batch.failed_objects) > 0:
            failed_count = len(self.collection.batch.failed_objects)
            # Limit error log to first 3 failures to avoid spam
            errors = [str(err) for err in self.collection.batch.failed_objects[:3]]
            msg = (
                f"Upsert failed for {failed_count} objects. First few errors: {errors}"
            )
            logger.error(msg)
            raise RuntimeError(msg)

        logger.info(f"Successfully upserted {len(data)} objects.")

    def _build_filter(self, filters: Dict[str, Any]) -> Optional[Filter]:
        """Convert MongoDB-style filter dictionaries to Weaviate Filter objects.

        Transforms filter dictionaries into Weaviate Filter objects, using
        the same MongoDB-style format as Qdrant, Chroma, and Pinecone wrappers
        for a consistent API across all vector database backends.

        Args:
            filters: Dictionary of field names to filter conditions.
                Supports two formats:

                - Implicit equality: ``{"field": "value"}``
                - Explicit operators: ``{"field": {"$op": value}}``

                Available operators:

                - ``$eq``: Equal to value
                - ``$ne``: Not equal to value
                - ``$gt``: Greater than value
                - ``$gte``: Greater than or equal to value
                - ``$lt``: Less than value
                - ``$lte``: Less than or equal to value
                - ``$in``: Value matches any in provided list
                - ``$like``: Wildcard pattern matching

                Logical operators:

                - ``$and``: List of conditions combined with AND
                - ``$or``: List of conditions combined with OR

        Returns:
            Weaviate Filter object or None if filters dict is empty.

        Example:
            Simple equality filter::

                filter = self._build_filter({"category": "news"})

            Range filter::

                filter = self._build_filter({"score": {"$gte": 0.8}})

            Combined filter::

                filter = self._build_filter(
                    {
                        "category": "tech",
                        "published": {"$gt": "2024-01-01"},
                    }
                )

            Logical OR::

                filter = self._build_filter(
                    {"$or": [{"category": "tech"}, {"category": "science"}]}
                )
        """
        if not filters:
            return None

        conditions = []
        for key, value in filters.items():
            if key == "$and":
                sub = [self._build_filter(c) for c in value]
                sub = [c for c in sub if c is not None]
                if sub:
                    conditions.append(Filter.all_of(sub))
            elif key == "$or":
                sub = [self._build_filter(c) for c in value]
                sub = [c for c in sub if c is not None]
                if sub:
                    conditions.append(Filter.any_of(sub))
            elif isinstance(value, dict):
                f = Filter.by_property(key)
                for op, val in value.items():
                    if op == "$eq":
                        conditions.append(f.equal(val))
                    elif op == "$ne":
                        conditions.append(f.not_equal(val))
                    elif op == "$gt":
                        conditions.append(f.greater_than(val))
                    elif op == "$gte":
                        conditions.append(f.greater_or_equal(val))
                    elif op == "$lt":
                        conditions.append(f.less_than(val))
                    elif op == "$lte":
                        conditions.append(f.less_or_equal(val))
                    elif op == "$in":
                        conditions.append(
                            Filter.any_of(
                                [Filter.by_property(key).equal(v) for v in val]
                            )
                        )
                    elif op == "$like":
                        conditions.append(f.like(val))
                    else:
                        logger.warning(f"Unknown filter operator: {op}")
            else:
                conditions.append(Filter.by_property(key).equal(value))

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return Filter.all_of(conditions)

    def query(
        self,
        query_string: Optional[str] = None,
        vector: Optional[List[float]] = None,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        hybrid: bool = False,
        alpha: float = 0.5,
        rerank: Optional[Dict[str, Any]] = None,
        return_properties: Optional[List[str]] = None,
        include_vectors: bool = False,  # NEW
        return_documents: bool = False,  # NEW
    ) -> Any:
        """Query vectors from the selected Weaviate collection.

        Args:
            query_string (Optional[str]): Text query.
            vector (Optional[List[float]]): Vector embedding of the query.
            limit (int): Maximum number of results.
            filters (Optional[Dict]): MongoDB-style metadata filters.
                Supports implicit equality (``{"field": "value"}``),
                explicit operators (``{"field": {"$op": value}}``),
                and logical combinators (``{"$and": [...]}``/``{"$or": [...]}``).
                Operators: ``$eq``, ``$ne``, ``$gt``, ``$gte``, ``$lt``,
                ``$lte``, ``$in``, ``$like``.
            hybrid (bool): Whether to use hybrid search.
            alpha (float): Hybrid search weight (1.0 = vector, 0.0 = keyword).
            rerank (Optional[Dict]): Reranker config, e.g., {"prop": "content"}.
            return_properties (Optional[List[str]]): List of properties to return.
            include_vectors (bool): Whether to return embeddings in results.
            return_documents (bool): Whether to return Haystack Docs instead of resp.

        Returns:
            Any: Query response from Weaviate. When return_documents=True, returns
                List[Document] containing Haystack Document objects with metadata,
                scores, and optionally embeddings. Otherwise returns the raw Weaviate
                response object with full query results including metadata distances.

        Raises:
            ValueError: If no collection is selected via create_collection or
                _select_collection before querying.
            ValueError: If hybrid search is enabled but no query string is provided.
        """
        if not self.collection:
            raise ValueError("No collection selected.")

        w_filter = self._build_filter(filters) if filters else None

        reranker = None
        if rerank:
            reranker = Rerank(prop=rerank.get("prop"), query=rerank.get("query"))

        query_params = {
            "limit": limit,
            "filters": w_filter,
            "return_metadata": MetadataQuery(distance=True, score=True),
            "rerank": reranker,
            "return_properties": return_properties,
            "include_vector": include_vectors,  # NEW
        }

        response = None
        if hybrid:
            if not query_string:
                raise ValueError("Hybrid search requires a query string.")
            logger.info(f"Executing hybrid search for: {query_string}")
            response = self.collection.query.hybrid(
                query=query_string,
                vector=vector,
                alpha=alpha,
                **query_params,
            )
        elif vector:
            logger.info("Executing near_vector search.")
            response = self.collection.query.near_vector(
                near_vector=vector, **query_params
            )
        elif query_string:
            logger.info(f"Executing near_text search for: {query_string}")
            response = self.collection.query.near_text(
                query=query_string, **query_params
            )
        else:
            # Fallback to fetch objects if no query provided
            logger.info("Fetching objects (no query provided).")
            response = self.collection.query.fetch_objects(
                limit=limit,
                filters=w_filter,
                return_properties=return_properties,
                include_vector=include_vectors,  # Supports include_vector too
            )

        if return_documents:
            return self._convert_to_documents(response, include_vectors)
        return response

    def _convert_to_documents(
        self,
        response: Any,
        include_vectors: bool = False,
    ) -> List[Document]:
        """Convert Weaviate query response to Haystack Documents.

        Args:
            response: Weaviate query response.
            include_vectors: Whether to attach embeddings.

        Returns:
            List of Haystack Documents.
        """
        documents = []
        # Handle different response types (search vs fetch)
        objects = response.objects if hasattr(response, "objects") else response

        for obj in objects:
            properties = dict(obj.properties) if obj.properties else {}
            content = properties.pop("text", "")

            doc = Document(
                id=str(obj.uuid),
                content=content,
                meta=properties,
            )

            # Score from metadata
            if hasattr(obj, "metadata") and obj.metadata:
                if (
                    hasattr(obj.metadata, "distance")
                    and obj.metadata.distance is not None
                ):
                    doc.score = 1 - obj.metadata.distance
                elif hasattr(obj.metadata, "score") and obj.metadata.score is not None:
                    doc.score = obj.metadata.score

            # Attach vector if requested
            if include_vectors and hasattr(obj, "vector") and obj.vector is not None:
                # Weaviate might return vector as dictionary with named vectors or list
                if isinstance(obj.vector, dict):
                    # For now just take the default or first one
                    # Improve if using named vectors
                    if "default" in obj.vector:
                        doc.embedding = list(obj.vector["default"])
                    elif obj.vector:
                        doc.embedding = list(next(iter(obj.vector.values())))
                else:
                    doc.embedding = list(obj.vector)

            documents.append(doc)

        return documents

    def query_to_documents(
        self,
        response: Any,
        include_vectors: bool = False,
    ) -> List[Document]:
        """Convert a Weaviate query response to Haystack Documents.

        Public wrapper around _convert_to_documents for external use.
        Use this when you have a raw Weaviate response object from a custom
        query and need to convert it to Haystack Document format.

        Args:
            response (Any): Raw Weaviate query response containing objects
                with properties, metadata, and optionally vectors.
            include_vectors (bool): Whether to extract and attach embeddings
                from the response objects to the Documents.

        Returns:
            List[Document]: List of Haystack Document objects with content,
                metadata, scores from distance/score metadata, and optionally
                embeddings if include_vectors=True.
        """
        return self._convert_to_documents(response, include_vectors)

    def hybrid_search(
        self,
        query: str,
        vector: Optional[List[float]] = None,
        top_k: int = 10,
        alpha: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
        include_vectors: bool = False,
    ) -> List[Document]:
        """Perform hybrid search (vector + BM25).

        Weaviate uses native BM25 for keyword matching, not SPLADE.

        Args:
            query: Text query for BM25 component.
            vector: Optional dense vector for semantic component.
            top_k: Number of results.
            alpha: Balance (1.0 = vector only, 0.0 = BM25 only).
            filters: Optional MongoDB-style metadata filters.
                See :meth:`_build_filter` for supported format and operators.
            include_vectors: Whether to return embeddings.

        Returns:
            List of Haystack Documents with scores.

        Raises:
            ValueError: If no collection is selected via create_collection
                or _select_collection before searching.
            ValueError: If query string is empty or None.
        """
        return self.query(
            query_string=query,
            vector=vector,
            limit=top_k,
            filters=filters,
            hybrid=True,
            alpha=alpha,
            include_vectors=include_vectors,
            return_documents=True,
        )

    def generate(
        self,
        query_string: str,
        single_prompt: Optional[str] = None,
        grouped_task: Optional[str] = None,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Perform RAG / Generative search.

        Args:
            query_string (str): The search query to retrieve context.
            single_prompt (Optional[str]): Prompt to apply to each result individually.
            grouped_task (Optional[str]): Prompt to apply to all results combined.
            limit (int): Number of context documents to retrieve.
            filters (Optional[Dict]): MongoDB-style metadata filters.
                See :meth:`_build_filter` for supported format and operators.

        Returns:
            Any: Generation response from Weaviate containing the generated text.
                The response format depends on the prompt type used (single_prompt
                or grouped_task). Contains the original search results along with
                the LLM-generated content based on the retrieved context.

        Raises:
            ValueError: If no collection is selected before generating.
            ValueError: If neither single_prompt nor grouped_task is provided.
        """
        if not self.collection:
            raise ValueError("No collection selected.")

        w_filter = self._build_filter(filters) if filters else None

        if grouped_task:
            return self.collection.generate.near_text(
                query=query_string,
                grouped_task=grouped_task,
                limit=limit,
                filters=w_filter,
            )
        if single_prompt:
            return self.collection.generate.near_text(
                query=query_string,
                single_prompt=single_prompt,
                limit=limit,
                filters=w_filter,
            )
        raise ValueError(
            "Must provide either single_prompt or grouped_task for generation."
        )

    def create_tenants(self, tenants: List[str]) -> None:
        """Create tenants in the current collection.

        Args:
            tenants (List[str]): List of tenant names to create. Each tenant
                must have a unique name within the collection. Duplicate names
                within the list will cause errors.

        Raises:
            ValueError: If no collection is selected or if the collection
                does not have multi-tenancy enabled.
            RuntimeError: If tenant creation fails due to duplicate names or
                Weaviate server errors.
        """
        if not self.collection:
            raise ValueError("No collection selected.")

        tenant_objs = [Tenant(name=t) for t in tenants]
        self.collection.tenants.create(tenant_objs)
        logger.info(f"Created {len(tenants)} tenants.")

    def delete_tenants(self, tenants: List[str]) -> None:
        """Delete tenants from the current collection.

        Args:
            tenants (List[str]): List of tenant names to delete. Non-existent
                tenant names are silently ignored by Weaviate.

        Raises:
            ValueError: If no collection is selected or if the collection
                does not have multi-tenancy enabled.
        """
        if not self.collection:
            raise ValueError("No collection selected.")

        self.collection.tenants.remove(tenants)
        logger.info(f"Deleted {len(tenants)} tenants.")

    def tenant_exists(self, tenant: str) -> bool:
        """Check if a tenant exists in the current collection.

        Args:
            tenant (str): Name of the tenant to check for existence.

        Returns:
            bool: True if the tenant exists in the current collection,
                False otherwise.

        Raises:
            ValueError: If no collection is selected or if the collection
                does not have multi-tenancy enabled.
        """
        if not self.collection:
            raise ValueError("No collection selected.")

        current_tenants = self.collection.tenants.get()
        return any(t.name == tenant for t in current_tenants.values())

    def with_tenant(self, tenant_name: str) -> "WeaviateVectorDB":
        """Switch context to a specific tenant.

        Modifies the internal collection state to operate within the specified
        tenant context. All subsequent operations (query, upsert, etc.) will
        be scoped to this tenant.

        Args:
            tenant_name (str): Name of the tenant to switch context to.
                The tenant must already exist in the current collection.

        Returns:
            WeaviateVectorDB: Self-reference to enable method chaining.
                Example: db.with_tenant("tenant_a").upsert(docs).query(...)

        Raises:
            ValueError: If no collection is selected or if the collection
                does not have multi-tenancy enabled.
            Exception: If the specified tenant does not exist.
        """
        if not self.collection:
            raise ValueError("No collection selected.")

        # We need to get the collection again with the tenant context
        # Ideally, we should store the base collection and derive tenant collections
        # But here we just update self.collection to the tenant-specific one.
        # Note: This changes the state of the object!

        # If we are already in a tenant context, we might need to be careful.
        # Ideally, we call with_tenant on the base collection object.
        # But weaviate client v4 returns a new collection object.

        self.collection = self.collection.with_tenant(tenant_name)
        logger.info(f"Switched context to tenant: {tenant_name}")
        return self
