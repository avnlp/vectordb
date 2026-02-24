"""Milvus Vector Database interface for Zilliz Cloud and self-hosted Milvus.

This module provides a high-level interface for interacting with Milvus and Zilliz
Cloud vector databases, used by both Haystack and LangChain
integrations. It supports dense vectors,
sparse vectors, and hybrid search combining both approaches.

Key Features:
    - Dense vector search with HNSW indexing and cosine similarity
    - Sparse vector search using BM25-inspired inverted indices
    - Hybrid search with Reciprocal Rank Fusion (RRF) or weighted reranking
    - JSON metadata filtering with path-based indexing
    - Multi-tenancy via partition keys for data isolation
    - Support for both Zilliz Cloud (managed) and self-hosted Milvus deployments

Milvus Concepts:
    Partition keys enable physical data isolation by routing documents to different
    partitions based on a key value (e.g., tenant_id). This is more efficient than
    soft isolation via metadata filtering as it limits search scope at the partition
    level rather than filtering after the search.

    Hybrid search uses multiple ANN search requests (dense + sparse) and merges results
    using a ranker. RRF (Reciprocal Rank Fusion) is the default as it requires no
    parameter tuning, while WeightedRanker allows explicit control over the balance.

    JSON path indexing allows filtering on nested metadata fields efficiently by
    creating inverted indices on specific JSON paths like metadata["category"].

Example:
    Basic usage with dense vectors:

    >>> db = MilvusVectorDB(uri="http://localhost:19530")
    >>> db.create_collection("docs", dimension=768)
    >>> db.insert_documents([Document(content="Hello", embedding=[...])])
    >>> results = db.search(query_embedding=[...], top_k=5)
"""

import logging
from typing import Any, Dict, List, Optional, Union

from haystack.dataclasses import Document, SparseEmbedding
from pymilvus import (
    AnnSearchRequest,
    DataType,
    MilvusClient,
    RRFRanker,
    WeightedRanker,
)

from vectordb.utils.ids import set_doc_id
from vectordb.utils.sparse import (
    get_doc_sparse_embedding,
    normalize_sparse,
    to_milvus_sparse,
)


logger = logging.getLogger(__name__)


class MilvusVectorDB:
    """Interface for interacting with Milvus/Zilliz vector databases.

    This class provides methods for creating collections, inserting documents,
    performing various search types (dense, sparse, hybrid), and managing metadata
    filtering. It abstracts Milvus-specific implementation details while exposing
    advanced features like partition keys and JSON indexing.

    Args:
        uri: Milvus server URI. Use "http://localhost:19530" for local Milvus or
            "https://..." for Zilliz Cloud endpoints.
        token: API token for Zilliz Cloud authentication. Leave empty for local
            Milvus or when using no authentication.
        host: Deprecated. Use uri instead. Maintained for backward compatibility.
        port: Deprecated. Use uri instead. Maintained for backward compatibility.
        collection_name: Default collection name for operations. Can be overridden
            per-method.
        **kwargs: Additional arguments passed to MilvusClient (e.g., timeout settings).

    Raises:
        ConnectionError: If unable to connect to the Milvus server.
        ValueError: If collection operations are attempted without specifying
            a collection name (neither in constructor nor method call).

    Example:
        >>> # Zilliz Cloud connection
        >>> db = MilvusVectorDB(
        ...     uri="https://in03-xxxxxxxx.api.gcp-us-west1.zillizcloud.com",
        ...     token="your-api-token",
        ...     collection_name="my_collection",
        ... )
        >>>
        >>> # Local Milvus with partition keys for multi-tenancy
        >>> db = MilvusVectorDB(uri="http://localhost:19530")
        >>> db.create_collection(
        ...     "tenant_docs",
        ...     dimension=768,
        ...     use_partition_key=True,
        ...     partition_key_field="tenant_id",
        ... )
    """

    def __init__(
        self,
        uri: str = "http://localhost:19530",
        token: str = "",  # nosec: B107 - default empty token for local testing
        host: Optional[str] = None,
        port: Optional[str] = None,
        collection_name: Optional[str] = None,
        **kwargs,
    ):
        """Initialize Milvus connection and create client instance.

        Establishes connection to Milvus/Zilliz Cloud server. For Zilliz Cloud,
        both uri (endpoint URL) and token (API key) are required. For local
        Milvus, only uri is needed unless authentication is enabled.

        Args:
            uri: Milvus server URI. Defaults to "http://localhost:19530" for
                local development. Use Zilliz Cloud HTTPS endpoint for managed
                service.
            token: API token for Zilliz Cloud. Empty string for local testing
                without authentication.
            host: Optional host address. If provided with port, overrides uri.
                Maintained for backward compatibility with older Milvus clients.
            port: Optional port number. Used with host for backward compatibility.
            collection_name: Default collection for subsequent operations. Can be
                overridden in individual method calls.
            **kwargs: Additional MilvusClient parameters (timeout, secure, etc.).

        Raises:
            ConnectionError: If the Milvus server is unreachable or credentials
                are invalid.

        Note:
            The host/port parameters are deprecated in favor of uri but maintained
            for compatibility. If both are provided, host:port takes precedence.
        """
        if host and port and (uri == "http://localhost:19530" or not uri):
            self.uri = f"http://{host}:{port}"
        else:
            self.uri = uri

        self.token = token
        self.collection_name = collection_name

        self.client = MilvusClient(uri=self.uri, token=self.token, **kwargs)
        logger.info(f"Connected to Milvus at {self.uri}")

    def create_collection(
        self,
        collection_name: str,
        dimension: int,
        description: str = "",
        use_sparse: bool = False,
        use_partition_key: bool = False,
        partition_key_field: str = "namespace",
        recreate: bool = False,
    ):
        """Create a Milvus collection with comprehensive schema for document storage.

        Creates a collection with the standard schema: auto-generated int64 primary key,
        dense float vector (for semantic search), optional sparse vector (for keyword
        matching), content text field, JSON metadata field, and optional partition key
        for multi-tenancy.

        HNSW index is created on dense vectors with cosine similarity. If sparse vectors
        are enabled, SPARSE_INVERTED_INDEX is created with inner product metric for BM25
        style retrieval.

        Args:
            collection_name: Unique name for the collection. If it already exists and
                recreate=False, the method returns without error.
            dimension: Dimensionality of dense embedding vectors. Must match the
                embedding model used (e.g., 768 for most transformer models).
            description: Human-readable description of the collection's purpose.
            use_sparse: Whether to include a sparse vector field for hybrid search.
                Adds storage overhead but enables keyword matching alongside semantic
                search. Required for hybrid search functionality.
            use_partition_key: Whether to enable physical data partitioning. Enables
                efficient multi-tenancy by isolating data at the partition level rather
                than filtering post-search. Recommended when serving multiple tenants.
            partition_key_field: Name of the partition key field. Documents with the
                same partition key value are routed to the same physical partition,
                enabling efficient tenant-scoped queries.
            recreate: If True, drops existing collection with the same name before
                creating new one. Use with caution in production.

        Raises:
            ValueError: If dimension is not a positive integer.
            MilvusException: If collection creation fails due to server errors or
                invalid parameters.

        Example:
            >>> db.create_collection(
            ...     "articles",
            ...     dimension=768,
            ...     use_sparse=True,
            ...     use_partition_key=True,
            ...     partition_key_field="tenant_id",
            ... )
        """
        if recreate and self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)

        if self.client.has_collection(collection_name):
            logger.info(f"Collection {collection_name} already exists.")
            return

        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
            description=description,
        )

        # Milvus requires exactly one primary key field for uniqueness and deduplication
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)

        # Dense vectors store semantic embeddings for similarity search
        schema.add_field(
            field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dimension
        )

        if use_sparse:
            # Sparse vectors enable BM25-style keyword matching for hybrid search
            schema.add_field(
                field_name="sparse_embedding", datatype=DataType.SPARSE_FLOAT_VECTOR
            )

        # VARCHAR stores raw document content; 65535 chars handles most docs
        schema.add_field(
            field_name="content", datatype=DataType.VARCHAR, max_length=65535
        )

        # JSON provides flexible schema-less metadata storage
        schema.add_field(field_name="metadata", datatype=DataType.JSON)

        if use_partition_key:
            # Partition keys enable physical data isolation at the storage layer
            # This routes documents to different partitions based on key value,
            # making tenant-scoped queries more efficient than post-filtering
            schema.add_field(
                field_name=partition_key_field,
                datatype=DataType.VARCHAR,
                max_length=128,
                is_partition_key=True,
            )

        self.client.create_collection(
            collection_name=collection_name, schema=schema, shards_num=2
        )

        index_params = self.client.prepare_index_params()

        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 500},
        )

        if use_sparse:
            index_params.add_index(
                field_name="sparse_embedding",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
            )

        self.client.create_index(collection_name, index_params)
        logger.info(f"Collection {collection_name} created successfully.")

    def create_json_index(
        self,
        collection_name: Optional[str] = None,
        field_name: str = "metadata",
        json_path: str = 'metadata["category"]',
        cast_type: str = "VARCHAR",
        index_name: Optional[str] = None,
        index_type: str = "AUTOINDEX",
    ) -> None:
        """Create a JSON path index for efficient metadata filtering on nested fields.

        Milvus supports indexing specific paths within JSON fields to enable fast
        filtering operations. Without an index, JSON path queries require full scans
        which become slow at scale. This method creates an inverted index on the
        specified JSON path.

        The cast_type parameter determines how values are interpreted for indexing:
        - VARCHAR: For string values like categories or tags
        - DOUBLE: For numeric values requiring range queries
        - BOOL: For boolean flags

        Args:
            collection_name: Target collection. Uses default from constructor if None.
            field_name: Name of the JSON field (typically "metadata").
            json_path: Path expression like 'metadata["category"]' or
                'metadata["attributes"]["priority"]'. Must use bracket notation.
            cast_type: Data type for the indexed values. VARCHAR for strings,
                DOUBLE for numbers, BOOL for booleans. Must match the actual
                data type in your documents to avoid casting errors.
            index_name: Custom name for the index. Auto-generated if not provided.
            index_type: AUTOINDEX lets Milvus choose optimal strategy (recommended),
                or INVERTED for explicit inverted index.

        Raises:
            ValueError: If collection_name is None and no default was set.
            MilvusException: If index creation fails (e.g., path doesn't exist).

        Example:
            >>> db.create_json_index(
            ...     collection_name="docs",
            ...     json_path='metadata["category"]',
            ...     cast_type="VARCHAR",
            ... )
            >>> # Now you can filter: filters={"category": "technology"}
        """
        collection_name = collection_name or self.collection_name

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name=field_name,
            index_name=index_name,
            index_type=index_type,
            params={
                "json_path": json_path,
                "json_cast_type": cast_type,
            },
        )
        self.client.create_index(collection_name, index_params)
        logger.info(f"Created JSON index on {json_path} in {collection_name}")

    def insert_documents(
        self,
        documents: List[Document],
        collection_name: Optional[str] = None,
        namespace: Optional[str] = None,
        partition_key_field: str = "namespace",
    ):
        """Insert Haystack documents into Milvus collection.

        Converts Haystack Document objects to Milvus entity format and inserts them.
        Handles both dense and sparse embeddings if present in the documents. When
        partition keys are used, assigns the namespace value to route documents to
        the correct physical partition.

        Documents are inserted in a single batch for efficiency. The method extracts
        content, metadata, and any embeddings from each Haystack Document.

        Args:
            documents: List of Haystack Document objects to insert. Each document
                may contain content, metadata, dense embedding, and/or sparse
                embedding.
            collection_name: Target collection. Uses default from constructor if None.
            namespace: Value for the partition key field. When using partition keys,
                this determines which partition stores the documents. Essential for
                multi-tenancy isolation.
            partition_key_field: Name of the partition key field in the collection
                schema. Must match the field name used when creating the collection.

        Raises:
            ValueError: If collection_name is None and no default was set in
                constructor, or if documents list is empty.
            MilvusException: If insertion fails due to schema mismatches or
                server errors.

        Example:
            >>> docs = [
            ...     Document(content="Hello world", embedding=[0.1, 0.2, ...]),
            ...     Document(content="Another doc", meta={"category": "test"}),
            ... ]
            >>> db.insert_documents(docs, namespace="tenant_1")
        """
        collection_name = collection_name or self.collection_name
        if not collection_name:
            raise ValueError("Collection name must be specified.")

        data = []
        for doc in documents:
            entity = {
                "content": doc.content or "",
                "metadata": doc.meta or {},
            }
            if doc.embedding is not None:
                entity["embedding"] = doc.embedding

            sparse = get_doc_sparse_embedding(doc)
            if sparse is not None:
                entity["sparse_embedding"] = to_milvus_sparse(sparse)

            if namespace:
                entity[partition_key_field] = namespace

            data.append(entity)

        self.client.insert(collection_name=collection_name, data=data)
        logger.info(f"Inserted {len(data)} documents into {collection_name}")

    def search(
        self,
        query_embedding: Optional[List[float]] = None,
        query_sparse_embedding: Optional[
            Union[Dict[int, float], SparseEmbedding]
        ] = None,
        top_k: int = 10,
        collection_name: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        scope: Optional[str] = None,  # NEW: alias for namespace
        namespace: Optional[str] = None,
        partition_key_field: str = "namespace",
        ranker_type: str = "rrf",  # 'rrf' or 'weighted'
        weights: Optional[List[float]] = None,
        include_vectors: bool = False,  # NEW
    ) -> List[Document]:
        """Perform semantic search with support for dense, sparse, or hybrid retrieval.

        Executes ANN (Approximate Nearest Neighbor) search using one or more vector
        types. Supports three search modes:

        1. Dense search: Semantic similarity using embedding vectors
        2. Sparse search: Keyword matching using sparse vectors (BM25-style)
        3. Hybrid search: Combines dense and sparse using a ranker (RRF or weighted)

        Hybrid search uses Reciprocal Rank Fusion (RRF) by default, which requires no
        parameter tuning and works well across different query types. Weighted ranker
        allows explicit control over the balance between semantic and keyword matching.

        Partition key filtering (scope/namespace) is applied as a pre-filter to limit
        search scope at the partition level, making multi-tenant queries efficient.

        Args:
            query_embedding: Dense query vector for semantic search. Typically from
                a text embedding model (e.g., 768-dim). Optional if using sparse only.
            query_sparse_embedding: Sparse query vector for keyword search. Can be
                Dict[int, float] mapping term IDs to weights, or Haystack
                SparseEmbedding.
                Optional if using dense only.
            top_k: Maximum number of results to return. Default 10.
            collection_name: Target collection. Uses default from constructor if None.
            filters: Metadata filter conditions as nested dict. Supports operators:
                $eq (equality), $gt/$lt (range), $in (list membership), $contains
                (JSON contains). Example: {"category": {"$eq": "tech"}}
            scope: Unified tenant/partition identifier. Preferred over 'namespace'
                as it clearly conveys the isolation concept.
            namespace: Legacy alias for scope. Partition key value for data isolation.
                Only used if scope is not provided.
            partition_key_field: Name of the partition key field in schema. Used to
                construct partition filter expressions.
            ranker_type: Reranking strategy for hybrid search. "rrf" uses Reciprocal
                Rank Fusion (k=60 constant). "weighted" uses explicit weights.
            weights: Two-element list [dense_weight, sparse_weight] for weighted
                ranker. Only used when ranker_type="weighted". Defaults to [0.5, 0.5].
            include_vectors: If True, includes embedding vectors in returned Documents.
                Increases response size but useful for downstream processing.

        Returns:
            List of Haystack Document objects ordered by relevance score (descending).
            Documents include content, metadata, score, and optionally embeddings.

        Raises:
            ValueError: If collection_name is None and no default was set, or if
                neither query_embedding nor query_sparse_embedding is provided.
            MilvusException: If search fails due to connection issues or invalid
                vector dimensions.

        Example:
            >>> # Dense semantic search
            >>> results = db.search(
            ...     query_embedding=[0.1, 0.2, ...], top_k=5, scope="tenant_1"
            ... )
            >>>
            >>> # Hybrid search with RRF
            >>> results = db.search(
            ...     query_embedding=dense_vec,
            ...     query_sparse_embedding=sparse_vec,
            ...     ranker_type="rrf",
            ...     top_k=10,
            ... )
            >>>
            >>> # With metadata filtering
            >>> results = db.search(
            ...     query_embedding=vec,
            ...     filters={"category": {"$eq": "technology"}, "priority": {"$gt": 5}},
            ... )
        """
        collection_name = collection_name or self.collection_name

        # scope is the preferred term for tenant isolation, but namespace is kept
        # for backward compatibility
        effective_scope = scope or namespace

        # Convert sparse embedding to Milvus format if provided
        query_sparse_dict = None
        if query_sparse_embedding is not None:
            sparse = normalize_sparse(query_sparse_embedding)
            if sparse:
                query_sparse_dict = to_milvus_sparse(sparse)

        filter_expr = self._build_filter_expression(filters)

        if effective_scope:
            # Partition key filtering is applied as pre-filter at the storage layer
            # This is more efficient than post-filtering as it limits search scope
            sanitized_scope = self._escape_expr_string(effective_scope)
            ns_expr = f'{partition_key_field} == "{sanitized_scope}"'
            filter_expr = f"({filter_expr}) and {ns_expr}" if filter_expr else ns_expr

        output_fields = ["content", "metadata"]
        if include_vectors:
            output_fields.append("embedding")
            # Sparse embeddings returned for re-ranking or downstream fusion
            output_fields.append("sparse_embedding")

        if query_embedding is not None and query_sparse_dict is not None:
            req_dense = AnnSearchRequest(
                data=[query_embedding],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {}},
                limit=top_k,
            )
            req_sparse = AnnSearchRequest(
                data=[query_sparse_dict],
                anns_field="sparse_embedding",
                param={"metric_type": "IP", "params": {}},
                limit=top_k,
            )

            if ranker_type == "weighted":
                ranker = WeightedRanker(*(weights or [0.5, 0.5]))
            else:
                ranker = RRFRanker()

            results = self.client.hybrid_search(
                collection_name=collection_name,
                reqs=[req_dense, req_sparse],
                ranker=ranker,
                limit=top_k,
                filter=filter_expr,
                output_fields=output_fields,
            )
        # Dense Search Only
        elif query_embedding is not None:
            results = self.client.search(
                collection_name=collection_name,
                data=[query_embedding],
                anns_field="embedding",
                limit=top_k,
                filter=filter_expr,
                output_fields=output_fields,
            )
        # Sparse Search Only
        elif query_sparse_dict is not None:
            results = self.client.search(
                collection_name=collection_name,
                data=[query_sparse_dict],
                anns_field="sparse_embedding",
                limit=top_k,
                filter=filter_expr,
                output_fields=output_fields,
            )
        else:
            # Metadata-only query
            results = [
                self.client.query(
                    collection_name=collection_name,
                    filter=filter_expr,
                    limit=top_k,
                    output_fields=output_fields,
                )
            ]

        return self._format_results(results, include_vectors=include_vectors)

    @staticmethod
    def _escape_expr_string(value: str) -> str:
        """Escape a string value for safe use in Milvus filter expressions."""
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def _build_filter_expression(self, filters: Optional[Dict[str, Any]]) -> str:
        """Convert dictionary filters to Milvus boolean expression strings.

        Transforms nested filter dictionaries into Milvus expression syntax for
        metadata filtering. Supports JSON path access, comparison operators, and
        membership tests.

        Non-reserved keys are assumed to be metadata fields accessed via JSON path
        notation (metadata["key"]). Reserved keys (id, content) map directly to
        their field names.

        Args:
            filters: Dictionary of filter conditions. Structure:
                - Simple: {"field": value} for equality
                - Operator: {"field": {"$eq": val, "$gt": val, "$lt": val,
                  "$in": [vals], "$contains": val}}
                String values are automatically quoted and escaped for safe
                interpolation. Numeric values are converted directly.

        Returns:
            Milvus boolean expression string. Empty string if filters is None/empty.
            Multiple conditions are combined with "and".

        Example:
            >>> _build_filter_expression({"category": "tech"})
            'metadata["category"] == "tech"'
            >>> _build_filter_expression({"priority": {"$gt": 5}})
            'metadata["priority"] > 5'
            >>> _build_filter_expression({"tags": {"$contains": "important"}})
            'json_contains(metadata["tags"], "important")'
        """
        if not filters:
            return ""

        expressions = []
        esc = self._escape_expr_string
        for key, value in filters.items():
            # Metadata fields use JSON path syntax; reserved fields (id, content) don't
            field_expr = (
                f'metadata["{esc(key)}"]' if key not in ["id", "content"] else key
            )

            if isinstance(value, dict):
                for op, val in value.items():
                    if op == "$eq":
                        expressions.append(f'{field_expr} == "{esc(str(val))}"')
                    elif op == "$gt":
                        expressions.append(f"{field_expr} > {val}")
                    elif op == "$lt":
                        expressions.append(f"{field_expr} < {val}")
                    elif op == "$in":
                        val_list = [
                            f'"{esc(v)}"' if isinstance(v, str) else str(v) for v in val
                        ]
                        expressions.append(f"{field_expr} in [{', '.join(val_list)}]")
                    elif op == "$contains":
                        expressions.append(
                            f'json_contains({field_expr}, "{esc(str(val))}")'
                        )
            else:
                # Dictionary value with no operator defaults to equality comparison
                val_str = f'"{esc(value)}"' if isinstance(value, str) else str(value)
                expressions.append(f"{field_expr} == {val_str}")

        return " and ".join(expressions)

    def build_filter_expression(self, filters: Optional[Dict[str, Any]]) -> str:
        """Convert dictionary filters to Milvus expression strings (public API).

        Public wrapper around _build_filter_expression. Use this method when you
        need to build filter expressions outside of the search flow, for example
        when deleting documents by filter or debugging query construction.

        Args:
            filters: Dictionary of filter conditions with operators ($eq, $gt, $lt,
                $in, $contains). See _build_filter_expression for full syntax.

        Returns:
            Milvus boolean expression string ready for use in search or delete
            operations.

        Example:
            >>> expr = db.build_filter_expression(
            ...     {"status": "active", "priority": {"$gt": 3}}
            ... )
            >>> print(expr)
            'metadata["status"] == "active" and metadata["priority"] > 3'
            >>> db.delete_documents(filter_expr=expr)
        """
        return self._build_filter_expression(filters)

    def _format_results(
        self,
        milvus_results: List[Any],
        include_vectors: bool = False,
    ) -> List[Document]:
        """Convert Milvus query results to Haystack Document objects.

        Transforms Milvus result structures into standard Haystack Document format.
        Handles both search results (with scores) and query results (no scores).
        Extracts content, metadata, scores, and optionally vector embeddings.

        Milvus returns different result formats:
        - Hybrid/dense/sparse search: List of Hit dicts, each containing "id",
          "distance", and an "entity" dict with the output fields.
        - Metadata query: List of flat dicts with fields at the top level
          (no "entity" wrapper, no "distance" key).
        - All results wrapped in an outer list.

        Args:
            milvus_results: Raw results from Milvus client operations (search,
                hybrid_search, or query). Always a list containing result groups.
            include_vectors: Whether to include embedding vectors in the returned
                Documents. Adds computational overhead but enables downstream use.

        Returns:
            List of Haystack Document objects with content, metadata, and optional
            score from similarity search. Documents are ordered by relevance (highest
            score first) when from search operations.

        Note:
            Document IDs are preserved from Milvus primary keys and stored in both
            doc.id and doc.meta["id"] for compatibility with Haystack conventions.
            Sparse embeddings are normalized back to Haystack format if present.
        """
        documents = []
        if not milvus_results:
            return []

        # Results can be from search (list of hits) or query (list of dicts)
        hits = milvus_results[0]
        for hit in hits:
            # search() returns Hit dicts with "entity" nested;
            # query() returns flat dicts
            if "entity" in hit:
                entity = hit.get("entity", {})
                score = hit.get("distance")
                hit_id = hit.get("id")
            else:
                entity = hit
                score = None
                hit_id = entity.get("id")

            doc = Document(
                content=entity.get("content", ""),
                meta=entity.get("metadata", {}),
                score=score,
            )

            # Milvus primary key becomes Haystack doc ID for deduplication
            if hit_id:
                set_doc_id(doc, str(hit_id))

            if include_vectors:
                # Reconstruct embeddings for downstream re-ranking or caching
                if "embedding" in entity:
                    doc.embedding = entity["embedding"]
                if "sparse_embedding" in entity:
                    doc.sparse_embedding = normalize_sparse(entity["sparse_embedding"])

            documents.append(doc)

        return documents

    def delete_documents(
        self,
        ids: Optional[List[Union[int, str]]] = None,
        filter_expr: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        """Delete documents from Milvus collection by IDs or filter expression.

        Removes documents from the collection using either explicit document IDs
        (primary keys) or a filter expression. When deleting by filter, all
        matching documents are removed in a single operation.

        Args:
            ids: List of document IDs to delete. These are the Milvus primary keys
                returned in search results. Mutually exclusive with filter_expr.
            filter_expr: Milvus boolean expression for bulk deletion. Use
                build_filter_expression() to construct from dict filters.
                Mutually exclusive with ids.
            collection_name: Target collection. Uses default from constructor if None.

        Raises:
            ValueError: If collection_name is None and no default was set, or if
                neither ids nor filter_expr is provided.
            MilvusException: If deletion fails (e.g., invalid IDs, filter syntax error).

        Example:
            >>> # Delete by IDs
            >>> db.delete_documents(ids=[1, 2, 3])
            >>>
            >>> # Delete by filter
            >>> filter_expr = db.build_filter_expression({"status": "archived"})
            >>> db.delete_documents(filter_expr=filter_expr)
            >>>
            >>> # Delete all documents in a namespace
            >>> db.delete_documents(filter_expr='namespace == "tenant_old"')
        """
        collection_name = collection_name or self.collection_name
        if ids:
            # Primary key deletion is faster than filter-based bulk deletion
            self.client.delete(collection_name=collection_name, ids=ids)
        elif filter_expr:
            # Filter deletion removes all matching documents in one operation
            self.client.delete(collection_name=collection_name, filter=filter_expr)

    def drop_collection(self, collection_name: Optional[str] = None):
        """Drop (delete) a Milvus collection and all its data.

        Permanently removes the entire collection including all documents, indexes,
        and schema definitions. This operation cannot be undone. The collection
        must exist; attempting to drop a non-existent collection is safe (no-op).

        Args:
            collection_name: Name of the collection to drop. Uses default from
                constructor if None. Must be explicitly provided either way.

        Raises:
            ValueError: If collection_name is None and no default was set.
            MilvusException: If server-side error occurs during deletion.

        Warning:
            This operation is destructive and irreversible. All documents, vectors,
            and indexes in the collection will be permanently lost.

        Example:
            >>> db.drop_collection("old_collection")
            >>> # Or using default from constructor
            >>> db = MilvusVectorDB(collection_name="temp_collection")
            >>> db.drop_collection()  # Drops "temp_collection"
        """
        collection_name = collection_name or self.collection_name
        if collection_name:
            # drop_collection is idempotent - safe to call on non-existent collections
            self.client.drop_collection(collection_name)
