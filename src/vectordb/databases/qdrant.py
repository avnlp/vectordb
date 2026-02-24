"""Qdrant vector database wrapper for Haystack and LangChain integrations.

This module provides a production-ready interface for Qdrant vector database,
enabling advanced RAG (Retrieval-Augmented Generation) pipelines
with Haystack and LangChain.

Features:
    - Multi-tenancy: Payload-based tenant isolation using Qdrant's is_tenant
      optimization for efficient filtering in multi-tenant environments.
    - Hybrid Search: Combines dense semantic vectors with sparse lexical vectors
      using Reciprocal Rank Fusion (RRF) for improved retrieval quality.
    - Named Vectors: Supports multiple vector spaces within a single collection,
      allowing separate dense and sparse vector configurations.
    - Quantization: Memory-efficient storage using scalar (INT8) or binary
      quantization to reduce memory footprint by 4-32x with minimal accuracy loss.
    - Maximal Marginal Relevance (MMR): Re-ranking strategy that balances
      relevance with diversity to reduce result redundancy.
    - Advanced Filtering: MongoDB-style filter syntax supporting equality,
      range, and set operations on document metadata.

Example:
    Basic usage with dense search::

        db = QdrantVectorDB()
        db.create_collection(dimension=768)
        db.index_documents(documents)
        results = db.search(query_vector=embedding, top_k=10)

    Hybrid search with multi-tenancy::

        db.create_collection(dimension=768, use_sparse=True)
        db.create_namespace_index()
        db.index_documents(documents, scope="tenant_1")
        results = db.search(
            query_vector={"dense": dense_vec, "sparse": sparse_vec},
            scope="tenant_1",
            search_type="hybrid",
        )

Note:
    Qdrant supports both gRPC (default, faster) and HTTP protocols.
    gRPC is preferred for production workloads due to lower latency.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import (
    BinaryQuantization,
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
    Range,
    ScalarQuantization,
    ScoredPoint,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from vectordb.utils.config import load_config, resolve_env_vars
from vectordb.utils.logging import LoggerFactory
from vectordb.utils.qdrant_document_converter import QdrantDocumentConverter
from vectordb.utils.sparse import get_doc_sparse_embedding


logger_factory = LoggerFactory(logger_name=__name__, log_level=logging.INFO)
logger = logger_factory.get_logger()


class QdrantVectorDB:
    """Production-ready Qdrant vector database interface for Haystack.

    Provides unified access to Qdrant's advanced features for building
    scalable RAG pipelines. Manages connection lifecycle, collection
    configuration, and document operations with tenant isolation.

    Attributes:
        client: QdrantClient instance for database operations.
        config: Resolved configuration dictionary with environment variables.
        url: Qdrant server URL (defaults to localhost:6333).
        api_key: Authentication token for cloud deployments.
        collection_name: Target collection for all operations.
        dense_vector_name: Identifier for dense vector space in hybrid mode.
        sparse_vector_name: Identifier for sparse vector space in hybrid mode.
        quantization_config: Quantization settings for memory optimization.

    Example:
        Initialize from config file::

            db = QdrantVectorDB(config_path="config/qdrant.yaml")

        Initialize programmatically::

            db = QdrantVectorDB(
                config={
                    "qdrant": {
                        "url": "https://cloud.qdrant.io",
                        "api_key": "your-key",
                        "collection_name": "my_docs",
                        "quantization": {"type": "scalar", "quantile": 0.99},
                    }
                }
            )
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
    ):
        """Initialize Qdrant connection and configuration.

        Configuration priority: config_path > config > empty dict.
        Environment variables are resolved within the configuration.

        Args:
            config: Direct configuration dictionary with "qdrant" key.
            config_path: Path to YAML file containing Qdrant settings.

        Raises:
            ConnectionError: If Qdrant server is unreachable.

        Example:
            Typical configuration structure::

                {
                    "qdrant": {
                        "url": "http://localhost:6333",
                        "api_key": None,
                        "collection_name": "documents",
                        "timeout": 60.0,
                        "prefer_grpc": True,
                        "dense_vector_name": "dense",
                        "sparse_vector_name": "sparse",
                        "quantization": {
                            "type": "scalar",
                            "quantile": 0.99,
                            "always_ram": True,
                        },
                    }
                }
        """
        # Load configuration from file or dict, with environment variable resolution
        if config_path:
            self.config = load_config(config_path)
        elif config:
            self.config = resolve_env_vars(config)
        else:
            self.config = {}

        qdrant_config = self.config.get("qdrant", {})

        # Connection parameters with environment fallback for cloud deployments
        self.url = qdrant_config.get("url") or os.environ.get(
            "QDRANT_URL", "http://localhost:6333"
        )
        self.api_key = qdrant_config.get("api_key") or os.environ.get("QDRANT_API_KEY")
        self.collection_name = qdrant_config.get(
            "collection_name", "haystack_collection"
        )

        # Initialize client with gRPC preferred for production performance
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=qdrant_config.get("timeout", 60.0),
            prefer_grpc=qdrant_config.get("prefer_grpc", True),
        )

        # Named vector configuration enables hybrid search with separate spaces
        self.dense_vector_name = qdrant_config.get("dense_vector_name", "dense")
        self.sparse_vector_name = qdrant_config.get("sparse_vector_name", "sparse")
        self.quantization_config = qdrant_config.get("quantization", {})

        logger.info(f"Initialized QdrantVectorDB connected to {self.url}")

    def create_collection(
        self,
        dimension: int,
        recreate: bool = False,
        use_sparse: bool = False,
        distance: str = "Cosine",
    ):
        """Create a Qdrant collection with configurable vector spaces.

        Supports both single-vector (dense-only) and multi-vector (hybrid)
        configurations. Named vectors enable hybrid search by storing dense
        and sparse embeddings in separate vector spaces within the same point.

        Args:
            dimension: Dimensionality of dense vectors (e.g., 768 for BERT).
            recreate: If True, delete existing collection before creating.
                Warning: Destructive operation that removes all data.
            use_sparse: If True, configure named vectors for hybrid search.
                Creates separate vector spaces for dense and sparse embeddings.
            distance: Similarity metric for dense vectors. Options: "Cosine",
                "Euclidean", "Dot". Cosine is standard for semantic similarity.

        Raises:
            ValueError: If distance metric is invalid.

        Note:
            Hybrid search requires documents with both dense (semantic) and
            sparse (lexical) embeddings. Use ``use_sparse=True`` when creating
            the collection, then index documents containing both vector types.
            The fusion happens at query time using RRF (Reciprocal Rank Fusion).

        Example:
            Dense-only collection for basic semantic search::

                db.create_collection(dimension=768, distance="Cosine")

            Hybrid collection combining semantic + lexical search::

                db.create_collection(dimension=768, use_sparse=True, distance="Cosine")
        """
        if recreate:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted existing collection '{self.collection_name}'")

        if self.client.collection_exists(self.collection_name):
            logger.info(f"Collection '{self.collection_name}' already exists.")
            return

        # Vector Configuration
        vectors_config = {}

        # 1. Dense Vector Config
        dist_metric = getattr(Distance, distance.upper(), Distance.COSINE)

        # Apply Quantization if configured
        quant_config = None
        if self.quantization_config:
            q_type = self.quantization_config.get("type")
            if q_type == "scalar":
                quant_config = ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=self.quantization_config.get("quantile", 0.99),
                        always_ram=self.quantization_config.get("always_ram", True),
                    )
                )
            elif q_type == "binary":
                quant_config = BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(
                        always_ram=self.quantization_config.get("always_ram", True)
                    )
                )

        if use_sparse:
            # For hybrid, we use named vectors
            vectors_config[self.dense_vector_name] = VectorParams(
                size=dimension,
                distance=dist_metric,
            )
            sparse_vectors_config = {
                self.sparse_vector_name: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            }
        else:
            # Standard single vector
            vectors_config = VectorParams(
                size=dimension,
                distance=dist_metric,
            )
            sparse_vectors_config = None

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
            quantization_config=quant_config,
        )
        logger.info(
            f"Created collection '{self.collection_name}' with dim={dimension}, "
            f"sparse={use_sparse}, quantization={bool(quant_config)}"
        )

    def create_payload_index(
        self,
        field_name: str,
        field_schema: str = "keyword",
        collection_name: Optional[str] = None,
        is_tenant: bool = False,
    ) -> None:
        """Create a payload index for fast filtering.

        Payload indexes dramatically improve query performance when filtering
        on metadata fields. Essential for multi-tenant setups and complex
        filtering requirements.

        Args:
            field_name: Name of the payload field to index.
            field_schema: Schema type (keyword, text, integer, float, bool,
                           geo, datetime).
            collection_name: Collection to create index on. Defaults to
                ``self.collection_name`` if not specified.
            is_tenant: If True, use Qdrant's tenant optimization (1.16+).
                Enables efficient tenant isolation filtering at scale.

        Returns:
            None

        Raises:
            ValueError: If field_schema is not a valid schema type.
            RuntimeError: If index creation fails due to collection not existing
                or insufficient permissions.

        Example:
            Create index for tenant filtering::

                db.create_payload_index("tenant_id", is_tenant=True)

            Create index for date filtering::

                db.create_payload_index("created_at", field_schema="datetime")
        """
        collection_name = collection_name or self.collection_name

        schema_map = {
            "keyword": models.PayloadSchemaType.KEYWORD,
            "text": models.PayloadSchemaType.TEXT,
            "integer": models.PayloadSchemaType.INTEGER,
            "float": models.PayloadSchemaType.FLOAT,
            "bool": models.PayloadSchemaType.BOOL,
            "geo": models.PayloadSchemaType.GEO,
            "datetime": models.PayloadSchemaType.DATETIME,
        }

        self.client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=schema_map.get(field_schema, models.PayloadSchemaType.KEYWORD),
            is_tenant=is_tenant,
        )
        logger.info(f"Created payload index on {field_name} (is_tenant={is_tenant})")

    def create_namespace_index(
        self,
        collection_name: Optional[str] = None,
        namespace_field: str = "tenant_id",
    ) -> None:
        """Create an optimized namespace/tenant index.

        Convenience method that wraps ``create_payload_index`` with tenant
        optimization enabled. This creates a specialized index that Qdrant
        optimizes for high-cardinality tenant filtering, dramatically improving
        query performance in multi-tenant scenarios.

        Args:
            collection_name: Collection to create index on. Defaults to
                ``self.collection_name`` if not specified.
            namespace_field: Name of the tenant identifier field in payloads.
                Defaults to "tenant_id".

        Returns:
            None

        Raises:
            RuntimeError: If index creation fails due to collection not existing.

        Note:
            Must be called after ``create_collection`` but before indexing
            documents with tenant IDs. The is_tenant optimization is only
            available in Qdrant 1.16+.

        Example:
            Setup for multi-tenant collection::

                db.create_collection(dimension=768)
                db.create_namespace_index()
                db.index_documents(docs, scope="tenant_1")
        """
        self.create_payload_index(
            field_name=namespace_field,
            field_schema="keyword",
            collection_name=collection_name,
            is_tenant=True,
        )

    def index_documents(
        self,
        documents: List[Any],
        tenant_id: Optional[str] = None,
        scope: Optional[str] = None,
        batch_size: int = 100,
    ):
        """Index documents with optional tenant isolation.

        Converts Haystack Documents to Qdrant points and upserts them into
        the collection. Automatically detects sparse vectors for hybrid search
        support. Documents are batched for efficient insertion.

        Args:
            documents: List of Haystack Documents to index. Each document
                should contain embeddings in its ``embedding`` field. For
                hybrid search, documents should also have sparse embeddings.
            tenant_id: Tenant ID for multi-tenancy (legacy parameter).
                Use ``scope`` for new code.
            scope: Scope/tenant ID for multi-tenancy. Injected into document
                metadata as ``tenant_id`` field for filtering.
            batch_size: Number of documents to upsert in each batch.
                Larger batches are more efficient but use more memory.

        Returns:
            None

        Raises:
            ValueError: If documents list is empty or contains invalid
                document objects.
            RuntimeError: If upsert operation fails due to connection issues
                or collection not existing.
            TypeError: If documents contain invalid embedding types.

        Note:
            The upsert operation is asynchronous (``wait=False``) for speed.
            This means documents may not be immediately searchable. Call
            ``client.wait_for_commit()`` if immediate consistency is required.

        Example:
            Index documents for a specific tenant::

                docs = [Document(content="Hello"), Document(content="World")]
                db.index_documents(docs, scope="tenant_1")

            Batch index large document set::

                db.index_documents(all_docs, batch_size=500)
        """
        effective_scope = scope or tenant_id

        # Inject scope/tenant_id into metadata before conversion
        if effective_scope:
            for doc in documents:
                if not doc.meta:
                    doc.meta = {}
                doc.meta["tenant_id"] = effective_scope

        has_sparse = any(get_doc_sparse_embedding(d) is not None for d in documents)

        points = QdrantDocumentConverter.prepare_haystack_documents_for_upsert(
            documents,
            dense_vector_name=self.dense_vector_name if has_sparse else None,
            sparse_vector_name=self.sparse_vector_name if has_sparse else None,
        )

        # Upsert in batches
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=False,  # Async for speed
            )

        logger.info(f"Indexed {len(points)} documents for scope '{effective_scope}'")

    def search(
        self,
        query_vector: Union[List[float], Dict[str, Any]],
        tenant_id: Optional[str] = None,
        scope: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        search_type: str = "dense",  # dense, sparse, hybrid, mmr
        mmr_diversity: float = 0.5,
        include_vectors: bool = False,
    ) -> List[Any]:
        """Unified search method supporting all retrieval strategies.

        Executes vector search with optional tenant isolation and metadata
        filtering. Supports multiple search strategies including dense semantic
        search, hybrid search combining dense and sparse vectors, and MMR for
        diverse results.

        Args:
            query_vector: Dense vector (list of floats) for standard search,
                or Dict with vector components for hybrid/sparse search.
                For hybrid: ``{"dense": [...], "sparse": [...]}`` or use
                configured vector names.
            tenant_id: Tenant isolation filter (legacy parameter).
                Use ``scope`` for new code.
            scope: Scope/tenant ID for multi-tenant filtering. Only documents
                with matching ``tenant_id`` in metadata are returned.
            top_k: Maximum number of results to return.
            filters: Metadata filters in MongoDB-style syntax. Supports
                operators: ``$eq``, ``$ne``, ``$gt``, ``$gte``, ``$lt``,
                ``$lte``, ``$in``, ``$nin``.
            search_type: Retrieval strategy to use. Options:

                - ``"dense"`` (default): Standard dense vector similarity search.
                - ``"sparse"``: Sparse vector lexical search using inverted index.
                - ``"hybrid"``: Combines dense and sparse via Reciprocal Rank
                  Fusion (RRF) for best of both semantic and lexical matching.
                - ``"mmr"``: Maximal Marginal Relevance re-ranking for diverse
                  results that balance relevance with novelty.

            mmr_diversity: Diversity factor for MMR search (0.0 to 1.0).
                Higher values favor more diverse results. Only used when
                ``search_type="mmr"``.
            include_vectors: If True, include embedding vectors in returned
                Document objects. Increases response size.

        Returns:
            List of Haystack Document objects sorted by relevance score
            (highest first). Each document includes metadata and score.

        Raises:
            ValueError: If ``search_type`` is invalid, or if hybrid search
                is requested without a dictionary query_vector containing
                both dense and sparse vectors.
            RuntimeError: If search fails due to connection issues, collection
                not existing, or invalid vector dimensions.
            TypeError: If query_vector format doesn't match search_type
                requirements.

        Example:
            Dense semantic search::

                results = db.search(query_vector=embedding, top_k=5)

            Hybrid search with tenant isolation::

                results = db.search(
                    query_vector={"dense": dense_vec, "sparse": sparse_vec},
                    scope="tenant_1",
                    search_type="hybrid",
                    top_k=10,
                )

            Search with metadata filters::

                results = db.search(
                    query_vector=embedding,
                    filters={"category": "tech", "score": {"$gt": 0.8}},
                    top_k=5,
                )
        """
        # Consolidate scope/tenant_id
        effective_scope = scope or tenant_id

        # 1. Build Filter
        final_filter = None
        if effective_scope:
            tenant_filter = self._get_tenant_filter(effective_scope)
            final_filter = Filter(must=[tenant_filter])

        if filters:
            user_filter = self._build_filter(filters)
            if final_filter:
                final_filter.must.append(user_filter)
            else:
                final_filter = user_filter

        results: List[ScoredPoint] = []

        # 2. Execute Search based on Type
        if search_type == "mmr":
            # Maximal Marginal Relevance: fetch extra candidates, then re-rank
            # for diversity using λ*relevance - (1-λ)*redundancy.
            if isinstance(query_vector, dict):
                query_vector = query_vector.get(self.dense_vector_name, query_vector)

            # Retrieve more candidates than needed so MMR has room to diversify
            candidate_multiplier = 4
            candidate_limit = max(top_k * candidate_multiplier, 20)

            candidates = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=final_filter,
                limit=candidate_limit,
                with_vectors=True,
            )

            if candidates:
                results = self._mmr_rerank(
                    query_vector=query_vector,
                    candidates=candidates,
                    top_k=top_k,
                    lambda_mult=mmr_diversity,
                )
            else:
                results = []

            # Strip vectors from results if caller didn't request them
            if not include_vectors:
                for point in results:
                    point.vector = None

        elif search_type == "hybrid":
            # RRF Fusion
            if not isinstance(query_vector, dict):
                raise ValueError(
                    "Hybrid search requires a dictionary with dense and sparse vectors."
                )

            dense_vec = query_vector.get("dense") or query_vector.get(
                self.dense_vector_name
            )
            sparse_vec = query_vector.get("sparse") or query_vector.get(
                self.sparse_vector_name
            )

            prefetch = [
                Prefetch(
                    query=dense_vec,
                    using=self.dense_vector_name,
                    filter=final_filter,
                    limit=top_k * 2,
                ),
                Prefetch(
                    query=sparse_vec,
                    using=self.sparse_vector_name,
                    filter=final_filter,
                    limit=top_k * 2,
                ),
            ]

            query_response = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=prefetch,
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                with_vectors=include_vectors,
            )
            results = query_response.points

        else:
            # Standard Dense Search (Default)
            vector_name = None
            if isinstance(query_vector, dict):
                # It's a named vector search
                name, vec = next(iter(query_vector.items()))
                vector_name = name
                query_vector = vec

            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                using=vector_name,
                query_filter=final_filter,
                limit=top_k,
                with_vectors=include_vectors,
            )

        # 3. Convert Results
        return QdrantDocumentConverter.convert_query_results_to_haystack_documents(
            results,
            include_embeddings=include_vectors,
        )

    @staticmethod
    def _mmr_rerank(
        query_vector: List[float],
        candidates: List[ScoredPoint],
        top_k: int,
        lambda_mult: float = 0.5,
    ) -> List[ScoredPoint]:
        """Re-rank candidates using Maximal Marginal Relevance.

        Greedily selects documents that balance relevance to the query with
        diversity among already-selected documents.

        Score = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected))

        Args:
            query_vector: Query embedding vector.
            candidates: ScoredPoint results with vectors from initial retrieval.
            top_k: Number of documents to select.
            lambda_mult: Trade-off between relevance (1.0) and diversity (0.0).

        Returns:
            Re-ranked list of ScoredPoint objects.
        """
        query_vec = np.array(query_vector, dtype=np.float64)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return candidates[:top_k]
        query_vec = query_vec / query_norm

        # Extract dense vectors from candidates
        doc_vectors: List[np.ndarray] = []
        for c in candidates:
            vec = c.vector
            if isinstance(vec, dict):
                # Named vectors — pick the first list-type vector (dense)
                for v in vec.values():
                    if isinstance(v, list):
                        vec = v
                        break
            if isinstance(vec, list):
                doc_vectors.append(np.array(vec, dtype=np.float64))
            else:
                # Fallback: zero vector so this candidate scores poorly
                doc_vectors.append(np.zeros_like(query_vec))

        # Normalise document vectors for cosine similarity via dot product
        doc_matrix = np.array(doc_vectors)
        norms = np.linalg.norm(doc_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        doc_matrix = doc_matrix / norms

        # Query-document relevance (cosine similarity)
        query_sims = doc_matrix @ query_vec  # shape (n_candidates,)

        selected_indices: List[int] = []
        remaining = set(range(len(candidates)))

        for _ in range(min(top_k, len(candidates))):
            best_idx = -1
            best_score = -np.inf

            for idx in remaining:
                relevance = float(query_sims[idx])

                # Max similarity to any already-selected document
                if selected_indices:
                    selected_vecs = doc_matrix[selected_indices]
                    redundancy = float(np.max(selected_vecs @ doc_matrix[idx]))
                else:
                    redundancy = 0.0

                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * redundancy
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx == -1:
                break
            selected_indices.append(best_idx)
            remaining.discard(best_idx)

        return [candidates[i] for i in selected_indices]

    def delete_documents(
        self,
        tenant_id: Optional[str] = None,
        scope: Optional[str] = None,
        filters: Optional[Dict] = None,
    ):
        """Delete documents for a specific tenant/scope.

        Removes documents matching the specified tenant/scope and optional
        metadata filters. Safety mechanism prevents deletion without any
        filters to avoid accidental data loss.

        Args:
            tenant_id: Tenant ID for deletion (legacy parameter).
                Use ``scope`` for new code.
            scope: Scope/tenant ID to delete documents for. Only documents
                with matching ``tenant_id`` metadata are removed.
            filters: Additional metadata filters to narrow deletion scope.
                Combined with tenant filter using AND logic.

        Returns:
            None

        Raises:
            RuntimeError: If deletion fails due to connection issues,
                collection not existing, or invalid filter syntax.

        Warning:
            Deletion is permanent and cannot be undone. Always verify
            filters before calling, especially in production.

        Example:
            Delete all documents for a tenant::

                db.delete_documents(scope="tenant_1")

            Delete specific documents by filter::

                db.delete_documents(scope="tenant_1", filters={"status": "archived"})
        """
        effective_scope = scope or tenant_id

        if not effective_scope and not filters:
            logger.warning(
                "No scope or filters provided for deletion. Aborting to prevent accidental data loss."
            )
            return

        final_filter = None
        if effective_scope:
            final_filter = self._get_tenant_filter(effective_scope)

        if filters:
            user_filter = self._build_filter(filters)
            if final_filter:
                final_filter = Filter(must=[final_filter.must[0], user_filter])
            else:
                final_filter = user_filter

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=final_filter),
        )
        logger.info(f"Deleted documents for scope '{effective_scope}'")

    def _get_tenant_filter(self, tenant_id: str) -> Filter:
        """Create the mandatory tenant isolation filter.

        Constructs a Qdrant Filter object that matches documents where
        the ``tenant_id`` metadata field equals the specified tenant ID.
        Used internally for multi-tenant data isolation.

        Args:
            tenant_id: The tenant identifier to filter by. Must match
                exactly with the ``tenant_id`` field stored in document
                metadata during indexing.

        Returns:
            Qdrant Filter object configured to match documents belonging
            to the specified tenant.

        Example:
            Internal usage in search::

                tenant_filter = self._get_tenant_filter("tenant_123")
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=vector,
                    query_filter=tenant_filter,
                )
        """
        return Filter(
            must=[
                FieldCondition(
                    key="tenant_id",
                    match=MatchValue(value=tenant_id),
                )
            ]
        )

    def _build_filter(self, filters: Dict[str, Any]) -> Filter:
        """Convert dictionary filters to Qdrant Filter objects.

        Transforms MongoDB-style filter dictionaries into Qdrant Filter
        objects for metadata-based document filtering. Supports a wide
        range of comparison operators for flexible querying.

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
                - ``$in``: Value is in provided list
                - ``$nin``: Value is not in provided list

        Returns:
            Qdrant Filter object with all conditions combined using AND logic.

        Raises:
            ValueError: If an unsupported operator is provided, or if
                filter values are of incompatible types.
            TypeError: If filters is not a dictionary.

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
                        "tags": {"$in": ["ai", "ml"]},
                    }
                )
        """
        conditions = []
        for key, value in filters.items():
            if isinstance(value, dict):
                # Handle operators
                for op, val in value.items():
                    if op == "$eq":
                        conditions.append(
                            FieldCondition(key=key, match=MatchValue(value=val))
                        )
                    elif op == "$ne":
                        # Must NOT match
                        conditions.append(
                            Filter(
                                must_not=[
                                    FieldCondition(key=key, match=MatchValue(value=val))
                                ]
                            )
                        )
                    elif op in ["$gt", "$gte", "$lt", "$lte"]:
                        range_args = {op[1:]: val}
                        conditions.append(
                            FieldCondition(key=key, range=Range(**range_args))
                        )
                    elif op == "$in":
                        conditions.append(
                            FieldCondition(key=key, match=models.MatchAny(any=val))
                        )
                    elif op == "$nin":
                        conditions.append(
                            Filter(
                                must_not=[
                                    FieldCondition(
                                        key=key, match=models.MatchAny(any=val)
                                    )
                                ]
                            )
                        )
            else:
                # Implicit equality
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        return Filter(must=conditions)

    def build_filter(self, filters: Dict[str, Any]) -> Filter:
        """Convert dictionary filters to Qdrant Filter objects.

        Public wrapper for external use in pipelines. Delegates to the
        internal ``_build_filter`` method but provides a public API for
        users who need to construct Qdrant filters for custom queries.

        Args:
            filters: MongoDB-style filter dictionary. Supports operators:
                ``$eq``, ``$ne``, ``$gt``, ``$gte``, ``$lt``, ``$lte``,
                ``$in``, ``$nin``. See ``_build_filter`` for full details.

        Returns:
            Qdrant Filter object ready for use in search queries.

        Raises:
            ValueError: If an unsupported operator is provided.
            TypeError: If filters is not a dictionary.

        Example:
            Build filter for custom query::

                from haystack_integrations.document_stores.qdrant import (
                    QdrantDocumentStore,
                )

                db = QdrantVectorDB()
                filter_obj = db.build_filter(
                    {"status": "active", "priority": {"$gt": 5}}
                )
                # Use filter_obj with custom Qdrant client operations
        """
        return self._build_filter(filters)

    def query(
        self,
        query_vector: Union[List[float], Dict[str, Any]],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Any]:
        """Query the collection for similar vectors (alias for search).

        This is an alias for the search method, provided for API consistency
        with other vector database wrappers that use 'query' as the primary
        search method name.

        Args:
            query_vector: Dense vector for standard search, or Dict with vector
                components for hybrid/sparse search.
            top_k: Maximum number of results to return. Default 10.
            filters: Metadata filters in MongoDB-style syntax.
            **kwargs: Additional arguments passed to search().

        Returns:
            List of Haystack Document objects ordered by relevance score.
        """
        return self.search(
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
            **kwargs,
        )
