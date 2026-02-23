"""Qdrant document converter for Haystack and LangChain integration.

This module provides bidirectional conversion between Haystack/LangChain document
formats and Qdrant's PointStruct storage format. It handles dense vectors, sparse
vectors for hybrid search, and metadata payload construction.

Key Responsibilities:
    - Convert Haystack Documents to Qdrant PointStruct objects
    - Convert LangChain Documents to Qdrant PointStruct objects
    - Transform Qdrant ScoredPoint results back to Haystack/LangChain Documents
    - Handle both dense-only and hybrid (dense + sparse) vector configurations
    - Support named vectors for multi-vector Qdrant collections

Qdrant-Specific Behavior:
    - Uses PointStruct for upsert operations with id, vector, and payload
    - Supports named vectors (e.g., {"dense": [...], "sparse": SparseVector})
    - Sparse embeddings use Qdrant's SparseVector model with indices and values
    - Payload includes document content and all metadata fields
    - Score is returned directly from ScoredPoint (no normalization needed)

Usage:
    >>> from vectordb.utils import QdrantDocumentConverter
    >>> converter = QdrantDocumentConverter()
    >>> points = converter.prepare_haystack_documents_for_upsert(
    ...     documents, dense_vector_name="dense", sparse_vector_name="sparse"
    ... )
"""

import logging
from typing import Any, List, Optional

from haystack import Document as HaystackDocument
from haystack.dataclasses import SparseEmbedding
from langchain_core.documents import Document as LangchainDocument
from qdrant_client import models
from qdrant_client.http.models import PointStruct, ScoredPoint

from vectordb.utils.ids import set_doc_id
from vectordb.utils.logging import LoggerFactory
from vectordb.utils.sparse import get_doc_sparse_embedding, normalize_sparse


logger_factory = LoggerFactory(logger_name=__name__, log_level=logging.INFO)
logger = logger_factory.get_logger()


class QdrantDocumentConverter:
    """Bidirectional converter between Haystack/LangChain and Qdrant formats.

    This converter transforms documents to Qdrant's PointStruct format for indexing
    and converts ScoredPoint results back to Haystack/LangChain Documents. It supports
    both dense-only and hybrid (dense + sparse) vector configurations.

    Key Transformations:
        - Haystack Document.id -> Qdrant PointStruct 'id' field
        - Haystack Document.embedding -> Qdrant 'vector' field (or named vector)
        - Haystack Document.sparse_embedding -> Qdrant SparseVector (named)
        - Haystack Document.content -> Qdrant payload['content'] field
        - Haystack Document.meta -> Qdrant payload (merged with content)
        - Qdrant score -> Document.score (direct, no normalization needed)

    Qdrant-Specific Notes:
        - Uses PointStruct for upsert operations with id, vector, and payload
        - Supports named vectors (e.g., {"dense": [...], "sparse": SparseVector})
        - Sparse embeddings use Qdrant's SparseVector model with indices and values
        - Payload includes document content and all metadata fields
        - Score is returned directly from ScoredPoint (no normalization needed)
        - Named vectors are automatically used when sparse embeddings are present

    The converter is stateless and all methods are static, allowing direct class
    method calls without instantiation.
    """

    @staticmethod
    def prepare_haystack_documents_for_upsert(
        documents: List[HaystackDocument],
        dense_vector_name: Optional[str] = None,
        sparse_vector_name: Optional[str] = None,
    ) -> List[PointStruct]:
        """Convert Haystack documents to Qdrant PointStruct objects.

        Args:
            documents (List[HaystackDocument]): A list of Haystack Document
                objects.
            dense_vector_name (Optional[str]): Name of the dense vector field.
                If provided along with sparse, vectors will be named.
            sparse_vector_name (Optional[str]): Name of the sparse vector field.

        Returns:
            List[PointStruct]: A list of PointStruct objects for upsertion.
        """
        points: List[PointStruct] = []

        for doc in documents:
            if doc.embedding is None:
                logger.warning(f"Document {doc.id} is missing embedding.")
                continue

            # Handle Vector (Dense & Sparse)
            vector: Any = doc.embedding

            # Use utility to get sparse embedding (standard or legacy meta)
            sparse_emb = get_doc_sparse_embedding(doc)

            if sparse_emb is not None:
                # Hybrid case: we must use named vectors
                # If names are not provided, default to "dense" and "sparse"
                d_name = dense_vector_name or "dense"
                s_name = sparse_vector_name or "sparse"

                sparse_vec = models.SparseVector(
                    indices=sparse_emb.indices, values=sparse_emb.values
                )

                vector = {d_name: doc.embedding, s_name: sparse_vec}
            elif dense_vector_name:
                # Dense only but with specific name
                vector = {dense_vector_name: doc.embedding}

            # Payload construction
            payload = doc.meta.copy() if doc.meta else {}
            if doc.content:
                payload["content"] = doc.content

            if "doc_id" not in payload and doc.id:
                payload["doc_id"] = str(doc.id)

            points.append(PointStruct(id=str(doc.id), vector=vector, payload=payload))

        logger.info(f"Prepared {len(points)} points for upsertion.")
        return points

    @staticmethod
    def prepare_langchain_documents_for_upsert(
        documents: List[LangchainDocument],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None,
    ) -> List[PointStruct]:
        """Prepare LangChain Document objects for upsertion.

        Args:
            documents (List[LangchainDocument]): LangChain documents.
            embeddings (List[List[float]]): Corresponding embeddings.
            ids (Optional[List[str]]): Optional list of IDs.

        Returns:
            List[PointStruct]: List of Qdrant points.
        """
        points = []
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            doc_id = ids[i] if ids else str(i)
            payload = doc.metadata.copy()
            payload["page_content"] = doc.page_content

            points.append(PointStruct(id=doc_id, vector=emb, payload=payload))
        return points

    @staticmethod
    def convert_query_results_to_haystack_documents(
        scored_points: List[ScoredPoint],
        include_embeddings: bool = False,
    ) -> List[HaystackDocument]:
        """Convert Qdrant search results to Haystack Documents.

        Handles both dense-only and hybrid vector configurations. For hybrid results,
        dense embeddings are extracted into doc.embedding and sparse embeddings into
        doc.sparse_embedding.

        Args:
            scored_points: ScoredPoint objects from Qdrant search operations.
            include_embeddings: Whether to include vector embeddings in output docs.

        Returns:
            List of Haystack Documents with content, metadata, score, and ID set.

        Note:
            For named vectors, the converter attempts to find dense vectors (list type)
            and sparse vectors (SparseVector objects with indices/values attributes)
            within the vector dictionary.
        """
        docs = []
        for point in scored_points:
            payload = point.payload or {}
            content = payload.pop("content", None) or payload.pop("text", "")

            # Extract score
            score = point.score

            doc = HaystackDocument(
                content=content,
                meta=payload,
                score=score,
            )

            # Use point.id as doc.id and set consistently
            set_doc_id(doc, str(point.id))

            if include_embeddings and point.vector:
                # Extract vector if present (it might be named or list)
                embedding = point.vector

                # Handle Dense
                dense_emb = None
                if isinstance(embedding, list):
                    dense_emb = embedding
                elif isinstance(embedding, dict):
                    # Try to find a list value (dense) and SparseVector (sparse)
                    for _k, v in embedding.items():
                        if isinstance(v, list):
                            dense_emb = v
                        elif hasattr(v, "indices") and hasattr(v, "values"):
                            # It's a sparse vector object
                            doc.sparse_embedding = SparseEmbedding(
                                indices=list(v.indices), values=list(v.values)
                            )
                        elif isinstance(v, dict) and "indices" in v:
                            # Dict representation of sparse
                            doc.sparse_embedding = normalize_sparse(v)

                if dense_emb:
                    doc.embedding = dense_emb

            docs.append(doc)
        return docs
