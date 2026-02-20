"""Weaviate document converter for Haystack and LangChain integration.

This module provides bidirectional conversion between Haystack/LangChain document
formats and Weaviate's native storage format. It handles the transformation of
embeddings, metadata, and content for both indexing and retrieval operations.

Key Responsibilities:
    - Convert Haystack Documents to Weaviate upsert format (UUID generation, vector
      packaging, metadata serialization)
    - Convert LangChain Documents to Weaviate upsert format
    - Transform Weaviate QueryReturn results back to Haystack/LangChain Documents
    - Handle score normalization (distance to similarity conversion)
    - Serialize list metadata fields using delimiter encoding

Weaviate-Specific Behavior:
    - Uses deterministic UUID5 generation for document IDs
    - Supports both 'distance' and 'score' metadata for relevance scoring
    - Handles named vectors for multi-vector configurations
    - Converts 'id' metadata key to 'document_id' to avoid conflicts with Weaviate's
      internal uuid field

Usage:
    >>> from vectordb.utils import WeaviateDocumentConverter
    >>> converter = WeaviateDocumentConverter()
    >>> upsert_data = converter.prepare_haystack_documents_for_upsert(documents)
    >>> haystack_docs = converter.convert_query_results_to_haystack_documents(results)
"""

import logging
from typing import Any, Dict, List

from haystack import Document as HaystackDocument
from langchain_core.documents import Document as LangchainDocument
from weaviate.collections.classes.internal import QueryReturn
from weaviate.util import generate_uuid5

from vectordb.utils.ids import get_doc_id, set_doc_id
from vectordb.utils.logging import LoggerFactory


logger_factory = LoggerFactory(logger_name=__name__, log_level=logging.INFO)
logger = logger_factory.get_logger()


class WeaviateDocumentConverter:
    """Bidirectional converter between Haystack/LangChain and Weaviate formats.

    This converter handles the format transformation required for storing documents
    in Weaviate and retrieving them back. Weaviate uses UUID-based identifiers and
    a specific object structure that differs from Haystack's Document format.

    Key Transformations:
        - Haystack Document.id -> Weaviate UUID5 (deterministic generation)
        - Haystack Document.embedding -> Weaviate 'vector' field
        - Haystack Document.content -> Weaviate 'text' property
        - Haystack Document.meta -> Weaviate 'metadata' property
        - Weaviate distance -> Similarity score (1.0 - distance)

    Weaviate-Specific Notes:
        - Uses deterministic UUID5 generation for consistent document IDs
        - Supports both 'distance' and 'score' metadata for relevance scoring
        - Handles named vectors for multi-vector configurations
        - Converts 'id' metadata key to 'document_id' to avoid conflicts with
          Weaviate's internal uuid field
        - List-of-string metadata values are joined with '$ $' delimiter because
          Weaviate requires uniform array types within a property

    The converter is stateless and all methods are static, allowing direct class
    method calls without instantiation.
    """

    @staticmethod
    def prepare_haystack_documents_for_upsert(
        documents: List[HaystackDocument],
    ) -> List[Dict[str, Any]]:
        """Convert Haystack documents to Weaviate upsert format.

        Transforms each document into a dictionary structure suitable for Weaviate's
        batch insert API, including deterministic UUID generation and metadata
        serialization.

        Args:
            documents: Haystack Documents with embeddings and metadata.

        Returns:
            List of dictionaries with 'uuid', 'vector', 'text', and 'metadata' keys.

        Raises:
            ValueError: If any document is missing its embedding vector.

        Note:
            List-of-string metadata values are joined with '$ $' delimiter to
            ensure Weaviate compatibility, as Weaviate requires uniform array types.
        """
        upsert_data: List[Dict[str, Any]] = []

        for document in documents:
            if document.embedding is None:
                logger.error(f"Document with missing embedding detected: {document}")
                msg = "Each document must have a dense embedding."
                raise ValueError(msg)

            # Use utility to get ID or generate one
            doc_id = get_doc_id(document)

            metadata = (document.meta or {}).copy()

            # Modify id to prevent conflicts
            if "id" in metadata:
                metadata["document_id"] = metadata.pop("id", None)

            # Handle list of strings in metadata
            for key, value in metadata.items():
                if isinstance(value, list) and value and isinstance(value[0], str):
                    metadata[key] = "$ $".join(value)

            prepared_doc = {
                "uuid": generate_uuid5(doc_id),
                "vector": document.embedding,
                "text": document.content,
                "metadata": metadata,
            }

            upsert_data.append(prepared_doc)
            logger.info(f"Prepared document for upsert: {doc_id}")

        logger.info(
            f"Prepared {len(upsert_data)} documents for vector store upsertion."
        )
        return upsert_data

    @staticmethod
    def prepare_langchain_documents_for_upsert(
        documents: List[LangchainDocument],
        embeddings: List[List[float]],
    ) -> List[Dict[str, Any]]:
        """Convert LangChain documents to Weaviate upsert format.

        Args:
            documents: LangChain Documents with page_content and metadata.
            embeddings: Embedding vectors corresponding to each document.

        Returns:
            List of dictionaries ready for Weaviate batch insertion.

        Note:
            Uses index-based UUID5 generation since LangChain Documents do not
            have stable IDs by default.
        """
        upsert_data: List[Dict[str, Any]] = []

        for index, document in enumerate(documents):
            metadata = document.metadata.copy()

            # Modify id to prevent conflicts
            if "id" in metadata:
                metadata["document_id"] = metadata.pop("id", None)

            # Handle list of strings in metadata
            for key, value in metadata.items():
                if isinstance(value, list) and value and isinstance(value[0], str):
                    metadata[key] = "$ $".join(value)

            prepared_doc = {
                "uuid": generate_uuid5(index),
                "vector": embeddings[index],
                "text": document.page_content,
                "metadata": metadata,
            }

            upsert_data.append(prepared_doc)

        logger.info(
            f"Prepared {len(upsert_data)} documents for vector store upsertion."
        )
        return upsert_data

    @staticmethod
    def convert_query_results_to_haystack_documents(
        query_results: QueryReturn,
        include_embeddings: bool = False,
    ) -> List[HaystackDocument]:
        """Convert Weaviate query results to Haystack Documents.

        Extracts content, metadata, scores, and optionally embeddings from Weaviate's
        QueryReturn structure. Handles score normalization from distance to similarity.

        Args:
            query_results: Weaviate QueryReturn object from a search operation.
            include_embeddings: Whether to include vector embeddings in output docs.

        Returns:
            List of Haystack Documents with content, metadata, score, and ID set.

        Note:
            Score normalization: Weaviate returns distance (lower is better), which
            is converted to similarity (higher is better) via `1.0 - distance`.
        """
        haystack_docs: List[HaystackDocument] = []

        results = query_results.objects
        if not results:
            logger.warning("Query results contain no matches.")
            return haystack_docs

        for result in results:
            properties = result.properties
            metadata = properties.get("metadata", {})
            text = properties.get("text", properties.get("content", ""))

            document = HaystackDocument(
                content=text,
                meta=metadata,
            )

            set_doc_id(document, str(result.uuid))

            # Extract score
            if result.metadata:
                if result.metadata.distance is not None:
                    document.score = 1.0 - result.metadata.distance
                elif result.metadata.score is not None:
                    document.score = result.metadata.score

            if include_embeddings and result.vector:
                # Handle named vectors or list
                if isinstance(result.vector, dict):
                    if "default" in result.vector:
                        document.embedding = list(result.vector["default"])
                    elif result.vector:
                        document.embedding = list(next(iter(result.vector.values())))
                else:
                    document.embedding = list(result.vector)

            haystack_docs.append(document)
            logger.info(f"Converted query result to HaystackDocument: {document.id}")

        logger.info(
            f"Converted {len(haystack_docs)} query results into HaystackDocument objects."
        )
        return haystack_docs

    @staticmethod
    def convert_query_results_to_langchain_documents(
        query_results: QueryReturn,
    ) -> List[LangchainDocument]:
        """Convert Weaviate query results to LangChain Documents.

        Args:
            query_results: Weaviate QueryReturn object from a search operation.

        Returns:
            List of LangChain Documents with page_content and metadata including
            the document ID.
        """
        langchain_docs: List[LangchainDocument] = []

        results = query_results.objects
        if not results:
            logger.warning("Query results contain no matches.")
            return langchain_docs

        for result in results:
            properties = result.properties
            metadata = properties.get("metadata", {})
            text = properties.get("text", properties.get("content", ""))

            document = LangchainDocument(
                page_content=text,
                metadata={**metadata, "id": str(result.uuid)},
            )

            langchain_docs.append(document)

        logger.info(
            f"Converted {len(langchain_docs)} query results into LangChainDocument objects."
        )
        return langchain_docs
