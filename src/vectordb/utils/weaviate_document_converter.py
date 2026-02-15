"""Weaviate Document Converter.

This module provides utilities for preparing and converting documents between
different formats for use in vector store operations with Weaviate. It includes
methods for handling Haystack and LangChain document formats and for converting
query results into Haystack documents.

Classes:
    - WeaviateDocumentConverter: Contains static methods for preparing and
      transforming documents.

Methods:
    - prepare_haystack_documents_for_upsert: Prepares Haystack Document objects
      for vector store upsertion.
    - prepare_langchain_documents_for_upsert: Placeholder for preparing
      LangChain Document objects for upsertion.
    - convert_query_results_to_haystack_documents: Converts query results into
      Haystack Document objects.

Logging:
    - Uses a centralized logging utility to provide detailed logs for all
      operations.

Exceptions:
    - Raises ValueError for invalid document inputs.
"""

import logging
from typing import Any

from haystack import Document as HaystackDocument
from langchain_core.documents import Document as LangchainDocument
from weaviate.collections.classes.internal import QueryReturn
from weaviate.util import generate_uuid5

from vectordb.utils.logging import LoggerFactory


logger_factory = LoggerFactory(logger_name=__name__, log_level=logging.INFO)
logger = logger_factory.get_logger()


class WeaviateDocumentConverter:
    """Utility class for preparing and converting documents for vector store ops."""

    @staticmethod
    def prepare_haystack_documents_for_upsert(
        documents: list[HaystackDocument],
    ) -> list[dict[str, Any]]:
        """Convert Haystack documents to Pinecone upsert format.

        This method takes a list of Haystack Document objects and prepares them
        for upsertion into a vector store. It ensures that each document has an
        ID and embeddings, and formats the data accordingly for the vector store.
        The method also handles sparse embeddings if they are present in the
        documents.

        Args:
            documents (List[HaystackDocument]): A list of Haystack Document
                objects, each containing embeddings, sparse embeddings, and
                metadata.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries formatted for vector
                store upsertion. Each dictionary includes:
                - 'id': The document ID.
                - 'vector': The dense embedding of the document.
                - 'text': The document content.
                - Additional metadata attributes.

        Raises:
            ValueError: If any document lacks an ID or embeddings.
        """
        upsert_data: list[dict[str, Any]] = []

        for document in documents:
            if document.embedding is None:
                logger.error(f"Document with missing embedding detected: {document}")
                msg = "Each document must have a dense embedding."
                raise ValueError(msg)

            metadata = document.meta

            # Modify id to prevent conflicts
            if "id" in metadata:
                metadata["document_id"] = metadata.pop("id", None)

            # Handle list of strings in metadata
            for key, value in metadata.items():
                if isinstance(value, list) and isinstance(value[0], str):
                    metadata[key] = "$ $".join(value)

            prepared_doc = {
                "uuid": generate_uuid5(document.id),
                "vector": document.embedding,
                "text": document.content,
                "metadata": metadata,
            }

            upsert_data.append(prepared_doc)
            logger.info(f"Prepared document for upsert: {document.id}")

        logger.info(
            f"Prepared {len(upsert_data)} documents for vector store upsertion."
        )
        return upsert_data

    @staticmethod
    def prepare_langchain_documents_for_upsert(
        documents: list[LangchainDocument],
        embeddings: list[list[float]],
    ) -> list[dict[str, Any]]:
        """Convert LangChain documents to Pinecone upsert format.

        This method takes a list of LangChain Document objects and prepares them
        for upsertion into a vector store.

        Args:
            documents (List[LangChainDocument]): A list of LangChain Document
                objects.
            embeddings (List[List[float]]): A list of embeddings for each
                document.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries formatted for vector
                store upsertion. Each dictionary includes:
                - 'id': The document ID.
                - 'vector': The dense embedding of the document.
                - 'text': The document content.
                - Additional metadata attributes.

        Raises:
            ValueError: If any document lacks an ID or embeddings.
        """
        upsert_data: list[dict[str, Any]] = []

        for index, document in enumerate(documents):
            metadata = document.metadata

            # Modify id to prevent conflicts
            if "id" in metadata:
                metadata["document_id"] = metadata.pop("id", None)

            # Handle list of strings in metadata
            for key, value in metadata.items():
                if isinstance(value, list) and isinstance(value[0], str):
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
    ) -> list[HaystackDocument]:
        """Convert query results into a list of Haystack Document objects.

        This method extracts relevant information from the query results and
        constructs Haystack Document objects.

        Args:
            query_results (QueryReturn): Query results containing match
                information.

        Returns:
            List[HaystackDocument]: A list of Haystack Document objects.
        """
        haystack_docs: list[HaystackDocument] = []

        results = query_results.objects
        if not results:
            logger.warning("Query results contain no matches.")
            return haystack_docs

        for result in results:
            metadata = result.properties["metadata"]
            text = result.properties["text"]

            document = HaystackDocument(
                content=text,
                meta=metadata,
                score=result.metadata.score,
            )

            haystack_docs.append(document)
            logger.info(f"Converted query result to HaystackDocument: {document.id}")

        logger.info(
            f"Converted {len(haystack_docs)} query results into HaystackDocument objects."
        )
        return haystack_docs

    @staticmethod
    def convert_query_results_to_langchain_documents(
        query_results: QueryReturn,
    ) -> list[LangchainDocument]:
        """Convert query results into a list of LangChain Document objects.

        This method extracts relevant information from the query results and
        constructs LangChain Document objects.

        Args:
            query_results (QueryReturn): Query results containing match
                information.

        Returns:
            List[LangChainDocument]: A list of LangChain Document objects.
        """
        langchain_docs: list[LangchainDocument] = []

        results = query_results.objects
        if not results:
            logger.warning("Query results contain no matches.")
            return langchain_docs

        for result in results:
            metadata = result.properties["metadata"]
            text = result.properties["text"]

            document = LangchainDocument(
                page_content=text,
                metadata=metadata,
            )

            langchain_docs.append(document)

        logger.info(
            f"Converted {len(langchain_docs)} query results into HaystackDocument objects."
        )
        return langchain_docs
