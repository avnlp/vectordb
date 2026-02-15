"""Chroma Document Converter.

This module provides utilities for preparing and converting documents between
different formats for use in vector store operations with Chroma. It includes
methods for handling Haystack and LangChain document formats and for converting
query results into Haystack documents.

Classes:
    - ChromaDocumentConverter: Contains static methods for preparing and
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

from vectordb.utils.logging import LoggerFactory


logger_factory = LoggerFactory(logger_name=__name__, log_level=logging.INFO)
logger = logger_factory.get_logger()


class ChromaDocumentConverter:
    """Utility class for preparing and converting documents for vector store ops."""

    @staticmethod
    def prepare_haystack_documents_for_upsert(
        documents: list[HaystackDocument],
    ) -> dict[str, Any]:
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
            Dict[str, Any]: A dictionary formatted for vector store upsertion.
                The dictionary includes:
                - 'texts': The document content.
                - 'embeddings': The dense embedding of the document.
                - 'metadatas': Metadata, including the document content and
                  additional attributes.
                - 'ids': The document ID.

        Raises:
            ValueError: If any document lacks an ID or embeddings.
        """
        if not documents:
            msg = "The document list is empty. Please provide valid documents."
            logger.error(msg)
            raise ValueError(msg)

        texts = []
        embeddings = []
        metadatas = []
        ids = []

        for doc in documents:
            if doc.embedding is None:
                msg = f"Document with ID '{doc.id}' lacks embeddings."
                logger.warning(msg)
                raise ValueError(msg)

            texts.append(doc.content)
            embeddings.append(doc.embedding)
            ids.append(doc.id)

            # Combine strings from metadata
            doc_meta = doc.meta
            for key, value in doc_meta.items():
                if isinstance(value, list) and isinstance(value[0], str):
                    doc_meta[key] = "$ $".join(value)

            metadatas.append(doc.meta)

        logger.info(f"Prepared {len(documents)} documents for upsertion.")
        return {
            "texts": texts,
            "embeddings": embeddings,
            "metadatas": metadatas,
            "ids": ids,
        }

    @staticmethod
    def prepare_langchain_documents_for_upsert(
        documents: list[LangchainDocument], embeddings: list[list[float]]
    ) -> list[dict[str, Any]]:
        """Prepare LangChain Document objects for upserting into a vector store.

        Args:
            documents (List[LangChainDocument]): A list of LangChain Document
                objects.
            embeddings (List[List[float]]): A list of embeddings corresponding
                to the documents.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries formatted for vector
                store upsertion.
        """
        if not documents:
            msg = "The document list is empty. Please provide valid documents."
            logger.error(msg)
            raise ValueError(msg)

        texts = []
        metadatas = []
        ids = []

        for index, document in enumerate(documents):
            texts.append(document.page_content)
            ids.append(str(index))

            # Combine strings from metadata
            document_metadata = document.metadata
            for key, value in document_metadata.items():
                if isinstance(value, list) and isinstance(value[0], str):
                    document_metadata[key] = "$ $".join(value)

            metadatas.append(document_metadata)

        logger.info(f"Prepared {len(documents)} documents for upsertion.")
        return {
            "texts": texts,
            "embeddings": embeddings,
            "metadatas": metadatas,
            "ids": ids,
        }

    @staticmethod
    def convert_query_results_to_haystack_documents(
        query_results: dict[str, Any],
    ) -> list[HaystackDocument]:
        """Convert query results into a list of Haystack Document objects.

        This method extracts relevant information from the query results and
        constructs Haystack Document objects.

        Args:
            query_results (Dict[str, Any]): Query results containing match
                information.

        Returns:
            List[HaystackDocument]: A list of Haystack Document objects.
        """
        haystack_docs: list[HaystackDocument] = []

        texts = query_results["documents"]
        metadatas = query_results["metadatas"]

        for text, metadata in zip(texts, metadatas):
            document = HaystackDocument(
                content=text[0],
                meta=metadata[0],
            )

            haystack_docs.append(document)

        logger.info(
            f"Converted {len(haystack_docs)} query results into HaystackDocument objects."
        )
        return haystack_docs

    @staticmethod
    def convert_query_results_to_langchain_documents(
        query_results: dict[str, Any],
    ) -> list[LangchainDocument]:
        """Convert query results into a list of LangChain Document objects.

        This method extracts relevant information from the query results and
        constructs LangChain Document objects.

        Args:
            query_results (Dict[str, Any]): Query results containing match
                information.

        Returns:
            List[LangChainDocument]: A list of LangChain Document objects.
        """
        langchain_docs: list[LangchainDocument] = []

        texts = query_results["documents"]
        metadatas = query_results["metadatas"]

        for text, metadata in zip(texts, metadatas):
            document = LangchainDocument(
                page_content=text[0],
                metadata=metadata[0],
            )

            langchain_docs.append(document)

        logger.info(
            f"Converted {len(langchain_docs)} query results into LangChainDocument objects."
        )
        return langchain_docs
