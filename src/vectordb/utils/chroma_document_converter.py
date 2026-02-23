"""Chroma document converter for Haystack and LangChain integration.

This module provides bidirectional conversion between Haystack/LangChain document
formats and Chroma's native storage format. It handles the transformation of
embeddings, metadata, and content for both indexing and retrieval operations.

Key Responsibilities:
    - Convert Haystack Documents to Chroma's batch format (texts, embeddings,
      metadatas, ids)
    - Convert LangChain Documents to Chroma's batch format
    - Transform Chroma query results back to Haystack/LangChain Documents
    - Handle score normalization (distance to similarity conversion)
    - Serialize list metadata fields using delimiter encoding

Chroma-Specific Behavior:
    - Uses batch dictionary format with parallel arrays for efficiency
    - Query results are nested in lists (for batch queries), first result set is used
    - Distance scores are converted to similarity via `1.0 - distance`
    - List-of-string metadata values are joined with '$ $' delimiter

Usage:
    >>> from vectordb.utils import ChromaDocumentConverter
    >>> converter = ChromaDocumentConverter()
    >>> batch = converter.prepare_haystack_documents_for_upsert(documents)
    >>> # batch contains {'texts': [...], 'embeddings': [...],
    ... #                'metadatas': [...], 'ids': [...]}
"""

import logging
from typing import Any, Dict, List

from haystack import Document as HaystackDocument
from langchain_core.documents import Document as LangchainDocument

from vectordb.utils.ids import get_doc_id, set_doc_id
from vectordb.utils.logging import LoggerFactory


logger_factory = LoggerFactory(logger_name=__name__, log_level=logging.INFO)
logger = logger_factory.get_logger()


class ChromaDocumentConverter:
    """Bidirectional converter between Haystack/LangChain and Chroma formats.

    This converter handles the format transformation required for storing documents
    in Chroma and retrieving them back. Chroma uses a batch-oriented API with
    parallel arrays, so this converter produces and consumes dictionaries with
    'texts', 'embeddings', 'metadatas', and 'ids' keys.

    Key Transformations:
        - Haystack Document.id -> Chroma 'ids' array element
        - Haystack Document.embedding -> Chroma 'embeddings' array element
        - Haystack Document.content -> Chroma 'texts' array element
        - Haystack Document.meta -> Chroma 'metadatas' array element
        - Chroma distance -> Similarity score (1.0 - distance)

    Chroma-Specific Notes:
        - Uses batch dictionary format with parallel arrays for efficiency
        - Query results are nested in lists (for batch queries), first result set used
        - Distance scores are converted to similarity via `1.0 - distance`
        - List-of-string metadata values are joined with '$ $' delimiter
        - All arrays (texts, embeddings, metadatas, ids) must have equal length

    The converter is stateless and all methods are static, allowing direct class
    method calls without instantiation.
    """

    @staticmethod
    def prepare_haystack_documents_for_upsert(
        documents: List[HaystackDocument],
    ) -> Dict[str, Any]:
        """Convert Haystack documents to Chroma upsert format.

        Args:
            documents (List[HaystackDocument]): A list of Haystack Document
                objects, each containing embeddings, sparse embeddings,
                and metadata.

        Returns:
            Dict[str, Any]: A dictionary formatted for vector store upsertion.
                Includes 'texts', 'embeddings', 'metadatas', and 'ids'.

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

            texts.append(doc.content or "")
            embeddings.append(doc.embedding)

            # Use utility to get ID consistently
            doc_id = get_doc_id(doc)
            ids.append(doc_id)

            # Combine strings from metadata if they are lists
            doc_meta = (doc.meta or {}).copy()
            for key, value in doc_meta.items():
                if isinstance(value, list) and value and isinstance(value[0], str):
                    doc_meta[key] = "$ $".join(value)

            metadatas.append(doc_meta)

        logger.info(f"Prepared {len(documents)} documents for upsertion.")
        return {
            "texts": texts,
            "embeddings": embeddings,
            "metadatas": metadatas,
            "ids": ids,
        }

    @staticmethod
    def prepare_langchain_documents_for_upsert(
        documents: List[LangchainDocument], embeddings: List[List[float]]
    ) -> Dict[str, Any]:
        """Convert LangChain documents to Chroma batch format.

        Args:
            documents: LangChain Documents with page_content and metadata.
            embeddings: Embedding vectors corresponding to each document.

        Returns:
            Dictionary with 'texts', 'embeddings', 'metadatas', and 'ids' keys.

        Raises:
            ValueError: If the document list is empty.

        Note:
            Uses index-based IDs since LangChain Documents do not have stable
            IDs by default.
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
            document_metadata = document.metadata.copy()
            for key, value in document_metadata.items():
                if isinstance(value, list) and value and isinstance(value[0], str):
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
        query_results: Dict[str, Any],
        include_embeddings: bool = False,
    ) -> List[HaystackDocument]:
        """Convert Chroma query results to Haystack Documents.

        Chroma returns results as nested lists to support batch queries. This method
        extracts the first result set and converts each match to a Haystack Document.

        Args:
            query_results: Chroma query result dictionary with 'ids', 'documents',
                'metadatas', 'distances', and optionally 'embeddings' keys.
            include_embeddings: Whether to include vector embeddings in output docs.

        Returns:
            List of Haystack Documents with content, metadata, score, and ID set.

        Note:
            Distance to similarity conversion: Chroma returns L2 distance where
            lower is better. We convert to similarity via `1.0 - distance`.
        """
        haystack_docs: List[HaystackDocument] = []

        # Chroma returns list of lists (for batch queries). We take first result set.
        ids = query_results.get("ids", [[]])[0]
        texts = query_results.get("documents", [[]])[0]
        metadatas = query_results.get("metadatas", [[]])[0]
        distances = query_results.get("distances", [[]])[0]
        embeddings = (
            query_results.get("embeddings", [[]])[0]
            if query_results.get("embeddings")
            else []
        )

        for i, doc_id in enumerate(ids):
            doc = HaystackDocument(
                content=texts[i] if texts and i < len(texts) else "",
                meta=metadatas[i] if metadatas and i < len(metadatas) else {},
            )

            set_doc_id(doc, str(doc_id))

            # Normalize distance to similarity
            if distances and i < len(distances):
                doc.score = 1.0 - distances[i]

            # Attach embedding
            if include_embeddings and embeddings and i < len(embeddings):
                doc.embedding = embeddings[i]

            haystack_docs.append(doc)

        logger.info(
            f"Converted {len(haystack_docs)} query results into HaystackDocument objects."
        )
        return haystack_docs

    @staticmethod
    def convert_query_results_to_langchain_documents(
        query_results: Dict[str, Any],
    ) -> List[LangchainDocument]:
        """Convert Chroma query results to LangChain Documents.

        Chroma returns results as nested lists to support batch queries. This method
        extracts the first result set and converts each match to a LangChain Document.

        Args:
            query_results: Chroma query result dictionary with 'ids', 'documents',
                'metadatas', and 'distances' keys. Results are nested in lists to
                support batch queries, e.g., {"ids": [["id1", "id2"]],
                "documents": [["text1", "text2"]], ...}

        Returns:
            List of LangChain Documents with page_content and metadata including
            the document ID. The document ID is placed in metadata['id'] for
            LangChain compatibility. The page_content is extracted from the
            'documents' field in the query results.

        Note:
            Unlike the Haystack version, this method does not include embeddings
            or scores in the output, as LangChain Documents do not have these
            fields by default. The ID is stored in metadata instead of a top-level
            ID field.
        """
        langchain_docs: List[LangchainDocument] = []

        ids = query_results.get("ids", [[]])[0]
        texts = query_results.get("documents", [[]])[0]
        metadatas = query_results.get("metadatas", [[]])[0]

        for i, doc_id in enumerate(ids):
            document = LangchainDocument(
                page_content=texts[i] if texts and i < len(texts) else "",
                metadata={
                    **(metadatas[i] if metadatas and i < len(metadatas) else {}),
                    "id": doc_id,
                },
            )
            langchain_docs.append(document)

        logger.info(
            f"Converted {len(langchain_docs)} query results into LangChainDocument objects."
        )
        return langchain_docs
