"""Pinecone agentic RAG pipeline for Haystack.

Implements agentic RAG with routing, self-reflection, and multi-hop for Pinecone.

Agentic Capabilities:
- Query routing: Automatically selects optimal tool based on query type
- Self-reflection: Evaluates answer quality and iteratively refines
- Multi-hop retrieval: Handles complex queries requiring multiple facts

Pinecone Integration:
Uses Pinecone's high-performance vector similarity search for document
retrieval within the agentic workflow. Supports:
- Dense retrieval with cosine similarity
- Metadata filtering for contextual search
- Batch indexing for large document corpora

The agentic workflow:
1. LLM router analyzes query intent
2. Routes to retrieval tool (this class) for factual lookups
3. Pinecone retrieves relevant context vectors
4. LLM generates answer from retrieved documents
5. Self-reflection loop validates and refines output
"""

import os

from haystack import Document
from pinecone import Pinecone

from vectordb.haystack.agentic_rag.base import BaseAgenticRAGPipeline


class PineconeAgenticRAGPipeline(BaseAgenticRAGPipeline):
    """Pinecone implementation of agentic RAG pipeline.

    Integrates Pinecone vector database into the agentic RAG workflow.

    Role in Agentic System:
        This class provides the retrieval capability for the agentic router.
        When the router selects "retrieval" as the optimal tool for a query,
        this class executes dense similarity search in Pinecone and returns
        relevant documents for answer generation.

    Agentic Workflow Integration:
        1. _retrieve(): Called by agentic router when tool=retrieval
        2. Embeds query using dense_embedder
        3. Queries Pinecone index for top_k similar vectors
        4. Returns Document objects with content, metadata, and similarity scores
        5. Generator LLM uses these documents to construct answers

    Self-reflection Support:
        If initial answer quality is insufficient, the router may call
        _retrieve() again with modified queries to gather additional context
        for answer refinement.

    Attributes:
        client: Pinecone client instance.
        index: Pinecone index for vector operations.
        index_name: Name of the Pinecone index.
    """

    def _connect(self) -> None:
        """Establish connection to Pinecone vector database."""
        pinecone_config = self.config.get("pinecone", {})
        api_key = pinecone_config.get("api_key")

        if not api_key:
            api_key = os.getenv("PINECONE_API_KEY")

        if not api_key:
            raise ValueError("Pinecone API key is required")

        self.client = Pinecone(api_key=api_key)
        self.logger.info("Connected to Pinecone")

    def _create_index(self) -> None:
        """Create or verify Pinecone index exists."""
        collection_config = self.config.get("collection", {})
        index_name = collection_config.get("name", "agentic-rag")

        try:
            existing_indexes = [idx.name for idx in self.client.list_indexes()]
            if index_name in existing_indexes:
                self.index = self.client.Index(index_name)
                stats = self.index.describe_index_stats()
                self.logger.info(
                    "Index '%s' exists with %d vectors",
                    index_name,
                    stats.total_vector_count,
                )
            else:
                self.logger.warning(
                    "Index '%s' not found. Will create during indexing.",
                    index_name,
                )
                self.index = None
        except Exception as e:
            self.logger.warning(
                "Could not check index '%s': %s",
                index_name,
                str(e),
            )
            self.index = None

        self.index_name = index_name

    def index_documents(self) -> int:
        """Index documents into Pinecone index.

        Returns:
            Number of documents indexed.
        """
        embedded_docs = self.embed_documents()

        if not embedded_docs:
            self.logger.warning("No documents to index")
            return 0

        if self.index is None:
            embedding_dim = len(embedded_docs[0].embedding)

            self.logger.info(
                "Creating Pinecone index '%s' with dimension %d",
                self.index_name,
                embedding_dim,
            )

            self.client.create_index(
                name=self.index_name,
                dimension=embedding_dim,
                metric="cosine",
            )

            self.index = self.client.Index(self.index_name)

        # Upsert documents in batches
        batch_size = 100
        indexed_count = 0

        for i in range(0, len(embedded_docs), batch_size):
            batch = embedded_docs[i : i + batch_size]
            vectors = [
                (
                    f"{indexed_count + j}",
                    doc.embedding,
                    {"content": doc.content, "metadata": doc.meta},
                )
                for j, doc in enumerate(batch)
            ]
            self.index.upsert(vectors=vectors)
            indexed_count += len(batch)
            self.logger.debug("Indexed %d documents", indexed_count)

        self.logger.info(
            "Indexed %d documents into Pinecone index '%s'",
            indexed_count,
            self.index_name,
        )
        return indexed_count

    def _retrieve(self, query: str, top_k: int) -> list[Document]:
        """Retrieve documents from Pinecone for agentic answer generation.

        Core retrieval method invoked by the agentic router when the
        "retrieval" tool is selected. Embeds the query and performs
        vector similarity search in Pinecone.

        Retrieval Process:
            1. Embed query using dense_embedder (sentence-transformers)
            2. Query Pinecone index with cosine similarity
            3. Convert matches to Haystack Document objects
            4. Include metadata and similarity scores for agent context

        Args:
            query: Search query text from user or agent decomposition.
            top_k: Maximum documents to retrieve for context building.

        Returns:
            List of Document objects with content, metadata, and scores.
            Empty list if retrieval fails (agent will handle gracefully).
        """
        if self.index is None:
            self.index = self.client.Index(self.index_name)

        try:
            query_embedding = self.dense_embedder.run(text=query)["embedding"]

            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
            )

            documents = []
            for match in results.matches:
                metadata = match.metadata or {}
                content = metadata.pop("content", "")

                doc = Document(
                    content=str(content),
                    meta=metadata,
                    score=match.score,
                )
                documents.append(doc)

            self.logger.info("Retrieved %d documents from Pinecone", len(documents))
            return documents

        except Exception as e:
            self.logger.error("Error during Pinecone retrieval: %s", str(e))
            return []
