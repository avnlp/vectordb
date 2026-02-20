"""Qdrant agentic RAG pipeline for Haystack.

Implements agentic RAG with routing, self-reflection, and multi-hop for Qdrant.

Agentic Capabilities:
- Query routing: Automatically selects optimal tool based on query type
- Self-reflection: Evaluates answer quality and iteratively refines
- Multi-hop retrieval: Handles complex queries requiring multiple facts

Qdrant Integration:
Uses Qdrant's production-ready vector database within the agentic workflow.
Supports:
- Advanced filtering with payload-based constraints
- Multiple vector types per point (multi-vector search)
- Distributed deployment for horizontal scaling
- Hybrid search combining dense and sparse vectors

The agentic workflow:
1. LLM router analyzes query intent
2. Routes to retrieval tool (this class) for vector search
3. Qdrant retrieves similar documents with payload metadata
4. LLM generates answer from retrieved documents
5. Self-reflection loop validates and refines output

Qdrant is ideal for:
- Production systems requiring payload filtering
- Applications needing multi-vector representations
- Cloud-native deployments with Kubernetes
"""

from haystack import Document
from qdrant_client import QdrantClient

from vectordb.haystack.agentic_rag.base import BaseAgenticRAGPipeline


class QdrantAgenticRAGPipeline(BaseAgenticRAGPipeline):
    """Qdrant implementation of agentic RAG pipeline.

    Integrates Qdrant vector database into the agentic RAG workflow.

    Role in Agentic System:
        Provides efficient vector retrieval with rich payload support.
        Ideal for queries requiring metadata filtering alongside
        semantic similarity search.

    Agentic Workflow Integration:
        1. _retrieve(): Called when router selects "retrieval" tool
        2. Embeds query text using configured embedder
        3. Searches Qdrant collection with cosine similarity
        4. Returns Document objects with payload metadata
        5. Supports filtered retrieval for precise context selection

    Qdrant Features:
        - PointStruct-based document storage with vectors and payloads
        - Cosine distance metric for semantic similarity
        - Automatic collection creation with inferred dimensions
        - Batch upsert for efficient document indexing

    Self-reflection Support:
        Enables multiple retrieval attempts with different query formulations
        to gather diverse context for answer refinement.

    Attributes:
        client: QdrantClient instance for database operations.
        collection_name: Name of the Qdrant collection.
    """

    def _connect(self) -> None:
        """Establish connection to Qdrant vector database."""
        qdrant_config = self.config.get("qdrant", {})
        host = qdrant_config.get("host", "localhost")
        port = qdrant_config.get("port", 6333)
        api_key = qdrant_config.get("api_key", None)

        self.client = QdrantClient(url=f"http://{host}:{port}", api_key=api_key)
        self.logger.info("Connected to Qdrant at %s:%s", host, port)

    def _create_index(self) -> None:
        """Create or verify Qdrant collection exists."""
        collection_config = self.config.get("collection", {})
        collection_name = collection_config.get("name", "agentic_rag")

        try:
            collection_info = self.client.get_collection(collection_name)
            self.logger.info(
                "Collection '%s' exists with %d vectors",
                collection_name,
                collection_info.points_count,
            )
        except Exception as e:
            self.logger.warning(
                "Collection '%s' not found: %s. Will create during indexing.",
                collection_name,
                str(e),
            )

        self.collection_name = collection_name

    def index_documents(self) -> int:
        """Index documents into Qdrant collection.

        Returns:
            Number of documents indexed.
        """
        embedded_docs = self.embed_documents()

        if not embedded_docs:
            self.logger.warning("No documents to index")
            return 0

        embedding_dim = len(embedded_docs[0].embedding)

        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            from qdrant_client.models import Distance, VectorParams

            self.logger.info(
                "Creating Qdrant collection '%s' with dimension %d",
                self.collection_name,
                embedding_dim,
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

        # Upsert documents in batches
        from qdrant_client.models import PointStruct

        batch_size = 100
        indexed_count = 0

        for i in range(0, len(embedded_docs), batch_size):
            batch = embedded_docs[i : i + batch_size]
            points = [
                PointStruct(
                    id=indexed_count + j,
                    vector=doc.embedding,
                    payload={
                        "content": doc.content,
                        "metadata": doc.meta,
                    },
                )
                for j, doc in enumerate(batch)
            ]
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            indexed_count += len(batch)
            self.logger.debug("Indexed %d documents", indexed_count)

        self.logger.info(
            "Indexed %d documents into Qdrant collection '%s'",
            indexed_count,
            self.collection_name,
        )
        return indexed_count

    def _retrieve(self, query: str, top_k: int) -> list[Document]:
        """Retrieve documents from Qdrant for agentic answer generation.

        Performs vector similarity search in Qdrant collection.

        Retrieval Process:
            1. Embed query into vector using dense_embedder
            2. Execute search against Qdrant collection
            3. Extract content and metadata from result payloads
            4. Preserve Qdrant similarity scores in Document objects
            5. Return ranked list for LLM context window

        Args:
            query: User query or decomposed sub-query from agent.
            top_k: Maximum number of documents to retrieve.

        Returns:
            List of Document objects with payload metadata and scores.
            Empty list on failure for graceful degradation.
        """
        try:
            query_embedding = self.dense_embedder.run(text=query)["embedding"]

            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
            )

            documents = []
            for result in search_results:
                doc = Document(
                    content=result.payload.get("content", ""),
                    meta=result.payload.get("metadata", {}),
                    score=result.score,
                )
                documents.append(doc)

            self.logger.info("Retrieved %d documents from Qdrant", len(documents))
            return documents

        except Exception as e:
            self.logger.error("Error during Qdrant retrieval: %s", str(e))
            return []
