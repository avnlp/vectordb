"""Milvus agentic RAG pipeline for Haystack.

Implements agentic RAG with routing, self-reflection, and multi-hop for Milvus.

Agentic Capabilities:
- Query routing: Automatically selects optimal tool based on query type
- Self-reflection: Evaluates answer quality and iteratively refines
- Multi-hop retrieval: Handles complex queries requiring multiple facts

Milvus Integration:
Uses Milvus's high-performance vector database within the agentic workflow.
Supports:
- Billion-scale vector storage and retrieval
- GPU acceleration for large-scale search
- Advanced indexing (IVF, HNSW) for fast queries
- Multi-tenancy with collection isolation

The agentic workflow:
1. LLM router analyzes query intent
2. Routes to retrieval tool (this class) for vector search
3. Milvus performs similarity search using optimized indexes
4. LLM generates answer from retrieved documents
5. Self-reflection loop validates and refines output

Milvus is ideal for:
- Production deployments at scale
- Large document corpora (millions+ vectors)
- High-throughput query workloads
"""

import json

from haystack import Document
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

from vectordb.haystack.agentic_rag.base import BaseAgenticRAGPipeline


class MilvusAgenticRAGPipeline(BaseAgenticRAGPipeline):
    """Milvus implementation of agentic RAG pipeline.

    Integrates Milvus vector database into the agentic RAG workflow.

    Role in Agentic System:
        Provides high-performance document retrieval for agentic processing.
        Optimized for large-scale deployments with millions of documents.

    Agentic Workflow Integration:
        1. _retrieve(): Invoked when router selects "retrieval" tool
        2. Embeds query using dense_embedder
        3. Searches Milvus collection with approximate nearest neighbors
        4. Returns Document objects with metadata and similarity scores
        5. Supports multiple retrieval cycles for multi-hop queries

    Scalability Features:
        - Collection-based isolation for multi-tenant setups
        - Schema-based document structure with typed fields
        - Batch insertion for efficient indexing
        - Metadata persistence as JSON for flexibility

    Self-reflection Support:
        Router can trigger additional retrieval passes with reformulated
        queries to improve answer quality through iterative refinement.

    Attributes:
        client: MilvusClient instance for database operations.
        collection_name: Name of the Milvus collection.
    """

    def _connect(self) -> None:
        """Establish connection to Milvus vector database."""
        milvus_config = self.config.get("milvus", {})
        host = milvus_config.get("host", "localhost")
        port = milvus_config.get("port", 19530)
        uri = milvus_config.get("uri", f"http://{host}:{port}")
        token = milvus_config.get("token", None)

        if token:
            self.client = MilvusClient(uri=uri, token=token)
        else:
            self.client = MilvusClient(uri=uri)

        self.logger.info("Connected to Milvus at %s", uri)

    def _create_index(self) -> None:
        """Create or verify Milvus collection exists."""
        collection_config = self.config.get("collection", {})
        collection_name = collection_config.get("name", "agentic_rag")

        try:
            if self.client.has_collection(collection_name):
                stats = self.client.get_collection_stats(collection_name)
                self.logger.info(
                    "Collection '%s' exists with %s rows",
                    collection_name,
                    stats.get("row_count", "unknown"),
                )
            else:
                self.logger.warning(
                    "Collection '%s' not found. Will create during indexing.",
                    collection_name,
                )
        except Exception as e:
            self.logger.warning(
                "Could not check collection '%s': %s",
                collection_name,
                str(e),
            )

        self.collection_name = collection_name

    def index_documents(self) -> int:
        """Index documents into Milvus collection.

        Returns:
            Number of documents indexed.
        """
        embedded_docs = self.embed_documents()

        if not embedded_docs:
            self.logger.warning("No documents to index")
            return 0

        if not self.client.has_collection(self.collection_name):
            embedding_dim = len(embedded_docs[0].embedding)

            self.logger.info(
                "Creating Milvus collection '%s' with dimension %d",
                self.collection_name,
                embedding_dim,
            )

            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.INT64,
                    is_primary=True,
                    auto_id=True,
                ),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=embedding_dim,
                ),
            ]
            schema = CollectionSchema(
                fields=fields,
                description="Agentic RAG collection",
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
            )

        # Insert documents in batches
        batch_size = self.config.get("indexing", {}).get("batch_size", 100)
        indexed_count = 0

        for i in range(0, len(embedded_docs), batch_size):
            batch = embedded_docs[i : i + batch_size]
            data = [
                [doc.content for doc in batch],
                [json.dumps(doc.meta) for doc in batch],
                [doc.embedding for doc in batch],
            ]
            self.client.insert(
                collection_name=self.collection_name,
                data=data,
            )
            indexed_count += len(batch)
            self.logger.debug("Indexed %d documents", indexed_count)

        self.logger.info(
            "Indexed %d documents into Milvus collection '%s'",
            indexed_count,
            self.collection_name,
        )
        return indexed_count

    def _retrieve(self, query: str, top_k: int) -> list[Document]:
        """Retrieve documents from Milvus for agentic answer generation.

        Executes vector similarity search against Milvus collection.

        Retrieval Process:
            1. Embed query text into dense vector
            2. Search Milvus collection with limit=top_k
            3. Retrieve content and metadata fields for each hit
            4. Parse JSON metadata string back to dictionary
            5. Construct Document objects with similarity scores

        Args:
            query: User query text for semantic search.
            top_k: Maximum documents to retrieve.

        Returns:
            List of Document objects with parsed metadata and scores.
            Empty list if search fails (enables graceful agent recovery).
        """
        try:
            query_embedding = self.dense_embedder.run(text=query)["embedding"]

            search_results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=top_k,
                output_fields=["content", "metadata"],
            )

            documents = []
            for hits in search_results:
                for hit in hits:
                    entity = hit.get("entity", {})
                    content = entity.get("content", "")
                    metadata = entity.get("metadata", {})
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except json.JSONDecodeError:
                            metadata = {}

                    doc = Document(
                        content=str(content),
                        meta=metadata,
                        score=1.0 - hit.get("distance", 1.0),
                    )
                    documents.append(doc)

            self.logger.info("Retrieved %d documents from Milvus", len(documents))
            return documents

        except Exception as e:
            self.logger.error("Error during Milvus retrieval: %s", str(e))
            return []
