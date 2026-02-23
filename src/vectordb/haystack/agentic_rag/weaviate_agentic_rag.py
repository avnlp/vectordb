"""Weaviate agentic RAG pipeline for Haystack.

Implements agentic RAG with routing, self-reflection, and multi-hop for Weaviate.

Agentic Capabilities:
- Query routing: Automatically selects optimal tool based on query type
- Self-reflection: Evaluates answer quality and iteratively refines
- Multi-hop retrieval: Handles complex queries requiring multiple facts

Weaviate Integration:
Uses Weaviate's semantic search capabilities within the agentic workflow.
Supports:
- Dense retrieval with vector similarity
- Hybrid search combining vector and BM25 (future enhancement)
- Graph-based multi-hop relationships for complex queries

The agentic workflow:
1. LLM router analyzes query intent
2. Routes to retrieval tool (this class) for semantic lookup
3. Weaviate retrieves semantically similar documents
4. LLM generates answer from retrieved context
5. Self-reflection loop validates and refines output
"""

from haystack import Document

from vectordb.haystack.agentic_rag.base import BaseAgenticRAGPipeline


class WeaviateAgenticRAGPipeline(BaseAgenticRAGPipeline):
    """Weaviate implementation of agentic RAG pipeline.

    Integrates Weaviate vector database into the agentic RAG workflow.

    Role in Agentic System:
        Provides semantic document retrieval for the agentic router.
        When queries require factual lookup, the router delegates to this
        class for vector-based document retrieval from Weaviate.

    Agentic Workflow Integration:
        1. _retrieve(): Called by agentic router when tool=retrieval
        2. Embeds query using dense_embedder
        3. Queries Weaviate collection using near_vector search
        4. Returns Document objects ranked by semantic similarity
        5. Supports multi-hop by enabling iterative retrieval calls

    Self-reflection Support:
        If answer quality is below threshold, router can call _retrieve()
        with reformulated queries to gather additional context for refinement.

    Attributes:
        client: Weaviate client instance.
        collection_name: Name of the Weaviate collection.
    """

    def _connect(self) -> None:
        """Establish connection to Weaviate vector database."""
        from haystack.lazy_imports import LazyImport

        with LazyImport("weaviate") as weaviate_import:
            import weaviate

        weaviate_import.check()

        weaviate_config = self.config.get("weaviate", {})
        host = weaviate_config.get("host", "localhost")
        port = weaviate_config.get("port", 8080)
        grpc_port = weaviate_config.get("grpc_port", 50051)
        api_key = weaviate_config.get("api_key", None)

        if api_key:
            auth_config = weaviate.auth.AuthApiKey(api_key=api_key)
            self.client = weaviate.connect_to_custom(
                http_host=host,
                http_port=port,
                http_secure=False,
                grpc_host=host,
                grpc_port=grpc_port,
                grpc_secure=False,
                auth_credentials=auth_config,
            )
        else:
            self.client = weaviate.connect_to_local(
                host=host, port=port, grpc_port=grpc_port
            )

        self.logger.info("Connected to Weaviate at %s:%s", host, port)

    def _create_index(self) -> None:
        """Create or verify Weaviate collection exists."""
        collection_config = self.config.get("collection", {})
        collection_name = collection_config.get("name", "AgenticRAG")

        try:
            if self.client.collections.exists(collection_name):
                self.logger.info("Collection '%s' already exists", collection_name)
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
        """Index documents into Weaviate collection.

        Returns:
            Number of documents indexed.
        """
        embedded_docs = self.embed_documents()

        if not embedded_docs:
            self.logger.warning("No documents to index")
            return 0

        if not self.client.collections.exists(self.collection_name):
            from weaviate.collections.classes.config import Configure

            self.logger.info(
                "Creating Weaviate collection '%s'",
                self.collection_name,
            )

            self.client.collections.create(
                name=self.collection_name,
                vectorizer_config=Configure.Vectorizer.none(),
            )

        collection = self.client.collections.get(self.collection_name)
        batch_size = self.config.get("indexing", {}).get("batch_size", 100)
        indexed_count = 0

        for i in range(0, len(embedded_docs), batch_size):
            batch = embedded_docs[i : i + batch_size]
            objects = [
                {
                    "properties": {
                        "content": doc.content,
                        "metadata": doc.meta,
                    },
                    "vector": doc.embedding,
                }
                for doc in batch
            ]
            collection.data.insert_many(objects)
            indexed_count += len(batch)
            self.logger.debug("Indexed %d documents", indexed_count)

        self.logger.info(
            "Indexed %d documents into Weaviate collection '%s'",
            indexed_count,
            self.collection_name,
        )
        return indexed_count

    def _retrieve(self, query: str, top_k: int) -> list[Document]:
        """Retrieve documents from Weaviate for agentic answer generation.

        Core retrieval method for the agentic workflow. Performs semantic
        vector search in Weaviate and returns documents for LLM context.

        Retrieval Process:
            1. Embed query text into vector representation
            2. Execute near_vector query against Weaviate collection
            3. Convert Weaviate objects to Haystack Document format
            4. Calculate similarity scores from distance metrics
            5. Return ranked documents for agent consumption

        Args:
            query: User query or sub-query from agent decomposition.
            top_k: Maximum number of documents to retrieve.

        Returns:
            List of Document objects with semantic similarity scores.
            Returns empty list on failure for graceful agent handling.
        """
        try:
            query_embedding = self.dense_embedder.run(text=query)["embedding"]

            collection = self.client.collections.get(self.collection_name)
            response = collection.query.near_vector(
                near_vector=query_embedding,
                limit=top_k,
                return_metadata=["distance"],
            )

            documents = []
            for obj in response.objects:
                properties = obj.properties or {}
                content = properties.get("content", "")
                metadata = {k: v for k, v in properties.items() if k != "content"}
                score = 1.0 - (obj.metadata.distance if obj.metadata else 0.0)

                doc = Document(
                    content=str(content),
                    meta=metadata,
                    score=score,
                )
                documents.append(doc)

            self.logger.info("Retrieved %d documents from Weaviate", len(documents))
            return documents

        except Exception as e:
            self.logger.error("Error during Weaviate retrieval: %s", str(e))
            return []
