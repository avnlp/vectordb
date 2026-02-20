"""Chroma agentic RAG pipeline for Haystack.

Implements agentic RAG with routing, self-reflection, and multi-hop for Chroma.

Agentic Capabilities:
- Query routing: Automatically selects optimal tool based on query type
- Self-reflection: Evaluates answer quality and iteratively refines
- Multi-hop retrieval: Handles complex queries requiring multiple facts

Chroma Integration:
Uses Chroma's lightweight embedding database within the agentic workflow.
Supports:
- In-memory and persistent storage options
- Local or server-based deployment
- Cosine similarity vector search

The agentic workflow:
1. LLM router analyzes query intent
2. Routes to retrieval tool (this class) for vector lookup
3. Chroma retrieves similar documents by embedding similarity
4. LLM generates answer from retrieved context
5. Self-reflection loop validates and refines output

Chroma is ideal for:
- Local development and testing
- Smaller document collections
- Rapid prototyping of agentic RAG systems
"""

from haystack import Document

from vectordb.haystack.agentic_rag.base import BaseAgenticRAGPipeline


class ChromaAgenticRAGPipeline(BaseAgenticRAGPipeline):
    """Chroma implementation of agentic RAG pipeline.

    Integrates Chroma vector database into the agentic RAG workflow.

    Role in Agentic System:
        Provides document retrieval for agentic query processing.
        Offers flexible deployment options (in-memory, persistent, or server).

    Agentic Workflow Integration:
        1. _retrieve(): Executes when router selects "retrieval" tool
        2. Embeds query text using configured embedder
        3. Queries Chroma collection with cosine similarity
        4. Returns documents with metadata and distance-based scores
        5. Supports iterative retrieval for multi-hop queries

    Deployment Options:
        - PersistentClient: Stores data on disk for persistence
        - HttpClient: Connects to external Chroma server
        - Ephemeral Client: In-memory for testing (fallback)

    Self-reflection Support:
        Router can invoke multiple retrieval cycles to gather additional
        context when answer quality needs improvement.

    Attributes:
        client: Chroma client instance.
        collection: Chroma collection for document storage.
        collection_name: Name of the Chroma collection.
    """

    def _connect(self) -> None:
        """Establish connection to Chroma vector database."""
        import chromadb

        chroma_config = self.config.get("chroma", {})
        host = chroma_config.get("host", "localhost")
        port = chroma_config.get("port", 8000)
        persist_directory = chroma_config.get("persist_directory", None)

        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.logger.info(
                "Connected to Chroma with persistence at %s", persist_directory
            )
        else:
            try:
                self.client = chromadb.HttpClient(host=host, port=port)
                self.logger.info("Connected to Chroma at %s:%s", host, port)
            except Exception:
                self.client = chromadb.Client()
                self.logger.info("Connected to ephemeral Chroma client")

    def _create_index(self) -> None:
        """Create or verify Chroma collection exists."""
        collection_config = self.config.get("collection", {})
        collection_name = collection_config.get("name", "agentic_rag")

        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            count = self.collection.count()
            self.logger.info(
                "Collection '%s' ready with %d documents",
                collection_name,
                count,
            )
        except Exception as e:
            self.logger.error(
                "Failed to create/get collection '%s': %s",
                collection_name,
                str(e),
            )
            raise

        self.collection_name = collection_name

    def index_documents(self) -> int:
        """Index documents into Chroma collection.

        Returns:
            Number of documents indexed.
        """
        embedded_docs = self.embed_documents()

        if not embedded_docs:
            self.logger.warning("No documents to index")
            return 0

        # Add documents in batches
        batch_size = 100
        indexed_count = 0

        for i in range(0, len(embedded_docs), batch_size):
            batch = embedded_docs[i : i + batch_size]
            ids = [f"{indexed_count + j}" for j in range(len(batch))]
            embeddings = [doc.embedding for doc in batch]
            documents = [doc.content for doc in batch]
            metadatas = [doc.meta for doc in batch]

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            indexed_count += len(batch)
            self.logger.debug("Indexed %d documents", indexed_count)

        self.logger.info(
            "Indexed %d documents into Chroma collection '%s'",
            indexed_count,
            self.collection_name,
        )
        return indexed_count

    def _retrieve(self, query: str, top_k: int) -> list[Document]:
        """Retrieve documents from Chroma for agentic answer generation.

        Performs vector similarity search in Chroma collection.

        Retrieval Process:
            1. Embed query into dense vector representation
            2. Query Chroma collection using query_embeddings
            3. Extract documents, metadatas, and distances from results
            4. Convert distances to similarity scores (1.0 - distance)
            5. Build Haystack Document objects with full metadata

        Args:
            query: User query or agent-decomposed sub-query.
            top_k: Maximum documents to retrieve for context.

        Returns:
            List of Document objects with content and similarity scores.
            Empty list on error for graceful agent handling.
        """
        try:
            query_embedding = self.dense_embedder.run(text=query)["embedding"]

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

            documents = []
            if results["documents"]:
                docs_list = results["documents"][0]
                metadatas = (
                    results["metadatas"][0]
                    if results["metadatas"]
                    else [{}] * len(docs_list)
                )
                distances = (
                    results["distances"][0]
                    if results["distances"]
                    else [0.0] * len(docs_list)
                )

                for content, metadata, distance in zip(docs_list, metadatas, distances):
                    score = 1.0 - distance
                    doc = Document(
                        content=str(content) if content else "",
                        meta=metadata or {},
                        score=score,
                    )
                    documents.append(doc)

            self.logger.info("Retrieved %d documents from Chroma", len(documents))
            return documents

        except Exception as e:
            self.logger.error("Error during Chroma retrieval: %s", str(e))
            return []
