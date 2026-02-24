"""Milvus search pipeline with query enhancement."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from haystack import Document

from vectordb.databases.milvus import MilvusVectorDB
from vectordb.haystack.components.query_enhancer import QueryEnhancer
from vectordb.haystack.query_enhancement.utils.config import (
    load_config,
    validate_config,
)
from vectordb.haystack.query_enhancement.utils.embeddings import create_text_embedder
from vectordb.haystack.query_enhancement.utils.fusion import (
    deduplicate_by_content,
    rrf_fusion_many,
)
from vectordb.haystack.query_enhancement.utils.llm import create_groq_generator
from vectordb.utils.logging import LoggerFactory


class MilvusQueryEnhancementSearchPipeline:
    """Multi-query search pipeline for Milvus.

    Generates query variations using LLM (Multi-Query, HyDE, Step-Back),
    executes parallel searches, fuses results with RRF, optionally generates
    RAG answer.
    """

    def __init__(self, config_path: str | Path) -> None:
        """Initialize pipeline from configuration.

        Args:
            config_path: Path to YAML configuration file.
        """
        self.config = load_config(config_path)
        validate_config(self.config)

        logger_factory = LoggerFactory("milvus_query_enhancement_search")
        self.logger = logger_factory.get_logger()

        self.embedder = create_text_embedder(self.config)
        self.query_enhancer = self._init_query_enhancer()
        self.db = self._init_db()
        self.rag_generator = self._init_rag_generator()

        self.logger.info("Milvus search pipeline initialized")

    def _init_db(self) -> MilvusVectorDB:
        """Initialize Milvus VectorDB from config."""
        milvus_config = self.config.get("milvus", {})
        return MilvusVectorDB(
            uri=milvus_config.get("uri", "http://localhost:19530"),
            user=milvus_config.get("user"),
            password=milvus_config.get("password"),
            collection_name=milvus_config.get("collection_name"),
            config=self.config,
        )

    def _init_query_enhancer(self) -> QueryEnhancer:
        """Initialize query enhancer from config."""
        qe_config = self.config.get("query_enhancement", {})
        llm_config = qe_config.get("llm", {})
        return QueryEnhancer(
            model=llm_config.get("model", "llama-3.3-70b-versatile"),
            api_key=llm_config.get("api_key"),
        )

    def _init_rag_generator(self) -> Any:
        """Initialize RAG generator if enabled."""
        rag_config = self.config.get("rag", {})
        if not rag_config.get("enabled", False):
            return None
        return create_groq_generator(self.config)

    def _search_single_query(self, query: str, top_k: int) -> list[Document]:
        """Execute search for a single query.

        Args:
            query: Query string.
            top_k: Number of results to return.

        Returns:
            List of Document objects.
        """
        # Embed the query
        embedded = self.embedder.run(text=query)
        query_embedding = embedded["embedding"]

        # Search in Milvus
        return self.db.query(query_embedding, top_k=top_k)

    def run(self, query: str, top_k: int = 10) -> dict[str, Any]:
        """Execute the query enhancement pipeline.

        Args:
            query: Input query string.
            top_k: Number of results to return.

        Returns:
            Dictionary with 'documents' and optional 'answer' keys.
        """
        self.logger.info(f"Processing query: {query}")

        # Determine enhancement type
        enhancement_config = self.config.get("query_enhancement", {})
        enhancement_type = enhancement_config.get("type", "multi_query")

        if enhancement_type == "multi_query":
            num_queries = enhancement_config.get("num_queries", 3)
            enhanced_queries = self.query_enhancer.generate_multi_queries(
                query, num_queries
            )
        elif enhancement_type == "hyde":
            num_docs = enhancement_config.get("num_hyde_docs", 3)
            hyde_docs = self.query_enhancer.generate_hypothetical_documents(
                query, num_docs
            )
            enhanced_queries = hyde_docs + [query]  # Include original query
        elif enhancement_type == "step_back":
            step_back_query = self.query_enhancer.generate_step_back_query(query)
            enhanced_queries = [query, step_back_query]
        else:
            enhanced_queries = [query]  # Default to original query

        self.logger.info(f"Generated {len(enhanced_queries)} queries for search")

        # Execute parallel searches
        all_results = []
        with ThreadPoolExecutor(max_workers=len(enhanced_queries)) as executor:
            futures = {
                executor.submit(self._search_single_query, q, top_k): q
                for q in enhanced_queries
            }

            for future in as_completed(futures):
                try:
                    results = future.result()
                    all_results.append(results)
                    query_text = futures[future]
                    self.logger.debug(
                        f"Search for '{query_text}' returned {len(results)} results"
                    )
                except Exception as e:
                    query_text = futures[future]
                    self.logger.error(f"Search failed for '{query_text}': {e}")

        # Fuse results using N-way RRF
        fused_results = rrf_fusion_many(all_results, top_k=top_k)

        # Deduplicate results
        deduplicated_results = deduplicate_by_content(fused_results)

        # Limit to top_k
        final_results = deduplicated_results[:top_k]

        self.logger.info(f"Returning {len(final_results)} final results")

        if self.rag_generator:
            context = "\n".join([doc.content for doc in final_results])
            rag_prompt = f"Answer the question based on the context:\n\nContext: {context}\n\nQuestion: {query}"

            try:
                rag_response = self.rag_generator.run(
                    [{"role": "user", "content": rag_prompt}]
                )
                answer = (
                    rag_response["replies"][0] if rag_response.get("replies") else ""
                )

                return {"documents": final_results, "answer": answer}
            except Exception as e:
                self.logger.error(f"RAG generation failed: {e}")
                return {"documents": final_results}
        else:
            return {"documents": final_results}
