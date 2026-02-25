"""Tests for BaseQueryEnhancementSearchPipeline (LangChain).

Covers shared logic that lives in the base class and is exercised through
the Chroma concrete implementation:
    - RAG answer generation when LLM is configured
    - 'enhanced_queries' key present in all results
    - RRF fusion across multiple enhanced queries
    - top_k truncation applied after fusion
    - Default ChatGroq fallback when RAGHelper returns None
"""

from unittest.mock import MagicMock, call, patch

from langchain_core.documents import Document

from vectordb.langchain.query_enhancement.search.chroma import (
    ChromaQueryEnhancementSearchPipeline,
)


BASE = "vectordb.langchain.query_enhancement.search.base"
CHROMA_DB = "vectordb.langchain.query_enhancement.search.chroma.ChromaVectorDB"


def _make_doc(content: str) -> Document:
    return Document(page_content=content, metadata={})


def _base_config() -> dict:
    return {
        "dataloader": {"type": "arc", "limit": 10},
        "embeddings": {"model": "test-model", "device": "cpu"},
        "chroma": {
            "path": "./test_chroma_data",
            "collection_name": "test_collection",
        },
        "rag": {"enabled": False},
    }


class TestBaseQueryEnhancementSearchRAG:
    """Tests for RAG answer generation path in the base class."""

    @patch("langchain_groq.ChatGroq")
    @patch(CHROMA_DB)
    @patch(f"{BASE}.EmbedderHelper.create_embedder")
    @patch(f"{BASE}.EmbedderHelper.embed_query")
    @patch(f"{BASE}.RAGHelper.create_llm")
    @patch(f"{BASE}.RAGHelper.generate")
    @patch(f"{BASE}.QueryEnhancer")
    def test_rag_answer_generated_when_llm_configured(
        self,
        mock_enhancer,
        mock_rag_generate,
        mock_create_llm,
        mock_embed_query,
        mock_create_embedder,
        mock_db,
        mock_groq,
        sample_documents,
    ):
        """RAG answer is included in result when LLM is configured."""
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm
        mock_embed_query.return_value = [0.1] * 384
        mock_db.return_value.query.return_value = sample_documents
        mock_rag_generate.return_value = "Generated answer"
        mock_enhancer.return_value.generate_queries.return_value = ["q1"]

        pipeline = ChromaQueryEnhancementSearchPipeline(_base_config())
        result = pipeline.search("test query", top_k=5)

        assert "answer" in result
        assert result["answer"] == "Generated answer"
        mock_rag_generate.assert_called_once()

    @patch("langchain_groq.ChatGroq")
    @patch(CHROMA_DB)
    @patch(f"{BASE}.EmbedderHelper.create_embedder")
    @patch(f"{BASE}.EmbedderHelper.embed_query")
    @patch(f"{BASE}.RAGHelper.create_llm")
    @patch(f"{BASE}.QueryEnhancer")
    def test_no_answer_when_llm_is_none(
        self,
        mock_enhancer,
        mock_create_llm,
        mock_embed_query,
        mock_create_embedder,
        mock_db,
        mock_groq,
        sample_documents,
    ):
        """'answer' key is absent when no LLM is configured."""
        mock_create_llm.return_value = None
        mock_embed_query.return_value = [0.1] * 384
        mock_db.return_value.query.return_value = sample_documents
        mock_enhancer.return_value.generate_queries.return_value = ["q1"]

        pipeline = ChromaQueryEnhancementSearchPipeline(_base_config())
        result = pipeline.search("test query", top_k=5)

        assert "answer" not in result


class TestBaseQueryEnhancementSearchResultShape:
    """Tests for the shape of the result dict produced by the base class."""

    @patch("langchain_groq.ChatGroq")
    @patch(CHROMA_DB)
    @patch(f"{BASE}.EmbedderHelper.create_embedder")
    @patch(f"{BASE}.EmbedderHelper.embed_query")
    @patch(f"{BASE}.RAGHelper.create_llm")
    @patch(f"{BASE}.QueryEnhancer")
    def test_result_contains_required_keys(
        self,
        mock_enhancer,
        mock_create_llm,
        mock_embed_query,
        mock_create_embedder,
        mock_db,
        mock_groq,
        sample_documents,
    ):
        """Result always contains 'documents', 'query', and 'enhanced_queries'."""
        mock_create_llm.return_value = None
        mock_embed_query.return_value = [0.1] * 384
        mock_db.return_value.query.return_value = sample_documents
        mock_enhancer.return_value.generate_queries.return_value = ["q1", "q2"]

        pipeline = ChromaQueryEnhancementSearchPipeline(_base_config())
        result = pipeline.search("original query", top_k=5)

        assert "documents" in result
        assert "query" in result
        assert "enhanced_queries" in result
        assert result["query"] == "original query"
        assert result["enhanced_queries"] == ["q1", "q2"]

    @patch("langchain_groq.ChatGroq")
    @patch(CHROMA_DB)
    @patch(f"{BASE}.EmbedderHelper.create_embedder")
    @patch(f"{BASE}.EmbedderHelper.embed_query")
    @patch(f"{BASE}.RAGHelper.create_llm")
    @patch(f"{BASE}.QueryEnhancer")
    def test_top_k_truncates_fused_results(
        self,
        mock_enhancer,
        mock_create_llm,
        mock_embed_query,
        mock_create_embedder,
        mock_db,
        mock_groq,
    ):
        """Results are truncated to top_k after RRF fusion."""
        mock_create_llm.return_value = None
        mock_embed_query.return_value = [0.1] * 384
        docs = [_make_doc(f"doc {i}") for i in range(10)]
        mock_db.return_value.query.return_value = docs
        mock_enhancer.return_value.generate_queries.return_value = ["q1"]

        pipeline = ChromaQueryEnhancementSearchPipeline(_base_config())
        result = pipeline.search("query", top_k=3)

        assert len(result["documents"]) <= 3


class TestBaseQueryEnhancementSearchFusion:
    """Tests for RRF fusion across multiple enhanced queries."""

    @patch("langchain_groq.ChatGroq")
    @patch(CHROMA_DB)
    @patch(f"{BASE}.EmbedderHelper.create_embedder")
    @patch(f"{BASE}.EmbedderHelper.embed_query")
    @patch(f"{BASE}.RAGHelper.create_llm")
    @patch(f"{BASE}.QueryEnhancer")
    def test_each_enhanced_query_is_searched(
        self,
        mock_enhancer,
        mock_create_llm,
        mock_embed_query,
        mock_create_embedder,
        mock_db,
        mock_groq,
        sample_documents,
    ):
        """DB query is called once per enhanced query."""
        mock_create_llm.return_value = None
        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst
        mock_enhancer.return_value.generate_queries.return_value = ["q1", "q2", "q3"]

        pipeline = ChromaQueryEnhancementSearchPipeline(_base_config())
        pipeline.search("query", top_k=5)

        assert mock_db_inst.query.call_count == 3

    @patch("langchain_groq.ChatGroq")
    @patch(CHROMA_DB)
    @patch(f"{BASE}.EmbedderHelper.create_embedder")
    @patch(f"{BASE}.EmbedderHelper.embed_query")
    @patch(f"{BASE}.RAGHelper.create_llm")
    @patch(f"{BASE}.QueryEnhancer")
    def test_embed_query_called_per_enhanced_query(
        self,
        mock_enhancer,
        mock_create_llm,
        mock_embed_query,
        mock_create_embedder,
        mock_db,
        mock_groq,
        sample_documents,
    ):
        """EmbedderHelper.embed_query is called once per enhanced query."""
        mock_create_llm.return_value = None
        mock_embed_query.return_value = [0.1] * 384
        mock_db.return_value.query.return_value = sample_documents
        enhanced = ["q1", "q2"]
        mock_enhancer.return_value.generate_queries.return_value = enhanced

        pipeline = ChromaQueryEnhancementSearchPipeline(_base_config())
        pipeline.search("query", top_k=5)

        assert mock_embed_query.call_count == len(enhanced)
        mock_embed_query.assert_has_calls(
            [call(pipeline.embedder, "q1"), call(pipeline.embedder, "q2")]
        )
