"""Tests for structured output types.

This module tests the dataclass structures used to represent retrieval
and pipeline execution results. These output types provide a consistent
interface for returning results from vector database operations.

Tested classes:
    RetrievedDocument: Individual document with content, score, and metadata.
    RetrievalOutput: Query results containing retrieved documents.
    PipelineOutput: Complete pipeline execution results with metrics.

Test coverage includes:
    - Minimal and full initialization of each dataclass
    - Serialization via to_dict() methods
    - Summary formatting for PipelineOutput
    - Handling of optional fields and edge cases
"""

from vectordb.utils.output import PipelineOutput, RetrievalOutput, RetrievedDocument


class TestRetrievedDocument:
    """Test suite for RetrievedDocument dataclass.

    Tests cover initialization, default values, and dictionary serialization
    for individual retrieved document representations.
    """

    def test_initialization_minimal(self) -> None:
        """Test RetrievedDocument with minimal fields."""
        doc = RetrievedDocument(content="Test content")
        assert doc.content == "Test content"
        assert doc.doc_id == ""
        assert doc.score == 0.0
        assert doc.metadata == {}
        assert doc.matched_children == []

    def test_initialization_full(self) -> None:
        """Test RetrievedDocument with all fields."""
        metadata = {"source": "wikipedia", "date": "2024-01-01"}
        children = [{"id": "child1", "score": 0.9}]
        doc = RetrievedDocument(
            content="Test content",
            doc_id="doc123",
            score=0.95,
            metadata=metadata,
            matched_children=children,
        )
        assert doc.content == "Test content"
        assert doc.doc_id == "doc123"
        assert doc.score == 0.95
        assert doc.metadata == metadata
        assert doc.matched_children == children

    def test_to_dict_minimal(self) -> None:
        """Test converting minimal document to dict."""
        doc = RetrievedDocument(content="Test content")
        result = doc.to_dict()
        assert result["content"] == "Test content"
        assert result["doc_id"] == ""
        assert result["score"] == 0.0
        assert result["metadata"] == {}
        assert "matched_children" not in result

    def test_to_dict_with_children(self) -> None:
        """Test converting document with matched children to dict."""
        children = [{"id": "child1", "score": 0.9}]
        doc = RetrievedDocument(
            content="Test content",
            doc_id="doc123",
            matched_children=children,
        )
        result = doc.to_dict()
        assert result["matched_children"] == children


class TestRetrievalOutput:
    """Test suite for RetrievalOutput dataclass.

    Tests cover query result representation, document list serialization,
    and retrieval mode handling.
    """

    def test_initialization_minimal(self) -> None:
        """Test RetrievalOutput with minimal fields."""
        output = RetrievalOutput(query="test query")
        assert output.query == "test query"
        assert output.documents == []
        assert output.retrieval_mode == "with_parents"
        assert output.top_k == 5
        assert output.total_retrieved == 0
        assert output.latency_ms == 0.0

    def test_initialization_full(self) -> None:
        """Test RetrievalOutput with all fields."""
        doc1 = RetrievedDocument(content="Doc 1", doc_id="1", score=0.9)
        doc2 = RetrievedDocument(content="Doc 2", doc_id="2", score=0.8)
        output = RetrievalOutput(
            query="test query",
            documents=[doc1, doc2],
            retrieval_mode="children_only",
            top_k=10,
            total_retrieved=2,
            latency_ms=150.5,
        )
        assert output.query == "test query"
        assert len(output.documents) == 2
        assert output.retrieval_mode == "children_only"
        assert output.top_k == 10
        assert output.total_retrieved == 2
        assert output.latency_ms == 150.5

    def test_to_dict_minimal(self) -> None:
        """Test converting minimal retrieval output to dict."""
        output = RetrievalOutput(query="test query")
        result = output.to_dict()
        assert result["query"] == "test query"
        assert result["documents"] == []
        assert result["retrieval_mode"] == "with_parents"
        assert result["top_k"] == 5
        assert result["total_retrieved"] == 0
        assert result["latency_ms"] == 0.0

    def test_to_dict_with_documents(self) -> None:
        """Test converting retrieval output with documents to dict."""
        doc1 = RetrievedDocument(content="Doc 1", doc_id="1", score=0.9)
        doc2 = RetrievedDocument(content="Doc 2", doc_id="2", score=0.8)
        output = RetrievalOutput(
            query="test query",
            documents=[doc1, doc2],
            total_retrieved=2,
        )
        result = output.to_dict()
        assert len(result["documents"]) == 2
        assert result["documents"][0]["content"] == "Doc 1"
        assert result["documents"][0]["score"] == 0.9
        assert result["documents"][1]["content"] == "Doc 2"

    def test_retrieval_modes(self) -> None:
        """Test different retrieval modes."""
        modes = ["with_parents", "children_only", "context_window"]
        for mode in modes:
            output = RetrievalOutput(query="test", retrieval_mode=mode)
            assert output.retrieval_mode == mode
            assert output.to_dict()["retrieval_mode"] == mode

    def test_document_serialization(self) -> None:
        """Test that documents are properly serialized in output."""
        doc = RetrievedDocument(
            content="Test", doc_id="1", score=0.9, metadata={"key": "value"}
        )
        output = RetrievalOutput(query="q", documents=[doc])
        result = output.to_dict()
        doc_dict = result["documents"][0]
        assert doc_dict["content"] == "Test"
        assert doc_dict["doc_id"] == "1"
        assert doc_dict["score"] == 0.9
        assert doc_dict["metadata"] == {"key": "value"}


class TestPipelineOutput:
    """Test suite for PipelineOutput dataclass.

    Tests cover complete pipeline result representation including index stats,
    retrieval results, evaluation metrics, and human-readable summary output.
    """

    def test_initialization_minimal(self) -> None:
        """Test PipelineOutput with minimal required fields."""
        output = PipelineOutput(
            pipeline_name="dense_retrieval",
            database_type="pinecone",
            dataset_name="triviaqa",
        )
        assert output.pipeline_name == "dense_retrieval"
        assert output.database_type == "pinecone"
        assert output.dataset_name == "triviaqa"
        assert output.index_stats == {}
        assert output.retrieval_results == []
        assert output.evaluation_metrics == {}

    def test_initialization_full(self) -> None:
        """Test PipelineOutput with all fields."""
        doc = RetrievedDocument(content="Test doc", doc_id="d1", score=0.85)
        retrieval = RetrievalOutput(
            query="test query",
            documents=[doc],
            retrieval_mode="with_parents",
            total_retrieved=1,
        )
        output = PipelineOutput(
            pipeline_name="hybrid_search",
            database_type="weaviate",
            dataset_name="arc",
            index_stats={"num_documents": 100, "num_parents": 20, "num_children": 80},
            retrieval_results=[retrieval],
            evaluation_metrics={"recall@5": 0.85, "mrr": 0.72},
        )
        assert output.pipeline_name == "hybrid_search"
        assert output.database_type == "weaviate"
        assert output.dataset_name == "arc"
        assert output.index_stats["num_documents"] == 100
        assert len(output.retrieval_results) == 1
        assert output.evaluation_metrics["recall@5"] == 0.85

    def test_to_dict_minimal(self) -> None:
        """Test converting minimal pipeline output to dict."""
        output = PipelineOutput(
            pipeline_name="dense",
            database_type="chroma",
            dataset_name="test",
        )
        result = output.to_dict()
        assert result["pipeline_name"] == "dense"
        assert result["database_type"] == "chroma"
        assert result["dataset_name"] == "test"
        assert result["index_stats"] == {}
        assert result["retrieval_results"] == []
        assert result["evaluation_metrics"] == {}

    def test_to_dict_with_retrieval_results(self) -> None:
        """Test to_dict properly serializes nested retrieval results."""
        doc = RetrievedDocument(content="Content", doc_id="id1", score=0.9)
        retrieval = RetrievalOutput(
            query="q1",
            documents=[doc],
            total_retrieved=1,
        )
        output = PipelineOutput(
            pipeline_name="test",
            database_type="qdrant",
            dataset_name="popqa",
            retrieval_results=[retrieval],
        )
        result = output.to_dict()
        assert len(result["retrieval_results"]) == 1
        assert result["retrieval_results"][0]["query"] == "q1"
        assert result["retrieval_results"][0]["documents"][0]["content"] == "Content"

    def test_summary_empty_metrics(self) -> None:
        """Test summary() with no evaluation metrics."""
        output = PipelineOutput(
            pipeline_name="dense_retrieval",
            database_type="pinecone",
            dataset_name="triviaqa",
            index_stats={"num_documents": 50, "num_parents": 10, "num_children": 40},
        )
        summary = output.summary()
        assert "Pipeline: dense_retrieval" in summary
        assert "Database: pinecone" in summary
        assert "Dataset: triviaqa" in summary
        assert "Indexed: 50 documents" in summary
        assert "Parents: 10" in summary
        assert "Children: 40" in summary
        assert "Queries evaluated: 0" in summary
        assert "Evaluation Metrics:" not in summary

    def test_summary_with_float_metrics(self) -> None:
        """Test summary() formats float metrics with 4 decimal places."""
        output = PipelineOutput(
            pipeline_name="hybrid",
            database_type="milvus",
            dataset_name="arc",
            evaluation_metrics={"recall@5": 0.85678, "precision": 0.123456789},
        )
        summary = output.summary()
        assert "Evaluation Metrics:" in summary
        assert "recall@5: 0.8568" in summary
        assert "precision: 0.1235" in summary

    def test_summary_with_non_float_metrics(self) -> None:
        """Test summary() handles non-float metric values."""
        output = PipelineOutput(
            pipeline_name="rag",
            database_type="chroma",
            dataset_name="factscore",
            evaluation_metrics={
                "accuracy": 0.9,
                "count": 100,
                "status": "passed",
            },
        )
        summary = output.summary()
        assert "accuracy: 0.9000" in summary
        assert "count: 100" in summary
        assert "status: passed" in summary

    def test_summary_with_missing_index_stats_keys(self) -> None:
        """Test summary() handles missing keys in index_stats gracefully."""
        output = PipelineOutput(
            pipeline_name="test",
            database_type="test_db",
            dataset_name="test_data",
            index_stats={},  # Empty - all keys missing
        )
        summary = output.summary()
        assert "Indexed: 0 documents" in summary
        assert "Parents: 0" in summary
        assert "Children: 0" in summary

    def test_summary_with_partial_index_stats(self) -> None:
        """Test summary() handles partial index_stats."""
        output = PipelineOutput(
            pipeline_name="test",
            database_type="test_db",
            dataset_name="test_data",
            index_stats={"num_documents": 25},  # Only one key
        )
        summary = output.summary()
        assert "Indexed: 25 documents" in summary
        assert "Parents: 0" in summary
        assert "Children: 0" in summary

    def test_summary_with_retrieval_results(self) -> None:
        """Test summary() includes correct query count."""
        results = [RetrievalOutput(query=f"query{i}") for i in range(5)]
        output = PipelineOutput(
            pipeline_name="test",
            database_type="test_db",
            dataset_name="test_data",
            retrieval_results=results,
        )
        summary = output.summary()
        assert "Queries evaluated: 5" in summary
