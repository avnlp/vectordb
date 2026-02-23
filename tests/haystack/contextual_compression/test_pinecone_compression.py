"""Tests for Pinecone contextual compression search pipeline."""

from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.contextual_compression import PineconeCompressionSearch


class TestPineconeCompressionSearch:
    """Unit tests for Pinecone contextual compression search pipeline."""

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.pinecone_search.Pinecone")
    def test_init_connects_to_pinecone(
        self,
        mock_pinecone_class: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        pinecone_config: dict,
    ) -> None:
        """Test initialization connects to Pinecone."""
        mock_load_config.return_value = pinecone_config
        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder

        mock_pc = MagicMock()
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_pc

        pipeline = PineconeCompressionSearch("config.yaml")

        mock_pinecone_class.assert_called_once_with(api_key="test-key")
        mock_pc.Index.assert_called_once_with("test-compression-index")
        assert pipeline.index == mock_index

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.pinecone_search.Pinecone")
    def test_init_raises_without_api_key(
        self,
        mock_pinecone_class: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        pinecone_config: dict,
    ) -> None:
        """Test initialization raises error without API key."""
        pinecone_config["pinecone"]["api_key"] = ""
        mock_load_config.return_value = pinecone_config
        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder

        with pytest.raises(ValueError, match="api_key"):
            PineconeCompressionSearch("config.yaml")

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.pinecone_search.Pinecone")
    def test_init_raises_without_index_name(
        self,
        mock_pinecone_class: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        pinecone_config: dict,
    ) -> None:
        """Test initialization raises error without index name."""
        pinecone_config["pinecone"]["index_name"] = ""
        mock_load_config.return_value = pinecone_config
        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder

        mock_pc = MagicMock()
        mock_pinecone_class.return_value = mock_pc

        with pytest.raises(ValueError, match="index_name"):
            PineconeCompressionSearch("config.yaml")

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.pinecone_search.Pinecone")
    def test_retrieve_base_results(
        self,
        mock_pinecone_class: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        pinecone_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test _retrieve_base_results queries Pinecone correctly."""
        mock_load_config.return_value = pinecone_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_index = MagicMock()
        mock_index.query.return_value = {
            "matches": [
                {
                    "id": "doc1",
                    "score": 0.95,
                    "metadata": {"content": "Test content 1", "source": "wiki"},
                },
                {
                    "id": "doc2",
                    "score": 0.85,
                    "metadata": {"content": "Test content 2", "source": "paper"},
                },
            ]
        }

        mock_pc = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_pc

        pipeline = PineconeCompressionSearch("config.yaml")
        results = pipeline._retrieve_base_results("test query", top_k=5)

        mock_embedder.run.assert_called_with(text="test query")
        mock_index.query.assert_called_once_with(
            vector=sample_embedding,
            top_k=5,
            include_metadata=True,
        )

        assert len(results) == 2
        assert results[0].content == "Test content 1"
        assert results[0].meta["score"] == 0.95
        assert results[0].meta["pinecone_id"] == "doc1"
        assert results[1].content == "Test content 2"

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.pinecone_search.Pinecone")
    def test_retrieve_base_results_empty(
        self,
        mock_pinecone_class: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        pinecone_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test _retrieve_base_results handles empty results."""
        mock_load_config.return_value = pinecone_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_index = MagicMock()
        mock_index.query.return_value = {"matches": []}

        mock_pc = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_pc

        pipeline = PineconeCompressionSearch("config.yaml")
        results = pipeline._retrieve_base_results("test query", top_k=5)

        assert len(results) == 0

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.pinecone_search.Pinecone")
    def test_run_retrieves_and_compresses(
        self,
        mock_pinecone_class: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        pinecone_config: dict,
        sample_embedding: list[float],
        sample_documents: list[Document],
    ) -> None:
        """Test run method retrieves and compresses documents."""
        mock_load_config.return_value = pinecone_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_index = MagicMock()
        mock_index.query.return_value = {
            "matches": [
                {
                    "id": f"doc{i}",
                    "score": 0.9 - i * 0.1,
                    "metadata": {"content": d.content},
                }
                for i, d in enumerate(sample_documents)
            ]
        }

        mock_pc = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_pc

        mock_compressor = MagicMock()
        mock_compressor.run.return_value = {"documents": sample_documents[:2]}
        mock_compressor_factory.return_value = mock_compressor

        pipeline = PineconeCompressionSearch("config.yaml")
        result = pipeline.run("test query", top_k=2)

        assert "documents" in result
        assert len(result["documents"]) == 2
        mock_compressor.run.assert_called_once()

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.pinecone_search.Pinecone")
    def test_run_returns_empty_when_no_results(
        self,
        mock_pinecone_class: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        pinecone_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test run returns empty documents when no results retrieved."""
        mock_load_config.return_value = pinecone_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_index = MagicMock()
        mock_index.query.return_value = {"matches": []}

        mock_pc = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_pc

        pipeline = PineconeCompressionSearch("config.yaml")
        result = pipeline.run("test query", top_k=5)

        assert result == {"documents": []}

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.pinecone_search.Pinecone")
    def test_run_handles_compressor_error(
        self,
        mock_pinecone_class: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        pinecone_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test run handles compressor errors gracefully."""
        mock_load_config.return_value = pinecone_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_index = MagicMock()
        mock_index.query.return_value = {
            "matches": [{"id": "doc1", "score": 0.9, "metadata": {"content": "Test"}}]
        }

        mock_pc = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_pc

        mock_compressor = MagicMock()
        mock_compressor.run.side_effect = Exception("Compressor error")
        mock_compressor_factory.return_value = mock_compressor

        pipeline = PineconeCompressionSearch("config.yaml")
        result = pipeline.run("test query", top_k=5)

        assert result == {"documents": []}

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.pinecone_search.Pinecone")
    def test_evaluate_runs_for_all_questions(
        self,
        mock_pinecone_class: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        pinecone_config: dict,
        sample_embedding: list[float],
        sample_documents: list[Document],
    ) -> None:
        """Test evaluate runs for all questions."""
        mock_load_config.return_value = pinecone_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_index = MagicMock()
        mock_index.query.return_value = {
            "matches": [{"id": "doc1", "score": 0.9, "metadata": {"content": "Test"}}]
        }

        mock_pc = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_pc

        mock_compressor = MagicMock()
        mock_compressor.run.return_value = {"documents": sample_documents[:1]}
        mock_compressor_factory.return_value = mock_compressor

        pipeline = PineconeCompressionSearch("config.yaml")

        questions = ["Q1?", "Q2?", "Q3?"]
        ground_truths = ["A1", "A2", "A3"]
        result = pipeline.evaluate(questions, ground_truths)

        assert result["questions"] == 3
        assert mock_index.query.call_count == 3
