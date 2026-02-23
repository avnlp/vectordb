"""Tests for Qdrant contextual compression search pipeline."""

from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.contextual_compression import QdrantCompressionSearch


class TestQdrantCompressionSearch:
    """Unit tests for Qdrant contextual compression search pipeline."""

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.qdrant_search.QdrantClient")
    def test_init_connects_to_qdrant(
        self,
        mock_qdrant_client_class: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        qdrant_config: dict,
    ) -> None:
        """Test initialization connects to Qdrant."""
        mock_load_config.return_value = qdrant_config
        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder

        mock_client = MagicMock()
        mock_qdrant_client_class.return_value = mock_client

        pipeline = QdrantCompressionSearch("config.yaml")

        mock_qdrant_client_class.assert_called_once_with(url="http://localhost:6333")
        mock_client.get_collection.assert_called_once_with("test_compression")
        assert pipeline.collection_name == "test_compression"

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.qdrant_search.QdrantClient")
    def test_init_raises_when_collection_not_found(
        self,
        mock_qdrant_client_class: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        qdrant_config: dict,
    ) -> None:
        """Test initialization raises error when collection not found."""
        mock_load_config.return_value = qdrant_config
        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder

        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_qdrant_client_class.return_value = mock_client

        with pytest.raises(Exception, match="Collection not found"):
            QdrantCompressionSearch("config.yaml")

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.qdrant_search.QdrantClient")
    def test_retrieve_base_results(
        self,
        mock_qdrant_client_class: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        qdrant_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test _retrieve_base_results queries Qdrant correctly."""
        mock_load_config.return_value = qdrant_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_point1 = MagicMock()
        mock_point1.id = "1"
        mock_point1.score = 0.95
        mock_point1.payload = {"content": "Test content 1", "source": "wiki"}

        mock_point2 = MagicMock()
        mock_point2.id = "2"
        mock_point2.score = 0.85
        mock_point2.payload = {"content": "Test content 2", "source": "paper"}

        mock_client = MagicMock()
        mock_client.search.return_value = [mock_point1, mock_point2]
        mock_qdrant_client_class.return_value = mock_client

        pipeline = QdrantCompressionSearch("config.yaml")
        results = pipeline._retrieve_base_results("test query", top_k=5)

        mock_embedder.run.assert_called_with(text="test query")
        mock_client.search.assert_called_once_with(
            collection_name="test_compression",
            query_vector=sample_embedding,
            limit=5,
        )

        assert len(results) == 2
        assert results[0].content == "Test content 1"
        assert results[0].meta["score"] == 0.95
        assert results[0].meta["qdrant_id"] == "1"
        assert results[0].meta["source"] == "wiki"

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.qdrant_search.QdrantClient")
    def test_retrieve_base_results_empty(
        self,
        mock_qdrant_client_class: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        qdrant_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test _retrieve_base_results handles empty results."""
        mock_load_config.return_value = qdrant_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_client = MagicMock()
        mock_client.search.return_value = []
        mock_qdrant_client_class.return_value = mock_client

        pipeline = QdrantCompressionSearch("config.yaml")
        results = pipeline._retrieve_base_results("test query", top_k=5)

        assert len(results) == 0

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.qdrant_search.QdrantClient")
    def test_retrieve_handles_nested_metadata(
        self,
        mock_qdrant_client_class: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        qdrant_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test _retrieve_base_results handles nested metadata correctly."""
        mock_load_config.return_value = qdrant_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_point = MagicMock()
        mock_point.id = "1"
        mock_point.score = 0.95
        mock_point.payload = {
            "content": "Test content",
            "metadata": "{'nested_key': 'nested_value'}",
        }

        mock_client = MagicMock()
        mock_client.search.return_value = [mock_point]
        mock_qdrant_client_class.return_value = mock_client

        pipeline = QdrantCompressionSearch("config.yaml")
        results = pipeline._retrieve_base_results("test query", top_k=5)

        assert len(results) == 1
        assert results[0].meta["nested_key"] == "nested_value"

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.qdrant_search.QdrantClient")
    def test_run_retrieves_and_compresses(
        self,
        mock_qdrant_client_class: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        qdrant_config: dict,
        sample_embedding: list[float],
        sample_documents: list[Document],
    ) -> None:
        """Test run method retrieves and compresses documents."""
        mock_load_config.return_value = qdrant_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_points = []
        for i, doc in enumerate(sample_documents):
            mock_point = MagicMock()
            mock_point.id = str(i + 1)
            mock_point.score = 0.9 - i * 0.1
            mock_point.payload = {"content": doc.content}
            mock_points.append(mock_point)

        mock_client = MagicMock()
        mock_client.search.return_value = mock_points
        mock_qdrant_client_class.return_value = mock_client

        mock_compressor = MagicMock()
        mock_compressor.run.return_value = {"documents": sample_documents[:2]}
        mock_compressor_factory.return_value = mock_compressor

        pipeline = QdrantCompressionSearch("config.yaml")
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
    @patch("vectordb.haystack.contextual_compression.search.qdrant_search.QdrantClient")
    def test_run_returns_empty_when_no_results(
        self,
        mock_qdrant_client_class: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        qdrant_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test run returns empty documents when no results retrieved."""
        mock_load_config.return_value = qdrant_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_client = MagicMock()
        mock_client.search.return_value = []
        mock_qdrant_client_class.return_value = mock_client

        pipeline = QdrantCompressionSearch("config.yaml")
        result = pipeline.run("test query", top_k=5)

        assert result == {"documents": []}

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.qdrant_search.QdrantClient")
    def test_run_handles_compressor_error(
        self,
        mock_qdrant_client_class: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        qdrant_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test run handles compressor errors gracefully."""
        mock_load_config.return_value = qdrant_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_point = MagicMock()
        mock_point.id = "1"
        mock_point.score = 0.9
        mock_point.payload = {"content": "Test"}

        mock_client = MagicMock()
        mock_client.search.return_value = [mock_point]
        mock_qdrant_client_class.return_value = mock_client

        mock_compressor = MagicMock()
        mock_compressor.run.side_effect = Exception("Compressor error")
        mock_compressor_factory.return_value = mock_compressor

        pipeline = QdrantCompressionSearch("config.yaml")
        result = pipeline.run("test query", top_k=5)

        assert result == {"documents": []}

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.qdrant_search.QdrantClient")
    def test_uses_default_collection_name(
        self,
        mock_qdrant_client_class: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        qdrant_config: dict,
    ) -> None:
        """Test uses default collection name when not specified."""
        del qdrant_config["qdrant"]["collection_name"]
        mock_load_config.return_value = qdrant_config

        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder

        mock_client = MagicMock()
        mock_qdrant_client_class.return_value = mock_client

        pipeline = QdrantCompressionSearch("config.yaml")

        assert pipeline.collection_name == "compression"
        mock_client.get_collection.assert_called_with("compression")

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.qdrant_search.QdrantClient")
    def test_evaluate_runs_for_all_questions(
        self,
        mock_qdrant_client_class: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        qdrant_config: dict,
        sample_embedding: list[float],
        sample_documents: list[Document],
    ) -> None:
        """Test evaluate runs for all questions."""
        mock_load_config.return_value = qdrant_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_point = MagicMock()
        mock_point.id = "1"
        mock_point.score = 0.9
        mock_point.payload = {"content": "Test"}

        mock_client = MagicMock()
        mock_client.search.return_value = [mock_point]
        mock_qdrant_client_class.return_value = mock_client

        mock_compressor = MagicMock()
        mock_compressor.run.return_value = {"documents": sample_documents[:1]}
        mock_compressor_factory.return_value = mock_compressor

        pipeline = QdrantCompressionSearch("config.yaml")

        questions = ["Q1?", "Q2?", "Q3?"]
        ground_truths = ["A1", "A2", "A3"]
        result = pipeline.evaluate(questions, ground_truths)

        assert result["questions"] == 3
        assert mock_client.search.call_count == 3
