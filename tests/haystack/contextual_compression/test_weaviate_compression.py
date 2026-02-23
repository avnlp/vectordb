"""Tests for Weaviate contextual compression search pipeline."""

from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.contextual_compression import WeaviateCompressionSearch


class TestWeaviateCompressionSearch:
    """Unit tests for Weaviate contextual compression search pipeline."""

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.weaviate_search.weaviate")
    def test_init_connects_to_weaviate(
        self,
        mock_weaviate_module: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        weaviate_config: dict,
    ) -> None:
        """Test initialization connects to Weaviate."""
        mock_load_config.return_value = weaviate_config
        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection
        mock_weaviate_module.connect_to_local.return_value = mock_client

        pipeline = WeaviateCompressionSearch("config.yaml")

        mock_weaviate_module.connect_to_local.assert_called_once_with(host="localhost")
        mock_client.collections.get.assert_called_once_with("TestCompression")
        assert pipeline.collection_name == "TestCompression"

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.weaviate_search.weaviate")
    def test_init_raises_when_collection_not_found(
        self,
        mock_weaviate_module: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        weaviate_config: dict,
    ) -> None:
        """Test initialization raises error when collection not found."""
        mock_load_config.return_value = weaviate_config
        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder

        mock_client = MagicMock()
        mock_client.collections.get.side_effect = Exception("Collection not found")
        mock_weaviate_module.connect_to_local.return_value = mock_client

        with pytest.raises(Exception, match="Collection not found"):
            WeaviateCompressionSearch("config.yaml")

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.weaviate_search.weaviate")
    def test_retrieve_base_results(
        self,
        mock_weaviate_module: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        weaviate_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test _retrieve_base_results queries Weaviate correctly."""
        mock_load_config.return_value = weaviate_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_obj1 = MagicMock()
        mock_obj1.properties = {"content": "Test content 1", "source": "wiki"}
        mock_obj1.metadata.distance = 0.05

        mock_obj2 = MagicMock()
        mock_obj2.properties = {"content": "Test content 2", "source": "paper"}
        mock_obj2.metadata.distance = 0.15

        mock_results = MagicMock()
        mock_results.objects = [mock_obj1, mock_obj2]

        mock_collection = MagicMock()
        mock_collection.query.near_vector.return_value = mock_results

        mock_client = MagicMock()
        mock_client.collections.get.return_value = mock_collection
        mock_weaviate_module.connect_to_local.return_value = mock_client

        pipeline = WeaviateCompressionSearch("config.yaml")
        results = pipeline._retrieve_base_results("test query", top_k=5)

        mock_embedder.run.assert_called_with(text="test query")
        mock_collection.query.near_vector.assert_called_once()

        call_kwargs = mock_collection.query.near_vector.call_args[1]
        assert call_kwargs["near_vector"] == sample_embedding
        assert call_kwargs["limit"] == 5

        assert len(results) == 2
        assert results[0].content == "Test content 1"
        assert results[0].meta["score"] == pytest.approx(0.95, rel=0.01)
        assert results[0].meta["weaviate_distance"] == 0.05

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.weaviate_search.weaviate")
    def test_retrieve_base_results_empty(
        self,
        mock_weaviate_module: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        weaviate_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test _retrieve_base_results handles empty results."""
        mock_load_config.return_value = weaviate_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_results = MagicMock()
        mock_results.objects = []

        mock_collection = MagicMock()
        mock_collection.query.near_vector.return_value = mock_results

        mock_client = MagicMock()
        mock_client.collections.get.return_value = mock_collection
        mock_weaviate_module.connect_to_local.return_value = mock_client

        pipeline = WeaviateCompressionSearch("config.yaml")
        results = pipeline._retrieve_base_results("test query", top_k=5)

        assert len(results) == 0

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.weaviate_search.weaviate")
    def test_retrieve_handles_nested_metadata(
        self,
        mock_weaviate_module: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        weaviate_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test _retrieve_base_results handles nested metadata correctly."""
        mock_load_config.return_value = weaviate_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_obj = MagicMock()
        mock_obj.properties = {
            "content": "Test content",
            "metadata": "{'nested_key': 'nested_value'}",
        }
        mock_obj.metadata.distance = 0.1

        mock_results = MagicMock()
        mock_results.objects = [mock_obj]

        mock_collection = MagicMock()
        mock_collection.query.near_vector.return_value = mock_results

        mock_client = MagicMock()
        mock_client.collections.get.return_value = mock_collection
        mock_weaviate_module.connect_to_local.return_value = mock_client

        pipeline = WeaviateCompressionSearch("config.yaml")
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
    @patch("vectordb.haystack.contextual_compression.search.weaviate_search.weaviate")
    def test_run_retrieves_and_compresses(
        self,
        mock_weaviate_module: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        weaviate_config: dict,
        sample_embedding: list[float],
        sample_documents: list[Document],
    ) -> None:
        """Test run method retrieves and compresses documents."""
        mock_load_config.return_value = weaviate_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_objs = []
        for i, doc in enumerate(sample_documents):
            mock_obj = MagicMock()
            mock_obj.properties = {"content": doc.content}
            mock_obj.metadata.distance = 0.1 * (i + 1)
            mock_objs.append(mock_obj)

        mock_results = MagicMock()
        mock_results.objects = mock_objs

        mock_collection = MagicMock()
        mock_collection.query.near_vector.return_value = mock_results

        mock_client = MagicMock()
        mock_client.collections.get.return_value = mock_collection
        mock_weaviate_module.connect_to_local.return_value = mock_client

        mock_compressor = MagicMock()
        mock_compressor.run.return_value = {"documents": sample_documents[:2]}
        mock_compressor_factory.return_value = mock_compressor

        pipeline = WeaviateCompressionSearch("config.yaml")
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
    @patch("vectordb.haystack.contextual_compression.search.weaviate_search.weaviate")
    def test_run_returns_empty_when_no_results(
        self,
        mock_weaviate_module: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        weaviate_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test run returns empty documents when no results retrieved."""
        mock_load_config.return_value = weaviate_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_results = MagicMock()
        mock_results.objects = []

        mock_collection = MagicMock()
        mock_collection.query.near_vector.return_value = mock_results

        mock_client = MagicMock()
        mock_client.collections.get.return_value = mock_collection
        mock_weaviate_module.connect_to_local.return_value = mock_client

        pipeline = WeaviateCompressionSearch("config.yaml")
        result = pipeline.run("test query", top_k=5)

        assert result == {"documents": []}

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.weaviate_search.weaviate")
    def test_run_handles_compressor_error(
        self,
        mock_weaviate_module: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        weaviate_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test run handles compressor errors gracefully."""
        mock_load_config.return_value = weaviate_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_obj = MagicMock()
        mock_obj.properties = {"content": "Test"}
        mock_obj.metadata.distance = 0.1

        mock_results = MagicMock()
        mock_results.objects = [mock_obj]

        mock_collection = MagicMock()
        mock_collection.query.near_vector.return_value = mock_results

        mock_client = MagicMock()
        mock_client.collections.get.return_value = mock_collection
        mock_weaviate_module.connect_to_local.return_value = mock_client

        mock_compressor = MagicMock()
        mock_compressor.run.side_effect = Exception("Compressor error")
        mock_compressor_factory.return_value = mock_compressor

        pipeline = WeaviateCompressionSearch("config.yaml")
        result = pipeline.run("test query", top_k=5)

        assert result == {"documents": []}

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.weaviate_search.weaviate")
    def test_uses_default_collection_name(
        self,
        mock_weaviate_module: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        weaviate_config: dict,
    ) -> None:
        """Test uses default collection name when not specified."""
        del weaviate_config["weaviate"]["collection_name"]
        mock_load_config.return_value = weaviate_config

        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder

        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.collections.get.return_value = mock_collection
        mock_weaviate_module.connect_to_local.return_value = mock_client

        pipeline = WeaviateCompressionSearch("config.yaml")

        assert pipeline.collection_name == "Compression"
        mock_client.collections.get.assert_called_with("Compression")

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.weaviate_search.weaviate")
    def test_evaluate_runs_for_all_questions(
        self,
        mock_weaviate_module: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        weaviate_config: dict,
        sample_embedding: list[float],
        sample_documents: list[Document],
    ) -> None:
        """Test evaluate runs for all questions."""
        mock_load_config.return_value = weaviate_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_obj = MagicMock()
        mock_obj.properties = {"content": "Test"}
        mock_obj.metadata.distance = 0.1

        mock_results = MagicMock()
        mock_results.objects = [mock_obj]

        mock_collection = MagicMock()
        mock_collection.query.near_vector.return_value = mock_results

        mock_client = MagicMock()
        mock_client.collections.get.return_value = mock_collection
        mock_weaviate_module.connect_to_local.return_value = mock_client

        mock_compressor = MagicMock()
        mock_compressor.run.return_value = {"documents": sample_documents[:1]}
        mock_compressor_factory.return_value = mock_compressor

        pipeline = WeaviateCompressionSearch("config.yaml")

        questions = ["Q1?", "Q2?", "Q3?"]
        ground_truths = ["A1", "A2", "A3"]
        result = pipeline.evaluate(questions, ground_truths)

        assert result["questions"] == 3
        assert mock_collection.query.near_vector.call_count == 3
