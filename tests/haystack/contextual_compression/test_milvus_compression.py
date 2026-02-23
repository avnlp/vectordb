"""Tests for Milvus contextual compression search pipeline."""

from unittest.mock import MagicMock, patch

from haystack import Document

from vectordb.haystack.contextual_compression import MilvusCompressionSearch


class TestMilvusCompressionSearch:
    """Unit tests for Milvus contextual compression search pipeline."""

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.milvus_search.Collection")
    @patch("vectordb.haystack.contextual_compression.search.milvus_search.connections")
    def test_init_connects_to_milvus(
        self,
        mock_connections: MagicMock,
        mock_collection_class: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        milvus_config: dict,
    ) -> None:
        """Test initialization connects to Milvus."""
        mock_load_config.return_value = milvus_config
        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder

        mock_collection = MagicMock()
        mock_collection_class.return_value = mock_collection

        pipeline = MilvusCompressionSearch("config.yaml")

        mock_connections.connect.assert_called_once_with(
            alias="default", host="localhost", port=19530
        )
        mock_collection.load.assert_called_once()
        assert pipeline.collection_name == "test_compression"

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.milvus_search.Collection")
    @patch("vectordb.haystack.contextual_compression.search.milvus_search.connections")
    def test_retrieve_base_results(
        self,
        mock_connections: MagicMock,
        mock_collection_class: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        milvus_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test _retrieve_base_results queries Milvus correctly."""
        mock_load_config.return_value = milvus_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_hit1 = MagicMock()
        mock_hit1.id = "1"
        mock_hit1.distance = 0.95
        mock_hit1.entity.get.side_effect = lambda k, d=None: {
            "content": "Test content 1",
            "metadata": "{'source': 'wiki'}",
        }.get(k, d)

        mock_hit2 = MagicMock()
        mock_hit2.id = "2"
        mock_hit2.distance = 0.85
        mock_hit2.entity.get.side_effect = lambda k, d=None: {
            "content": "Test content 2",
            "metadata": "{'source': 'paper'}",
        }.get(k, d)

        mock_collection = MagicMock()
        mock_collection.search.return_value = [[mock_hit1, mock_hit2]]
        mock_collection_class.return_value = mock_collection

        pipeline = MilvusCompressionSearch("config.yaml")
        results = pipeline._retrieve_base_results("test query", top_k=5)

        mock_embedder.run.assert_called_with(text="test query")
        mock_collection.search.assert_called_once()

        call_kwargs = mock_collection.search.call_args[1]
        assert call_kwargs["limit"] == 5
        assert call_kwargs["output_fields"] == ["content", "metadata"]

        assert len(results) == 2
        assert results[0].content == "Test content 1"
        assert results[0].meta["distance"] == 0.95
        assert results[0].meta["milvus_id"] == "1"

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.milvus_search.Collection")
    @patch("vectordb.haystack.contextual_compression.search.milvus_search.connections")
    def test_retrieve_base_results_empty(
        self,
        mock_connections: MagicMock,
        mock_collection_class: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        milvus_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test _retrieve_base_results handles empty results."""
        mock_load_config.return_value = milvus_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_collection = MagicMock()
        mock_collection.search.return_value = [[]]
        mock_collection_class.return_value = mock_collection

        pipeline = MilvusCompressionSearch("config.yaml")
        results = pipeline._retrieve_base_results("test query", top_k=5)

        assert len(results) == 0

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.milvus_search.Collection")
    @patch("vectordb.haystack.contextual_compression.search.milvus_search.connections")
    def test_run_retrieves_and_compresses(
        self,
        mock_connections: MagicMock,
        mock_collection_class: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        milvus_config: dict,
        sample_embedding: list[float],
        sample_documents: list[Document],
    ) -> None:
        """Test run method retrieves and compresses documents."""
        mock_load_config.return_value = milvus_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_hits = []
        for i, doc in enumerate(sample_documents):
            mock_hit = MagicMock()
            mock_hit.id = str(i + 1)
            mock_hit.distance = 0.9 - i * 0.1
            mock_hit.entity.get.side_effect = lambda k, d=None, content=doc.content: {
                "content": content,
                "metadata": "{}",
            }.get(k, d)
            mock_hits.append(mock_hit)

        mock_collection = MagicMock()
        mock_collection.search.return_value = [mock_hits]
        mock_collection_class.return_value = mock_collection

        mock_compressor = MagicMock()
        mock_compressor.run.return_value = {"documents": sample_documents[:2]}
        mock_compressor_factory.return_value = mock_compressor

        pipeline = MilvusCompressionSearch("config.yaml")
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
    @patch("vectordb.haystack.contextual_compression.search.milvus_search.Collection")
    @patch("vectordb.haystack.contextual_compression.search.milvus_search.connections")
    def test_run_returns_empty_when_no_results(
        self,
        mock_connections: MagicMock,
        mock_collection_class: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        milvus_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test run returns empty documents when no results retrieved."""
        mock_load_config.return_value = milvus_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_collection = MagicMock()
        mock_collection.search.return_value = [[]]
        mock_collection_class.return_value = mock_collection

        pipeline = MilvusCompressionSearch("config.yaml")
        result = pipeline.run("test query", top_k=5)

        assert result == {"documents": []}

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.milvus_search.Collection")
    @patch("vectordb.haystack.contextual_compression.search.milvus_search.connections")
    def test_run_handles_compressor_error(
        self,
        mock_connections: MagicMock,
        mock_collection_class: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        milvus_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test run handles compressor errors gracefully."""
        mock_load_config.return_value = milvus_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_hit = MagicMock()
        mock_hit.id = "1"
        mock_hit.distance = 0.9
        mock_hit.entity.get.side_effect = lambda k, d=None: {
            "content": "Test",
            "metadata": "{}",
        }.get(k, d)

        mock_collection = MagicMock()
        mock_collection.search.return_value = [[mock_hit]]
        mock_collection_class.return_value = mock_collection

        mock_compressor = MagicMock()
        mock_compressor.run.side_effect = Exception("Compressor error")
        mock_compressor_factory.return_value = mock_compressor

        pipeline = MilvusCompressionSearch("config.yaml")
        result = pipeline.run("test query", top_k=5)

        assert result == {"documents": []}

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.milvus_search.Collection")
    @patch("vectordb.haystack.contextual_compression.search.milvus_search.connections")
    def test_uses_default_collection_name(
        self,
        mock_connections: MagicMock,
        mock_collection_class: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        milvus_config: dict,
    ) -> None:
        """Test uses default collection name when not specified."""
        del milvus_config["milvus"]["collection_name"]
        mock_load_config.return_value = milvus_config

        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder

        mock_collection = MagicMock()
        mock_collection_class.return_value = mock_collection

        pipeline = MilvusCompressionSearch("config.yaml")

        assert pipeline.collection_name == "compression"
        mock_collection_class.assert_called_with("compression", using="default")

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.milvus_search.Collection")
    @patch("vectordb.haystack.contextual_compression.search.milvus_search.connections")
    def test_evaluate_runs_for_all_questions(
        self,
        mock_connections: MagicMock,
        mock_collection_class: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        milvus_config: dict,
        sample_embedding: list[float],
        sample_documents: list[Document],
    ) -> None:
        """Test evaluate runs for all questions."""
        mock_load_config.return_value = milvus_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_hit = MagicMock()
        mock_hit.id = "1"
        mock_hit.distance = 0.9
        mock_hit.entity.get.side_effect = lambda k, d=None: {
            "content": "Test",
            "metadata": "{}",
        }.get(k, d)

        mock_collection = MagicMock()
        mock_collection.search.return_value = [[mock_hit]]
        mock_collection_class.return_value = mock_collection

        mock_compressor = MagicMock()
        mock_compressor.run.return_value = {"documents": sample_documents[:1]}
        mock_compressor_factory.return_value = mock_compressor

        pipeline = MilvusCompressionSearch("config.yaml")

        questions = ["Q1?", "Q2?", "Q3?"]
        ground_truths = ["A1", "A2", "A3"]
        result = pipeline.evaluate(questions, ground_truths)

        assert result["questions"] == 3
        assert mock_collection.search.call_count == 3
