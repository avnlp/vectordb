"""Tests for Chroma contextual compression search pipeline."""

from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.contextual_compression import ChromaCompressionSearch


class TestChromaCompressionSearch:
    """Unit tests for Chroma contextual compression search pipeline."""

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.chroma_search.chromadb")
    def test_init_connects_to_chroma(
        self,
        mock_chromadb_module: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        chroma_config: dict,
    ) -> None:
        """Test initialization connects to Chroma."""
        mock_load_config.return_value = chroma_config
        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder

        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chromadb_module.PersistentClient.return_value = mock_client

        pipeline = ChromaCompressionSearch("config.yaml")

        mock_chromadb_module.PersistentClient.assert_called_once_with(
            path="/tmp/test_chroma_compression"
        )
        mock_client.get_collection.assert_called_once_with(name="test_compression")
        assert pipeline.collection == mock_collection

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.chroma_search.chromadb")
    def test_init_raises_when_collection_not_found(
        self,
        mock_chromadb_module: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        chroma_config: dict,
    ) -> None:
        """Test initialization raises error when collection not found."""
        mock_load_config.return_value = chroma_config
        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder

        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_chromadb_module.PersistentClient.return_value = mock_client

        with pytest.raises(Exception, match="Collection not found"):
            ChromaCompressionSearch("config.yaml")

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.chroma_search.chromadb")
    def test_retrieve_base_results(
        self,
        mock_chromadb_module: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        chroma_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test _retrieve_base_results queries Chroma correctly."""
        mock_load_config.return_value = chroma_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Test content 1", "Test content 2"]],
            "metadatas": [[{"source": "wiki"}, {"source": "paper"}]],
            "distances": [[0.05, 0.15]],
        }

        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chromadb_module.PersistentClient.return_value = mock_client

        pipeline = ChromaCompressionSearch("config.yaml")
        results = pipeline._retrieve_base_results("test query", top_k=5)

        mock_embedder.run.assert_called_with(text="test query")
        mock_collection.query.assert_called_once_with(
            query_embeddings=[sample_embedding],
            n_results=5,
            include=["documents", "metadatas", "distances"],
        )

        assert len(results) == 2
        assert results[0].content == "Test content 1"
        assert results[0].meta["score"] == pytest.approx(0.95, rel=0.01)
        assert results[0].meta["chroma_distance"] == 0.05
        assert results[0].meta["source"] == "wiki"

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.chroma_search.chromadb")
    def test_retrieve_base_results_empty(
        self,
        mock_chromadb_module: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        chroma_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test _retrieve_base_results handles empty results."""
        mock_load_config.return_value = chroma_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chromadb_module.PersistentClient.return_value = mock_client

        pipeline = ChromaCompressionSearch("config.yaml")
        results = pipeline._retrieve_base_results("test query", top_k=5)

        assert len(results) == 0

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.chroma_search.chromadb")
    def test_retrieve_handles_nested_metadata(
        self,
        mock_chromadb_module: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        chroma_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test _retrieve_base_results handles nested metadata correctly."""
        mock_load_config.return_value = chroma_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Test content"]],
            "metadatas": [[{"metadata": "{'nested_key': 'nested_value'}"}]],
            "distances": [[0.1]],
        }

        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chromadb_module.PersistentClient.return_value = mock_client

        pipeline = ChromaCompressionSearch("config.yaml")
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
    @patch("vectordb.haystack.contextual_compression.search.chroma_search.chromadb")
    def test_run_retrieves_and_compresses(
        self,
        mock_chromadb_module: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        chroma_config: dict,
        sample_embedding: list[float],
        sample_documents: list[Document],
    ) -> None:
        """Test run method retrieves and compresses documents."""
        mock_load_config.return_value = chroma_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [[doc.content for doc in sample_documents]],
            "metadatas": [[{} for _ in sample_documents]],
            "distances": [[0.1 * (i + 1) for i in range(len(sample_documents))]],
        }

        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chromadb_module.PersistentClient.return_value = mock_client

        mock_compressor = MagicMock()
        mock_compressor.run.return_value = {"documents": sample_documents[:2]}
        mock_compressor_factory.return_value = mock_compressor

        pipeline = ChromaCompressionSearch("config.yaml")
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
    @patch("vectordb.haystack.contextual_compression.search.chroma_search.chromadb")
    def test_run_returns_empty_when_no_results(
        self,
        mock_chromadb_module: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        chroma_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test run returns empty documents when no results retrieved."""
        mock_load_config.return_value = chroma_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chromadb_module.PersistentClient.return_value = mock_client

        pipeline = ChromaCompressionSearch("config.yaml")
        result = pipeline.run("test query", top_k=5)

        assert result == {"documents": []}

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.chroma_search.chromadb")
    def test_run_handles_compressor_error(
        self,
        mock_chromadb_module: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        chroma_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test run handles compressor errors gracefully."""
        mock_load_config.return_value = chroma_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Test"]],
            "metadatas": [[{}]],
            "distances": [[0.1]],
        }

        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chromadb_module.PersistentClient.return_value = mock_client

        mock_compressor = MagicMock()
        mock_compressor.run.side_effect = Exception("Compressor error")
        mock_compressor_factory.return_value = mock_compressor

        pipeline = ChromaCompressionSearch("config.yaml")
        result = pipeline.run("test query", top_k=5)

        assert result == {"documents": []}

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.chroma_search.chromadb")
    def test_uses_default_collection_name(
        self,
        mock_chromadb_module: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        chroma_config: dict,
    ) -> None:
        """Test uses default collection name when not specified."""
        del chroma_config["chroma"]["collection_name"]
        mock_load_config.return_value = chroma_config

        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder

        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chromadb_module.PersistentClient.return_value = mock_client

        ChromaCompressionSearch("config.yaml")

        mock_client.get_collection.assert_called_with(name="compression")

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.chroma_search.chromadb")
    @patch("tempfile.gettempdir")
    def test_uses_temp_dir_as_default_path(
        self,
        mock_gettempdir: MagicMock,
        mock_chromadb_module: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        chroma_config: dict,
    ) -> None:
        """Test uses temp directory as default path when not specified."""
        del chroma_config["chroma"]["path"]
        mock_load_config.return_value = chroma_config
        mock_gettempdir.return_value = "/tmp"

        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder

        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chromadb_module.PersistentClient.return_value = mock_client

        ChromaCompressionSearch("config.yaml")

        mock_chromadb_module.PersistentClient.assert_called_once_with(
            path="/tmp/chroma"
        )

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.chroma_search.chromadb")
    def test_evaluate_runs_for_all_questions(
        self,
        mock_chromadb_module: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        chroma_config: dict,
        sample_embedding: list[float],
        sample_documents: list[Document],
    ) -> None:
        """Test evaluate runs for all questions."""
        mock_load_config.return_value = chroma_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Test"]],
            "metadatas": [[{}]],
            "distances": [[0.1]],
        }

        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chromadb_module.PersistentClient.return_value = mock_client

        mock_compressor = MagicMock()
        mock_compressor.run.return_value = {"documents": sample_documents[:1]}
        mock_compressor_factory.return_value = mock_compressor

        pipeline = ChromaCompressionSearch("config.yaml")

        questions = ["Q1?", "Q2?", "Q3?"]
        ground_truths = ["A1", "A2", "A3"]
        result = pipeline.evaluate(questions, ground_truths)

        assert result["questions"] == 3
        assert mock_collection.query.call_count == 3

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.chroma_search.chromadb")
    def test_handles_none_metadata(
        self,
        mock_chromadb_module: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        chroma_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test _retrieve_base_results handles None metadata gracefully."""
        mock_load_config.return_value = chroma_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Test content"]],
            "metadatas": None,
            "distances": [[0.1]],
        }

        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chromadb_module.PersistentClient.return_value = mock_client

        pipeline = ChromaCompressionSearch("config.yaml")
        results = pipeline._retrieve_base_results("test query", top_k=5)

        assert len(results) == 1
        assert results[0].content == "Test content"
        assert results[0].meta["score"] == pytest.approx(0.9, rel=0.01)

    @patch(
        "vectordb.haystack.contextual_compression.base.CompressorFactory.create_compressor"
    )
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.search.chroma_search.chromadb")
    def test_handles_none_distances(
        self,
        mock_chromadb_module: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
        mock_compressor_factory: MagicMock,
        chroma_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test _retrieve_base_results handles None distances gracefully."""
        mock_load_config.return_value = chroma_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Test content"]],
            "metadatas": [[{}]],
            "distances": None,
        }

        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chromadb_module.PersistentClient.return_value = mock_client

        pipeline = ChromaCompressionSearch("config.yaml")
        results = pipeline._retrieve_base_results("test query", top_k=5)

        assert len(results) == 1
        assert results[0].meta["score"] == 1.0
        assert results[0].meta["chroma_distance"] == 0.0
