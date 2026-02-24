"""Parametrized base tests for query enhancement pipelines.

This module contains common tests that run against all database provider
implementations. Provider-specific tests remain in their respective files.
"""

from importlib import import_module
from unittest.mock import MagicMock, patch

from haystack import Document

from tests.haystack.query_enhancement.conftest import ProviderFixture


class TestIndexingPipelineBase:
    """Common tests for indexing pipeline implementations."""

    @patch("vectordb.haystack.query_enhancement.indexing.base.DataloaderCatalog")
    @patch("vectordb.haystack.query_enhancement.indexing.base.create_document_embedder")
    @patch("vectordb.haystack.query_enhancement.indexing.base.validate_config")
    @patch("vectordb.haystack.query_enhancement.indexing.base.load_config")
    def test_indexing_pipeline_initialization(
        self,
        mock_load_config: MagicMock,
        mock_validate: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_dataloader_catalog: MagicMock,
        provider_fixture: ProviderFixture,
    ) -> None:
        """Test that indexing pipeline initializes correctly."""
        with patch(provider_fixture.indexing_db_mock_path):
            mock_load_config.return_value = provider_fixture.config
            mock_dataset = MagicMock()
            mock_loader = MagicMock()
            mock_loader.load.return_value = mock_dataset
            mock_dataloader_catalog.create.return_value = mock_loader
            mock_embedder = MagicMock()
            mock_embedder_factory.return_value = mock_embedder

            module_path, class_name = provider_fixture.indexing_pipeline_path.rsplit(
                ".", 1
            )
            module = import_module(module_path)
            pipeline_cls = getattr(module, class_name)
            pipeline = pipeline_cls("/tmp/fake_config.yaml")

            mock_load_config.assert_called_once_with("/tmp/fake_config.yaml")
            mock_validate.assert_called_once_with(provider_fixture.config)
            mock_dataloader_catalog.create.assert_called_once()
            mock_embedder_factory.assert_called_once_with(provider_fixture.config)
            assert pipeline.config == provider_fixture.config

    @patch("vectordb.haystack.query_enhancement.indexing.base.DataloaderCatalog")
    @patch("vectordb.haystack.query_enhancement.indexing.base.create_document_embedder")
    @patch("vectordb.haystack.query_enhancement.indexing.base.validate_config")
    @patch("vectordb.haystack.query_enhancement.indexing.base.load_config")
    def test_indexing_pipeline_run(
        self,
        mock_load_config: MagicMock,
        mock_validate: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_dataloader_catalog: MagicMock,
        provider_fixture: ProviderFixture,
        mock_documents: list[Document],
    ) -> None:
        """Test that indexing pipeline runs and indexes documents."""
        with patch(provider_fixture.indexing_db_mock_path) as mock_db_cls:
            mock_load_config.return_value = provider_fixture.config

            mock_dataset = MagicMock()
            mock_dataset.to_haystack.return_value = mock_documents
            mock_loader = MagicMock()
            mock_loader.load.return_value = mock_dataset
            mock_dataloader_catalog.create.return_value = mock_loader

            mock_embedder = MagicMock()
            mock_embedder.run.return_value = {"documents": mock_documents}
            mock_embedder_factory.return_value = mock_embedder

            mock_db = MagicMock()
            mock_db.upsert.return_value = len(mock_documents)
            mock_db_cls.return_value = mock_db

            module_path, class_name = provider_fixture.indexing_pipeline_path.rsplit(
                ".", 1
            )
            module = import_module(module_path)
            pipeline_cls = getattr(module, class_name)
            pipeline = pipeline_cls("/tmp/fake_config.yaml")
            result = pipeline.run()

            assert result["documents_indexed"] == len(mock_documents)
            mock_loader.load.assert_called_once()
            mock_dataset.to_haystack.assert_called_once()
            mock_embedder.run.assert_called_once_with(documents=mock_documents)
            mock_db.upsert.assert_called_once()

    @patch("vectordb.haystack.query_enhancement.indexing.base.DataloaderCatalog")
    @patch("vectordb.haystack.query_enhancement.indexing.base.create_document_embedder")
    @patch("vectordb.haystack.query_enhancement.indexing.base.validate_config")
    @patch("vectordb.haystack.query_enhancement.indexing.base.load_config")
    def test_indexing_pipeline_creates_index(
        self,
        mock_load_config: MagicMock,
        mock_validate: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_dataloader_catalog: MagicMock,
        provider_fixture: ProviderFixture,
        mock_documents: list[Document],
    ) -> None:
        """Test that indexing pipeline creates index with correct dimension."""
        with patch(provider_fixture.indexing_db_mock_path) as mock_db_cls:
            mock_load_config.return_value = provider_fixture.config

            mock_dataset = MagicMock()
            mock_dataset.to_haystack.return_value = mock_documents
            mock_loader = MagicMock()
            mock_loader.load.return_value = mock_dataset
            mock_dataloader_catalog.create.return_value = mock_loader

            mock_embedder = MagicMock()
            mock_embedder.run.return_value = {"documents": mock_documents}
            mock_embedder_factory.return_value = mock_embedder

            mock_db = MagicMock()
            mock_db.upsert.return_value = len(mock_documents)
            mock_db_cls.return_value = mock_db

            module_path, class_name = provider_fixture.indexing_pipeline_path.rsplit(
                ".", 1
            )
            module = import_module(module_path)
            pipeline_cls = getattr(module, class_name)
            pipeline = pipeline_cls("/tmp/fake_config.yaml")
            pipeline.run()

            mock_db.create_index.assert_called_once_with(dimension=384)


class TestSearchPipelineBase:
    """Common tests for search pipeline implementations."""

    @patch("vectordb.haystack.query_enhancement.search.base.create_groq_generator")
    @patch("vectordb.haystack.query_enhancement.search.base.QueryEnhancer")
    @patch("vectordb.haystack.query_enhancement.search.base.create_text_embedder")
    @patch("vectordb.haystack.query_enhancement.search.base.validate_config")
    @patch("vectordb.haystack.query_enhancement.search.base.load_config")
    def test_search_pipeline_initialization(
        self,
        mock_load_config: MagicMock,
        mock_validate: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_query_enhancer_cls: MagicMock,
        mock_groq: MagicMock,
        provider_fixture: ProviderFixture,
    ) -> None:
        """Test that search pipeline initializes correctly."""
        with patch(provider_fixture.search_db_mock_path):
            mock_load_config.return_value = provider_fixture.config

            module_path, class_name = provider_fixture.search_pipeline_path.rsplit(
                ".", 1
            )
            module = import_module(module_path)
            pipeline_cls = getattr(module, class_name)
            pipeline = pipeline_cls("/tmp/fake_config.yaml")

            mock_load_config.assert_called_once_with("/tmp/fake_config.yaml")
            mock_validate.assert_called_once_with(provider_fixture.config)
            mock_embedder_factory.assert_called_once_with(provider_fixture.config)
            assert pipeline.config == provider_fixture.config
            rag_enabled = provider_fixture.config.get("rag", {}).get("enabled", False)
            if not rag_enabled:
                assert pipeline.rag_generator is None

    @patch("vectordb.haystack.query_enhancement.search.base.create_groq_generator")
    @patch("vectordb.haystack.query_enhancement.search.base.QueryEnhancer")
    @patch("vectordb.haystack.query_enhancement.search.base.create_text_embedder")
    @patch("vectordb.haystack.query_enhancement.search.base.validate_config")
    @patch("vectordb.haystack.query_enhancement.search.base.load_config")
    def test_search_pipeline_multi_query(
        self,
        mock_load_config: MagicMock,
        mock_validate: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_query_enhancer_cls: MagicMock,
        mock_groq: MagicMock,
        provider_fixture: ProviderFixture,
        mock_documents: list[Document],
    ) -> None:
        """Test search pipeline with multi-query enhancement."""
        with patch(provider_fixture.search_db_mock_path) as mock_db_cls:
            mock_load_config.return_value = provider_fixture.config

            mock_embedder = MagicMock()
            mock_embedder.run.return_value = {"embedding": [0.1] * 384}
            mock_embedder_factory.return_value = mock_embedder

            mock_db = MagicMock()
            mock_db.query.return_value = mock_documents
            mock_db_cls.return_value = mock_db

            mock_query_enhancer = MagicMock()
            mock_query_enhancer.generate_multi_queries.return_value = [
                "query 1",
                "query 2",
                "query 3",
            ]
            mock_query_enhancer_cls.return_value = mock_query_enhancer

            module_path, class_name = provider_fixture.search_pipeline_path.rsplit(
                ".", 1
            )
            module = import_module(module_path)
            pipeline_cls = getattr(module, class_name)
            pipeline = pipeline_cls("/tmp/fake_config.yaml")
            result = pipeline.run("test query", top_k=5)

            assert "documents" in result
            mock_query_enhancer.generate_multi_queries.assert_called_once_with(
                "test query", 3
            )

    @patch("vectordb.haystack.query_enhancement.search.base.create_groq_generator")
    @patch("vectordb.haystack.query_enhancement.search.base.QueryEnhancer")
    @patch("vectordb.haystack.query_enhancement.search.base.create_text_embedder")
    @patch("vectordb.haystack.query_enhancement.search.base.validate_config")
    @patch("vectordb.haystack.query_enhancement.search.base.load_config")
    def test_search_pipeline_with_rag(
        self,
        mock_load_config: MagicMock,
        mock_validate: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_query_enhancer_cls: MagicMock,
        mock_groq: MagicMock,
        provider_fixture: ProviderFixture,
        mock_documents: list[Document],
    ) -> None:
        """Test search pipeline with RAG generation enabled."""
        with patch(provider_fixture.search_db_mock_path) as mock_db_cls:
            mock_load_config.return_value = provider_fixture.config_with_rag

            mock_embedder = MagicMock()
            mock_embedder.run.return_value = {"embedding": [0.1] * 384}
            mock_embedder_factory.return_value = mock_embedder

            mock_db = MagicMock()
            mock_db.query.return_value = mock_documents
            mock_db_cls.return_value = mock_db

            mock_query_enhancer = MagicMock()
            mock_query_enhancer.generate_multi_queries.return_value = ["query 1"]
            mock_query_enhancer_cls.return_value = mock_query_enhancer

            mock_generator = MagicMock()
            mock_generator.run.return_value = {"replies": ["Generated answer"]}
            mock_groq.return_value = mock_generator

            module_path, class_name = provider_fixture.search_pipeline_path.rsplit(
                ".", 1
            )
            module = import_module(module_path)
            pipeline_cls = getattr(module, class_name)
            pipeline = pipeline_cls("/tmp/fake_config.yaml")
            result = pipeline.run("test query", top_k=5)

            assert "documents" in result
            assert "answer" in result
            assert result["answer"] == "Generated answer"

    @patch("vectordb.haystack.query_enhancement.search.base.create_groq_generator")
    @patch("vectordb.haystack.query_enhancement.search.base.QueryEnhancer")
    @patch("vectordb.haystack.query_enhancement.search.base.create_text_embedder")
    @patch("vectordb.haystack.query_enhancement.search.base.validate_config")
    @patch("vectordb.haystack.query_enhancement.search.base.load_config")
    def test_search_pipeline_without_rag(
        self,
        mock_load_config: MagicMock,
        mock_validate: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_query_enhancer_cls: MagicMock,
        mock_groq: MagicMock,
        provider_fixture: ProviderFixture,
        mock_documents: list[Document],
    ) -> None:
        """Test search pipeline without RAG enabled."""
        with patch(provider_fixture.search_db_mock_path) as mock_db_cls:
            config_no_rag = provider_fixture.config.copy()
            config_no_rag["rag"] = {"enabled": False}
            mock_load_config.return_value = config_no_rag

            mock_embedder = MagicMock()
            mock_embedder.run.return_value = {"embedding": [0.1] * 384}
            mock_embedder_factory.return_value = mock_embedder

            mock_query_enhancer = MagicMock()
            mock_query_enhancer.generate_multi_queries.return_value = ["query 1"]
            mock_query_enhancer_cls.return_value = mock_query_enhancer

            mock_db = MagicMock()
            mock_db.query.return_value = mock_documents
            mock_db_cls.return_value = mock_db

            module_path, class_name = provider_fixture.search_pipeline_path.rsplit(
                ".", 1
            )
            module = import_module(module_path)
            pipeline_cls = getattr(module, class_name)
            pipeline = pipeline_cls("/tmp/fake_config.yaml")
            result = pipeline.run("test query", top_k=5)

            assert "documents" in result
            assert "answer" not in result

    @patch("vectordb.haystack.query_enhancement.search.base.create_groq_generator")
    @patch("vectordb.haystack.query_enhancement.search.base.QueryEnhancer")
    @patch("vectordb.haystack.query_enhancement.search.base.create_text_embedder")
    @patch("vectordb.haystack.query_enhancement.search.base.validate_config")
    @patch("vectordb.haystack.query_enhancement.search.base.load_config")
    def test_search_pipeline_hyde_enhancement(
        self,
        mock_load_config: MagicMock,
        mock_validate: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_query_enhancer_cls: MagicMock,
        mock_groq: MagicMock,
        provider_fixture: ProviderFixture,
        mock_documents: list[Document],
    ) -> None:
        """Test search pipeline with HyDE enhancement."""
        with patch(provider_fixture.search_db_mock_path) as mock_db_cls:
            hyde_config = provider_fixture.config.copy()
            hyde_config["query_enhancement"] = {
                "type": "hyde",
                "num_hyde_docs": 2,
                "llm": {"model": "llama-3.3-70b-versatile", "api_key": "test-key"},
            }
            hyde_config["rag"] = {"enabled": False}
            mock_load_config.return_value = hyde_config

            mock_embedder = MagicMock()
            mock_embedder.run.return_value = {"embedding": [0.1] * 384}
            mock_embedder_factory.return_value = mock_embedder

            mock_db = MagicMock()
            mock_db.query.return_value = mock_documents
            mock_db_cls.return_value = mock_db

            mock_query_enhancer = MagicMock()
            mock_query_enhancer.generate_hypothetical_documents.return_value = [
                "hypothetical doc 1",
                "hypothetical doc 2",
            ]
            mock_query_enhancer_cls.return_value = mock_query_enhancer

            module_path, class_name = provider_fixture.search_pipeline_path.rsplit(
                ".", 1
            )
            module = import_module(module_path)
            pipeline_cls = getattr(module, class_name)
            pipeline = pipeline_cls("/tmp/fake_config.yaml")
            result = pipeline.run("test query", top_k=5)

            assert "documents" in result
            mock_query_enhancer.generate_hypothetical_documents.assert_called_once_with(
                "test query", 2
            )

    @patch("vectordb.haystack.query_enhancement.search.base.create_groq_generator")
    @patch("vectordb.haystack.query_enhancement.search.base.QueryEnhancer")
    @patch("vectordb.haystack.query_enhancement.search.base.create_text_embedder")
    @patch("vectordb.haystack.query_enhancement.search.base.validate_config")
    @patch("vectordb.haystack.query_enhancement.search.base.load_config")
    def test_search_pipeline_step_back_enhancement(
        self,
        mock_load_config: MagicMock,
        mock_validate: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_query_enhancer_cls: MagicMock,
        mock_groq: MagicMock,
        provider_fixture: ProviderFixture,
        mock_documents: list[Document],
    ) -> None:
        """Test search pipeline with step-back enhancement."""
        with patch(provider_fixture.search_db_mock_path) as mock_db_cls:
            step_back_config = provider_fixture.config.copy()
            step_back_config["query_enhancement"] = {
                "type": "step_back",
                "llm": {"model": "llama-3.3-70b-versatile", "api_key": "test-key"},
            }
            step_back_config["rag"] = {"enabled": False}
            mock_load_config.return_value = step_back_config

            mock_embedder = MagicMock()
            mock_embedder.run.return_value = {"embedding": [0.1] * 384}
            mock_embedder_factory.return_value = mock_embedder

            mock_db = MagicMock()
            mock_db.query.return_value = mock_documents
            mock_db_cls.return_value = mock_db

            mock_query_enhancer = MagicMock()
            mock_query_enhancer.generate_step_back_query.return_value = (
                "step back query"
            )
            mock_query_enhancer_cls.return_value = mock_query_enhancer

            module_path, class_name = provider_fixture.search_pipeline_path.rsplit(
                ".", 1
            )
            module = import_module(module_path)
            pipeline_cls = getattr(module, class_name)
            pipeline = pipeline_cls("/tmp/fake_config.yaml")
            result = pipeline.run("test query", top_k=5)

            assert "documents" in result
            mock_query_enhancer.generate_step_back_query.assert_called_once_with(
                "test query"
            )

    @patch("vectordb.haystack.query_enhancement.search.base.create_groq_generator")
    @patch("vectordb.haystack.query_enhancement.search.base.QueryEnhancer")
    @patch("vectordb.haystack.query_enhancement.search.base.create_text_embedder")
    @patch("vectordb.haystack.query_enhancement.search.base.validate_config")
    @patch("vectordb.haystack.query_enhancement.search.base.load_config")
    def test_search_pipeline_default_enhancement(
        self,
        mock_load_config: MagicMock,
        mock_validate: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_query_enhancer_cls: MagicMock,
        mock_groq: MagicMock,
        provider_fixture: ProviderFixture,
        mock_documents: list[Document],
    ) -> None:
        """Test search pipeline default branch when type is unknown."""
        with patch(provider_fixture.search_db_mock_path) as mock_db_cls:
            default_config = provider_fixture.config.copy()
            default_config["query_enhancement"] = {"type": "unknown"}
            default_config["rag"] = {"enabled": False}
            mock_load_config.return_value = default_config

            mock_embedder = MagicMock()
            mock_embedder.run.return_value = {"embedding": [0.1] * 384}
            mock_embedder_factory.return_value = mock_embedder

            mock_db = MagicMock()
            mock_db.query.return_value = mock_documents
            mock_db_cls.return_value = mock_db

            module_path, class_name = provider_fixture.search_pipeline_path.rsplit(
                ".", 1
            )
            module = import_module(module_path)
            pipeline_cls = getattr(module, class_name)
            pipeline = pipeline_cls("/tmp/fake_config.yaml")
            result = pipeline.run("test query", top_k=5)

            assert "documents" in result
            mock_db.query.assert_called_once()

    @patch("vectordb.haystack.query_enhancement.search.base.create_groq_generator")
    @patch("vectordb.haystack.query_enhancement.search.base.QueryEnhancer")
    @patch("vectordb.haystack.query_enhancement.search.base.create_text_embedder")
    @patch("vectordb.haystack.query_enhancement.search.base.validate_config")
    @patch("vectordb.haystack.query_enhancement.search.base.load_config")
    def test_search_pipeline_handles_search_failure(
        self,
        mock_load_config: MagicMock,
        mock_validate: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_query_enhancer_cls: MagicMock,
        mock_groq: MagicMock,
        provider_fixture: ProviderFixture,
        mock_documents: list[Document],
    ) -> None:
        """Test search pipeline logs when a search task fails."""
        with patch(provider_fixture.search_db_mock_path) as mock_db_cls:
            config_no_rag = provider_fixture.config.copy()
            config_no_rag["rag"] = {"enabled": False}
            mock_load_config.return_value = config_no_rag

            mock_embedder_factory.return_value = MagicMock()
            mock_db_cls.return_value = MagicMock()
            mock_query_enhancer = MagicMock()
            mock_query_enhancer.generate_multi_queries.return_value = ["good", "bad"]
            mock_query_enhancer_cls.return_value = mock_query_enhancer

            module_path, class_name = provider_fixture.search_pipeline_path.rsplit(
                ".", 1
            )
            module = import_module(module_path)
            pipeline_cls = getattr(module, class_name)
            pipeline = pipeline_cls("/tmp/fake_config.yaml")
            pipeline.logger = MagicMock()

            def search_side_effect(query: str, top_k: int) -> list[Document]:
                if query == "bad":
                    raise RuntimeError("boom")
                return mock_documents[:1]

            pipeline._search_single_query = MagicMock(side_effect=search_side_effect)
            result = pipeline.run("test query", top_k=5)

            assert len(result["documents"]) == 1
            pipeline.logger.error.assert_called_once()
            pipeline._search_single_query.assert_any_call("good", 5)
            pipeline._search_single_query.assert_any_call("bad", 5)

    @patch("vectordb.haystack.query_enhancement.search.base.create_groq_generator")
    @patch("vectordb.haystack.query_enhancement.search.base.QueryEnhancer")
    @patch("vectordb.haystack.query_enhancement.search.base.create_text_embedder")
    @patch("vectordb.haystack.query_enhancement.search.base.validate_config")
    @patch("vectordb.haystack.query_enhancement.search.base.load_config")
    def test_search_pipeline_handles_rag_failure(
        self,
        mock_load_config: MagicMock,
        mock_validate: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_query_enhancer_cls: MagicMock,
        mock_groq: MagicMock,
        provider_fixture: ProviderFixture,
        mock_documents: list[Document],
    ) -> None:
        """Test search pipeline returns documents when RAG generation fails."""
        with patch(provider_fixture.search_db_mock_path) as mock_db_cls:
            mock_load_config.return_value = provider_fixture.config_with_rag

            mock_embedder_factory.return_value = MagicMock()
            mock_db_cls.return_value = MagicMock()
            mock_query_enhancer = MagicMock()
            mock_query_enhancer.generate_multi_queries.return_value = ["query 1"]
            mock_query_enhancer_cls.return_value = mock_query_enhancer

            mock_generator = MagicMock()
            mock_generator.run.side_effect = RuntimeError("RAG failed")
            mock_groq.return_value = mock_generator

            module_path, class_name = provider_fixture.search_pipeline_path.rsplit(
                ".", 1
            )
            module = import_module(module_path)
            pipeline_cls = getattr(module, class_name)
            pipeline = pipeline_cls("/tmp/fake_config.yaml")
            pipeline.logger = MagicMock()
            pipeline._search_single_query = MagicMock(return_value=mock_documents[:1])

            result = pipeline.run("test query", top_k=5)

            assert "documents" in result
            assert "answer" not in result
            pipeline.logger.error.assert_called_once()
