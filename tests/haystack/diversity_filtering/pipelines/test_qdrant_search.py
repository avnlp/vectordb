"""Tests for Qdrant search pipeline in diversity filtering."""

from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.diversity_filtering.pipelines.qdrant_search import run_search


class TestRunSearch:
    """Tests for run_search function."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create a mock configuration."""
        config = MagicMock()
        config.embedding.model = "sentence-transformers/all-MiniLM-L6-v2"
        config.embedding.dimension = 384
        config.embedding.device = None
        config.index.name = "test_index"
        config.retrieval.top_k_candidates = 100
        config.diversity.algorithm = "maximum_margin_relevance"
        config.diversity.top_k = 10
        config.diversity.similarity_metric = "cosine"
        config.rag.enabled = False
        config.vectordb.qdrant.url = "http://localhost:6333"
        config.vectordb.qdrant.api_key = None
        return config

    @pytest.fixture
    def sample_candidates(self) -> list[Document]:
        """Create sample candidate documents."""
        return [
            Document(
                content=f"Candidate {i}",
                meta={"source": f"doc{i}"},
                score=0.9 - i * 0.05,
            )
            for i in range(20)
        ]

    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.qdrant_search.SentenceTransformersDiversityRanker"
    )
    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.qdrant_search.QdrantVectorDB"
    )
    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.qdrant_search.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.diversity_filtering.pipelines.qdrant_search.ConfigLoader")
    def test_run_search_success_no_diversity(
        self,
        mock_config_loader: MagicMock,
        mock_embedder_class: MagicMock,
        mock_db_class: MagicMock,
        mock_ranker_class: MagicMock,
        mock_config: MagicMock,
        sample_candidates: list[Document],
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """Test successful search without diversity filtering."""
        mock_config.diversity.algorithm = "greedy_diversity_order"

        config_file = tmp_path / "config.yaml"
        config_file.write_text("dummy: config")

        mock_config_loader.load.return_value = mock_config

        # Setup embedder
        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder.warm_up = MagicMock()

        # Setup database
        mock_db = MagicMock()
        mock_db_class.return_value = mock_db
        mock_db.search.return_value = sample_candidates

        # Setup ranker
        mock_ranker = MagicMock()
        mock_ranker_class.return_value = mock_ranker
        mock_ranker.warm_up = MagicMock()
        mock_ranker.run.return_value = {"documents": sample_candidates[:10]}

        result = run_search(str(config_file), "test query")

        assert result["query"] == "test query"
        assert result["num_diverse"] == 10
        assert len(result["documents"]) == 10
        assert result["answer"] is None

        mock_db.search.assert_called_once_with(
            query_embedding=[0.1] * 384,
            top_k=100,
        )

    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.qdrant_search.SentenceTransformersDiversityRanker"
    )
    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.qdrant_search.QdrantVectorDB"
    )
    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.qdrant_search.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.diversity_filtering.pipelines.qdrant_search.ConfigLoader")
    def test_run_search_with_mmr_diversity(
        self,
        mock_config_loader: MagicMock,
        mock_embedder_class: MagicMock,
        mock_db_class: MagicMock,
        mock_ranker_class: MagicMock,
        mock_config: MagicMock,
        sample_candidates: list[Document],
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """Test search with MMR diversity filtering."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("dummy: config")

        mock_config_loader.load.return_value = mock_config

        # Setup embedder
        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder.warm_up = MagicMock()

        # Setup database
        mock_db = MagicMock()
        mock_db_class.return_value = mock_db
        mock_db.search.return_value = sample_candidates

        # Setup ranker
        mock_ranker = MagicMock()
        mock_ranker_class.return_value = mock_ranker
        diverse_docs = sample_candidates[:5]
        mock_ranker.run.return_value = {"documents": diverse_docs}

        result = run_search(str(config_file), "test query")

        assert result["num_diverse"] == 5
        mock_ranker_class.assert_called_once_with(
            model="sentence-transformers/all-MiniLM-L6-v2",
            top_k=10,
            similarity="cosine",
            strategy="maximum_margin_relevance",
        )
        mock_ranker.run.assert_called_once_with(
            documents=sample_candidates,
            query="test query",
        )

    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.qdrant_search.SentenceTransformersDiversityRanker"
    )
    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.qdrant_search.QdrantVectorDB"
    )
    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.qdrant_search.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.diversity_filtering.pipelines.qdrant_search.ConfigLoader")
    def test_run_search_with_dot_product_similarity(
        self,
        mock_config_loader: MagicMock,
        mock_embedder_class: MagicMock,
        mock_db_class: MagicMock,
        mock_ranker_class: MagicMock,
        mock_config: MagicMock,
        sample_candidates: list[Document],
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """Test search with dot product similarity metric."""
        mock_config.diversity.similarity_metric = "dot_product"

        config_file = tmp_path / "config.yaml"
        config_file.write_text("dummy: config")

        mock_config_loader.load.return_value = mock_config

        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder.warm_up = MagicMock()

        mock_db = MagicMock()
        mock_db_class.return_value = mock_db
        mock_db.search.return_value = sample_candidates

        mock_ranker = MagicMock()
        mock_ranker_class.return_value = mock_ranker
        mock_ranker.run.return_value = {"documents": sample_candidates[:5]}

        run_search(str(config_file), "test query")

        mock_ranker_class.assert_called_once_with(
            model="sentence-transformers/all-MiniLM-L6-v2",
            top_k=10,
            similarity="dot_product",
            strategy="maximum_margin_relevance",
        )

    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.qdrant_search.QdrantVectorDB"
    )
    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.qdrant_search.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.diversity_filtering.pipelines.qdrant_search.ConfigLoader")
    def test_run_search_no_candidates(
        self,
        mock_config_loader: MagicMock,
        mock_embedder_class: MagicMock,
        mock_db_class: MagicMock,
        mock_config: MagicMock,
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """Test search with no candidates found."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("dummy: config")

        mock_config_loader.load.return_value = mock_config

        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder.warm_up = MagicMock()

        mock_db = MagicMock()
        mock_db_class.return_value = mock_db
        mock_db.search.return_value = []

        result = run_search(str(config_file), "test query")

        assert result["documents"] == []
        assert result["num_diverse"] == 0
        assert result["answer"] is None
        assert result["query"] == "test query"

    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.qdrant_search.OpenAIGenerator"
    )
    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.qdrant_search.PromptBuilder"
    )
    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.qdrant_search.format_documents"
    )
    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.qdrant_search.get_prompt_template"
    )
    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.qdrant_search.SentenceTransformersDiversityRanker"
    )
    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.qdrant_search.QdrantVectorDB"
    )
    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.qdrant_search.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.diversity_filtering.pipelines.qdrant_search.ConfigLoader")
    def test_run_search_with_rag(
        self,
        mock_config_loader: MagicMock,
        mock_embedder_class: MagicMock,
        mock_db_class: MagicMock,
        mock_ranker_class: MagicMock,
        mock_get_template: MagicMock,
        mock_format_docs: MagicMock,
        mock_prompt_builder_class: MagicMock,
        mock_generator_class: MagicMock,
        mock_config: MagicMock,
        sample_candidates: list[Document],
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """Test search with RAG enabled."""
        mock_config.rag.enabled = True
        mock_config.rag.provider = "groq"
        mock_config.rag.model = "llama-3.3-70b-versatile"
        mock_config.rag.temperature = 0.7
        mock_config.rag.max_tokens = 2048
        mock_config.dataset.name = "triviaqa"

        config_file = tmp_path / "config.yaml"
        config_file.write_text("dummy: config")

        mock_config_loader.load.return_value = mock_config

        # Setup embedder
        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder.warm_up = MagicMock()

        # Setup database
        mock_db = MagicMock()
        mock_db_class.return_value = mock_db
        mock_db.search.return_value = sample_candidates

        # Setup ranker
        mock_ranker = MagicMock()
        mock_ranker_class.return_value = mock_ranker
        diverse_docs = sample_candidates[:3]
        mock_ranker.run.return_value = {"documents": diverse_docs}

        # Setup RAG components
        mock_get_template.return_value = "Template with {query} and {documents}"
        mock_format_docs.return_value = "Formatted documents"

        mock_prompt_builder = MagicMock()
        mock_prompt_builder_class.return_value = mock_prompt_builder
        mock_prompt_builder.run.return_value = {"prompt": "Final prompt"}

        mock_generator = MagicMock()
        mock_generator_class.return_value = mock_generator
        mock_generator.run.return_value = {"replies": ["Generated answer"]}

        result = run_search(str(config_file), "test query")

        assert result["answer"] == "Generated answer"
        mock_get_template.assert_called_once_with("triviaqa")
        mock_generator_class.assert_called_once_with(
            api_key_env_var="GROQ_API_KEY",
            model="llama-3.3-70b-versatile",
            generation_kwargs={
                "temperature": 0.7,
                "max_tokens": 2048,
            },
        )

    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.qdrant_search.OpenAIGenerator"
    )
    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.qdrant_search.PromptBuilder"
    )
    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.qdrant_search.format_documents"
    )
    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.qdrant_search.get_prompt_template"
    )
    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.qdrant_search.SentenceTransformersDiversityRanker"
    )
    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.qdrant_search.QdrantVectorDB"
    )
    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.qdrant_search.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.diversity_filtering.pipelines.qdrant_search.ConfigLoader")
    def test_run_search_rag_error(
        self,
        mock_config_loader: MagicMock,
        mock_embedder_class: MagicMock,
        mock_db_class: MagicMock,
        mock_ranker_class: MagicMock,
        mock_get_template: MagicMock,
        mock_format_docs: MagicMock,
        mock_prompt_builder_class: MagicMock,
        mock_generator_class: MagicMock,
        mock_config: MagicMock,
        sample_candidates: list[Document],
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """Test search with RAG error handling."""
        mock_config.rag.enabled = True
        mock_config.rag.provider = "openai"

        config_file = tmp_path / "config.yaml"
        config_file.write_text("dummy: config")

        mock_config_loader.load.return_value = mock_config

        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder.warm_up = MagicMock()

        mock_db = MagicMock()
        mock_db_class.return_value = mock_db
        mock_db.search.return_value = sample_candidates

        mock_ranker = MagicMock()
        mock_ranker_class.return_value = mock_ranker
        mock_ranker.run.return_value = {"documents": sample_candidates[:3]}

        mock_generator = MagicMock()
        mock_generator_class.return_value = mock_generator
        mock_generator.run.side_effect = Exception("API Error")

        result = run_search(str(config_file), "test query")

        assert "Error generating answer" in result["answer"]
        assert "API Error" in result["answer"]

    @patch("vectordb.haystack.diversity_filtering.pipelines.qdrant_search.ConfigLoader")
    def test_run_search_config_not_found(
        self,
        mock_config_loader: MagicMock,
    ) -> None:
        """Test search with missing config file."""
        mock_config_loader.load.side_effect = FileNotFoundError("Config not found")

        with pytest.raises(FileNotFoundError, match="Config not found"):
            run_search("/nonexistent/config.yaml", "test query")

    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.qdrant_search.SentenceTransformersDiversityRanker"
    )
    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.qdrant_search.QdrantVectorDB"
    )
    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.qdrant_search.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.diversity_filtering.pipelines.qdrant_search.ConfigLoader")
    def test_run_search_db_initialization(
        self,
        mock_config_loader: MagicMock,
        mock_embedder_class: MagicMock,
        mock_db_class: MagicMock,
        mock_ranker_class: MagicMock,
        mock_config: MagicMock,
        sample_candidates: list[Document],
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """Test QdrantVectorDB initialization in search."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("dummy: config")

        mock_config_loader.load.return_value = mock_config

        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder.warm_up = MagicMock()

        mock_db = MagicMock()
        mock_db_class.return_value = mock_db
        mock_db.search.return_value = sample_candidates

        mock_ranker = MagicMock()
        mock_ranker_class.return_value = mock_ranker
        mock_ranker.run.return_value = {"documents": sample_candidates[:5]}

        run_search(str(config_file), "test query")

        mock_db_class.assert_called_once_with(
            url="http://localhost:6333",
            api_key=None,
            index="test_index",
            embedding_dim=384,
        )

    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.qdrant_search.SentenceTransformersDiversityRanker"
    )
    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.qdrant_search.QdrantVectorDB"
    )
    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.qdrant_search.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.diversity_filtering.pipelines.qdrant_search.ConfigLoader")
    def test_run_search_with_api_key(
        self,
        mock_config_loader: MagicMock,
        mock_embedder_class: MagicMock,
        mock_db_class: MagicMock,
        mock_ranker_class: MagicMock,
        mock_config: MagicMock,
        sample_candidates: list[Document],
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """Test search with API key."""
        mock_config.vectordb.qdrant.api_key = "secret-api-key"

        config_file = tmp_path / "config.yaml"
        config_file.write_text("dummy: config")

        mock_config_loader.load.return_value = mock_config

        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder.warm_up = MagicMock()

        mock_db = MagicMock()
        mock_db_class.return_value = mock_db
        mock_db.search.return_value = sample_candidates

        mock_ranker = MagicMock()
        mock_ranker_class.return_value = mock_ranker
        mock_ranker.run.return_value = {"documents": sample_candidates[:5]}

        run_search(str(config_file), "test query")

        mock_db_class.assert_called_once_with(
            url="http://localhost:6333",
            api_key="secret-api-key",
            index="test_index",
            embedding_dim=384,
        )
