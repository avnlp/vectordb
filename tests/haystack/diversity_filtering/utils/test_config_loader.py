"""Additional tests for config loader module to improve coverage."""

import os
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from vectordb.haystack.diversity_filtering.utils.config_loader import (
    ConfigLoader,
    DatasetConfig,
    DiversityConfig,
    DiversityFilteringConfig,
    EmbeddingConfig,
    IndexConfig,
    MilvusConfig,
    PineconeConfig,
    QdrantConfig,
    RetrievalConfig,
    VectorDBConfig,
    WeaviateConfig,
)


class TestConfigModels:
    """Test individual configuration model classes."""

    def test_dataset_config_defaults(self) -> None:
        """Test DatasetConfig default values."""
        config = DatasetConfig(name="triviaqa")
        assert config.name == "triviaqa"
        assert config.split == "test"
        assert config.max_documents is None

    def test_dataset_config_all_datasets(self) -> None:
        """Test DatasetConfig with all valid dataset names."""
        for name in ["triviaqa", "arc", "popqa", "factscore", "earnings_calls"]:
            config = DatasetConfig(name=name)
            assert config.name == name

    def test_dataset_config_invalid_dataset(self) -> None:
        """Test DatasetConfig with invalid dataset name."""
        with pytest.raises(ValidationError):
            DatasetConfig(name="invalid_dataset")

    def test_embedding_config_defaults(self) -> None:
        """Test EmbeddingConfig default values."""
        config = EmbeddingConfig()
        assert config.model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.dimension == 384
        assert config.batch_size == 32
        assert config.device is None

    def test_embedding_config_custom(self) -> None:
        """Test EmbeddingConfig with custom values."""
        config = EmbeddingConfig(
            model="custom-model",
            dimension=768,
            batch_size=64,
            device="cuda",
        )
        assert config.model == "custom-model"
        assert config.dimension == 768
        assert config.batch_size == 64
        assert config.device == "cuda"

    def test_index_config(self) -> None:
        """Test IndexConfig."""
        config = IndexConfig(name="my_index")
        assert config.name == "my_index"
        assert (
            config.recreate is False
        )  # Default is False for safe incremental indexing

    def test_index_config_recreate_false(self) -> None:
        """Test IndexConfig with recreate=False for incremental indexing."""
        config = IndexConfig(name="my_index", recreate=False)
        assert config.name == "my_index"
        assert config.recreate is False

    def test_index_config_recreate_true(self) -> None:
        """Test IndexConfig with recreate=True for destructive reindexing."""
        config = IndexConfig(name="my_index", recreate=True)
        assert config.name == "my_index"
        assert config.recreate is True

    def test_retrieval_config_defaults(self) -> None:
        """Test RetrievalConfig default values."""
        config = RetrievalConfig()
        assert config.top_k_candidates == 100

    def test_diversity_config_defaults(self) -> None:
        """Test DiversityConfig default values."""
        config = DiversityConfig()
        assert config.algorithm == "maximum_margin_relevance"
        assert config.top_k == 10
        assert config.mmr_lambda == 0.5
        assert config.similarity_metric == "cosine"

    def test_diversity_config_invalid_algorithm(self) -> None:
        """Test DiversityConfig with invalid algorithm."""
        with pytest.raises(ValidationError):
            DiversityConfig(algorithm="invalid_algorithm")

    def test_diversity_config_invalid_similarity(self) -> None:
        """Test DiversityConfig with invalid similarity metric."""
        with pytest.raises(ValidationError):
            DiversityConfig(similarity_metric="euclidean")

    def test_diversity_config_mmr_lambda_bounds(self) -> None:
        """Test DiversityConfig MMR lambda bounds."""
        # Valid values
        DiversityConfig(mmr_lambda=0.0)
        DiversityConfig(mmr_lambda=1.0)
        DiversityConfig(mmr_lambda=0.5)

        # Invalid values
        with pytest.raises(ValidationError):
            DiversityConfig(mmr_lambda=-0.1)
        with pytest.raises(ValidationError):
            DiversityConfig(mmr_lambda=1.1)

    def test_diversity_config_top_k_bounds(self) -> None:
        """Test DiversityConfig top_k bounds."""
        with pytest.raises(ValidationError):
            DiversityConfig(top_k=0)
        with pytest.raises(ValidationError):
            DiversityConfig(top_k=-1)

    def test_qdrant_config(self) -> None:
        """Test QdrantConfig."""
        config = QdrantConfig(url="http://localhost:6333")
        assert config.url == "http://localhost:6333"
        assert config.api_key is None
        assert config.timeout == 30

        config_with_key = QdrantConfig(
            url="http://qdrant:6333", api_key="secret", timeout=60
        )
        assert config_with_key.api_key == "secret"
        assert config_with_key.timeout == 60

    def test_pinecone_config(self) -> None:
        """Test PineconeConfig."""
        config = PineconeConfig(api_key="test-key", index_name="test-index")
        assert config.api_key == "test-key"
        assert config.index_name == "test-index"
        assert config.environment == "production"

    def test_weaviate_config(self) -> None:
        """Test WeaviateConfig."""
        config = WeaviateConfig(url="http://localhost:8080")
        assert config.url == "http://localhost:8080"
        assert config.api_key is None

    def test_milvus_config_defaults(self) -> None:
        """Test MilvusConfig default values."""
        config = MilvusConfig()
        assert config.host == "localhost"
        assert config.port == 19530
        assert config.db_name == "default"

    def test_vectordb_config_qdrant(self) -> None:
        """Test VectorDBConfig with Qdrant."""
        config = VectorDBConfig(
            type="qdrant",
            qdrant=QdrantConfig(url="http://localhost:6333"),
        )
        assert config.type == "qdrant"
        assert config.qdrant is not None
        assert config.qdrant.url == "http://localhost:6333"

    def test_vectordb_config_pinecone(self) -> None:
        """Test VectorDBConfig with Pinecone."""
        config = VectorDBConfig(
            type="pinecone",
            pinecone=PineconeConfig(api_key="key", index_name="idx"),
        )
        assert config.type == "pinecone"
        assert config.pinecone is not None

    def test_vectordb_config_weaviate(self) -> None:
        """Test VectorDBConfig with Weaviate."""
        config = VectorDBConfig(
            type="weaviate",
            weaviate=WeaviateConfig(url="http://localhost:8080"),
        )
        assert config.type == "weaviate"
        assert config.weaviate is not None

    def test_vectordb_config_chroma(self) -> None:
        """Test VectorDBConfig with Chroma."""
        from vectordb.haystack.diversity_filtering.utils.config_loader import (
            ChromaConfig,
        )

        config = VectorDBConfig(
            type="chroma",
            chroma=ChromaConfig(host="localhost", port=8000),
        )
        assert config.type == "chroma"
        assert config.chroma is not None

    def test_vectordb_config_milvus(self) -> None:
        """Test VectorDBConfig with Milvus."""
        config = VectorDBConfig(
            type="milvus",
            milvus=MilvusConfig(host="localhost", port=19530),
        )
        assert config.type == "milvus"
        assert config.milvus is not None

    def test_vectordb_config_invalid_type(self) -> None:
        """Test VectorDBConfig with invalid type."""
        with pytest.raises(ValidationError):
            VectorDBConfig(type="elasticsearch")


class TestConfigLoaderEnvVars:
    """Test ConfigLoader environment variable resolution."""

    def test_resolve_env_vars_in_dict(self) -> None:
        """Test resolving env vars in nested dict."""
        os.environ["TEST_VAR1"] = "value1"
        os.environ["TEST_VAR2"] = "value2"

        data = {
            "key1": "${TEST_VAR1}",
            "key2": {
                "nested": "${TEST_VAR2}",
                "normal": "unchanged",
            },
            "key3": ["${TEST_VAR1}", "normal"],
        }

        result = ConfigLoader._resolve_env_vars(data)

        assert result["key1"] == "value1"
        assert result["key2"]["nested"] == "value2"
        assert result["key2"]["normal"] == "unchanged"
        assert result["key3"] == ["value1", "normal"]

    def test_resolve_env_vars_missing(self) -> None:
        """Test resolving missing env vars."""
        # Ensure variable doesn't exist
        if "MISSING_VAR_XYZ" in os.environ:
            del os.environ["MISSING_VAR_XYZ"]

        data = {"key": "${MISSING_VAR_XYZ}"}
        with pytest.raises(
            ValueError,
            match="Required environment variable 'MISSING_VAR_XYZ' is not set",
        ):
            ConfigLoader._resolve_env_vars(data)

    def test_resolve_env_vars_non_string(self) -> None:
        """Test resolving env vars with non-string values."""
        data = {
            "int": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
        }
        result = ConfigLoader._resolve_env_vars(data)
        assert result["int"] == 42
        assert result["float"] == 3.14
        assert result["bool"] is True
        assert result["none"] is None

    def test_resolve_env_vars_partial_string(self) -> None:
        """Test that partial env var strings are not resolved."""
        os.environ["TEST_PARTIAL"] = "partial_value"

        data = {
            "prefix": "prefix_${TEST_PARTIAL}",
            "suffix": "${TEST_PARTIAL}_suffix",
            "middle": "before_${TEST_PARTIAL}_after",
        }

        result = ConfigLoader._resolve_env_vars(data)
        # Only exact ${VAR} patterns should be resolved
        assert result["prefix"] == "prefix_${TEST_PARTIAL}"
        assert result["suffix"] == "${TEST_PARTIAL}_suffix"
        assert result["middle"] == "before_${TEST_PARTIAL}_after"


class TestConfigLoaderLoad:
    """Test ConfigLoader.load method."""

    def test_load_non_dict_yaml(self, tmp_path: Path) -> None:
        """Test loading YAML that doesn't contain a dict."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("- item1\n- item2")  # List instead of dict

        with pytest.raises(ValueError, match="Expected YAML to contain dict"):
            ConfigLoader.load(str(config_file))

    def test_load_empty_yaml(self, tmp_path: Path) -> None:
        """Test loading empty YAML file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")

        # Empty YAML loads as None
        with pytest.raises(ValueError, match="Expected YAML to contain dict"):
            ConfigLoader.load(str(config_file))

    def test_load_with_env_substitution(self, tmp_path: Path) -> None:
        """Test loading config with env var substitution."""
        os.environ["TEST_QDRANT_URL"] = "http://custom:6333"

        config_dict = {
            "dataset": {"name": "triviaqa"},
            "index": {"name": "test"},
            "vectordb": {
                "type": "qdrant",
                "qdrant": {"url": "${TEST_QDRANT_URL}"},
            },
        }

        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_dict))

        config = ConfigLoader.load(str(config_file))
        assert config.vectordb.qdrant.url == "http://custom:6333"


class TestDiversityFilteringConfig:
    """Test DiversityFilteringConfig complete configuration."""

    def test_full_config_creation(self) -> None:
        """Test creating full configuration."""
        config = DiversityFilteringConfig(
            dataset=DatasetConfig(name="triviaqa"),
            embedding=EmbeddingConfig(model="custom-model", dimension=768),
            index=IndexConfig(name="my_index"),
            retrieval=RetrievalConfig(top_k_candidates=50),
            diversity=DiversityConfig(algorithm="clustering", top_k=5),
            vectordb=VectorDBConfig(
                type="qdrant",
                qdrant=QdrantConfig(url="http://localhost:6333"),
            ),
        )

        assert config.dataset.name == "triviaqa"
        assert config.embedding.dimension == 768
        assert config.index.name == "my_index"
        assert config.retrieval.top_k_candidates == 50
        assert config.diversity.algorithm == "clustering"
        assert config.diversity.top_k == 5
        assert config.vectordb.type == "qdrant"

    def test_config_with_rag(self) -> None:
        """Test configuration with RAG enabled."""
        from vectordb.haystack.diversity_filtering.utils.config_loader import RAGConfig

        config = DiversityFilteringConfig(
            dataset=DatasetConfig(name="triviaqa"),
            index=IndexConfig(name="test"),
            vectordb=VectorDBConfig(
                type="qdrant",
                qdrant=QdrantConfig(url="http://localhost:6333"),
            ),
            rag=RAGConfig(
                enabled=True,
                provider="openai",
                model="gpt-4",
                temperature=0.5,
                max_tokens=1024,
            ),
        )

        assert config.rag.enabled is True
        assert config.rag.provider == "openai"
        assert config.rag.model == "gpt-4"
        assert config.rag.temperature == 0.5
        assert config.rag.max_tokens == 1024

    def test_rag_temperature_bounds(self) -> None:
        """Test RAG temperature bounds."""
        from vectordb.haystack.diversity_filtering.utils.config_loader import RAGConfig

        # Valid values
        RAGConfig(temperature=0.0)
        RAGConfig(temperature=2.0)
        RAGConfig(temperature=1.0)

        # Invalid values
        with pytest.raises(ValidationError):
            RAGConfig(temperature=-0.1)
        with pytest.raises(ValidationError):
            RAGConfig(temperature=2.1)

    def test_rag_invalid_provider(self) -> None:
        """Test RAG with invalid provider."""
        from vectordb.haystack.diversity_filtering.utils.config_loader import RAGConfig

        with pytest.raises(ValidationError):
            RAGConfig(provider="anthropic")


class TestConfigLoaderDict:
    """Test ConfigLoader.load_dict method."""

    def test_load_dict_minimal(self) -> None:
        """Test loading minimal valid config dict."""
        data = {
            "dataset": {"name": "triviaqa"},
            "index": {"name": "test"},
            "vectordb": {
                "type": "qdrant",
                "qdrant": {"url": "http://localhost:6333"},
            },
        }

        config = ConfigLoader.load_dict(data)
        assert isinstance(config, DiversityFilteringConfig)
        assert config.dataset.name == "triviaqa"

    def test_load_dict_with_env_vars(self) -> None:
        """Test load_dict with environment variables."""
        os.environ["TEST_API_KEY"] = "secret123"

        data = {
            "dataset": {"name": "arc"},
            "index": {"name": "test"},
            "vectordb": {
                "type": "pinecone",
                "pinecone": {
                    "api_key": "${TEST_API_KEY}",
                    "index_name": "test-idx",
                },
            },
        }

        config = ConfigLoader.load_dict(data)
        assert config.vectordb.pinecone.api_key == "secret123"


class TestChromaConfig:
    """Test ChromaConfig model."""

    def test_chroma_config_defaults(self) -> None:
        """Test ChromaConfig default values."""
        from vectordb.haystack.diversity_filtering.utils.config_loader import (
            ChromaConfig,
        )

        config = ChromaConfig()
        assert config.host == "localhost"
        assert config.port == 8000
        assert config.is_persistent is False

    def test_chroma_config_custom(self) -> None:
        """Test ChromaConfig with custom values."""
        from vectordb.haystack.diversity_filtering.utils.config_loader import (
            ChromaConfig,
        )

        config = ChromaConfig(host="chroma-server", port=9000, is_persistent=True)
        assert config.host == "chroma-server"
        assert config.port == 9000
        assert config.is_persistent is True
