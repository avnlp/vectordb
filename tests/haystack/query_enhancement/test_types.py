"""Tests for query enhancement configuration TypedDicts."""

from vectordb.haystack.query_enhancement.utils.types import (
    DataLoaderConfig,
    EmbeddingConfig,
    LLMConfig,
    QueryEnhancementConfig,
    QueryEnhancementPipelineConfig,
    VectorDBConfig,
)


class TestTypes:
    """Test suite for query enhancement configuration TypedDicts."""

    def test_data_loader_config(self) -> None:
        """Test DataLoaderConfig TypedDict."""
        config: DataLoaderConfig = {
            "type": "arc",
            "params": {"split": "train", "batch_size": 32},
        }

        assert config["type"] == "arc"
        assert config["params"] == {"split": "train", "batch_size": 32}

        # Test with different params
        config2: DataLoaderConfig = {"type": "csv", "params": {}}
        assert config2["type"] == "csv"
        assert config2["params"] == {}

    def test_embedding_config(self) -> None:
        """Test EmbeddingConfig TypedDict."""
        config: EmbeddingConfig = {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "params": {"pooling": "mean", "normalize": True},
        }

        assert config["model"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert config["params"] == {"pooling": "mean", "normalize": True}

        # Test with minimal params
        config2: EmbeddingConfig = {
            "model": "openai/text-embedding-ada-002",
            "params": {},
        }
        assert config2["model"] == "openai/text-embedding-ada-002"
        assert config2["params"] == {}

    def test_llm_config(self) -> None:
        """Test LLMConfig TypedDict."""
        # Test with API key
        config_with_key: LLMConfig = {
            "model": "gpt-3.5-turbo",
            "api_key": "test-key-123",
        }
        assert config_with_key["model"] == "gpt-3.5-turbo"
        assert config_with_key["api_key"] == "test-key-123"

        # Test with None API key
        config_without_key: LLMConfig = {"model": "local-model", "api_key": None}
        assert config_without_key["model"] == "local-model"
        assert config_without_key["api_key"] is None

    def test_query_enhancement_config_multi_query(self) -> None:
        """Test QueryEnhancementConfig with multi_query type."""
        llm_config: LLMConfig = {"model": "gpt-3.5-turbo", "api_key": "test-key"}

        config: QueryEnhancementConfig = {
            "type": "multi_query",
            "num_queries": 3,
            "num_hyde_docs": 1,
            "llm": llm_config,
            "fusion_method": "rrf",
            "rrf_k": 20,
            "top_k": 10,
        }

        assert config["type"] == "multi_query"
        assert config["num_queries"] == 3
        assert config["num_hyde_docs"] == 1
        assert config["llm"] == llm_config
        assert config["fusion_method"] == "rrf"
        assert config["rrf_k"] == 20
        assert config["top_k"] == 10

    def test_query_enhancement_config_hyde(self) -> None:
        """Test QueryEnhancementConfig with hyde type."""
        llm_config: LLMConfig = {"model": "claude-2", "api_key": "anthropic-key"}

        config: QueryEnhancementConfig = {
            "type": "hyde",
            "num_queries": 1,
            "num_hyde_docs": 5,
            "llm": llm_config,
            "fusion_method": "weighted",
            "rrf_k": 60,
            "top_k": 20,
        }

        assert config["type"] == "hyde"
        assert config["num_queries"] == 1
        assert config["num_hyde_docs"] == 5
        assert config["llm"] == llm_config
        assert config["fusion_method"] == "weighted"
        assert config["rrf_k"] == 60
        assert config["top_k"] == 20

    def test_query_enhancement_config_step_back(self) -> None:
        """Test QueryEnhancementConfig with step_back type."""
        llm_config: LLMConfig = {"model": "gpt-4", "api_key": None}

        config: QueryEnhancementConfig = {
            "type": "step_back",
            "num_queries": 2,
            "num_hyde_docs": 3,
            "llm": llm_config,
            "fusion_method": "rrf",
            "rrf_k": 50,
            "top_k": 15,
        }

        assert config["type"] == "step_back"
        assert config["num_queries"] == 2
        assert config["num_hyde_docs"] == 3
        assert config["llm"] == llm_config
        assert config["fusion_method"] == "rrf"
        assert config["rrf_k"] == 50
        assert config["top_k"] == 15

    def test_vector_db_config(self) -> None:
        """Test VectorDBConfig TypedDict."""
        config: VectorDBConfig = {
            "api_key": "vector-db-key",
            "index_name": "test-index",
            "namespace": "test-namespace",
        }

        assert config["api_key"] == "vector-db-key"
        assert config["index_name"] == "test-index"
        assert config["namespace"] == "test-namespace"

        # Test with different values
        config2: VectorDBConfig = {
            "api_key": "another-key",
            "index_name": "prod-index",
            "namespace": "production",
        }
        assert config2["api_key"] == "another-key"
        assert config2["index_name"] == "prod-index"
        assert config2["namespace"] == "production"

    def test_query_enhancement_pipeline_config(self) -> None:
        """Test QueryEnhancementPipelineConfig TypedDict with all required configs."""
        # Create all nested configs
        dataloader_config: DataLoaderConfig = {
            "type": "arc",
            "params": {"split": "train"},
        }

        embedding_config: EmbeddingConfig = {
            "model": "all-MiniLM-L6-v2",
            "params": {"normalize": True},
        }

        llm_config: LLMConfig = {"model": "gpt-3.5-turbo", "api_key": "test-api-key"}

        query_enhancement_config: QueryEnhancementConfig = {
            "type": "multi_query",
            "num_queries": 3,
            "num_hyde_docs": 1,
            "llm": llm_config,
            "fusion_method": "rrf",
            "rrf_k": 20,
            "top_k": 10,
        }

        vector_db_config: VectorDBConfig = {
            "api_key": "db-key",
            "index_name": "test-index",
            "namespace": "test-ns",
        }

        pipeline_config: QueryEnhancementPipelineConfig = {
            "dataloader": dataloader_config,
            "embeddings": embedding_config,
            "query_enhancement": query_enhancement_config,
            "pinecone": vector_db_config,
            "qdrant": vector_db_config,
            "milvus": vector_db_config,
            "chroma": vector_db_config,
            "weaviate": vector_db_config,
            "logging": {"level": "INFO", "format": "json"},
            "rag": {"chunk_size": 512, "overlap": 128},
        }

        # Validate all fields
        assert pipeline_config["dataloader"] == dataloader_config
        assert pipeline_config["embeddings"] == embedding_config
        assert pipeline_config["query_enhancement"] == query_enhancement_config
        assert pipeline_config["pinecone"] == vector_db_config
        assert pipeline_config["qdrant"] == vector_db_config
        assert pipeline_config["milvus"] == vector_db_config
        assert pipeline_config["chroma"] == vector_db_config
        assert pipeline_config["weaviate"] == vector_db_config
        assert pipeline_config["logging"] == {"level": "INFO", "format": "json"}
        assert pipeline_config["rag"] == {"chunk_size": 512, "overlap": 128}

    def test_query_enhancement_pipeline_config_different_values(self) -> None:
        """Test QueryEnhancementPipelineConfig with different configuration values."""
        # Create configs with different values
        dataloader_config: DataLoaderConfig = {
            "type": "csv",
            "params": {"delimiter": ",", "header": 0},
        }

        embedding_config: EmbeddingConfig = {
            "model": "openai/text-embedding-ada-002",
            "params": {"dimensions": 1536},
        }

        llm_config: LLMConfig = {"model": "claude-3-opus", "api_key": None}

        query_enhancement_config: QueryEnhancementConfig = {
            "type": "hyde",
            "num_queries": 1,
            "num_hyde_docs": 5,
            "llm": llm_config,
            "fusion_method": "weighted",
            "rrf_k": 100,
            "top_k": 5,
        }

        vector_db_config: VectorDBConfig = {
            "api_key": "different-db-key",
            "index_name": "different-index",
            "namespace": "different-namespace",
        }

        pipeline_config: QueryEnhancementPipelineConfig = {
            "dataloader": dataloader_config,
            "embeddings": embedding_config,
            "query_enhancement": query_enhancement_config,
            "pinecone": vector_db_config,
            "qdrant": vector_db_config,
            "milvus": vector_db_config,
            "chroma": vector_db_config,
            "weaviate": vector_db_config,
            "logging": {"level": "DEBUG"},
            "rag": {"strategy": "adaptive", "max_chunks": 10},
        }

        # Validate all fields
        assert pipeline_config["dataloader"] == dataloader_config
        assert pipeline_config["embeddings"] == embedding_config
        assert pipeline_config["query_enhancement"] == query_enhancement_config
        assert pipeline_config["logging"] == {"level": "DEBUG"}
        assert pipeline_config["rag"] == {"strategy": "adaptive", "max_chunks": 10}

    def test_literal_values_validation(self) -> None:
        """Test that literal values are correctly constrained."""
        llm_config: LLMConfig = {"model": "test-model", "api_key": "test-key"}

        # Valid type values
        multi_query_config: QueryEnhancementConfig = {
            "type": "multi_query",
            "num_queries": 1,
            "num_hyde_docs": 1,
            "llm": llm_config,
            "fusion_method": "rrf",
            "rrf_k": 20,
            "top_k": 10,
        }
        assert multi_query_config["type"] == "multi_query"

        hyde_config: QueryEnhancementConfig = {
            "type": "hyde",
            "num_queries": 1,
            "num_hyde_docs": 1,
            "llm": llm_config,
            "fusion_method": "weighted",
            "rrf_k": 20,
            "top_k": 10,
        }
        assert hyde_config["type"] == "hyde"

        step_back_config: QueryEnhancementConfig = {
            "type": "step_back",
            "num_queries": 1,
            "num_hyde_docs": 1,
            "llm": llm_config,
            "fusion_method": "rrf",
            "rrf_k": 20,
            "top_k": 10,
        }
        assert step_back_config["type"] == "step_back"

        # Valid fusion methods
        rrf_config: QueryEnhancementConfig = {
            "type": "multi_query",
            "num_queries": 1,
            "num_hyde_docs": 1,
            "llm": llm_config,
            "fusion_method": "rrf",
            "rrf_k": 20,
            "top_k": 10,
        }
        assert rrf_config["fusion_method"] == "rrf"

        weighted_config: QueryEnhancementConfig = {
            "type": "multi_query",
            "num_queries": 1,
            "num_hyde_docs": 1,
            "llm": llm_config,
            "fusion_method": "weighted",
            "rrf_k": 20,
            "top_k": 10,
        }
        assert weighted_config["fusion_method"] == "weighted"

    def test_edge_cases_and_validation(self) -> None:
        """Test edge cases and validation for TypedDicts."""
        # Test with minimum integer values
        llm_config: LLMConfig = {"model": "test-model", "api_key": "test-key"}

        min_values_config: QueryEnhancementConfig = {
            "type": "multi_query",
            "num_queries": 1,  # Minimum positive integer
            "num_hyde_docs": 0,  # Minimum value
            "llm": llm_config,
            "fusion_method": "rrf",
            "rrf_k": 1,  # Minimum positive integer
            "top_k": 1,  # Minimum positive integer
        }
        assert min_values_config["num_queries"] == 1
        assert min_values_config["num_hyde_docs"] == 0
        assert min_values_config["rrf_k"] == 1
        assert min_values_config["top_k"] == 1

        # Test with maximum realistic values
        max_values_config: QueryEnhancementConfig = {
            "type": "hyde",
            "num_queries": 100,  # Large number of queries
            "num_hyde_docs": 50,  # Large number of hyde docs
            "llm": llm_config,
            "fusion_method": "weighted",
            "rrf_k": 1000,  # Large k value
            "top_k": 100,  # Large top_k value
        }
        assert max_values_config["num_queries"] == 100
        assert max_values_config["num_hyde_docs"] == 50
        assert max_values_config["rrf_k"] == 1000
        assert max_values_config["top_k"] == 100

        # Test with complex nested parameters
        complex_params_config: DataLoaderConfig = {
            "type": "complex-dataset",
            "params": {
                "nested_param": {
                    "deeply_nested": {
                        "value": "deep_value",
                        "list_param": [1, 2, 3, {"key": "value"}],
                    }
                },
                "list_of_dicts": [{"a": 1}, {"b": 2}],
                "mixed_types": [1, "string", True, None, 3.14],
            },
        }
        expected_params = {
            "nested_param": {
                "deeply_nested": {
                    "value": "deep_value",
                    "list_param": [1, 2, 3, {"key": "value"}],
                }
            },
            "list_of_dicts": [{"a": 1}, {"b": 2}],
            "mixed_types": [1, "string", True, None, 3.14],
        }
        assert complex_params_config["params"] == expected_params

    def test_empty_and_none_values(self) -> None:
        """Test handling of empty and None values in TypedDicts."""
        # Test with empty strings and empty dicts
        empty_config: VectorDBConfig = {
            "api_key": "",  # Empty string
            "index_name": "",
            "namespace": "",
        }
        assert empty_config["api_key"] == ""
        assert empty_config["index_name"] == ""
        assert empty_config["namespace"] == ""

        # Test with empty params in other configs
        empty_params_config: DataLoaderConfig = {
            "type": "test-type",
            "params": {},  # Empty dict
        }
        assert empty_params_config["params"] == {}

        # Test with complex empty structures in logging and rag
        complex_empty_config: QueryEnhancementPipelineConfig = {
            "dataloader": {"type": "test", "params": {}},
            "embeddings": {"model": "test", "params": {}},
            "query_enhancement": {
                "type": "multi_query",
                "num_queries": 1,
                "num_hyde_docs": 1,
                "llm": {"model": "test", "api_key": "key"},
                "fusion_method": "rrf",
                "rrf_k": 20,
                "top_k": 10,
            },
            "pinecone": {"api_key": "key", "index_name": "idx", "namespace": "ns"},
            "qdrant": {"api_key": "key", "index_name": "idx", "namespace": "ns"},
            "milvus": {"api_key": "key", "index_name": "idx", "namespace": "ns"},
            "chroma": {"api_key": "key", "index_name": "idx", "namespace": "ns"},
            "weaviate": {"api_key": "key", "index_name": "idx", "namespace": "ns"},
            "logging": {},  # Empty logging dict
            "rag": {},  # Empty rag dict
        }
        assert complex_empty_config["logging"] == {}
        assert complex_empty_config["rag"] == {}

    def test_special_character_values(self) -> None:
        """Test TypedDicts with special characters and unicode values."""
        special_config: VectorDBConfig = {
            "api_key": "key_with_underscores-and.dashes@123!",
            "index_name": "index-with.special.chars_123",
            "namespace": "namespace with spaces and ünïcödë",
        }
        assert special_config["api_key"] == "key_with_underscores-and.dashes@123!"
        assert special_config["index_name"] == "index-with.special.chars_123"
        assert special_config["namespace"] == "namespace with spaces and ünïcödë"

        # Test with special characters in other configs
        special_char_config: EmbeddingConfig = {
            "model": "model/with/slashes-and-hyphens_v1@latest",
            "params": {
                "special_key!@#$%": "special_value!@#$%",
                "unicode_test": "üñíçødé",
                "emoji_test": "🚀🌟✨",
            },
        }
        assert (
            special_char_config["model"] == "model/with/slashes-and-hyphens_v1@latest"
        )
        assert special_char_config["params"]["special_key!@#$%"] == "special_value!@#$%"
        assert special_char_config["params"]["unicode_test"] == "üñíçødé"
        assert special_char_config["params"]["emoji_test"] == "🚀🌟✨"

    def test_numeric_edge_cases(self) -> None:
        """Test TypedDicts with numeric edge cases."""
        llm_config: LLMConfig = {"model": "test-model", "api_key": "key"}

        # Test with zero and negative values where appropriate
        numeric_config: QueryEnhancementConfig = {
            "type": "step_back",
            "num_queries": 5,
            "num_hyde_docs": 0,  # Zero is valid
            "llm": llm_config,
            "fusion_method": "weighted",
            "rrf_k": 50,
            "top_k": 25,
        }
        assert numeric_config["num_hyde_docs"] == 0
        assert numeric_config["num_queries"] == 5
        assert numeric_config["rrf_k"] == 50
        assert numeric_config["top_k"] == 25
