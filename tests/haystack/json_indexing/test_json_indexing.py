"""Unit tests for Haystack JSON indexing/search common utilities.

Tests validate shared infrastructure used by JSON-based indexing pipelines
across all supported vector databases (Milvus, Pinecone, Qdrant, Weaviate,
Chroma). These utilities provide:
- Configuration loading with environment variable resolution
- Metadata flattening for vector database compatibility
- Filter builders for each database's query DSL

The filter builders translate generic filter dictionaries into database-specific
query formats, enabling consistent filtering across different backends.
"""

from vectordb.haystack.json_indexing.common.config import load_config
from vectordb.haystack.json_indexing.common.filters.chroma import (
    build_chroma_filter,
)
from vectordb.haystack.json_indexing.common.filters.milvus import (
    build_milvus_filter,
)
from vectordb.haystack.json_indexing.common.filters.pinecone import (
    build_pinecone_filter,
)
from vectordb.haystack.json_indexing.common.filters.qdrant import (
    build_qdrant_filter,
)
from vectordb.haystack.json_indexing.common.filters.weaviate import (
    build_weaviate_filter,
)
from vectordb.haystack.json_indexing.common.metadata import flatten_metadata


class TestJSONIndexing:
    """Test suite for JSON indexing common utilities.

    Validates:
    - Configuration loading with env var substitution
    - Default value resolution for missing env vars
    - Metadata flattening for vector DB compatibility
    - Filter builders for all 5 supported databases
    - Package imports for indexer/searcher classes
    """

    def test_config_loading_dict(self) -> None:
        """Test that load_config accepts and returns dictionary configs.

        Validates passthrough behavior when config is already a dict,
        ensuring no transformation occurs.

        Returns:
            None
        """
        config = {"key": "value", "nested": {"key2": "value2"}}
        result = load_config(config)
        assert result == config

    def test_config_loading_with_env_vars(self) -> None:
        """Test environment variable resolution in configuration.

        Validates that ${VAR} syntax is replaced with actual
        environment variable values during config loading.

        Returns:
            None
        """
        import os

        os.environ["TEST_VAR"] = "resolved_value"
        config = {"key": "${TEST_VAR}"}
        result = load_config(config)
        assert result["key"] == "resolved_value"

    def test_config_loading_with_defaults(self) -> None:
        """Test default value resolution for unset environment variables.

        Validates ${VAR:-default} syntax provides fallback values
        when environment variables are not defined.

        Returns:
            None
        """
        config = {"key": "${NONEXISTENT_VAR:-default_value}"}
        result = load_config(config)
        assert result["key"] == "default_value"

    def test_flatten_metadata(self) -> None:
        """Test metadata flattening for vector database compatibility.

        Validates that complex metadata structures are flattened:
        - Strings preserved as-is
        - Integers and floats preserved
        - Booleans converted to values
        - Nested dicts converted to string representation
        - None values are filtered out

        Returns:
            None
        """
        metadata = {
            "name": "test",
            "count": 42,
            "score": 3.14,
            "active": True,
            "nested": {"key": "value"},
            "none_value": None,
        }

        result = flatten_metadata(metadata)

        assert result["name"] == "test"
        assert result["count"] == 42
        assert result["score"] == 3.14
        assert result["active"] is True
        assert result["nested"] == "{'key': 'value'}"
        assert "none_value" not in result

    def test_milvus_filter_simple_equality(self) -> None:
        """Test Milvus filter builder with simple equality condition.

        Validates that {"field": "value"} converts to Milvus expression
        syntax: metadata["field"] == "value".

        Returns:
            None
        """
        filters = {"category": "science"}
        result = build_milvus_filter(filters)
        assert 'metadata["category"]' in result
        assert '== "science"' in result

    def test_milvus_filter_with_operators(self) -> None:
        """Test Milvus filter builder with comparison operators.

        Validates operator translation:
        - $gt (greater than)
        - $lte (less than or equal)

        Returns:
            None
        """
        filters = {"age": {"$gt": 25}, "score": {"$lte": 100}}
        result = build_milvus_filter(filters)
        assert 'metadata["age"]' in result
        assert "> 25" in result
        assert 'metadata["score"]' in result
        assert "<= 100" in result

    def test_milvus_filter_none(self) -> None:
        """Test Milvus filter builder with None input.

        Validates that None filters return None without error,
        indicating no filtering should be applied.

        Returns:
            None
        """
        result = build_milvus_filter(None)
        assert result is None

    def test_pinecone_filter_simple(self) -> None:
        """Test Pinecone filter builder with equality condition.

        Validates that {"field": "value"} converts to Pinecone's
        metadata filter format: {"field": {"$eq": "value"}}.

        Returns:
            None
        """
        filters = {"category": "science"}
        result = build_pinecone_filter(filters)
        assert result == {"category": {"$eq": "science"}}

    def test_pinecone_filter_with_operators(self) -> None:
        """Test Pinecone filter builder with comparison operators.

        Validates that operators like $gt are passed through
        to Pinecone's filter format unchanged.

        Returns:
            None
        """
        filters = {"age": {"$gt": 25}}
        result = build_pinecone_filter(filters)
        assert result == {"age": {"$gt": 25}}

    def test_qdrant_filter_simple(self) -> None:
        """Test Qdrant filter builder with equality condition.

        Validates that simple filters are converted to Qdrant's
        Filter object with must conditions.

        Returns:
            None
        """
        filters = {"category": "science"}
        result = build_qdrant_filter(filters)
        assert result is not None
        assert hasattr(result, "must")

    def test_qdrant_filter_none(self) -> None:
        """Test Qdrant filter builder with None input.

        Validates that None filters return None without error.

        Returns:
            None
        """
        result = build_qdrant_filter(None)
        assert result is None

    def test_weaviate_filter_simple(self) -> None:
        """Test Weaviate filter builder with equality condition.

        Validates that simple filters are converted to Weaviate's
        GraphQL filter format.

        Returns:
            None
        """
        filters = {"category": "science"}
        result = build_weaviate_filter(filters)
        assert result is not None

    def test_weaviate_filter_none(self) -> None:
        """Test Weaviate filter builder with None input.

        Validates that None filters return None without error.

        Returns:
            None
        """
        result = build_weaviate_filter(None)
        assert result is None

    def test_chroma_filter_simple(self) -> None:
        """Test Chroma filter builder with equality condition.

        Validates that {"field": "value"} converts to Chroma's
        metadata filter format: {"field": {"$eq": "value"}}.

        Returns:
            None
        """
        filters = {"category": "science"}
        result = build_chroma_filter(filters)
        assert result == {"category": {"$eq": "science"}}

    def test_chroma_filter_multiple_conditions(self) -> None:
        """Test Chroma filter builder with multiple AND conditions.

        Validates that multiple filter conditions are combined
        with $and operator in Chroma's format.

        Returns:
            None
        """
        filters = {"category": "science", "active": True}
        result = build_chroma_filter(filters)
        assert "$and" in result
        assert len(result["$and"]) == 2

    def test_chroma_filter_none(self) -> None:
        """Test Chroma filter builder with None input.

        Validates that None filters return None without error.

        Returns:
            None
        """
        result = build_chroma_filter(None)
        assert result is None

    def test_imports(self) -> None:
        """Test that all JSON indexer and searcher classes are importable.

        Validates package structure by importing indexer/searcher pairs
        for all 5 supported vector databases (Milvus, Pinecone, Qdrant,
        Weaviate, Chroma).

        Returns:
            None
        """
        from vectordb.haystack.json_indexing import (
            ChromaJSONIndexer,
            ChromaJSONSearcher,
            MilvusJSONIndexer,
            MilvusJSONSearcher,
            PineconeJSONIndexer,
            PineconeJSONSearcher,
            QdrantJSONIndexer,
            QdrantJSONSearcher,
            WeaviateJSONIndexer,
            WeaviateJSONSearcher,
        )

        assert callable(MilvusJSONIndexer)
        assert callable(MilvusJSONSearcher)
        assert callable(PineconeJSONIndexer)
        assert callable(PineconeJSONSearcher)
        assert callable(QdrantJSONIndexer)
        assert callable(QdrantJSONSearcher)
        assert callable(WeaviateJSONIndexer)
        assert callable(WeaviateJSONSearcher)
        assert callable(ChromaJSONIndexer)
        assert callable(ChromaJSONSearcher)
