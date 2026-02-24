"""Tests for base multi-tenancy pipeline."""

import os
import tempfile
from unittest.mock import patch

import pytest
import yaml

from vectordb.haystack.multi_tenancy.base import (
    BaseMultitenancyPipeline,
    Timer,
    _resolve_env_vars,
)
from vectordb.haystack.multi_tenancy.common.tenant_context import TenantContext
from vectordb.haystack.multi_tenancy.vectordb_multitenancy_type import (
    TenantIsolationStrategy,
)


class TestTimer:
    """Tests for Timer context manager."""

    def test_timer_initial_state(self):
        """Test timer initializes with zero values."""
        timer = Timer()
        assert timer._start == 0.0
        assert timer._end == 0.0

    def test_timer_context_manager(self):
        """Test timer as context manager."""
        with Timer() as timer:
            assert timer._start > 0.0
        assert timer._end > 0.0

    def test_timer_elapsed_ms(self):
        """Test elapsed time calculation."""
        with Timer() as timer:
            timer._start = timer._start
        assert timer.elapsed_ms >= 0.0

    def test_timer_elapsed_ms_with_work(self):
        """Test elapsed time with actual work."""
        import time

        with Timer() as timer:
            time.sleep(0.01)  # 10ms sleep
        assert timer.elapsed_ms >= 10.0


class TestResolveEnvVars:
    """Tests for _resolve_env_vars function."""

    def test_resolve_string_no_vars(self):
        """Test resolving string with no environment variables."""
        result = _resolve_env_vars("simple string")
        assert result == "simple string"

    def test_resolve_integer(self):
        """Test resolving integer value."""
        result = _resolve_env_vars(42)
        assert result == 42

    def test_resolve_boolean(self):
        """Test resolving boolean value."""
        result = _resolve_env_vars(True)
        assert result is True

    def test_resolve_list(self):
        """Test resolving list."""
        result = _resolve_env_vars(["string", 42, True])
        assert result == ["string", 42, True]

    def test_resolve_dict(self):
        """Test resolving dictionary."""
        result = _resolve_env_vars({"key": "value", "number": 123})
        assert result == {"key": "value", "number": 123}

    def test_resolve_env_var(self):
        """Test resolving environment variable."""
        os.environ["TEST_VAR"] = "resolved_value"
        try:
            result = _resolve_env_vars("${TEST_VAR}")
            assert result == "resolved_value"
        finally:
            del os.environ["TEST_VAR"]

    def test_resolve_env_var_with_default(self):
        """Test resolving with default value."""
        result = _resolve_env_vars("${NONEXISTENT:-default}")
        assert result == "default"

    def test_resolve_nested_dict(self):
        """Test resolving nested dictionary."""
        os.environ["NESTED_VAR"] = "nested_value"
        try:
            result = _resolve_env_vars({"outer": {"inner": "${NESTED_VAR}"}})
            assert result == {"outer": {"inner": "nested_value"}}
        finally:
            del os.environ["NESTED_VAR"]

    def test_resolve_nested_list(self):
        """Test resolving nested list."""
        os.environ["LIST_VAR"] = "list_value"
        try:
            result = _resolve_env_vars([["${LIST_VAR}"], ["other"]])
            assert result == [["list_value"], ["other"]]
        finally:
            del os.environ["LIST_VAR"]

    def test_resolve_none(self):
        """Test resolving None value."""
        result = _resolve_env_vars(None)
        assert result is None


class TestBaseMultitenancyPipeline:
    """Tests for BaseMultitenancyPipeline abstract class."""

    def test_default_constants(self):
        """Test class default constants."""
        assert (
            BaseMultitenancyPipeline.DEFAULT_EMBEDDING_MODEL
            == "Qwen/Qwen3-Embedding-0.6B"
        )
        assert BaseMultitenancyPipeline.DEFAULT_EMBEDDING_DIMENSION == 1024

    def test_load_config_static_file_not_found(self):
        """Test that missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            BaseMultitenancyPipeline._load_config_static(
                "/nonexistent/path/config.yaml"
            )

    def test_load_config_static_valid_yaml(self):
        """Test loading valid YAML config."""
        yaml_content = """
pipeline:
  name: test_pipeline
database:
  type: milvus
tenant:
  id: test_tenant
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                config = BaseMultitenancyPipeline._load_config_static(f.name)
                assert config["pipeline"]["name"] == "test_pipeline"
                assert config["database"]["type"] == "milvus"
                assert config["tenant"]["id"] == "test_tenant"
            finally:
                os.unlink(f.name)

    def test_load_config_static_resolves_env_vars(self):
        """Test that config loading resolves environment variables."""
        os.environ["CONFIG_TEST_VAR"] = "env_value"
        yaml_content = """
key: ${CONFIG_TEST_VAR}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                config = BaseMultitenancyPipeline._load_config_static(f.name)
                assert config["key"] == "env_value"
            finally:
                os.unlink(f.name)
                del os.environ["CONFIG_TEST_VAR"]

    def test_load_config_static_malformed_yaml_raises_error(self):
        """Test that _load_config_static raises yaml.YAMLError for malformed YAML."""
        malformed_yaml = """invalid_yaml: [
    unclosed: quote
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(malformed_yaml)
            f.flush()
            config_path = f.name

        try:
            with pytest.raises(yaml.YAMLError):
                BaseMultitenancyPipeline._load_config_static(config_path)
        finally:
            os.unlink(config_path)

    def test_get_embedding_dimension_from_config(self):
        """Test getting embedding dimension from config."""

        # Create a concrete subclass for testing
        class ConcretePipeline(BaseMultitenancyPipeline):
            @property
            def isolation_strategy(self) -> TenantIsolationStrategy:
                return TenantIsolationStrategy.PARTITION_KEY

            def _connect(self) -> None:
                self.client = object()

            def close(self) -> None:
                self.client = None

            def _get_document_store(self) -> None:
                return None

            def create_tenant(self, tenant_id: str) -> bool:
                return True

            def tenant_exists(self, tenant_id: str) -> bool:
                return True

            def get_tenant_stats(self, tenant_id: str):
                return {}

            def list_tenants(self) -> list[str]:
                return []

            def delete_tenant(self, tenant_id: str) -> bool:
                return True

            def index_documents(self, documents, tenant_id=None) -> int:
                return 0

            def query(self, query, top_k=10, tenant_id=None):
                return []

        yaml_content = """
pipeline:
  name: test_pipeline
embedding:
  dimension: 512
  output_dimension: 256
tenant:
  id: test_tenant
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                tenant_context = TenantContext(tenant_id="test_tenant")
                with (
                    patch.object(TenantContext, "resolve", return_value=tenant_context),
                    patch.object(ConcretePipeline, "_connect"),
                ):
                    pipeline = ConcretePipeline(
                        config_path=f.name, tenant_context=tenant_context
                    )
                    # Test with output_dimension set
                    dim = pipeline._get_embedding_dimension()
                    assert dim == 256
            finally:
                os.unlink(f.name)

    def test_get_embedding_dimension_default(self):
        """Test getting default embedding dimension."""

        class ConcretePipeline(BaseMultitenancyPipeline):
            @property
            def isolation_strategy(self) -> TenantIsolationStrategy:
                return TenantIsolationStrategy.PARTITION_KEY

            def _connect(self) -> None:
                self.client = object()

            def close(self) -> None:
                self.client = None

            def _get_document_store(self) -> None:
                return None

            def create_tenant(self, tenant_id: str) -> bool:
                return True

            def tenant_exists(self, tenant_id: str) -> bool:
                return True

            def get_tenant_stats(self, tenant_id: str):
                return {}

            def list_tenants(self) -> list[str]:
                return []

            def delete_tenant(self, tenant_id: str) -> bool:
                return True

            def index_documents(self, documents, tenant_id=None) -> int:
                return 0

            def query(self, query, top_k=10, tenant_id=None):
                return []

        yaml_content = """
pipeline:
  name: test_pipeline
tenant:
  id: test_tenant
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                tenant_context = TenantContext(tenant_id="test_tenant")
                with (
                    patch.object(TenantContext, "resolve", return_value=tenant_context),
                    patch.object(ConcretePipeline, "_connect"),
                ):
                    pipeline = ConcretePipeline(
                        config_path=f.name, tenant_context=tenant_context
                    )
                    dim = pipeline._get_embedding_dimension()
                    assert dim == 1024  # DEFAULT_EMBEDDING_DIMENSION
            finally:
                os.unlink(f.name)

    def test_get_tenant_isolation_config_defaults(self):
        """Test default tenant isolation config."""

        class ConcretePipeline(BaseMultitenancyPipeline):
            @property
            def isolation_strategy(self) -> TenantIsolationStrategy:
                return TenantIsolationStrategy.PARTITION_KEY

            def _connect(self) -> None:
                self.client = object()

            def close(self) -> None:
                self.client = None

            def _get_document_store(self) -> None:
                return None

            def create_tenant(self, tenant_id: str) -> bool:
                return True

            def tenant_exists(self, tenant_id: str) -> bool:
                return True

            def get_tenant_stats(self, tenant_id: str):
                return {}

            def list_tenants(self) -> list[str]:
                return []

            def delete_tenant(self, tenant_id: str) -> bool:
                return True

            def index_documents(self, documents, tenant_id=None) -> int:
                return 0

            def query(self, query, top_k=10, tenant_id=None):
                return []

        yaml_content = """
pipeline:
  name: test_pipeline
tenant:
  id: test_tenant
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                tenant_context = TenantContext(tenant_id="test_tenant")
                with (
                    patch.object(TenantContext, "resolve", return_value=tenant_context),
                    patch.object(ConcretePipeline, "_connect"),
                ):
                    pipeline = ConcretePipeline(
                        config_path=f.name, tenant_context=tenant_context
                    )
                    config = pipeline._get_tenant_isolation_config()
                    assert config.strategy == "partition_key"
                    assert config.field_name == "tenant_id"
                    assert config.auto_create_tenant is True
            finally:
                os.unlink(f.name)

    def test_get_tenant_isolation_config_custom(self):
        """Test custom tenant isolation config."""

        class ConcretePipeline(BaseMultitenancyPipeline):
            @property
            def isolation_strategy(self) -> TenantIsolationStrategy:
                return TenantIsolationStrategy.NAMESPACE

            def _connect(self) -> None:
                self.client = object()

            def close(self) -> None:
                self.client = None

            def _get_document_store(self) -> None:
                return None

            def create_tenant(self, tenant_id: str) -> bool:
                return True

            def tenant_exists(self, tenant_id: str) -> bool:
                return True

            def get_tenant_stats(self, tenant_id: str):
                return {}

            def list_tenants(self) -> list[str]:
                return []

            def delete_tenant(self, tenant_id: str) -> bool:
                return True

            def index_documents(self, documents, tenant_id=None) -> int:
                return 0

            def query(self, query, top_k=10, tenant_id=None):
                return []

        yaml_content = """
pipeline:
  name: test_pipeline
multitenancy:
  strategy: namespace
  field_name: org_id
  auto_create_tenant: false
  partition_key_isolation: true
  num_partitions: 128
tenant:
  id: test_tenant
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                tenant_context = TenantContext(tenant_id="test_tenant")
                with (
                    patch.object(TenantContext, "resolve", return_value=tenant_context),
                    patch.object(ConcretePipeline, "_connect"),
                ):
                    pipeline = ConcretePipeline(
                        config_path=f.name, tenant_context=tenant_context
                    )
                    config = pipeline._get_tenant_isolation_config()
                    assert config.strategy == "namespace"
                    assert config.field_name == "org_id"
                    assert config.auto_create_tenant is False
                    assert config.partition_key_isolation is True
                    assert config.num_partitions == 128
            finally:
                os.unlink(f.name)

    def test_get_collection_name(self):
        """Test getting collection name from config."""

        class ConcretePipeline(BaseMultitenancyPipeline):
            @property
            def isolation_strategy(self) -> TenantIsolationStrategy:
                return TenantIsolationStrategy.PARTITION_KEY

            def _connect(self) -> None:
                self.client = object()

            def close(self) -> None:
                self.client = None

            def _get_document_store(self) -> None:
                return None

            def create_tenant(self, tenant_id: str) -> bool:
                return True

            def tenant_exists(self, tenant_id: str) -> bool:
                return True

            def get_tenant_stats(self, tenant_id: str):
                return {}

            def list_tenants(self) -> list[str]:
                return []

            def delete_tenant(self, tenant_id: str) -> bool:
                return True

            def index_documents(self, documents, tenant_id=None) -> int:
                return 0

            def query(self, query, top_k=10, tenant_id=None):
                return []

        yaml_content = """
pipeline:
  name: test_pipeline
collection:
  name: my_collection
tenant:
  id: test_tenant
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                tenant_context = TenantContext(tenant_id="test_tenant")
                with (
                    patch.object(TenantContext, "resolve", return_value=tenant_context),
                    patch.object(ConcretePipeline, "_connect"),
                ):
                    pipeline = ConcretePipeline(
                        config_path=f.name, tenant_context=tenant_context
                    )
                    name = pipeline._get_collection_name()
                    assert name == "my_collection"
            finally:
                os.unlink(f.name)

    def test_create_timing_metrics(self):
        """Test creating timing metrics."""

        class ConcretePipeline(BaseMultitenancyPipeline):
            @property
            def isolation_strategy(self) -> TenantIsolationStrategy:
                return TenantIsolationStrategy.PARTITION_KEY

            def _connect(self) -> None:
                self.client = object()

            def close(self) -> None:
                self.client = None

            def _get_document_store(self) -> None:
                return None

            def create_tenant(self, tenant_id: str) -> bool:
                return True

            def tenant_exists(self, tenant_id: str) -> bool:
                return True

            def get_tenant_stats(self, tenant_id: str):
                return {}

            def list_tenants(self) -> list[str]:
                return []

            def delete_tenant(self, tenant_id: str) -> bool:
                return True

            def index_documents(self, documents, tenant_id=None) -> int:
                return 0

            def query(self, query, top_k=10, tenant_id=None):
                return []

        yaml_content = """
pipeline:
  name: test_pipeline
tenant:
  id: test_tenant
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                tenant_context = TenantContext(tenant_id="test_tenant")
                with (
                    patch.object(TenantContext, "resolve", return_value=tenant_context),
                    patch.object(ConcretePipeline, "_connect"),
                ):
                    pipeline = ConcretePipeline(
                        config_path=f.name, tenant_context=tenant_context
                    )
                    metrics = pipeline._create_timing_metrics(
                        tenant_resolution_ms=10.0,
                        index_operation_ms=100.0,
                        retrieval_ms=50.0,
                        total_ms=160.0,
                        num_documents=10,
                    )
                    assert metrics.tenant_id == "test_tenant"
                    assert metrics.tenant_resolution_ms == 10.0
                    assert metrics.num_documents == 10
            finally:
                os.unlink(f.name)

    def test_context_manager(self):
        """Test pipeline as context manager."""

        class ConcretePipeline(BaseMultitenancyPipeline):
            closed = False

            @property
            def isolation_strategy(self) -> TenantIsolationStrategy:
                return TenantIsolationStrategy.PARTITION_KEY

            def _connect(self) -> None:
                self.client = object()

            def close(self) -> None:
                ConcretePipeline.closed = True

            def _get_document_store(self) -> None:
                return None

            def create_tenant(self, tenant_id: str) -> bool:
                return True

            def tenant_exists(self, tenant_id: str) -> bool:
                return True

            def get_tenant_stats(self, tenant_id: str):
                return {}

            def list_tenants(self) -> list[str]:
                return []

            def delete_tenant(self, tenant_id: str) -> bool:
                return True

            def index_documents(self, documents, tenant_id=None) -> int:
                return 0

            def query(self, query, top_k=10, tenant_id=None):
                return []

        yaml_content = """
pipeline:
  name: test_pipeline
tenant:
  id: test_tenant
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                tenant_context = TenantContext(tenant_id="test_tenant")
                with (
                    patch.object(TenantContext, "resolve", return_value=tenant_context),
                    patch.object(ConcretePipeline, "_connect"),
                ):
                    with ConcretePipeline(
                        config_path=f.name, tenant_context=tenant_context
                    ) as pipeline:
                        assert pipeline is not None
                    assert ConcretePipeline.closed is True
            finally:
                os.unlink(f.name)

    def test_logger_lazy_initialization(self):
        """Test logger property lazy initialization."""

        class ConcretePipeline(BaseMultitenancyPipeline):
            @property
            def isolation_strategy(self) -> TenantIsolationStrategy:
                return TenantIsolationStrategy.PARTITION_KEY

            def _connect(self) -> None:
                self.client = object()

            def close(self) -> None:
                pass

            def _get_document_store(self) -> None:
                return None

            def create_tenant(self, tenant_id: str) -> bool:
                return True

            def tenant_exists(self, tenant_id: str) -> bool:
                return True

            def get_tenant_stats(self, tenant_id: str):
                return {}

            def list_tenants(self) -> list[str]:
                return []

            def delete_tenant(self, tenant_id: str) -> bool:
                return True

            def index_documents(self, documents, tenant_id=None) -> int:
                return 0

            def query(self, query, top_k=10, tenant_id=None):
                return []

        yaml_content = """
pipeline:
  name: my_pipeline_name
tenant:
  id: test_tenant
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                tenant_context = TenantContext(tenant_id="test_tenant")
                with (
                    patch.object(TenantContext, "resolve", return_value=tenant_context),
                    patch.object(ConcretePipeline, "_connect"),
                    patch(
                        "vectordb.haystack.multi_tenancy.base.LoggerFactory"
                    ) as mock_factory_class,
                ):
                    mock_factory = mock_factory_class.configure_from_env.return_value
                    mock_logger = mock_factory.get_logger.return_value

                    pipeline = ConcretePipeline(
                        config_path=f.name, tenant_context=tenant_context
                    )

                    assert pipeline._logger is None

                    logger1 = pipeline.logger

                    mock_factory_class.configure_from_env.assert_called_once_with(
                        "my_pipeline_name"
                    )
                    mock_factory.get_logger.assert_called_once()
                    assert logger1 == mock_logger

                    logger2 = pipeline.logger
                    assert logger2 == logger1
                    assert mock_factory_class.configure_from_env.call_count == 1
            finally:
                os.unlink(f.name)

    def test_exit_calls_close(self):
        """Test __exit__ method calls close()."""

        class ConcretePipeline(BaseMultitenancyPipeline):
            close_called = False

            @property
            def isolation_strategy(self) -> TenantIsolationStrategy:
                return TenantIsolationStrategy.PARTITION_KEY

            def _connect(self) -> None:
                self.client = object()

            def close(self) -> None:
                ConcretePipeline.close_called = True

            def _get_document_store(self) -> None:
                return None

            def create_tenant(self, tenant_id: str) -> bool:
                return True

            def tenant_exists(self, tenant_id: str) -> bool:
                return True

            def get_tenant_stats(self, tenant_id: str):
                return {}

            def list_tenants(self) -> list[str]:
                return []

            def delete_tenant(self, tenant_id: str) -> bool:
                return True

            def index_documents(self, documents, tenant_id=None) -> int:
                return 0

            def query(self, query, top_k=10, tenant_id=None):
                return []

        yaml_content = """
pipeline:
  name: test_pipeline
tenant:
  id: test_tenant
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                tenant_context = TenantContext(tenant_id="test_tenant")
                with (
                    patch.object(TenantContext, "resolve", return_value=tenant_context),
                    patch.object(ConcretePipeline, "_connect"),
                ):
                    ConcretePipeline.close_called = False

                    pipeline = ConcretePipeline(
                        config_path=f.name, tenant_context=tenant_context
                    )

                    pipeline.__exit__(None, None, None)

                    assert ConcretePipeline.close_called is True
            finally:
                os.unlink(f.name)

    def test_logger_uses_class_name_when_pipeline_name_not_in_config(self):
        """Test logger uses class name when pipeline name is not in config."""

        class MyCustomPipeline(BaseMultitenancyPipeline):
            @property
            def isolation_strategy(self) -> TenantIsolationStrategy:
                return TenantIsolationStrategy.PARTITION_KEY

            def _connect(self) -> None:
                self.client = object()

            def close(self) -> None:
                pass

            def _get_document_store(self) -> None:
                return None

            def create_tenant(self, tenant_id: str) -> bool:
                return True

            def tenant_exists(self, tenant_id: str) -> bool:
                return True

            def get_tenant_stats(self, tenant_id: str):
                return {}

            def list_tenants(self) -> list[str]:
                return []

            def delete_tenant(self, tenant_id: str) -> bool:
                return True

            def index_documents(self, documents, tenant_id=None) -> int:
                return 0

            def query(self, query, top_k=10, tenant_id=None):
                return []

        yaml_content = """
tenant:
  id: test_tenant
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                tenant_context = TenantContext(tenant_id="test_tenant")
                with (
                    patch.object(TenantContext, "resolve", return_value=tenant_context),
                    patch.object(MyCustomPipeline, "_connect"),
                    patch(
                        "vectordb.haystack.multi_tenancy.base.LoggerFactory"
                    ) as mock_factory_class,
                ):
                    pipeline = MyCustomPipeline(
                        config_path=f.name, tenant_context=tenant_context
                    )

                    _ = pipeline.logger

                    mock_factory_class.configure_from_env.assert_called_once_with(
                        "MyCustomPipeline"
                    )
            finally:
                os.unlink(f.name)
