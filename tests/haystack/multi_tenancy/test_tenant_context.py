"""Tests for tenant context module."""

import os

import pytest

from vectordb.haystack.multi_tenancy.tenant_context import (
    TenantContext,
    TenantContextError,
)


class TestTenantContextError:
    """Tests for TenantContextError exception."""

    def test_error_is_value_error(self):
        """Test that TenantContextError inherits from ValueError."""
        assert issubclass(TenantContextError, ValueError)

    def test_raise_error(self):
        """Test raising TenantContextError."""
        with pytest.raises(TenantContextError) as exc_info:
            raise TenantContextError("Test error message")
        assert str(exc_info.value) == "Test error message"


class TestTenantContext:
    """Tests for TenantContext dataclass."""

    def test_create_context(self):
        """Test creating tenant context."""
        context = TenantContext(tenant_id="tenant_123")
        assert context.tenant_id == "tenant_123"
        assert context.tenant_name is None
        assert context.metadata == {}

    def test_create_context_with_name(self):
        """Test creating tenant context with name."""
        context = TenantContext(tenant_id="tenant_123", tenant_name="Acme Corp")
        assert context.tenant_id == "tenant_123"
        assert context.tenant_name == "Acme Corp"

    def test_create_context_with_metadata(self):
        """Test creating tenant context with metadata."""
        context = TenantContext(
            tenant_id="tenant_123",
            metadata={"department": "engineering", "tier": "enterprise"},
        )
        assert context.tenant_id == "tenant_123"
        assert context.metadata == {"department": "engineering", "tier": "enterprise"}

    def test_empty_tenant_id_raises_error(self):
        """Test that empty tenant_id raises TenantContextError."""
        with pytest.raises(TenantContextError) as exc_info:
            TenantContext(tenant_id="")
        assert "tenant_id cannot be empty" in str(exc_info.value)

    def test_non_string_tenant_id_raises_error(self):
        """Test that non-string tenant_id raises TenantContextError."""
        with pytest.raises(TenantContextError) as exc_info:
            TenantContext(tenant_id=123)  # type: ignore
        assert "tenant_id must be a string" in str(exc_info.value)

    def test_immutability(self):
        """Test that TenantContext is frozen (immutable)."""
        context = TenantContext(tenant_id="tenant_123")
        with pytest.raises(AttributeError):
            context.tenant_id = "different_tenant"  # type: ignore


class TestTenantContextFromEnvironment:
    """Tests for from_environment class method."""

    def test_from_environment_success(self):
        """Test creating context from environment variables."""
        os.environ["TENANT_ID"] = "env_tenant"
        os.environ["TENANT_NAME"] = "Environment Tenant"
        try:
            context = TenantContext.from_environment()
            assert context.tenant_id == "env_tenant"
            assert context.tenant_name == "Environment Tenant"
        finally:
            del os.environ["TENANT_ID"]
            del os.environ["TENANT_NAME"]

    def test_from_environment_only_id(self):
        """Test creating context with only tenant ID."""
        os.environ["TENANT_ID"] = "env_tenant"
        try:
            context = TenantContext.from_environment()
            assert context.tenant_id == "env_tenant"
            assert context.tenant_name is None
        finally:
            del os.environ["TENANT_ID"]

    def test_from_environment_missing_var(self):
        """Test that missing environment variable raises error."""
        if "TENANT_ID" in os.environ:
            del os.environ["TENANT_ID"]
        with pytest.raises(TenantContextError) as exc_info:
            TenantContext.from_environment()
        assert "TENANT_ID" in str(exc_info.value)

    def test_from_environment_custom_env_var(self):
        """Test from_environment with custom environment variable name."""
        os.environ["CUSTOM_TENANT_ID"] = "custom_tenant"
        try:
            context = TenantContext.from_environment(env_var="CUSTOM_TENANT_ID")
            assert context.tenant_id == "custom_tenant"
        finally:
            del os.environ["CUSTOM_TENANT_ID"]

    def test_from_environment_custom_name_var(self):
        """Test from_environment with custom name environment variable."""
        os.environ["CUSTOM_TENANT_ID"] = "custom_tenant"
        os.environ["CUSTOM_TENANT_DISPLAY"] = "Custom Tenant Name"
        try:
            context = TenantContext.from_environment(
                env_var="CUSTOM_TENANT_ID",
                name_env_var="CUSTOM_TENANT_DISPLAY",
            )
            assert context.tenant_id == "custom_tenant"
            assert context.tenant_name == "Custom Tenant Name"
        finally:
            del os.environ["CUSTOM_TENANT_ID"]
            del os.environ["CUSTOM_TENANT_DISPLAY"]

    def test_from_environment_no_name_var(self):
        """Test from_environment with name_env_var set to None."""
        os.environ["TENANT_ID"] = "env_tenant"
        try:
            context = TenantContext.from_environment(name_env_var=None)
            assert context.tenant_id == "env_tenant"
            assert context.tenant_name is None
        finally:
            del os.environ["TENANT_ID"]


class TestTenantContextFromConfig:
    """Tests for from_config class method."""

    def test_from_config_success(self):
        """Test creating context from configuration dictionary."""
        config = {
            "tenant": {
                "id": "config_tenant",
                "name": "Config Tenant",
                "metadata": {"department": "sales"},
            }
        }
        context = TenantContext.from_config(config)
        assert context.tenant_id == "config_tenant"
        assert context.tenant_name == "Config Tenant"
        assert context.metadata == {"department": "sales"}

    def test_from_config_minimal(self):
        """Test creating context with minimal config."""
        config = {"tenant": {"id": "config_tenant"}}
        context = TenantContext.from_config(config)
        assert context.tenant_id == "config_tenant"
        assert context.tenant_name is None
        assert context.metadata == {}

    def test_from_config_missing_tenant_section(self):
        """Test that missing tenant section raises error."""
        config = {"database": {"type": "milvus"}}
        with pytest.raises(TenantContextError) as exc_info:
            TenantContext.from_config(config)
        assert "tenant.id" in str(exc_info.value)

    def test_from_config_missing_tenant_id(self):
        """Test that missing tenant.id raises error."""
        config = {"tenant": {"name": "No ID Tenant"}}
        with pytest.raises(TenantContextError) as exc_info:
            TenantContext.from_config(config)
        assert "tenant.id" in str(exc_info.value)

    def test_from_config_empty_tenant_id(self):
        """Test that empty tenant.id raises error."""
        config = {"tenant": {"id": ""}}
        with pytest.raises(TenantContextError):
            TenantContext.from_config(config)


class TestTenantContextResolve:
    """Tests for resolve class method."""

    def test_resolve_explicit_context(self):
        """Test that explicit context takes priority."""
        explicit = TenantContext(tenant_id="explicit")
        resolved = TenantContext.resolve(tenant_context=explicit)
        assert resolved.tenant_id == "explicit"

    def test_resolve_from_environment(self):
        """Test resolution from environment variable."""
        os.environ["TENANT_ID"] = "env_tenant"
        try:
            resolved = TenantContext.resolve()
            assert resolved.tenant_id == "env_tenant"
        finally:
            del os.environ["TENANT_ID"]

    def test_resolve_from_config(self):
        """Test resolution from configuration."""
        config = {"tenant": {"id": "config_tenant"}}
        resolved = TenantContext.resolve(config=config)
        assert resolved.tenant_id == "config_tenant"

    def test_resolve_priority_explicit_over_env(self):
        """Test that explicit context takes priority over environment."""
        explicit = TenantContext(tenant_id="explicit")
        os.environ["TENANT_ID"] = "env_tenant"
        try:
            resolved = TenantContext.resolve(
                tenant_context=explicit, config={"tenant": {"id": "config_tenant"}}
            )
            assert resolved.tenant_id == "explicit"
        finally:
            del os.environ["TENANT_ID"]

    def test_resolve_priority_env_over_config(self):
        """Test that environment takes priority over config."""
        os.environ["TENANT_ID"] = "env_tenant"
        config = {"tenant": {"id": "config_tenant"}}
        try:
            resolved = TenantContext.resolve(config=config)
            assert resolved.tenant_id == "env_tenant"
        finally:
            del os.environ["TENANT_ID"]

    def test_resolve_failure_no_sources(self):
        """Test that resolution fails when no sources are available."""
        if "TENANT_ID" in os.environ:
            del os.environ["TENANT_ID"]
        with pytest.raises(TenantContextError) as exc_info:
            TenantContext.resolve()
        assert "Cannot resolve tenant context" in str(exc_info.value)


class TestTenantContextToDict:
    """Tests for to_dict method."""

    def test_to_dict_basic(self):
        """Test converting context to dictionary."""
        context = TenantContext(tenant_id="tenant_123")
        result = context.to_dict()
        assert result == {
            "tenant_id": "tenant_123",
            "tenant_name": None,
            "metadata": {},
        }

    def test_to_dict_with_name(self):
        """Test converting context with name to dictionary."""
        context = TenantContext(tenant_id="tenant_123", tenant_name="Acme Corp")
        result = context.to_dict()
        assert result == {
            "tenant_id": "tenant_123",
            "tenant_name": "Acme Corp",
            "metadata": {},
        }

    def test_to_dict_with_metadata(self):
        """Test converting context with metadata to dictionary."""
        context = TenantContext(
            tenant_id="tenant_123",
            metadata={"key": "value"},
        )
        result = context.to_dict()
        assert result == {
            "tenant_id": "tenant_123",
            "tenant_name": None,
            "metadata": {"key": "value"},
        }


class TestTenantContextStr:
    """Tests for __str__ method."""

    def test_str_with_id_only(self):
        """Test string representation with ID only."""
        context = TenantContext(tenant_id="tenant_123")
        assert str(context) == "TenantContext(tenant_123)"

    def test_str_with_name(self):
        """Test string representation with name."""
        context = TenantContext(tenant_id="tenant_123", tenant_name="Acme Corp")
        assert str(context) == "TenantContext(tenant_123, name=Acme Corp)"
