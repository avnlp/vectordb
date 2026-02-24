"""Tests for common tenant context utilities."""

import pytest

from vectordb.haystack.multi_tenancy.common.tenant_context import (
    TenantContext,
    TenantContextError,
)


class TestTenantContextValidation:
    """Tests for TenantContext validation logic."""

    def test_empty_tenant_id_raises_error(self) -> None:
        """Ensure empty tenant_id raises error."""
        with pytest.raises(TenantContextError, match="tenant_id cannot be empty"):
            TenantContext(tenant_id="")

    def test_non_string_tenant_id_raises_error(self) -> None:
        """Ensure non-string tenant_id raises error."""
        with pytest.raises(TenantContextError, match="tenant_id must be a string"):
            TenantContext(tenant_id=123)  # type: ignore[arg-type]


class TestTenantContextFromEnvironment:
    """Tests for from_environment classmethod."""

    def test_from_environment_requires_env_var(self) -> None:
        """Ensure missing env var raises error."""
        with pytest.raises(
            TenantContextError, match="Environment variable 'TENANT_ID'"
        ):
            TenantContext.from_environment()

    def test_from_environment_reads_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ensure name is read when name_env_var is set."""
        monkeypatch.setenv("TENANT_ID", "tenant-env")
        monkeypatch.setenv("TENANT_NAME", "Tenant Name")

        context = TenantContext.from_environment()

        assert context.tenant_id == "tenant-env"
        assert context.tenant_name == "Tenant Name"

    def test_from_environment_ignores_name_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ensure name is not read when name_env_var is None."""
        monkeypatch.setenv("TENANT_ID", "tenant-env")
        monkeypatch.setenv("TENANT_NAME", "Tenant Name")

        context = TenantContext.from_environment(name_env_var=None)

        assert context.tenant_id == "tenant-env"
        assert context.tenant_name is None

    def test_from_environment_custom_vars(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ensure custom env vars are respected."""
        monkeypatch.setenv("CUSTOM_TENANT_ID", "custom-id")
        monkeypatch.setenv("CUSTOM_TENANT_NAME", "Custom Name")

        context = TenantContext.from_environment(
            env_var="CUSTOM_TENANT_ID",
            name_env_var="CUSTOM_TENANT_NAME",
        )

        assert context.tenant_id == "custom-id"
        assert context.tenant_name == "Custom Name"


class TestTenantContextFromConfig:
    """Tests for from_config classmethod."""

    def test_from_config_missing_tenant_id(self) -> None:
        """Ensure missing tenant.id raises error."""
        config = {"tenant": {"name": "Missing ID"}}

        with pytest.raises(TenantContextError, match="tenant.id"):
            TenantContext.from_config(config)


class TestTenantContextResolve:
    """Tests for resolve classmethod."""

    def test_resolve_uses_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ensure TENANT_ID env var is used when no explicit context is given."""
        monkeypatch.setenv("TENANT_ID", "env-tenant")

        context = TenantContext.resolve()

        assert context.tenant_id == "env-tenant"

    def test_resolve_uses_config_when_present(self) -> None:
        """Ensure config tenant is used when env var is missing."""
        config = {"tenant": {"id": "config-tenant"}}

        context = TenantContext.resolve(config=config)

        assert context.tenant_id == "config-tenant"

    def test_resolve_raises_when_no_sources(self) -> None:
        """Ensure resolve raises when no sources provided."""
        with pytest.raises(TenantContextError, match="Cannot resolve tenant context"):
            TenantContext.resolve()


class TestTenantContextOutput:
    """Tests for output helpers."""

    def test_to_dict_includes_metadata(self) -> None:
        """Ensure to_dict outputs all fields."""
        context = TenantContext(
            tenant_id="tenant",
            tenant_name="Tenant",
            metadata={"tier": "gold"},
        )

        assert context.to_dict() == {
            "tenant_id": "tenant",
            "tenant_name": "Tenant",
            "metadata": {"tier": "gold"},
        }

    def test_str_includes_name_when_present(self) -> None:
        """Ensure __str__ includes tenant name when provided."""
        context = TenantContext(tenant_id="tenant", tenant_name="Tenant")

        assert str(context) == "TenantContext(tenant, name=Tenant)"

    def test_str_without_name(self) -> None:
        """Ensure __str__ excludes name when missing."""
        context = TenantContext(tenant_id="tenant")

        assert str(context) == "TenantContext(tenant)"
