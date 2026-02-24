"""Shared fixtures for multi-tenancy pipeline tests.

This module provides pytest fixtures for testing multi-tenancy implementations
in Haystack. Multi-tenancy enables data isolation between tenants within
shared vector database deployments using namespaces, partitions, or metadata.

Fixtures:
    sample_tenant_context: TenantContext instance for tenant isolation testing.
    sample_documents: Haystack Documents for tenant-scoped indexing tests.
    mock_config: Configuration dictionary for multi-tenancy pipeline tests.

Note:
    Multi-tenancy tests validate that documents indexed for one tenant
    are not accessible to queries from other tenants, ensuring proper
    data isolation in shared infrastructure deployments.
"""

import pytest

from vectordb.haystack.multi_tenancy.common.tenant_context import TenantContext


@pytest.fixture
def sample_tenant_context() -> TenantContext:
    """Create a sample tenant context for isolation testing.

    Returns:
        TenantContext with test tenant ID and name for validating
        tenant-scoped document operations and query routing.
    """
    return TenantContext(tenant_id="test-tenant", tenant_name="Test Tenant")


@pytest.fixture
def sample_documents() -> list:
    """Create sample documents for multi-tenancy testing.

    Returns:
        List of Haystack Documents with metadata for testing
        tenant-scoped indexing and retrieval operations.
    """
    from haystack import Document

    return [
        Document(content="This is a test document", meta={"source": "test"}),
        Document(content="Another test document", meta={"source": "test"}),
        Document(content="Third test document", meta={"category": "test"}),
    ]


@pytest.fixture
def mock_config() -> dict:
    """Create mock configuration for multi-tenancy pipeline tests.

    Returns:
        Configuration dictionary with database, embedding, and collection
        settings for testing multi-tenancy pipeline initialization.
    """
    return {
        "database": {"type": "milvus", "host": "localhost", "port": 19530},
        "embedding": {"model": "Qwen/Qwen3-Embedding-0.6B", "dimension": 1024},
        "collection": {"name": "test_collection"},
    }
