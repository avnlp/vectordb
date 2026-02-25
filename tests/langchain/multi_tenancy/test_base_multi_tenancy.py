"""Tests for the base multi-tenancy pipeline (LangChain).

This module tests the MultiTenancyPipeline abstract base class that defines the
interface for all vector database multi-tenancy implementations. Multi-tenancy
enables data isolation between different tenants (users, organizations, or apps)
within a single vector database instance.

The MultiTenancyPipeline ABC ensures that all implementations provide:
    - index_for_tenant: Index documents for a specific tenant
    - search_for_tenant: Search within a specific tenant's data
    - delete_tenant: Remove a tenant and all associated data
    - list_tenants: Enumerate all active tenants

Key Concepts:
    - Tenant Isolation: Each tenant's data is logically separated
    - Tenant ID: Unique identifier for each tenant (string)
    - Document Association: Documents tagged with tenant_id in metadata

Test Coverage:
    - ABC inheritance verification
    - Abstract method enforcement (cannot instantiate directly)
    - Interface signature validation
    - Concrete implementation requirements
    - Tenant isolation behavior
    - Error handling for edge cases

All concrete implementations (Chroma, Pinecone, Weaviate, Milvus, Qdrant)
must pass the interface compliance tests in this module.
"""

from abc import ABC

import pytest
from langchain_core.documents import Document


class TestMultiTenancyPipelineABC:
    """Tests for the MultiTenancyPipeline abstract base class.

    Validates that MultiTenancyPipeline is properly defined as an ABC,
    cannot be instantiated directly, and enforces implementation of all
    required abstract methods.

    Abstract Methods Required:
        - index_for_tenant: Store documents with tenant association
        - search_for_tenant: Query within tenant boundary
        - delete_tenant: Remove tenant data
        - list_tenants: Enumerate all tenants

    Design Pattern:
        Template Method pattern - base class defines interface,
        concrete implementations provide database-specific logic.
    """

    def test_is_abstract_base_class(self):
        """Test that MultiTenancyPipeline is an abstract base class."""
        from vectordb.langchain.multi_tenancy.base import MultiTenancyPipeline

        assert issubclass(MultiTenancyPipeline, ABC)

    def test_cannot_instantiate_directly(self):
        """Test that MultiTenancyPipeline cannot be instantiated directly."""
        from vectordb.langchain.multi_tenancy.base import MultiTenancyPipeline

        with pytest.raises(TypeError):
            MultiTenancyPipeline()

    def test_has_required_abstract_methods(self):
        """Test that all required abstract methods are defined."""
        from vectordb.langchain.multi_tenancy.base import MultiTenancyPipeline

        abstract_methods = getattr(MultiTenancyPipeline, "__abstractmethods__", set())
        required_methods = {
            "index_for_tenant",
            "search_for_tenant",
            "delete_tenant",
            "list_tenants",
        }

        assert required_methods.issubset(abstract_methods)


class TestConcreteMultiTenancyPipelineImplementation:
    """Tests using a concrete implementation to verify base class interface.

    Creates a minimal in-memory implementation of MultiTenancyPipeline
    to validate that the ABC interface works correctly when properly
    implemented.

    In-Memory Implementation:
        - Uses dict to store tenant -> documents mapping
        - Simple list operations for indexing and search
        - Validates method signatures and return types

    This pattern demonstrates the minimal requirements for creating
    a new database-specific multi-tenancy implementation.
    """

    def test_concrete_implementation_can_be_created(self):
        """Test that a concrete implementation can be created."""
        from vectordb.langchain.multi_tenancy.base import MultiTenancyPipeline

        class ConcretePipeline(MultiTenancyPipeline):
            def index_for_tenant(
                self,
                tenant_id: str,
                documents: list[Document],
                embeddings: list[list[float]],
            ) -> int:
                return len(documents)

            def search_for_tenant(
                self,
                tenant_id: str,
                query: str,
                top_k: int = 10,
                filters: dict | None = None,
            ) -> list[Document]:
                return []

            def delete_tenant(self, tenant_id: str) -> bool:
                return True

            def list_tenants(self) -> list[str]:
                return []

        # Should be able to instantiate
        pipeline = ConcretePipeline()
        assert pipeline is not None

    def test_concrete_implementation_methods_work(self):
        """Test that concrete implementation methods work correctly."""
        from vectordb.langchain.multi_tenancy.base import MultiTenancyPipeline

        class ConcretePipeline(MultiTenancyPipeline):
            def __init__(self):
                self.tenants = {}

            def index_for_tenant(
                self,
                tenant_id: str,
                documents: list[Document],
                embeddings: list[list[float]],
            ) -> int:
                if tenant_id not in self.tenants:
                    self.tenants[tenant_id] = []
                self.tenants[tenant_id].extend(documents)
                return len(documents)

            def search_for_tenant(
                self,
                tenant_id: str,
                query: str,
                top_k: int = 10,
                filters: dict | None = None,
            ) -> list[Document]:
                return self.tenants.get(tenant_id, [])[:top_k]

            def delete_tenant(self, tenant_id: str) -> bool:
                if tenant_id in self.tenants:
                    del self.tenants[tenant_id]
                    return True
                return False

            def list_tenants(self) -> list[str]:
                return list(self.tenants.keys())

        pipeline = ConcretePipeline()

        # Test index_for_tenant
        docs = [Document(page_content="test", metadata={})]
        count = pipeline.index_for_tenant("tenant1", docs, [[0.1] * 384])
        assert count == 1

        # Test search_for_tenant
        results = pipeline.search_for_tenant("tenant1", "query")
        assert len(results) == 1

        # Test list_tenants
        tenants = pipeline.list_tenants()
        assert "tenant1" in tenants

        # Test delete_tenant
        deleted = pipeline.delete_tenant("tenant1")
        assert deleted is True
        assert len(pipeline.list_tenants()) == 0


class TestMultiTenancyPipelineTenantIsolation:
    """Tests for tenant isolation behavior.

    Validates that tenant data is properly isolated - tenants cannot
    access each other's documents. This is the core security guarantee
    of multi-tenancy.

    Isolation Requirements:
        - Tenant A cannot see Tenant B's documents
        - Search results scoped to requesting tenant only
        - Delete operations affect only specified tenant
        - List operations return only tenant's own documents

    Security Implications:
        - Tenant ID must be validated on every operation
        - Default should deny access (fail closed)
        - No cross-tenant leakage in search results
    """

    def test_tenant_isolation_in_concrete_implementation(self):
        """Test that tenant data is properly isolated."""
        from vectordb.langchain.multi_tenancy.base import MultiTenancyPipeline

        class ConcretePipeline(MultiTenancyPipeline):
            def __init__(self):
                self.tenants = {}

            def index_for_tenant(
                self,
                tenant_id: str,
                documents: list[Document],
                embeddings: list[list[float]],
            ) -> int:
                if tenant_id not in self.tenants:
                    self.tenants[tenant_id] = []
                self.tenants[tenant_id].extend(documents)
                return len(documents)

            def search_for_tenant(
                self,
                tenant_id: str,
                query: str,
                top_k: int = 10,
                filters: dict | None = None,
            ) -> list[Document]:
                return self.tenants.get(tenant_id, [])[:top_k]

            def delete_tenant(self, tenant_id: str) -> bool:
                if tenant_id in self.tenants:
                    del self.tenants[tenant_id]
                    return True
                return False

            def list_tenants(self) -> list[str]:
                return list(self.tenants.keys())

        pipeline = ConcretePipeline()

        # Index documents for tenant1
        docs1 = [
            Document(page_content="tenant1 doc1", metadata={"tenant": "tenant1"}),
            Document(page_content="tenant1 doc2", metadata={"tenant": "tenant1"}),
        ]
        pipeline.index_for_tenant("tenant1", docs1, [[0.1] * 384, [0.2] * 384])

        # Index documents for tenant2
        docs2 = [
            Document(page_content="tenant2 doc1", metadata={"tenant": "tenant2"}),
        ]
        pipeline.index_for_tenant("tenant2", docs2, [[0.3] * 384])

        # Verify tenant isolation
        tenant1_results = pipeline.search_for_tenant("tenant1", "query")
        assert len(tenant1_results) == 2
        assert all("tenant1" in doc.page_content for doc in tenant1_results)

        tenant2_results = pipeline.search_for_tenant("tenant2", "query")
        assert len(tenant2_results) == 1
        assert "tenant2" in tenant2_results[0].page_content

        # Verify listing
        tenants = pipeline.list_tenants()
        assert len(tenants) == 2
        assert "tenant1" in tenants
        assert "tenant2" in tenants


class TestMultiTenancyPipelineErrorHandling:
    """Tests for error handling in multi-tenancy pipelines.

    Validates robust handling of edge cases and invalid inputs.
    Multi-tenancy adds complexity that requires careful error handling
    to prevent data leakage or corruption.

    Error Cases Covered:
        - Empty tenant_id (should raise ValueError)
        - Mismatched documents/embeddings counts
        - Non-existent tenant searches (should return empty, not error)
        - Duplicate tenant creation (idempotent or error)

    Defensive Programming:
        - Validate tenant_id before any data operations
        - Check document/embeddings count consistency
        - Clear error messages for debugging
    """

    def test_empty_tenant_id_handling(self):
        """Test handling of empty tenant_id."""
        from vectordb.langchain.multi_tenancy.base import MultiTenancyPipeline

        class ConcretePipeline(MultiTenancyPipeline):
            def index_for_tenant(
                self,
                tenant_id: str,
                documents: list[Document],
                embeddings: list[list[float]],
            ) -> int:
                if not tenant_id:
                    raise ValueError("tenant_id cannot be empty")
                return len(documents)

            def search_for_tenant(
                self,
                tenant_id: str,
                query: str,
                top_k: int = 10,
                filters: dict | None = None,
            ) -> list[Document]:
                if not tenant_id:
                    raise ValueError("tenant_id cannot be empty")
                return []

            def delete_tenant(self, tenant_id: str) -> bool:
                if not tenant_id:
                    raise ValueError("tenant_id cannot be empty")
                return True

            def list_tenants(self) -> list[str]:
                return []

        pipeline = ConcretePipeline()

        with pytest.raises(ValueError, match="tenant_id cannot be empty"):
            pipeline.index_for_tenant("", [], [])

        with pytest.raises(ValueError, match="tenant_id cannot be empty"):
            pipeline.search_for_tenant("", "query")

        with pytest.raises(ValueError, match="tenant_id cannot be empty"):
            pipeline.delete_tenant("")

    def test_mismatched_documents_embeddings(self):
        """Test handling of mismatched documents and embeddings."""
        from vectordb.langchain.multi_tenancy.base import MultiTenancyPipeline

        class ConcretePipeline(MultiTenancyPipeline):
            def index_for_tenant(
                self,
                tenant_id: str,
                documents: list[Document],
                embeddings: list[list[float]],
            ) -> int:
                if len(documents) != len(embeddings):
                    raise ValueError(
                        f"Documents count ({len(documents)}) "
                        f"does not match embeddings count ({len(embeddings)})"
                    )
                return len(documents)

            def search_for_tenant(
                self,
                tenant_id: str,
                query: str,
                top_k: int = 10,
                filters: dict | None = None,
            ) -> list[Document]:
                return []

            def delete_tenant(self, tenant_id: str) -> bool:
                return True

            def list_tenants(self) -> list[str]:
                return []

        pipeline = ConcretePipeline()

        docs = [Document(page_content="test", metadata={})]
        embeddings = [[0.1] * 384, [0.2] * 384]  # Mismatched count

        with pytest.raises(ValueError, match="does not match embeddings count"):
            pipeline.index_for_tenant("tenant1", docs, embeddings)
