"""Unit tests for ParentDocumentStore."""

import pickle
import tempfile
from pathlib import Path

import pytest

from vectordb.langchain.parent_document_retrieval.parent_store import (
    ParentDocumentStore,
)


class TestParentDocumentStoreInitialization:
    """Test ParentDocumentStore initialization."""

    def test_init_without_cache_dir(self) -> None:
        """Test initialization without cache directory."""
        store = ParentDocumentStore()
        assert store.parent_map == {}
        assert store.chunk_to_parent == {}
        assert store.cache_dir is None

    def test_init_with_cache_dir(self) -> None:
        """Test initialization with cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ParentDocumentStore(cache_dir=tmpdir)
            assert store.cache_dir == tmpdir
            assert Path(tmpdir).exists()

    def test_init_creates_cache_dir(self) -> None:
        """Test that cache_dir is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = str(Path(tmpdir) / "nested" / "cache" / "dir")
            store = ParentDocumentStore(cache_dir=nested_path)
            assert Path(nested_path).exists()
            assert store.cache_dir == nested_path


class TestAddParent:
    """Test add_parent method."""

    def test_add_single_parent(self) -> None:
        """Test adding a single parent document."""
        store = ParentDocumentStore()
        parent_doc = {"text": "Parent content", "id": "p1"}
        store.add_parent("parent_1", parent_doc)

        assert "parent_1" in store.parent_map
        assert store.parent_map["parent_1"] == parent_doc

    def test_add_multiple_parents(self) -> None:
        """Test adding multiple parent documents."""
        store = ParentDocumentStore()
        parents = {
            "parent_1": {"text": "Content 1", "metadata": {"source": "doc1"}},
            "parent_2": {"text": "Content 2", "metadata": {"source": "doc2"}},
            "parent_3": {"text": "Content 3", "metadata": {"source": "doc3"}},
        }

        for parent_id, parent_doc in parents.items():
            store.add_parent(parent_id, parent_doc)

        assert len(store.parent_map) == 3
        for parent_id, parent_doc in parents.items():
            assert store.parent_map[parent_id] == parent_doc

    def test_overwrite_existing_parent(self) -> None:
        """Test overwriting an existing parent document."""
        store = ParentDocumentStore()
        store.add_parent("parent_1", {"text": "Original content"})
        new_content = {"text": "Updated content"}
        store.add_parent("parent_1", new_content)

        assert len(store.parent_map) == 1
        assert store.parent_map["parent_1"] == new_content


class TestAddChunkMapping:
    """Test add_chunk_mapping method."""

    def test_add_single_mapping(self) -> None:
        """Test adding a single chunk-to-parent mapping."""
        store = ParentDocumentStore()
        store.add_chunk_mapping("chunk_1", "parent_1")

        assert "chunk_1" in store.chunk_to_parent
        assert store.chunk_to_parent["chunk_1"] == "parent_1"

    def test_add_multiple_mappings(self) -> None:
        """Test adding multiple chunk-to-parent mappings."""
        store = ParentDocumentStore()
        mappings = {
            "chunk_1": "parent_1",
            "chunk_2": "parent_1",
            "chunk_3": "parent_2",
            "chunk_4": "parent_2",
        }

        for chunk_id, parent_id in mappings.items():
            store.add_chunk_mapping(chunk_id, parent_id)

        assert len(store.chunk_to_parent) == 4
        for chunk_id, parent_id in mappings.items():
            assert store.chunk_to_parent[chunk_id] == parent_id

    def test_overwrite_existing_mapping(self) -> None:
        """Test overwriting an existing chunk mapping."""
        store = ParentDocumentStore()
        store.add_chunk_mapping("chunk_1", "parent_1")
        store.add_chunk_mapping("chunk_1", "parent_2")

        assert store.chunk_to_parent["chunk_1"] == "parent_2"


class TestGetParent:
    """Test get_parent method."""

    def test_get_existing_parent(self) -> None:
        """Test retrieving an existing parent by chunk ID."""
        store = ParentDocumentStore()
        parent_doc = {"text": "Parent content", "id": "p1"}
        store.add_parent("parent_1", parent_doc)
        store.add_chunk_mapping("chunk_1", "parent_1")

        result = store.get_parent("chunk_1")
        assert result == parent_doc

    def test_get_parent_nonexistent_chunk(self) -> None:
        """Test retrieving parent for nonexistent chunk returns None."""
        store = ParentDocumentStore()
        store.add_parent("parent_1", {"text": "Parent content"})

        result = store.get_parent("nonexistent_chunk")
        assert result is None

    def test_get_parent_unmapped_chunk(self) -> None:
        """Test retrieving parent when chunk has no mapping."""
        store = ParentDocumentStore()
        store.add_parent("parent_1", {"text": "Parent content"})
        store.add_chunk_mapping("chunk_1", "parent_1")

        result = store.get_parent("chunk_2")
        assert result is None

    def test_get_parent_missing_parent_doc(self) -> None:
        """Test retrieving parent when mapping points to missing parent."""
        store = ParentDocumentStore()
        store.chunk_to_parent["chunk_1"] = "parent_1"  # Add mapping without parent

        result = store.get_parent("chunk_1")
        assert result is None


class TestGetParentById:
    """Test get_parent_by_id method."""

    def test_get_existing_parent_by_id(self) -> None:
        """Test retrieving an existing parent by parent ID."""
        store = ParentDocumentStore()
        parent_doc = {"text": "Parent content", "id": "p1"}
        store.add_parent("parent_1", parent_doc)

        result = store.get_parent_by_id("parent_1")
        assert result == parent_doc

    def test_get_nonexistent_parent_by_id(self) -> None:
        """Test retrieving nonexistent parent by ID returns None."""
        store = ParentDocumentStore()
        store.add_parent("parent_1", {"text": "Parent content"})

        result = store.get_parent_by_id("nonexistent_parent")
        assert result is None

    def test_get_parent_by_id_empty_store(self) -> None:
        """Test retrieving parent from empty store returns None."""
        store = ParentDocumentStore()
        result = store.get_parent_by_id("any_parent")
        assert result is None


class TestGetParentsForChunks:
    """Test get_parents_for_chunks method."""

    def test_get_single_chunk_single_parent(self) -> None:
        """Test retrieving parent for single chunk."""
        store = ParentDocumentStore()
        parent_doc = {"text": "Parent content"}
        store.add_parent("parent_1", parent_doc)
        store.add_chunk_mapping("chunk_1", "parent_1")

        result = store.get_parents_for_chunks(["chunk_1"])
        assert len(result) == 1
        assert result[0] == parent_doc

    def test_get_multiple_chunks_multiple_parents(self) -> None:
        """Test retrieving parents for multiple chunks with different parents."""
        store = ParentDocumentStore()
        parent_1 = {"text": "Parent 1"}
        parent_2 = {"text": "Parent 2"}
        parent_3 = {"text": "Parent 3"}

        store.add_parent("parent_1", parent_1)
        store.add_parent("parent_2", parent_2)
        store.add_parent("parent_3", parent_3)

        store.add_chunk_mapping("chunk_1", "parent_1")
        store.add_chunk_mapping("chunk_2", "parent_2")
        store.add_chunk_mapping("chunk_3", "parent_3")

        result = store.get_parents_for_chunks(["chunk_1", "chunk_2", "chunk_3"])
        assert len(result) == 3
        assert parent_1 in result
        assert parent_2 in result
        assert parent_3 in result

    def test_get_parents_for_chunks_deduplication(self) -> None:
        """Test that duplicate parents are deduplicated."""
        store = ParentDocumentStore()
        parent_doc = {"text": "Parent content"}
        store.add_parent("parent_1", parent_doc)

        # Map multiple chunks to same parent
        store.add_chunk_mapping("chunk_1", "parent_1")
        store.add_chunk_mapping("chunk_2", "parent_1")
        store.add_chunk_mapping("chunk_3", "parent_1")

        result = store.get_parents_for_chunks(["chunk_1", "chunk_2", "chunk_3"])
        assert len(result) == 1
        assert result[0] == parent_doc

    def test_get_parents_for_chunks_mixed_existing_missing(self) -> None:
        """Test retrieving parents when some chunks don't have parents."""
        store = ParentDocumentStore()
        parent_1 = {"text": "Parent 1"}
        parent_2 = {"text": "Parent 2"}

        store.add_parent("parent_1", parent_1)
        store.add_parent("parent_2", parent_2)

        store.add_chunk_mapping("chunk_1", "parent_1")
        store.add_chunk_mapping("chunk_2", "parent_2")
        # chunk_3 has no mapping

        result = store.get_parents_for_chunks(["chunk_1", "chunk_2", "chunk_3"])
        assert len(result) == 2
        assert parent_1 in result
        assert parent_2 in result

    def test_get_parents_for_chunks_empty_list(self) -> None:
        """Test retrieving parents for empty chunk list."""
        store = ParentDocumentStore()
        store.add_parent("parent_1", {"text": "Parent content"})

        result = store.get_parents_for_chunks([])
        assert result == []

    def test_get_parents_for_chunks_all_nonexistent(self) -> None:
        """Test retrieving parents when no chunks have mappings."""
        store = ParentDocumentStore()
        store.add_parent("parent_1", {"text": "Parent content"})

        result = store.get_parents_for_chunks(["chunk_1", "chunk_2", "chunk_3"])
        assert result == []

    def test_get_parents_for_chunks_complex_scenario(self) -> None:
        """Test complex scenario with mixed mappings and deduplication."""
        store = ParentDocumentStore()
        parent_1 = {"text": "Parent 1", "id": "p1"}
        parent_2 = {"text": "Parent 2", "id": "p2"}
        parent_3 = {"text": "Parent 3", "id": "p3"}

        store.add_parent("parent_1", parent_1)
        store.add_parent("parent_2", parent_2)
        store.add_parent("parent_3", parent_3)

        # Multiple chunks map to same parents
        store.add_chunk_mapping("chunk_1", "parent_1")
        store.add_chunk_mapping("chunk_2", "parent_1")
        store.add_chunk_mapping("chunk_3", "parent_2")
        store.add_chunk_mapping("chunk_4", "parent_3")
        store.add_chunk_mapping("chunk_5", "parent_2")

        result = store.get_parents_for_chunks(
            ["chunk_1", "chunk_2", "chunk_3", "chunk_4", "chunk_5"]
        )
        assert len(result) == 3
        assert parent_1 in result
        assert parent_2 in result
        assert parent_3 in result


class TestSave:
    """Test save method."""

    def test_save_with_cache_dir(self) -> None:
        """Test saving parent store to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ParentDocumentStore(cache_dir=tmpdir)
            store.add_parent("parent_1", {"text": "Parent content"})
            store.add_chunk_mapping("chunk_1", "parent_1")

            filepath = store.save("test_store.pkl")

            assert Path(filepath).exists()
            assert filepath.endswith("test_store.pkl")

    def test_save_without_cache_dir_raises_error(self) -> None:
        """Test that save raises ValueError when cache_dir is not set."""
        store = ParentDocumentStore()
        store.add_parent("parent_1", {"text": "Parent content"})

        with pytest.raises(ValueError, match="cache_dir not set"):
            store.save("test_store.pkl")

    def test_save_preserves_data(self) -> None:
        """Test that saved data can be loaded back correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ParentDocumentStore(cache_dir=tmpdir)
            parent_1 = {"text": "Parent 1", "metadata": {"source": "doc1"}}
            parent_2 = {"text": "Parent 2", "metadata": {"source": "doc2"}}

            store.add_parent("parent_1", parent_1)
            store.add_parent("parent_2", parent_2)
            store.add_chunk_mapping("chunk_1", "parent_1")
            store.add_chunk_mapping("chunk_2", "parent_2")

            filepath = store.save("test_store.pkl")

            # Load and verify
            with open(filepath, "rb") as f:
                loaded_data = pickle.load(f)

            assert loaded_data["parent_map"] == store.parent_map
            assert loaded_data["chunk_to_parent"] == store.chunk_to_parent

    def test_save_multiple_times(self) -> None:
        """Test saving the same store multiple times."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ParentDocumentStore(cache_dir=tmpdir)

            store.add_parent("parent_1", {"text": "Parent 1"})
            filepath1 = store.save("store_1.pkl")

            store.add_parent("parent_2", {"text": "Parent 2"})
            filepath2 = store.save("store_2.pkl")

            assert Path(filepath1).exists()
            assert Path(filepath2).exists()

            # Verify both files have different content
            with open(filepath1, "rb") as f:
                data1 = pickle.load(f)
            with open(filepath2, "rb") as f:
                data2 = pickle.load(f)

            assert len(data1["parent_map"]) == 1
            assert len(data2["parent_map"]) == 2


class TestLoad:
    """Test load method."""

    def test_load_saved_store(self) -> None:
        """Test loading a previously saved store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save
            store1 = ParentDocumentStore(cache_dir=tmpdir)
            parent_doc = {"text": "Parent content"}
            store1.add_parent("parent_1", parent_doc)
            store1.add_chunk_mapping("chunk_1", "parent_1")
            filepath = store1.save("test_store.pkl")

            # Load
            store2 = ParentDocumentStore.load(filepath)

            assert store2.parent_map == store1.parent_map
            assert store2.chunk_to_parent == store1.chunk_to_parent
            assert store2.cache_dir == tmpdir

    def test_load_nonexistent_file_raises_error(self) -> None:
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ParentDocumentStore.load("/nonexistent/path/store.pkl")

    def test_load_preserves_complex_data(self) -> None:
        """Test that loading preserves complex parent documents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store1 = ParentDocumentStore(cache_dir=tmpdir)

            complex_parent = {
                "text": "Complex parent",
                "metadata": {
                    "source": "document",
                    "page": 5,
                    "tags": ["tag1", "tag2"],
                },
                "nested": {"key1": "value1", "key2": [1, 2, 3]},
            }

            store1.add_parent("parent_1", complex_parent)
            store1.add_chunk_mapping("chunk_1", "parent_1")
            filepath = store1.save("test_store.pkl")

            store2 = ParentDocumentStore.load(filepath)

            assert store2.get_parent("chunk_1") == complex_parent
            assert store2.get_parent_by_id("parent_1") == complex_parent

    def test_load_multiple_documents(self) -> None:
        """Test loading store with multiple documents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store1 = ParentDocumentStore(cache_dir=tmpdir)

            for i in range(5):
                parent_id = f"parent_{i}"
                parent_doc = {"text": f"Parent {i}", "id": parent_id}
                store1.add_parent(parent_id, parent_doc)

                for j in range(3):
                    chunk_id = f"chunk_{i}_{j}"
                    store1.add_chunk_mapping(chunk_id, parent_id)

            filepath = store1.save("test_store.pkl")
            store2 = ParentDocumentStore.load(filepath)

            assert len(store2.parent_map) == 5
            assert len(store2.chunk_to_parent) == 15

            for i in range(5):
                assert store2.get_parent_by_id(f"parent_{i}") is not None
                for j in range(3):
                    assert store2.get_parent(f"chunk_{i}_{j}") is not None


class TestClear:
    """Test clear method."""

    def test_clear_empty_store(self) -> None:
        """Test clearing an empty store."""
        store = ParentDocumentStore()
        store.clear()

        assert len(store.parent_map) == 0
        assert len(store.chunk_to_parent) == 0

    def test_clear_populated_store(self) -> None:
        """Test clearing a populated store."""
        store = ParentDocumentStore()
        store.add_parent("parent_1", {"text": "Parent content"})
        store.add_chunk_mapping("chunk_1", "parent_1")

        assert len(store.parent_map) == 1
        assert len(store.chunk_to_parent) == 1

        store.clear()

        assert len(store.parent_map) == 0
        assert len(store.chunk_to_parent) == 0

    def test_clear_multiple_documents(self) -> None:
        """Test clearing store with multiple documents."""
        store = ParentDocumentStore()

        for i in range(5):
            store.add_parent(f"parent_{i}", {"text": f"Parent {i}"})

        for i in range(10):
            store.add_chunk_mapping(f"chunk_{i}", f"parent_{i % 5}")

        assert len(store.parent_map) == 5
        assert len(store.chunk_to_parent) == 10

        store.clear()

        assert len(store.parent_map) == 0
        assert len(store.chunk_to_parent) == 0


class TestLen:
    """Test __len__ method."""

    def test_len_empty_store(self) -> None:
        """Test length of empty store."""
        store = ParentDocumentStore()
        assert len(store) == 0

    def test_len_single_parent(self) -> None:
        """Test length with single parent."""
        store = ParentDocumentStore()
        store.add_parent("parent_1", {"text": "Parent content"})
        assert len(store) == 1

    def test_len_multiple_parents(self) -> None:
        """Test length with multiple parents."""
        store = ParentDocumentStore()

        for i in range(5):
            store.add_parent(f"parent_{i}", {"text": f"Parent {i}"})

        assert len(store) == 5

    def test_len_after_clear(self) -> None:
        """Test length after clearing."""
        store = ParentDocumentStore()
        store.add_parent("parent_1", {"text": "Parent content"})
        assert len(store) == 1

        store.clear()
        assert len(store) == 0


class TestContains:
    """Test __contains__ method."""

    def test_contains_existing_chunk(self) -> None:
        """Test checking if chunk mapping exists."""
        store = ParentDocumentStore()
        store.add_chunk_mapping("chunk_1", "parent_1")

        assert "chunk_1" in store

    def test_contains_nonexistent_chunk(self) -> None:
        """Test checking if nonexistent chunk mapping exists."""
        store = ParentDocumentStore()
        store.add_chunk_mapping("chunk_1", "parent_1")

        assert "chunk_2" not in store

    def test_contains_empty_store(self) -> None:
        """Test contains on empty store."""
        store = ParentDocumentStore()

        assert "any_chunk" not in store

    def test_contains_multiple_chunks(self) -> None:
        """Test contains with multiple chunks."""
        store = ParentDocumentStore()

        for i in range(5):
            store.add_chunk_mapping(f"chunk_{i}", f"parent_{i % 2}")

        for i in range(5):
            assert f"chunk_{i}" in store

        assert "chunk_5" not in store


class TestIntegration:
    """Integration tests for ParentDocumentStore."""

    def test_full_workflow(self) -> None:
        """Test complete workflow of store creation and retrieval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ParentDocumentStore(cache_dir=tmpdir)

            # Add multiple documents
            docs = {
                "parent_1": {"text": "First document", "id": 1},
                "parent_2": {"text": "Second document", "id": 2},
                "parent_3": {"text": "Third document", "id": 3},
            }

            for parent_id, doc in docs.items():
                store.add_parent(parent_id, doc)

            # Create chunk mappings
            mappings = [
                ("chunk_1", "parent_1"),
                ("chunk_2", "parent_1"),
                ("chunk_3", "parent_2"),
                ("chunk_4", "parent_2"),
                ("chunk_5", "parent_3"),
            ]

            for chunk_id, parent_id in mappings:
                store.add_chunk_mapping(chunk_id, parent_id)

            # Verify retrieval
            assert len(store) == 3
            assert len(store.chunk_to_parent) == 5

            # Test specific retrievals
            assert store.get_parent("chunk_1") == docs["parent_1"]
            assert store.get_parent("chunk_3") == docs["parent_2"]
            assert store.get_parent_by_id("parent_3") == docs["parent_3"]

            # Test batch retrieval with deduplication
            parents = store.get_parents_for_chunks(
                ["chunk_1", "chunk_2", "chunk_3", "chunk_4"]
            )
            assert len(parents) == 2

            # Save and load
            filepath = store.save("workflow_test.pkl")
            loaded_store = ParentDocumentStore.load(filepath)

            assert len(loaded_store) == 3
            assert loaded_store.get_parent("chunk_1") == docs["parent_1"]
            assert loaded_store.get_parent("chunk_5") == docs["parent_3"]

            # Clear
            loaded_store.clear()
            assert len(loaded_store) == 0
            assert loaded_store.chunk_to_parent == {}

    def test_multiple_loads_save_cycle(self) -> None:
        """Test multiple load-save cycles maintain data integrity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First cycle
            store1 = ParentDocumentStore(cache_dir=tmpdir)
            store1.add_parent("p1", {"text": "Content 1"})
            store1.add_chunk_mapping("c1", "p1")
            filepath1 = store1.save("store.pkl")

            # Load and modify
            store2 = ParentDocumentStore.load(filepath1)
            store2.add_parent("p2", {"text": "Content 2"})
            store2.add_chunk_mapping("c2", "p2")
            filepath2 = store2.save("store.pkl")

            # Load again
            store3 = ParentDocumentStore.load(filepath2)

            assert len(store3) == 2
            assert store3.get_parent("c1") == {"text": "Content 1"}
            assert store3.get_parent("c2") == {"text": "Content 2"}
