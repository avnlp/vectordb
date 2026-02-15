from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, PointStruct, ScoredPoint, VectorParams


class QdrantVectorDB:
    """Interface for interacting with Qdrant vector databases.

    Provides functionalities for creating collections, inserting vectors,
    querying vectors, and deleting collections.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
        timeout: Optional[float] = 60.0,
    ):
        """Initialize the Qdrant client with the given parameters.

        Args:
            host (str): Host address of the Qdrant server
                (default: "localhost").
            port (int): Port number for the Qdrant server (default: 6333).
            api_key (Optional[str]): API key for Qdrant
                (if using managed service).
            collection_name (Optional[str]): Name of the Qdrant collection
                to use (optional).
            timeout (Optional[float]): Timeout for client requests
                in seconds (default: 60.0).
            retries (Optional[int]): Number of retries for failed requests
                (default: 3).
        """
        self.client = QdrantClient(
            host=host, port=port, api_key=api_key, timeout=timeout
        )
        self.collection_name = collection_name

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "Cosine",
    ):
        """Create a Qdrant collection with the specified configuration.

        Args:
            collection_name (str): Name of the new collection.
            vector_size (int): Dimensionality of the vectors.
            distance (str): Distance metric for similarity search (default: "Cosine").
        """
        self.collection_name = collection_name
        if self.client.get_collection(collection_name, raise_on_not_found=False):
            logger.info(
                f"Collection '{collection_name}' already exists. Skipping creation."
            )
            return

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )
        logger.info(f"Collection '{collection_name}' created successfully.")

    def upsert_vectors(
        self,
        vectors: List[Dict[str, Any]],
    ):
        """Insert or update vectors in the specified Qdrant collection.

        Args:
            vectors (List[Dict[str, Any]]): List of vectors to upsert,
                where each vector contains `id`, `vector`,
                and optional `payload`.
        """
        if not self.collection_name:
            raise ValueError(
                "No collection selected. Create or specify a collection first."
            )

        points = [
            PointStruct(
                id=vector["id"], vector=vector["vector"], payload=vector.get("payload")
            )
            for vector in vectors
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(
            f"Upserted {len(vectors)} vectors into collection '{self.collection_name}'."
        )

    def query_vectors(
        self,
        query_vector: List[float],
        top_k: int = 5,
        query_filter: Optional[Filter] = None,
        include_payload: bool = True,
    ) -> List[ScoredPoint]:
        """Query the collection for nearest vectors to the given query vector.

        Args:
            query_vector (List[float]): Query vector.
            top_k (int): Number of nearest neighbors to return
                (default: 5).
            query_filter (Optional[Filter]): Filter to apply to the query
                (default: None).
            include_payload (bool): Whether to include payload in the
                query result (default: True).

        Returns:
            List[ScoredPoint]: List of points with their similarity scores.
        """
        if not self.collection_name:
            raise ValueError(
                "No collection selected. Create or specify a collection first."
            )

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=include_payload,
        )
        logger.info(f"Query returned {len(results)} results.")
        return results

    def delete_collection(self):
        """Delete the current Qdrant collection."""
        if not self.collection_name:
            raise ValueError("No collection selected for deletion.")

        self.client.delete_collection(self.collection_name)
        logger.info(f"Collection '{self.collection_name}' deleted successfully.")
        self.collection_name = None

    def list_collections(self) -> List[str]:
        """List all collections in the Qdrant database.

        Returns:
            List[str]: List of collection names.
        """
        collections = self.client.get_collections().collections
        return [collection.name for collection in collections]
