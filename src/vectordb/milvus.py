from typing import Any, List, Optional

import weave
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections


@weave.model
class MilvusVectorDB:
    """Implements an interface for interacting with Milvus vector databases, with Weave tracking.
    Provides functionalities for creating collections, inserting vectors, querying vectors, and deleting collections.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        collection_name: Optional[str] = None,
    ):
        """Initializes the Milvus client and connects to the server.

        :param host: The Milvus server host.
        :param port: The Milvus server port.
        :param collection_name: Name of the collection to operate on.
        """
        self.host = host
        self.port = port
        self.collection_name = weave.track(collection_name, name="collection_name")
        self.collection = None

        connections.connect(alias="default", host=host, port=port)
        weave.track(
            f"Connected to Milvus at {host}:{port}", name="milvus_connection_status"
        )

    def _initialize_weave(self, **weave_params) -> None:
        """Initialize Weave with the specified tracing project name.

        Sets up the Weave environment and creates a tracer for monitoring pipeline
        execution.

        Args:
            weave_params (dict[str, Any]): Additional parameters for configuring Weave.
        """
        weave.init("milvus", **weave_params)

    @weave.op()
    def create_collection(
        self,
        collection_name: str,
        dimension: int,
        metric_type: str = "L2",
        description: str = "",
    ):
        """Create a Milvus collection with the specified configuration.

        :param collection_name: Name of the new collection.
        :param dimension: Dimensionality of the vectors.
        :param metric_type: Metric type for similarity search (default: "L2").
        :param description: Description of the collection.
        """
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
        ]
        schema = CollectionSchema(fields, description=description)
        collection = Collection(name=collection_name, schema=schema)

        self.collection_name = weave.track(
            collection_name, name="created_collection_name"
        )
        self.collection = collection
        weave.track(
            f"Collection '{collection_name}' created.",
            name="collection_creation_status",
        )

    @weave.op()
    def insert_vectors(
        self, vectors: List[List[float]], ids: Optional[List[int]] = None
    ):
        """Insert vectors into the selected Milvus collection.

        :param vectors: List of vectors to insert.
        :param ids: Optional list of IDs for the vectors.
        """
        if not self.collection:
            raise ValueError(
                "No collection selected. Create or specify a collection before inserting vectors."
            )

        entities = {
            "id": ids if ids else list(range(len(vectors))),
            "vector": vectors,
        }
        self.collection.insert([entities])
        weave.track(
            f"Inserted {len(vectors)} vectors into collection '{self.collection_name}'.",
            name="insertion_status",
        )

    @weave.op()
    def query_vectors(
        self,
        vector: List[float],
        top_k: int = 5,
        metric_type: str = "L2",
    ) -> Any:
        """Query the Milvus collection for the nearest neighbors of the given vector.

        :param vector: The query vector.
        :param top_k: Number of nearest neighbors to return (default: 5).
        :param metric_type: Metric type for similarity search (default: "L2").
        :return: Query results.
        """
        if not self.collection:
            raise ValueError(
                "No collection selected. Create or specify a collection before querying."
            )

        search_params = {"metric_type": metric_type, "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
        )
        weave.track(
            f"Queried top {top_k} neighbors for the vector.", name="query_status"
        )
        return results

    @weave.op()
    def delete_collection(self):
        """Delete the current Milvus collection."""
        if not self.collection_name:
            raise ValueError("No collection specified for deletion.")

        self.collection.drop()
        weave.track(
            f"Collection '{self.collection_name}' deleted.", name="deletion_status"
        )
        self.collection = None
        self.collection_name = None

    def list_collections(self) -> List[str]:
        """List all collections in the connected Milvus instance.

        :return: List of collection names.
        """
        return weave.track(connections.list_collections(), name="list_collections")
