# Vector Databases

- Vector databases store and manage data as high-dimensional vectors, enabling similarity-based retrieval.
- They are designed to efficiently find the most relevant entries by comparing vector embeddings, which represent the semantic meaning of data.
- Vector databases are integral to Retrieval-Augmented Generation (RAG) pipelines, where they enhance the retrieval of contextually relevant information before the response generation stage.

The main goal of this repo is to compare and contrast the functionality of the vector databases. We compare the following vector databases:

- Pinecone
- Weaviate
- Chroma

## Installation

```bash
pip install git+https://github.com/avnlp/vectordb
```

## Vector Embeddings

There are two main types of vector embeddings:

- Dense Embeddings
- Sparse Embeddings

### Dense Embeddings

- A dense embedding represents the semantic meaning of a piece of text in a high-dimensional numerical representation, where each element (or dimension) in the vector contains a real-valued number that contributes to the overall representation of the data's features.
- Pinecone, Weaviate, and Chroma support creation of indexes/collections with dense embeddings.

### Sparse Embeddings

- Sparse vectors have very large number of dimensions, where only a small proportion of values are non-zero.
- When used for keywords search, each sparse vector represents a document; the dimensions represent words from a dictionary, and the values represent the importance of these words in the document.
- Keyword search algorithms like the BM25 algorithm compute the relevance of text documents based on the number of keyword matches, their frequency, and other factors based on token presence in the document.

### Hybrid Search

- Hybrid search in Pinecone leverages a single sparse-dense index, enabling simultaneous retrieval based on both keyword relevance (sparse vector) and semantic context (dense vector). Querying this index requires providing both the sparse and dense vector representations of the query.
- During Index creation both dense and sparse embeddings need to be computed and specified for each document. This is an additional step that can be time-consuming for large datasets.
- Weaviate’s Hybrid search combines BM25-based keyword search and vector-based semantic search by merging the results from both methods. To enable hybrid search, the query must specify `hybrid=True`, allowing for retrieval that balances exact term matching and contextual understanding.
- Weaviate does not allow you to use custom sparse vectors for hybrid search. Only dense embeddings are required when creating the collection.
- Chroma does not currently support hybrid search capabilities.  

## Storing vector embeddings in a vector database

- The Pinecone vector database stores vector embeddings in an Index. Each index can be partitioned into multiple namespaces.
- Every index is made up of one or more namespaces and they are uniquely identified by a namespace name. Every record exists in exactly one namespace.
- Queries and other operations are confined to one namespace, so different requests can search different subsets of your index.

Example: Creating an Index with multiple namespaces

```python
pinecone_vector_db = PineconeVectorDB(api_key=pinecone_api_key)
pinecone_vector_db.create_index(
    index_name="arc_index",
    dimension=768,
    metric="dotproduct",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
pinecone_vector_db.upsert(data=arc_train, namespace="train")
pinecone_vector_db.upsert(data=arc_dev, namespace="dev")
```

Example: Querying a specific namespace

```python
pinecone_vector_db.query(vector=query_embedding, namespace="dev", top_k=5)
```

- The Weaviate vector database stores vector embeddings in a Collection.
- Multiple independent collections can be part of a single Weaviate Cluster.

Example: Creating multiple independent collections in a Weaviate Cluster

```python
weaviate_vector_db = WeaviateVectorDB(cluster_url=weaviate_cluster_url, api_key=weaviate_api_key)
weaviate_vector_db.create_collection(collection_name="arc_train")
weaviate_vector_db.create_collection(collection_name="arc_dev")
```

- The Chroma vector database stores vector embeddings in a Collection. It does not support namespaces. Each collection is independent.

Example: Creating a Chroma Collection

```python
chroma_vector_db = ChromaVectorDB(path="./chroma")
chroma_vector_db.create_collection(name="arc_train")
```

## Metadata Filtering

- Metadata fields are a way to add information to individual vectors to give them more meaning. By adding metadata to your vectors, you can filter by those fields at query time. You can limit your vector search based on metadata.
- All three vector databases let you attach metadata key-value pairs to vectors in an index/collection, and specify filter expressions when you query it.
- The metadata is included in the payload when you add your vectors.

Example: Metadata Filters in Pinecone

Pinecone supports metadata filtering using the `filter` parameter. The `filter` parameter takes a dictionary that specifies a filter expression.

```python
query_response = pinecone_vector_db.query(
    vector=dense_question_embedding,
    sparse_vector=sparse_question_embedding,
    top_k=10,
    include_metadata=True,
    namespace="test_namespace",
    filter={"$and": [{"id": {"$eq": "752235"}}, {"title": {"$eq": "Pete Sampras"}}]},
)
```

Example: Metadata Filters in Weaviate

Metadata filters in Weaviate are passed using the `filters` parameter. A list of `Filter` objects can be passed to the `filters` parameter.

```python
query_response = weaviate_vector_db.query(
    vector=dense_question_embedding,
    query_string=question,
    limit=10,
    hybrid=True,
    alpha=0.5,
    filters=Filter.by_property("text").like("Pete Sampras"),
)
```

Example: Metadata Filters in Chroma

Metadata filters in ChromaDB can be used to filter documents based on specific metadata or content criteria during a query. They are passed using `where_document` parameter.

```python
query_response = chroma_vector_db.query(
    query_embedding=dense_question_embedding, n_results=10, where_document={"$contains": "Pete Sampras"}
)
```
