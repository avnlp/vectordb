from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from weaviate.classes.query import Filter

from vectordb import WeaviateDocumentConverter, WeaviateVectorDB

weaviate_vector_db = WeaviateVectorDB(
    cluster_url="https://kzlfxowbtpn5oyhcqbag.c0.us-west3.gcp.weaviate.cloud",
    api_key="1N2AxyRUtlPyQYjFlc9tdWvzc6PD8xRKHlXa",
    collection_name="test_collection_dense1",
)


text_embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


question = "Who was the man behind The Chipmunks?"
dense_question_embedding = text_embedder.embed_query(question)

query_response = weaviate_vector_db.query(
    vector=dense_question_embedding,
    query_string=question,
    limit=10,
    hybrid=True,
    alpha=0.5,
    filters=Filter.by_property("text").like("Chipmunks"),
)

retrieval_results = WeaviateDocumentConverter.convert_query_results_to_haystack_documents(query_response)
print(retrieval_results)
