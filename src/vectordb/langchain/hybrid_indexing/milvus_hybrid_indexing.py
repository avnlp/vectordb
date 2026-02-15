import argparse
from ast import literal_eval

from dataloaders import ARCDataloader, EdgarDataloader, FactScoreDataloader, PopQADataloader, TriviaQADataloader
from dataloaders.llms import ChatGroqGenerator
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from pymilvus import CollectionSchema, FieldSchema, DataType, MilvusClient

from vectordb import MilvusDocumentConverter

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Run a query using Milvus VectorDB with HuggingFace embeddings.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["triviaqa", "arc", "popqa", "factscore", "edgar"],
        required=True,
        help="The dataset name.",
    )
    parser.add_argument(
        "--split", type=str, default="test[:5]", help="The split of the dataset to load (default is 'test[:5]')."
    )
    parser.add_argument(
        "--model", type=str, default="llama-3.1-8b-instant", help="LLM model name (default is 'llama-3.1-8b-instant')."
    )
    parser.add_argument("--api_key", type=str, required=True, help="API key for the LLM service.")
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Embedding model name (default is 'sentence-transformers/all-mpnet-base-v2').",
    )
    parser.add_argument("--top_k", type=int, default=10, help="Number of top results to retrieve (default is 10).")
    parser.add_argument(
        "--collection_name",
        type=str,
        default="test_collection_dense1",
        help="Milvus collection name (default is 'test_collection_dense1').",
    )
    parser.add_argument(
        "--milvus_host",
        type=str,
        default="localhost",
        help="Milvus host address (default is 'localhost').",
    )
    parser.add_argument(
        "--milvus_port",
        type=str,
        default="19530",
        help="Milvus port (default is '19530').",
    )
    args = parser.parse_args()

    # Initialize the generator
    generator = ChatGroqGenerator(
        model=args.model,
        api_key=args.api_key,
        llm_params={"temperature": 0, "max_tokens": 1024, "timeout": 360, "max_retries": 100},
    )

    # Choose the appropriate dataloader class based on the dataset name
    dataloader_map = {
        "triviaqa": TriviaQADataloader,
        "arc": ARCDataloader,
        "popqa": PopQADataloader,
        "factscore": FactScoreDataloader,
        "edgar": EdgarDataloader,
    }
    dataloader_cls = dataloader_map[args.dataset_name]
    dataloader = dataloader_cls(
        dataset_name=f"awinml/{args.dataset_name}", split=args.split, answer_summary_generator=generator
    )

    # Load the data
    dataloader.load_data()
    dataloader.get_questions()
    langchain_documents = dataloader.get_langchain_documents()

    # Load the embedding model
    embedder = HuggingFaceEmbeddings(model_name=args.embedding_model)

    # Create the embeddings for each document
    texts = [doc.page_content for doc in langchain_documents]
    embeddings = embedder.embed_documents(texts=texts)

    # Initialize Milvus client
    client = MilvusClient(uri=f"http://{args.milvus_host}:{args.milvus_port}")

    # Create a schema for the Milvus collection
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=len(embeddings[0])),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500),
        ]
    )

    # Create or get the collection
    if not client.has_collection(args.collection_name):
        client.create_collection(collection_name=args.collection_name, schema=schema)

    # Prepare the data for insertion
    data_for_milvus = MilvusDocumentConverter.prepare_langchain_documents_for_upsert(langchain_documents, embeddings)

    # Insert data into Milvus
    client.insert(
        collection_name=args.collection_name,
        data={
            "embedding": [doc["embedding"] for doc in data_for_milvus],
            "text": [doc["text"] for doc in data_for_milvus],
        },
    )

    print(f"Inserted {len(data_for_milvus)} documents into Milvus collection '{args.collection_name}'.")

if __name__ == "__main__":
    main()
