import argparse
from ast import literal_eval
from dataloaders import TriviaQADataloader
from dataloaders.llms import ChatGroqGenerator
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType

def main():
    """Main function to handle Milvus indexing and querying."""
    # Argument parser for user inputs
    parser = argparse.ArgumentParser(description="Script for processing and indexing data with Milvus.")
    
    # Dataloader parameters
    parser.add_argument(
        "--dataloader",
        required=True,
        choices=["triviaqa", "arc", "popqa", "factscore", "edgar"],
        help="Dataloader to use for loading datasets."
    )
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset to be used by the dataloader.")
    parser.add_argument("--split", default="test", help="Dataset split to process (e.g., 'test', 'train').")
    parser.add_argument(
        "--text_splitter",
        default="RecursiveCharacterTextSplitter",
        help="Text splitter method to preprocess documents."
    )
    parser.add_argument("--text_splitter_params", type=str, help="JSON string of parameters for configuring the text splitter.")
    
    # Generator parameters
    parser.add_argument("--generator_model", type=str, help="Model name for the dataloader's generator.")
    parser.add_argument("--generator_api_key", help="API key for the dataloader generator.")
    parser.add_argument("--generator_llm_params", type=str, help="JSON string of parameters for the generator LLM.")
    
    # Embedder parameters
    parser.add_argument(
        "--embedding_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model to use for generating document embeddings."
    )
    parser.add_argument("--embedding_model_params", type=str, help="JSON string of parameters for the embedding model.")
    
    # Milvus parameters
    parser.add_argument("--milvus_uri", default="http://localhost:19530", help="URI for Milvus server.")
    parser.add_argument("--milvus_token", default="root:Milvus", help="Token for Milvus authentication.")
    parser.add_argument("--collection_name", required=True, help="Milvus collection name.")
    parser.add_argument("--partition_name", required=True, help="Partition name in Milvus collection.")
    parser.add_argument("--query", type=str, required=True, help="Query string for Milvus search.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Parse JSON strings
    text_splitter_params = literal_eval(args.text_splitter_params) if args.text_splitter_params else {}
    generator_params = literal_eval(args.generator_llm_params) if args.generator_llm_params else {}
    embedding_model_params = literal_eval(args.embedding_model_params) if args.embedding_model_params else {}
    
    # Initialize generator if model and API key are provided
    generator = None
    if args.generator_model and args.generator_api_key:
        generator = ChatGroqGenerator(
            model=args.generator_model,
            api_key=args.generator_api_key,
            llm_params=generator_params,
        )
    
    # Initialize dataloader
    dataloader = TriviaQADataloader(
        answer_summary_generator=generator,
        dataset_name=args.dataset_name,
        split=args.split,
        text_splitter=args.text_splitter,
        text_splitter_params=text_splitter_params,
    )
    
    # Load data and preprocess
    dataloader.load_data()
    langchain_documents = dataloader.get_langchain_documents()
    
    # Initialize the embedding model
    text_embedder = HuggingFaceEmbeddings(model_name=args.embedding_model)
    
    # Generate embeddings for documents
    docs_with_embeddings = [
        {"text": doc.text, "embedding": text_embedder.embed_query(doc.text)}
        for doc in langchain_documents
    ]
    
    # Connect to Milvus
    connections.connect("default", uri=args.milvus_uri, token=args.milvus_token)
    
    # Define Milvus schema and collection
    if len(docs_with_embeddings) == 0:
        raise ValueError("No documents found to embed and index.")
    embedding_dim = len(docs_with_embeddings[0]["embedding"])
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
    ]
    schema = CollectionSchema(fields, "Collection for document embeddings")
    
    collection = Collection(name=args.collection_name, schema=schema, using="default")
    if args.partition_name not in [partition.name for partition in collection.partitions]:
        collection.create_partition(args.partition_name)
    
    # Insert data into Milvus
    embeddings = [doc["embedding"] for doc in docs_with_embeddings]
    ids = list(range(len(embeddings)))
    collection.insert([ids, embeddings], partition_name=args.partition_name)
    
    # Generate query embedding
    query_embedding = text_embedder.embed_query(args.query)
    
    # Query data from Milvus
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=10,
        partition_names=[args.partition_name],
    )
    
    # Print results
    for result in results[0]:
        print(f"ID: {result.id}, Distance: {result.distance}")

if __name__ == "__main__":
    main()

