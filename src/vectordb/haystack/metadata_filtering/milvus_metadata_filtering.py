import argparse
from ast import literal_eval
from haystack.components.embedders import SentenceTransformersTextEmbedder

from vectordb import MilvusVectorDB


def main():
    """Perform Metadata Filtering using Milvus."""
    # Argument parser for user inputs
    parser = argparse.ArgumentParser(description="Script for querying Milvus with metadata filtering.")

    # Dataloader parameters
    parser.add_argument(
        "--dataloader",
        required=True,
        choices=["triviaqa", "arc", "popqa", "factscore", "edgar"],
        help="Dataloader to use for loading datasets.",
    )
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset to be used by the dataloader.")
    parser.add_argument("--split", default="test", help="Dataset split to process (e.g., 'test', 'train').")
    parser.add_argument(
        "--text_splitter",
        default="RecursiveCharacterTextSplitter",
        help="Text splitter method to preprocess documents.",
    )
    parser.add_argument(
        "--text_splitter_params", type=str, help="JSON string of parameters for configuring the text splitter."
    )

    # Generator parameters
    parser.add_argument("--generator_model", type=str, help="Model name for the dataloader's generator.")
    parser.add_argument("--generator_api_key", help="API key for the dataloader generator.")
    parser.add_argument("--generator_llm_params", type=str, help="JSON string of parameters for the generator LLM.")

    # Embedder parameters
    parser.add_argument(
        "--embedding_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model to use for generating document embeddings.",
    )
    parser.add_argument("--embedding_model_params", type=str, help="JSON string of parameters for the embedding model.")

    # Milvus VectorDB parameters
    parser.add_argument("--milvus_host", required=True, help="Milvus server host.")
    parser.add_argument("--milvus_port", default="19530", help="Milvus server port.")
    parser.add_argument("--collection_name", required=True, help="Name of the collection to query.")
    parser.add_argument("--filter_expression", required=True, help="Filter expression for metadata filtering.")
    parser.add_argument("--limit", type=int, default=10, help="Number of results to retrieve.")

    args = parser.parse_args()

    # Parse parameters
    text_splitter_params = literal_eval(args.text_splitter_params) if args.text_splitter_params else {}
    generator_params = literal_eval(args.generator_llm_params) if args.generator_llm_params else {}
    embedding_model_params = literal_eval(args.embedding_model_params) if args.embedding_model_params else {}

    # Instantiate generator if model and API key are provided
    generator = None
    if args.generator_model and args.generator_api_key:
        generator = ChatGroqGenerator(
            model=args.generator_model,
            api_key=args.generator_api_key,
            llm_params=generator_params,
        )

    # Map the dataloader name to its corresponding class
    dataloader_map = {
        "triviaqa": TriviaQADataloader,
        "arc": ARCDataloader,
        "popqa": PopQADataloader,
        "factscore": FactScoreDataloader,
        "edgar": EdgarDataloader,
    }
    dataloader_cls = dataloader_map[args.dataloader]
    dataloader_kwargs = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "text_splitter": args.text_splitter,
        "text_splitter_params": text_splitter_params,
    }
    if generator:
        dataloader_kwargs["answer_summary_generator"] = generator
    dataloader = dataloader_cls(**dataloader_kwargs)

    # Instantiate the text embedder
    text_embedder = SentenceTransformersTextEmbedder(model=args.embedding_model)
    text_embedder.warm_up()

    # Get the embedding for the question
    question = "Your Query Text Here"  # Replace with actual query text if needed
    dense_question_embedding = text_embedder.run(text=question)["embedding"]

    # Initialize Milvus VectorDB
    milvus_vectordb = MilvusVectorDB(host=args.milvus_host, port=args.milvus_port)

    # Perform the query with metadata filtering
    query_results = milvus_vectordb.query(
        collection_name=args.collection_name,
        vector=dense_question_embedding,
        limit=args.limit,
        filter_expression=args.filter_expression,  # Pass the filter expression for metadata filtering
    )

    # Log the results
    for result in query_results:
        print(result)


if __name__ == "__main__":
    main()

