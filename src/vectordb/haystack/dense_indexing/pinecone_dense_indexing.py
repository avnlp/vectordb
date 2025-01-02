import argparse
from ast import literal_eval

from dataloaders import ARCDataloader, EdgarDataloader, FactScoreDataloader, PopQADataloader, TriviaQADataloader
from dataloaders.llms import ChatGroqGenerator
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from pinecone import ServerlessSpec

from vectordb import PineconeDocumentConverter, PineconeVectorDB


def main():
    """Process data, generate embeddings, and index them into a Pinecone vector database.

    This script is designed to:
    - Load datasets using specified dataloaders.
    - Optionally use an LLM-based generator for answer summarization.
    - Split and preprocess text data.
    - Generate embeddings for documents.
    - Index processed data into a Pinecone vector database.
    """
    # Argument parser for user inputs
    parser = argparse.ArgumentParser(description="Script for processing and indexing data with Pinecone.")

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

    # Pinecone VectorDB parameters
    parser.add_argument("--api_key", required=True, help="API key for accessing Pinecone.")
    parser.add_argument("--host", help="Host URL for Pinecone.")
    parser.add_argument("--index_name", required=True, help="Name of the Pinecone index.")
    parser.add_argument("--proxy_url", help="Proxy URL for Pinecone.")
    parser.add_argument("--proxy_headers", type=str, help="JSON string of proxy headers for configuring Pinecone.")
    parser.add_argument("--ssl_ca_certs", help="Path to SSL CA certificates.")
    parser.add_argument("--ssl_verify", type=bool, default=True, help="Enable or disable SSL verification.")
    parser.add_argument(
        "--additional_headers", type=str, help="JSON string of additional headers for Pinecone requests."
    )
    parser.add_argument("--pool_threads", type=int, default=1, help="Number of threads for connection pooling.")
    parser.add_argument("--namespace", required=True, help="Namespace for data upserts in Pinecone.")
    parser.add_argument("--dimension", type=int, default=768, help="Vector dimension for the Pinecone index.")
    parser.add_argument("--metric", default="cosine", help="Similarity metric to use in the Pinecone index.")
    parser.add_argument("--cloud", default="aws", help="Cloud provider hosting the Pinecone database.")
    parser.add_argument("--region", default="us-east-1", help="Region where the Pinecone index is hosted.")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for upserting data to Pinecone.")
    parser.add_argument("--show_progress", type=bool, default=True, help="Show progress bar while upserting data.")
    parser.add_argument("--tracing_project_name", type=str, help="Name of the tracing project.")
    parser.add_argument("--weave_params", type=str, help="JSON string of parameters for configuring Weave.")

    args = parser.parse_args()

    # Parse parameters
    text_splitter_params = literal_eval(args.text_splitter_params) if args.text_splitter_params else {}
    generator_params = literal_eval(args.generator_llm_params) if args.generator_llm_params else {}
    proxy_headers = literal_eval(args.proxy_headers) if args.proxy_headers else {}
    additional_headers = literal_eval(args.additional_headers) if args.additional_headers else {}
    embedding_model_params = literal_eval(args.embedding_model_params) if args.embedding_model_params else {}
    weave_params = literal_eval(args.weave_params) if args.weave_params else {}

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

    if args.dataloader not in dataloader_map:
        msg = f"Invalid dataloader name '{args.dataloader}'. Available options are: {', '.join(dataloader_map.keys())}"
        raise ValueError(msg)

    # Initialize the dataloader
    dataloader_cls = dataloader_map[args.dataloader]
    dataloader_kwargs = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "text_splitter": args.text_splitter,
        "text_splitter_params": text_splitter_params,
    }

    # Only add generator if instantiated
    if generator:
        dataloader_kwargs["answer_summary_generator"] = generator

    dataloader = dataloader_cls(**dataloader_kwargs)

    # Load data and preprocess
    dataloader.load_data()
    haystack_documents = dataloader.get_haystack_documents()

    # Create embeddings
    embedder = SentenceTransformersDocumentEmbedder(model=args.embedding_model, **embedding_model_params)
    embedder.warm_up()
    docs_with_embeddings = embedder.run(documents=haystack_documents)["documents"]

    # Initialize Pinecone
    pinecone_vector_db = PineconeVectorDB(
        api_key=args.api_key,
        host=args.host,
        proxy_url=args.proxy_url,
        proxy_headers=proxy_headers,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_verify=args.ssl_verify,
        additional_headers=additional_headers,
        pool_threads=args.pool_threads,
        index_name=args.index_name,
        tracing_project_name=args.tracing_project_name,
        weave_params=weave_params,
    )
    pinecone_vector_db.create_index(
        index_name=args.index_name,
        dimension=args.dimension,
        metric=args.metric,
        spec=ServerlessSpec(cloud=args.cloud, region=args.region),
    )

    # Prepare and upsert data into Pinecone
    data_for_pinecone = PineconeDocumentConverter.prepare_haystack_documents_for_upsert(docs_with_embeddings)
    pinecone_vector_db.upsert(
        data=data_for_pinecone, namespace=args.namespace, batch_size=args.batch_size, show_progress=args.show_progress
    )


if __name__ == "__main__":
    main()
