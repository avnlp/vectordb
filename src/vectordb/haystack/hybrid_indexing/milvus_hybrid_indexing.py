import argparse
from ast import literal_eval

from dataloaders import ARCDataloader, FactScoreDataloader, PopQADataloader, TriviaQADataloader
from dataloaders.llms import ChatGroqGenerator
from sklearn.feature_extraction.text import TfidfVectorizer

from vectordb import MilvusVectorDB


def main():
    """Process data, generate sparse embeddings, and index them into a Milvus vector database.

    This script is designed to:
    - Load datasets using specified dataloaders.
    - Optionally use an LLM-based generator for answer summarization.
    - Split and preprocess text data.
    - Generate sparse embeddings for documents.
    - Index processed data into a Milvus vector database.
    """
    # Argument parser for user inputs
    parser = argparse.ArgumentParser(description="Script for processing and indexing data with Milvus.")

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

    # Milvus VectorDB parameters
    parser.add_argument("--milvus_host", default="localhost", help="Milvus server host.")
    parser.add_argument("--milvus_port", default="19530", help="Milvus server port.")
    parser.add_argument("--collection_name", required=True, help="Name of the collection to interact with.")
    parser.add_argument("--dimension", type=int, required=True, help="Dimensionality of the sparse vectors.")
    parser.add_argument("--metric_type", default="IP", help="Metric type for similarity search (default: 'IP').")

    # Sparse Index parameters
    parser.add_argument("--index_name", default="sparse_index", help="Name of the sparse index.")
    parser.add_argument("--drop_ratio_build", type=float, default=0.2, help="Drop ratio for small vector values.")

    args = parser.parse_args()

    # Parse parameters
    text_splitter_params = literal_eval(args.text_splitter_params) if args.text_splitter_params else {}
    generator_params = literal_eval(args.generator_llm_params) if args.generator_llm_params else {}

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

    # Load data and preprocess
    dataloader.load_data()
    haystack_documents = dataloader.get_haystack_documents()

    # Prepare corpus for sparse embedding
    corpus = [doc.content for doc in haystack_documents]

    # Generate sparse embeddings using TF-IDF
    vectorizer = TfidfVectorizer(max_features=args.dimension)
    sparse_embeddings = vectorizer.fit_transform(corpus).toarray()

    # Initialize Milvus VectorDB
    milvus_vectordb = MilvusVectorDB(host=args.milvus_host, port=args.milvus_port)

    # Create the collection in Milvus
    milvus_vectordb.create_collection(
        collection_name=args.collection_name,
        dimension=args.dimension,
        metric_type=args.metric_type,
        description="Collection for sparse indexing.",
        is_sparse=True  
    )

    # Add the sparse index
    milvus_vectordb.add_index(
        collection_name=args.collection_name,
        field_name="sparse",
        index_name=args.index_name,
        index_type="SPARSE_INVERTED_INDEX",
        metric_type=args.metric_type,
        params={"drop_ratio_build": args.drop_ratio_build}
    )

    # Insert the sparse vectors into Milvus
    milvus_vectordb.insert_vectors(sparse_embeddings)

    print(f"Indexed {len(sparse_embeddings)} sparse vectors into the collection: {args.collection_name}")


if __name__ == "__main__":
    main()

