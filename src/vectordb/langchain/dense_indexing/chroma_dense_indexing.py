import argparse

from dataloaders import ARCDataloader, EdgarDataloader, FactScoreDataloader, PopQADataloader, TriviaQADataloader
from dataloaders.llms import ChatGroqGenerator
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from vectordb import ChromaDocumentConverter, ChromaVectorDB


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Run a query using Chroma VectorDB with HuggingFace embeddings.")
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
        help="Chroma collection name (default is 'test_collection_dense1').",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="./chroma_database_files",
        help="Path to store the Chroma database (default is './chroma_database_files').",
    )
    parser.add_argument("--tracing_project_name", default="weaviate", help="Name of the Weave project for tracing.")
    parser.add_argument("--weave_params", type=str, help="JSON string of additional Weave parameters.")
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

    # Initialize Chroma VectorDB
    chroma_vector_db = ChromaVectorDB(persistent=True, path=args.db_path)

    # Create a collection in Chroma
    chroma_vector_db.create_collection(name=args.collection_name)

    # Prepare the data for upsert
    data_for_chroma = ChromaDocumentConverter.prepare_langchain_documents_for_upsert(langchain_documents, embeddings)

    # Upsert the data into Chroma
    chroma_vector_db.upsert(data=data_for_chroma)


if __name__ == "__main__":
    main()

