import argparse

from dataloaders import (
    ARCDataloader,
    EdgarDataloader,
    FactScoreDataloader,
    PopQADataloader,
    TriviaQADataloader,
)
from dataloaders.llms import ChatGroqGenerator
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)

from vectordb import ChromaDocumentConverter, ChromaVectorDB


def main():
    """Hybrid embedding upsert and retrieval using Chroma.

    This script:
    - Loads data using specified dataloaders and optionally generates summaries.
    - Embeds documents using a dense embedding model.
    - Upserts documents into Chroma collections.
    - Retrieves and prints results for a given query from two collections.
    """
    # Argument parser for user inputs
    parser = argparse.ArgumentParser(
        description="Script for processing and indexing data with Chroma."
    )

    # Dataloader parameters
    parser.add_argument(
        "--dataloader",
        required=True,
        choices=["triviaqa", "arc", "popqa", "factscore", "edgar"],
        help="Dataloader to use for loading datasets.",
    )
    parser.add_argument(
        "--dataset_name",
        required=True,
        help="Name of the dataset to be used by the dataloader.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to process (e.g., 'test', 'train').",
    )
    parser.add_argument(
        "--text_splitter",
        default="RecursiveCharacterTextSplitter",
        help="Text splitter method to preprocess documents.",
    )
    parser.add_argument(
        "--text_splitter_params",
        type=str,
        help="JSON string of parameters for configuring the text splitter.",
    )

    # Generator parameters
    parser.add_argument(
        "--generator_model", type=str, help="Model name for the dataloader's generator."
    )
    parser.add_argument(
        "--generator_api_key", help="API key for the dataloader generator."
    )
    parser.add_argument(
        "--generator_llm_params",
        type=str,
        help="JSON string of parameters for the generator LLM.",
    )

    # Embedder parameters
    parser.add_argument(
        "--embedding_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model to use for generating document embeddings.",
    )
    parser.add_argument(
        "--embedding_model_params",
        type=str,
        help="JSON string of parameters for the embedding model.",
    )

    # Chroma VectorDB arguments
    parser.add_argument(
        "--chroma_path",
        default="./chroma_database_files",
        help="Path for Chroma database files.",
    )
    parser.add_argument(
        "--collection_name",
        default="test_collection_dense1",
        help="Name of the Chroma collection.",
    )
    parser.add_argument(
        "--tracing_project_name",
        default="chroma",
        help="Name of the Weave project for tracing.",
    )
    parser.add_argument(
        "--weave_params",
        type=str,
        help="JSON string of parameters for initializing Weave.",
    )

    args = parser.parse_args()

    # Initialize the generator
    generator_params = literal_eval(args.generator_llm_params)
    generator = ChatGroqGenerator(
        model=args.generator_model,
        api_key=args.generator_api_key,
        llm_params=generator_params,
    )

    # Map dataloader names to classes
    dataloader_map = {
        "triviaqa": TriviaQADataloader,
        "arc": ARCDataloader,
        "popqa": PopQADataloader,
        "factscore": FactScoreDataloader,
        "edgar": EdgarDataloader,
    }

    dataloader_cls = dataloader_map[args.dataloader]
    dataloader = dataloader_cls(
        dataset_name=args.dataset_name,
        split=args.split,
        answer_summary_generator=generator,
    )

    # Load the data
    dataloader.load_data()
    dataloader.get_questions()
    dataloader.get_haystack_documents()

    # Chroma collection arguments
    parser.add_argument(
        "--collection_name1",
        type=str,
        required=True,
        help="Name of the first Chroma collection.",
    )
    parser.add_argument(
        "--collection_name2",
        type=str,
        required=True,
        help="Name of the second Chroma collection.",
    )

    # Embedding model arguments
    parser.add_argument(
        "--dense_model", type=str, required=True, help="Dense embedding model name."
    )

    parser.add_argument(
        "--dataset_name",
        required=True,
        help="Name of the dataset to be used by the dataloader.",
    )
    parser.add_argument(
        "--split1", required=True, help="Dataset split for the first collection."
    )
    parser.add_argument(
        "--split2", required=True, help="Dataset split for the second collection."
    )
    parser.add_argument(
        "--tracing_project_name",
        default="chroma",
        help="Name of the Weave project for tracing.",
    )
    parser.add_argument(
        "--weave_params",
        type=str,
        help="JSON string of parameters for initializing Weave.",
    )

    # Generator arguments
    parser.add_argument(
        "--generator_model", type=str, help="Model name for the dataloader's generator."
    )
    parser.add_argument(
        "--generator_api_key", help="API key for the dataloader generator."
    )
    parser.add_argument(
        "--generator_llm_params",
        type=str,
        help="JSON string of parameters for the generator LLM.",
    )

    # Query arguments
    parser.add_argument(
        "--question", type=str, required=True, help="Query/question text."
    )
    parser.add_argument(
        "--top_k", type=int, default=10, help="Number of top results to retrieve."
    )

    args = parser.parse_args()

    # Initialize the generator if provided
    generator = None
    if args.generator_model and args.generator_api_key:
        generator_params = (
            eval(args.generator_llm_params) if args.generator_llm_params else {}
        )
        generator = ChatGroqGenerator(
            model=args.generator_model,
            api_key=args.generator_api_key,
            llm_params=generator_params,
        )

    # Map dataloader names to classes
    dataloader_map = {
        "triviaqa": TriviaQADataloader,
        "arc": ARCDataloader,
        "popqa": PopQADataloader,
        "factscore": FactScoreDataloader,
        "edgar": EdgarDataloader,
    }
    dataloader_cls = dataloader_map[args.dataloader]

    # Load data for both splits
    dataloader_split1 = dataloader_cls(
        dataset_name=args.dataset_name,
        split=args.split1,
        answer_summary_generator=generator,
    )
    dataloader_split2 = dataloader_cls(
        dataset_name=args.dataset_name,
        split=args.split2,
        answer_summary_generator=generator,
    )

    dataloader_split1.load_data()
    dataloader_split2.load_data()

    haystack_documents_split1 = dataloader_split1.get_haystack_documents()
    haystack_documents_split2 = dataloader_split2.get_haystack_documents()

    # Initialize and warm up the embedding model
    embedder = SentenceTransformersDocumentEmbedder(model=args.dense_model)
    embedder.warm_up()

    # Generate embeddings for both splits
    docs_with_embeddings_split1 = embedder.run(documents=haystack_documents_split1)[
        "documents"
    ]
    docs_with_embeddings_split2 = embedder.run(documents=haystack_documents_split2)[
        "documents"
    ]

    # Initialize Chroma VectorDB
    weave_params = literal_eval(args.weave_params) if args.weave_params else {}
    chroma_vector_db = ChromaVectorDB(
        persistent=True,
        path=args.chroma_path,
        tracing_project_name=args.tracing_project_name,
        weave_params=weave_params,
    )
    chroma_vector_db.create_collection(name=args.collection_name)

    # Prepare data for Chroma
    docs_for_chroma_split1 = (
        ChromaDocumentConverter.prepare_haystack_documents_for_upsert(
            docs_with_embeddings_split1
        )
    )
    docs_for_chroma_split2 = (
        ChromaDocumentConverter.prepare_haystack_documents_for_upsert(
            docs_with_embeddings_split2
        )
    )

    # Initialize Chroma and upsert data
    weave_params = literal_eval(args.weave_params) if args.weave_params else {}
    chroma_vector_db = ChromaVectorDB(
        persistent=True,
        path=args.chroma_path,
        tracing_project_name=args.tracing_project_name,
        weave_params=weave_params,
    )

    chroma_vector_db.create_collection(name=args.collection_name1)
    chroma_vector_db.upsert(
        data=docs_for_chroma_split1, collection_name=args.collection_name1
    )

    chroma_vector_db.create_collection(name=args.collection_name2)
    chroma_vector_db.upsert(
        data=docs_for_chroma_split2, collection_name=args.collection_name2
    )

    # Query Chroma collections
    text_embedder = SentenceTransformersTextEmbedder(model=args.dense_model)
    text_embedder.warm_up()

    dense_question_embedding = text_embedder.run(text=args.question)["embedding"]

    # Query collection 1
    query_response_split1 = chroma_vector_db.query(
        query_embedding=dense_question_embedding,
        n_results=args.top_k,
        collection_name=args.collection_name1,
    )
    retrieval_results_split1 = (
        ChromaDocumentConverter.convert_query_results_to_haystack_documents(
            query_response_split1
        )
    )
    print("Results from collection 1:")
    print(retrieval_results_split1)

    # Query collection 2
    query_response_split2 = chroma_vector_db.query(
        query_embedding=dense_question_embedding,
        n_results=args.top_k,
        collection_name=args.collection_name2,
    )
    retrieval_results_split2 = (
        ChromaDocumentConverter.convert_query_results_to_haystack_documents(
            query_response_split2
        )
    )
    print("Results from collection 2:")
    print(retrieval_results_split2)


if __name__ == "__main__":
    main()
